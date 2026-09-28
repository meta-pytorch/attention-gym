"""Check which Attention Gym cuDNN fixes are still needed after an upstream rebase.

Reads fixes.toml, the authoritative ledger. --write-ledger regenerates the compact
MAINTENANCE.md index. Two independent verification modes:

  --tree [PATH]          Run every fix's guarding pytest node ids inside an Attention Gym checkout
                         (bare flag: this repo, with the current interpreter; another checkout
                         uses its own .venv unless --tree-python is given). Reports
                         pass/fail/missing per fix.
  --upstream-python PY   Run each upstream repro (upstream/repro_*.py) against the stock cudnn
                         frontend installed in PY's environment. Reports per fix whether the bug
                         is still present upstream or looks fixed. --upstream-pythonpath prepends
                         a patched package tree as a positive control.

Every run is wrapped in `timeout -k 10 <T>`; --gpu-run additionally prefixes
`gpu-run --timeout 900 auto --` (or a custom prefix) to reserve a GPU per run. Without it, wrap
the whole command in gpu-run yourself. Writes markdown to stdout and JSON plus per-run logs under
agent_space/cudnn_fe_verify/ (gitignored) unless --out/--logdir say otherwise.

Examples (from the repo root):
  gpu-run --timeout 900 auto -- python tools/cudnn_fe/verify_fixes.py --tree .
  python tools/cudnn_fe/verify_fixes.py --gpu-run --upstream-python <env>/bin/python
  python tools/cudnn_fe/verify_fixes.py --tree . --only B6 B7 --no-stress
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import tomllib

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIXES = HERE / "fixes.toml"
OUT_DIR = REPO_ROOT / "agent_space" / "cudnn_fe_verify"
GPU_RUN = "gpu-run --timeout 900 auto --"
GPU_BUSY = 75  # gpu-run: every GPU reserved
TIMEOUT_CODES = {124, 137}

# A failing repro that printed one of these reproduced the bug (assertion-style failure).
# Anything else nonzero (ImportError, TypeError, crash) is reported as an error to inspect:
# after a rebase that usually means the upstream API moved and the repro needs updating.
BUG_PATTERNS = [
    r"AssertionError",
    r"^FAIL",
    r"^RESULT:? .*\b[1-9]\d* failed",
    r"Tensor-likes are not equal",
]


def load_fixes(only: set[str] | None = None, path: Path = FIXES) -> list[dict]:
    """Ledger fixes plus superseded guards, each tagged with its group."""
    data = tomllib.loads(path.read_text())
    entries = [dict(f, group="fix") for f in data["fix"]]
    entries += [
        dict(f, group="superseded", upstream_status="superseded-by-v1.30", kind="superseded")
        for f in data.get("superseded", [])
    ]
    if only:
        entries = [f for f in entries if f["id"] in only]
    return entries


LEDGER_START = "<!-- BEGIN GENERATED FIX LEDGER -->"
LEDGER_END = "<!-- END GENERATED FIX LEDGER -->"


def maintenance_path() -> Path:
    spec = importlib.util.find_spec("attn_gym")
    assert spec is not None and spec.origin is not None, "install Attention Gym first"
    return Path(spec.origin).parent / "linear/_delta_rule/cudnn_fe/MAINTENANCE.md"


def ledger_table() -> str:
    """Render the documentation index from the authoritative machine ledger."""
    fixes = [f for f in load_fixes() if f["group"] == "fix"]
    lines = [
        LEDGER_START,
        f"<!-- Generated from tools/cudnn_fe/fixes.toml: {len(fixes)} rows. -->",
        "| ID | Title | Guarding test / gate |",
        "|---|---|---|",
    ]
    for fix in fixes:
        guards = [f"`{node}`" for node in fix["guarding_tests"]]
        if fix.get("gate"):
            guards.append(fix["gate"])
        cells = [fix["id"], fix["title"], "; ".join(guards)]
        lines.append("| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |")
    return "\n".join([*lines, LEDGER_END])


def write_ledger() -> None:
    path = maintenance_path()
    text = path.read_text()
    before, rest = text.split(LEDGER_START, 1)
    _, after = rest.split(LEDGER_END, 1)
    path.write_text(before + ledger_table() + after)


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict | None = None, log: Path | None = None
) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if log is not None:
        log.write_text(f"$ {shlex.join(cmd)}\n{proc.stdout}\nEXIT_STATUS={proc.returncode}\n")
    return proc.returncode, proc.stdout


# ------------------------------------------------------------------------------------ tree


def collect(python: Path | str, tree: Path, files: list[str]) -> set[str]:
    """Collected node ids for the given test files (CPU only; no GPU reservation)."""
    _, out = run(
        [str(python), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *files],
        cwd=tree,
    )
    return {line.rstrip() for line in out.splitlines() if "::" in line and not line[0].isspace()}


def node_matches(node: str, collected: set[str]) -> list[str]:
    """Collected ids for a node id, including its parametrizations."""
    return [c for c in collected if c == node or c.startswith(node + "[")]


def wrap(args, seconds: int) -> list[str]:
    return [*shlex.split(args.gpu_run or ""), "timeout", "-k", "10", str(seconds)]


def tree_python(args, tree: Path) -> Path:
    if args.tree_python:
        return args.tree_python
    venv = tree / ".venv/bin/python"
    if tree == REPO_ROOT or not venv.exists():
        return Path(sys.executable)
    return venv


def junit_outcomes(xml_path: Path) -> dict[str, str]:
    """Map collected node id (with params) -> passed/failed/error/skipped."""
    outcomes = {}
    if not xml_path.exists():
        return outcomes
    for case in ET.parse(xml_path).iter("testcase"):
        cls, name = case.get("classname", ""), case.get("name", "")
        path = cls.replace(".", "/") + ".py"
        node = f"{path}::{name}"
        state = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                state = "failed" if child.tag == "failure" else "error"
                break
            if child.tag == "skipped":
                state = "skipped"
        outcomes[node] = state
    return outcomes


def run_tree(args, fixes: list[dict], logdir: Path) -> dict:
    tree = args.tree.resolve()
    python = tree_python(args, tree)
    assert python.exists(), f"no interpreter at {python}"
    rev = run(["git", "-C", str(tree), "log", "-1", "--format=%h %s"])[1].strip()
    branch = run(["git", "-C", str(tree), "branch", "--show-current"])[1].strip()

    wanted = {n for f in fixes for n in f.get("guarding_tests", [])}
    files = sorted({n.split("::")[0] for n in wanted if (tree / n.split("::")[0]).exists()})
    collected = collect(python, tree, files) if files else set()

    def matches(node: str) -> list[str]:
        return node_matches(node, collected)

    missing = {n for n in wanted if not matches(n)}
    # Group runnable nodes by extra env so stress tests get ATTN_GYM_RUN_STRESS_TESTS=1.
    groups: dict[tuple, set[str]] = {}
    for f in fixes:
        extra = {} if args.no_stress else f.get("env", {})
        key = tuple(sorted(extra.items()))
        groups.setdefault(key, set()).update(
            n for n in f.get("guarding_tests", []) if n not in missing
        )
    # A node wanted both with and without stress env only needs the stress run.
    stress_nodes = set().union(*(v for k, v in groups.items() if k)) if groups else set()
    if () in groups:
        groups[()] -= stress_nodes

    outcomes: dict[str, str] = {}
    runs = []
    for i, (key, nodes) in enumerate(sorted(groups.items())):
        if not nodes:
            continue
        env = dict(os.environ, **dict(key))
        xml = logdir / f"tree_{i}.xml"
        xdist = ["-n", str(args.workers)] if args.workers > 0 and len(nodes) > 1 else []
        cmd = [
            *wrap(args, args.test_timeout),
            str(python),
            "-m",
            "pytest",
            *xdist,
            "-rA",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
            *sorted(nodes),
        ]
        log = logdir / f"tree_{i}.log"
        print(f"[tree] running {len(nodes)} node ids env={dict(key)} -> {log}", file=sys.stderr)
        code, _ = run(cmd, cwd=tree, env=env, log=log)
        runs.append({"env": dict(key), "nodes": len(nodes), "exit": code, "log": str(log)})
        outcomes.update(junit_outcomes(xml))
        if code == GPU_BUSY or code in TIMEOUT_CODES:
            print(f"[tree] run {i} exit {code} (gpu busy / timeout)", file=sys.stderr)

    results = []
    for f in fixes:
        per_node = {}
        for node in f.get("guarding_tests", []):
            if node in missing:
                per_node[node] = "missing"
                continue
            states = [outcomes.get(c, "not-run") for c in matches(node)]
            if any(s in ("failed", "error") for s in states):
                per_node[node] = "failed"
            elif "not-run" in states:
                per_node[node] = "not-run"
            elif all(s == "skipped" for s in states):
                per_node[node] = "skipped"
            else:
                per_node[node] = "passed"
            per_node[node] += f" ({states.count('passed')}/{len(states)})"
        verdicts = [v.split()[0] for v in per_node.values()]
        if not verdicts:
            verdict = "no pytest guard"
        elif "failed" in verdicts:
            verdict = "FAIL"
        elif "missing" in verdicts:
            verdict = "MISSING (test renamed/removed?)"
        elif "not-run" in verdicts:
            verdict = "NOT-RUN (gpu busy/timeout?)"
        elif all(v == "skipped" for v in verdicts):
            verdict = "skipped"
        else:
            verdict = "pass"
        results.append(
            {
                "id": f["id"],
                "group": f["group"],
                "title": f["title"],
                "verdict": verdict,
                "tests": per_node,
                "gate": f.get("gate"),
            }
        )
    return {
        "tree": str(tree),
        "python": str(python),
        "branch": branch,
        "head": rev,
        "runs": runs,
        "fixes": results,
    }


# -------------------------------------------------------------------------------- upstream


def classify(expect: str, code: int, output: str, patterns: list[str]) -> str:
    if code == GPU_BUSY:
        return "gpu-unavailable"
    if code in TIMEOUT_CODES:
        return "timeout"
    if expect in ("smoke", "scope"):
        return "pass" if code == 0 else "error"
    if code == 0:
        return "fixed"
    if any(re.search(p, output, re.MULTILINE) for p in BUG_PATTERNS + patterns):
        return "bug"
    return "error"


UPSTREAM_VERDICT = {
    ("bug", "bug"): "bug still present (keep our fix / file upstream)",
    ("bug", "fixed"): "fixed upstream (can drop ours after confirming)",
    ("smoke", "pass"): "no isolated bug (hardening; smoke passes on stock)",
    ("smoke", "error"): "smoke FAILED on stock upstream: investigate",
    ("scope", "pass"): "upstream still lacks the ABI (AG-only fix stays; nothing to file)",
    ("scope", "error"): "scope check changed: upstream may now support it, re-evaluate",
}


def run_upstream(args, fixes: list[dict], logdir: Path) -> dict:
    python = args.upstream_python
    env = dict(os.environ)
    if args.upstream_pythonpath:
        env["PYTHONPATH"] = str(args.upstream_pythonpath.resolve())
    code, version = run(
        [str(python), "-c", "import cudnn; print(cudnn.__version__, cudnn.__file__)"], env=env
    )
    version = version.strip().splitlines()[-1] if code == 0 else f"import failed: {version[-300:]}"
    results = []
    for f in fixes:
        repro = f.get("upstream_repro")
        if not repro:
            continue
        expect = f.get("upstream_expect", "bug")
        run_states = []
        for j, extra in enumerate(f.get("upstream_runs", [[]])):
            log = logdir / f"upstream_{Path(repro).stem}_{j}.log"
            cmd = [
                *wrap(args, args.repro_timeout),
                str(python),
                str(HERE / repro),
                *extra,
            ]
            print(
                f"[upstream] {f['id']}: {Path(repro).name} {' '.join(extra)} -> {log}",
                file=sys.stderr,
            )
            code, out = run(cmd, cwd=HERE, env=env, log=log)
            state = classify(expect, code, out, f.get("upstream_bug_patterns", []))
            run_states.append({"args": extra, "exit": code, "state": state, "log": str(log)})
        states = [r["state"] for r in run_states]
        if expect == "bug":
            state = (
                "bug"
                if "bug" in states
                else "fixed"
                if all(s == "fixed" for s in states)
                else next(s for s in states if s != "fixed")
            )
        else:
            state = (
                "pass"
                if all(s == "pass" for s in states)
                else next(s for s in states if s != "pass")
            )
        verdict = UPSTREAM_VERDICT.get((expect, state), f"{state}: inspect logs")
        results.append(
            {
                "id": f["id"],
                "patch": f.get("upstream_patch"),
                "repro": repro,
                "expect": expect,
                "state": state,
                "verdict": verdict,
                "runs": run_states,
            }
        )
    return {"python": str(python), "cudnn": version, "results": results}


# ---------------------------------------------------------------------------------- report


def report(tree_res: dict | None, up_res: dict | None) -> str:
    lines = [
        f"# verify_fixes.py report ({dt.datetime.now(dt.UTC).isoformat(timespec='seconds')})",
        "",
    ]
    if tree_res:
        fx = tree_res["fixes"]
        counts: dict[str, int] = {}
        for r in fx:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        lines += [
            f"## Guarding tests: `{tree_res['tree']}`",
            "",
            (
                f"branch `{tree_res['branch']}` @ `{tree_res['head']}`, python "
                f"`{tree_res['python']}`"
            ),
            "",
            "Summary: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
            "",
            "| ID | group | verdict | tests (passed/collected) | non-pytest gate |",
            "|---|---|---|---|---|",
        ]
        for r in fx:
            tests = "<br>".join(f"`{n.split('::')[-1]}` {s}" for n, s in r["tests"].items())
            lines.append(
                f"| {r['id']} | {r['group']} | **{r['verdict']}** | {tests or '-'} | "
                f"{r['gate'] or ''} |"
            )
        lines += [
            "",
            "pytest runs: "
            + "; ".join(
                f"env={x['env']} nodes={x['nodes']} exit={x['exit']}" for x in tree_res["runs"]
            ),
            "",
        ]
    if up_res:
        lines += [
            f"## Upstream repros: `{up_res['python']}`",
            "",
            f"cudnn: `{up_res['cudnn']}`",
            "",
            "| fix | upstream patch | repro | runs (exit/state) | verdict |",
            "|---|---|---|---|---|",
        ]
        for r in up_res["results"]:
            runs = "<br>".join(
                f"`{' '.join(x['args']) or '(default)'}` {x['exit']}/{x['state']}"
                for x in r["runs"]
            )
            patch = Path(r["patch"]).name if r["patch"] else "none (excluded)"
            lines.append(
                f"| {r['id']} | {patch} | {Path(r['repro']).name} | {runs} | **{r['verdict']}** |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--tree",
        type=Path,
        nargs="?",
        const=REPO_ROOT,
        default=None,
        help="Attention Gym checkout to test (bare flag: this repo)",
    )
    p.add_argument(
        "--tree-python",
        type=Path,
        help="interpreter for --tree (default: current one for this repo, else <tree>/.venv)",
    )
    p.add_argument(
        "--upstream-python",
        type=Path,
        help="python of an env with the stock upstream cudnn frontend installed",
    )
    p.add_argument(
        "--upstream-pythonpath",
        type=Path,
        help="prepend a patched cudnn package tree (<patched checkout>/python) as a positive "
        "control; B10 stays 'bug' because patch 04 is opt-in (--skip-empty)",
    )
    p.add_argument(
        "--gpu-run",
        nargs="?",
        const=GPU_RUN,
        default=None,
        help=f"prefix each run with a GPU reservation (bare flag: '{GPU_RUN}')",
    )
    p.add_argument("--only", nargs="+", help="restrict to these fix ids")
    p.add_argument(
        "--no-stress",
        action="store_true",
        help="do not set ATTN_GYM_RUN_STRESS_TESTS=1 (stress tests then skip)",
    )
    p.add_argument("-n", "--workers", type=int, default=6, help="pytest-xdist workers")
    p.add_argument("--test-timeout", type=int, default=1800, help="seconds per pytest group")
    p.add_argument("--repro-timeout", type=int, default=600, help="seconds per repro run")
    p.add_argument("--out", type=Path, default=OUT_DIR / "results.json")
    p.add_argument("--logdir", type=Path, default=OUT_DIR / "logs")
    p.add_argument("--write-ledger", action="store_true", help="regenerate the MAINTENANCE index")
    args = p.parse_args()
    if args.write_ledger:
        write_ledger()
        return 0
    if args.tree is None and args.upstream_python is None:
        p.error("pass --tree and/or --upstream-python")
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixes = load_fixes(set(args.only) if args.only else None)

    tree_res = run_tree(args, fixes, args.logdir) if args.tree else None
    up_res = run_upstream(args, fixes, args.logdir) if args.upstream_python else None
    args.out.write_text(json.dumps({"tree": tree_res, "upstream": up_res}, indent=2))
    print(report(tree_res, up_res))
    bad = [
        r
        for r in (tree_res or {}).get("fixes", [])
        if r["verdict"] not in ("pass", "no pytest guard")
    ]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
