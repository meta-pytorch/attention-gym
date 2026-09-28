"""Vendor the cudnn-frontend GDN + KDA kernel closure at one upstream tag, verbatim.

Modes (see MAINTENANCE.md in the vendored package for the full upgrade procedure)::

    # Vendor a tag into a scratch dir (default agent_space/cudnn_fe_vendor/<rev>).
    python -m tools.cudnn_fe.vendor --rev v1.31.0 --upstream <cudnn-frontend clone>
    # Check the machinery: regenerate v1.30.0 and compare with 88eb5ce's verbatim drop.
    python -m tools.cudnn_fe.vendor --rev v1.30.0 --upstream <clone> --verify 88eb5ce
    # Per-file upstream churn of the closure between two tags.
    python -m tools.cudnn_fe.vendor --upstream <clone> --diff-upstream v1.30.0 v1.31.0

Output: ``{kernel,common,tile_dsl}/`` from the upstream closure of the modules the Attention Gym
drivers import, with ``cudnn.frost.*`` imports relocated into the package, a modification notice
after each file's leading comment block, the upstream license files and ``_compat.py``. The
documented GDP fork cut is applied whenever the chain prologue reaches it. Nothing is written into
``attn_gym/`` unless ``--dest`` names it; symlinked destinations are refused.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.cudnn_fe import closure

LICENSE_FILES = (
    "LICENSE.txt",
    "LICENSE-MIT.txt",
    "NOTICE",
    "LICENSING.md",
    "THIRD_PARTY_LICENSES.txt",
)
NOTICE = (
    "# Modified by Attention Gym in 2026: vendored from cudnn-frontend {rev}; imports relocated into\n"
    "# {package}.\n"
)
# Torch-backed replacements for the cudnn.frost host utilities v1.30 kernels reach, byte-identical
# to 88eb5ce's (its docstring names that drop's script); written into scratch output so it
# imports. The behavior layer deletes the host code that needs it.
COMPAT = '''# SPDX-License-Identifier: BSD-3-Clause
"""Torch-backed replacements for the cudnn.frost host utilities the vendored kernels reach.

Surface required by the v1.30 kernel/common closure (see import_closure.py):
cudnn.frost.buffers.{DeviceView, probe}, cudnn.frost.device.{current_device, multiprocessor_count}.
"""

from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def DeviceView(ptr: int, shape, dtype: str, device_id: int) -> torch.Tensor:
    """Compile-time placeholder tensor (upstream wraps a raw pointer; only shape/dtype matter)."""
    del ptr
    return torch.empty(
        tuple(int(s) for s in shape), dtype=_DTYPES[dtype], device=f"cuda:{device_id}"
    )


def probe(buf: torch.Tensor):
    """(ptr, shape, strides_in_elements, dtype_name, device_id), matching cudnn.frost.buffers.probe."""
    name = {v: k for k, v in _DTYPES.items()}[buf.dtype]
    return buf.data_ptr(), tuple(buf.shape), tuple(buf.stride()), name, buf.device.index


def current_device() -> int:
    return torch.cuda.current_device()


def multiprocessor_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
'''


def relocate(text: str, label: str) -> str:
    """Rewrite absolute ``cudnn.frost`` imports to package-relative ones; fail on any other."""
    text = re.sub(r"\bfrom cudnn\.frost\.tile_dsl\.", "from ..tile_dsl.", text)
    text = re.sub(r"\bfrom cudnn\.frost\.tile_dsl import\b", "from ..tile_dsl import", text)
    text = re.sub(
        r"\bfrom cudnn\.frost\.(?:buffers|device) import\b", "from .._compat import", text
    )
    leftover = re.findall(r"^\s*(?:from|import) cudnn\b.*$", text, re.MULTILINE)
    if leftover:
        raise ValueError(f"{label}: unhandled cudnn import(s): {leftover}")
    return text


def add_notice(text: str, rev: str, package: str = closure.PACKAGE) -> str:
    """Insert the modification notice after the leading ``#`` comment block (SPDX/license)."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    notice = "#\n" + NOTICE.format(rev=rev, package=package)
    return "".join(lines[:i]) + notice + "".join(lines[i:])


def subpackage_init(package: str, rev: str) -> str:
    return f'# SPDX-License-Identifier: Apache-2.0\n"""Vendored cudnn-frontend {rev} {package} subset."""\n'


def prune_gdp_fork(text: str) -> str:
    """Cut the compact_qdo (GDP d_v=64) fork out of kernel/gdn_chain_prologue_f16.py.

    The fork is a constexpr branch: drop the module import and replace the branch body with a
    trace-time error, so every scalar-GDN path traces unchanged. Fails if the upstream shape moved.
    """
    text = text.replace(", gdp_bprop_v64_f16\n", "\n", 1)
    pattern = re.compile(
        r"(?P<indent>[ ]+)if cutlass\.const_expr\(compact_qdo\):\n"
        r"(?P<body>(?:(?P=indent)[ ]+.*\n|\n)+?)"
        r"(?=(?P=indent)(?:else|elif|\S))",
    )

    def repl(m: re.Match) -> str:
        if "gdp_bprop_v64_f16" not in m.group("body"):
            return m.group(0)
        ind = m.group("indent")
        return (
            f"{ind}if cutlass.const_expr(compact_qdo):\n"
            f"{ind}    # Attention Gym: the GDP compact-Q/dO d_v=64 backward fork is not vendored.\n"
            f"{ind}    raise NotImplementedError('compact_qdo requires the GDP d_v=64 backward')\n"
        )

    text = pattern.sub(repl, text)
    if "gdp_bprop_v64_f16" in text:
        raise ValueError("prune_gdp_fork: residual gdp_bprop_v64_f16 references; cut by hand")
    return text


def compute(upstream: closure.Upstream, roots: str) -> closure.Closure:
    """Closure of the driver roots with the GDP cut, or of the engine roots 88eb5ce vendored.

    ``engines`` is only the ``--verify`` reference: gdn_engine reaches GDP directly, so that
    closure keeps the fork, as 88eb5ce did.
    """
    if roots == "engines":
        return closure.compute(upstream, list(closure.ENGINE_ROOTS), skip=closure.ENGINE_SKIP)
    return closure.compute(upstream, closure.driver_roots(), prune=True)


def check_dest(dest: Path) -> None:
    """Refuse a destination that is, or contains, a symlink: writes could land elsewhere."""
    if dest.is_symlink():
        raise SystemExit(f"refusing to write through symlinked destination {dest}")
    for top, dirs, files in os.walk(dest):
        for name in dirs + files:
            if (Path(top) / name).is_symlink():
                raise SystemExit(f"refusing to write into {dest}: {Path(top) / name} is a symlink")


def default_dest(rev: str) -> Path:
    """``agent_space/cudnn_fe_vendor/<rev>``, emptied; refuses paths that escape through links."""
    base = closure.repo_root() / "agent_space" / "cudnn_fe_vendor"
    dest = base / rev
    for path in (dest, *dest.parents):
        if path.is_symlink():
            raise SystemExit(f"refusing default destination {dest}: {path} is a symlink")
        if path == closure.repo_root():
            break
    resolved = dest.resolve()
    if not resolved.is_relative_to(base.resolve()) or resolved.is_relative_to(
        closure.package_dir().resolve()
    ):
        raise SystemExit(f"refusing default destination {dest}: resolves to {resolved}")
    if dest.exists():
        shutil.rmtree(dest)
    return dest


def vendor(upstream: closure.Upstream, dest: Path, roots: str = "drivers") -> closure.Closure:
    """Write the closure of ``roots`` at ``upstream.rev`` into ``dest``; return the closure."""
    cl = compute(upstream, roots)
    if cl.external:
        raise SystemExit(f"closure reaches unvendorable upstream modules: {sorted(cl.external)}")
    dest.mkdir(parents=True, exist_ok=True)
    check_dest(dest)
    for package in closure.SOURCES:
        (dest / package).mkdir(exist_ok=True)
        (dest / package / "__init__.py").write_text(subpackage_init(package, upstream.rev))
    for rel in sorted(cl.files):
        text = relocate(upstream.show(closure.upstream_path(rel)), rel)
        if rel == closure.PRUNE_ENTRY and cl.cut:
            text = prune_gdp_fork(text)
        (dest / rel).write_text(add_notice(text, upstream.rev))
    for name in LICENSE_FILES:
        (dest / name).write_bytes(upstream.show_bytes(name))
    if cl.compat and not (dest / "_compat.py").exists():
        (dest / "_compat.py").write_text(COMPAT)
    return cl


def report(cl: closure.Closure, rev: str, roots: str) -> None:
    print(f"{rev}: {len(cl.files)} vendored modules from the {roots} roots")
    if cl.not_upstream:
        print(f"  roots absent upstream (AG-authored or renamed): {', '.join(cl.not_upstream)}")
    nested = sorted(rel for rel in cl.files if cl.nested_only(rel))
    if nested:
        print(f"  reached only through nested imports (prune candidates): {', '.join(nested)}")
    if cl.compat:
        print(f"  host utilities relocated to _compat: {', '.join(sorted(cl.compat))}")
    if cl.cut:
        print(f"  GDP fork cut out of {closure.PRUNE_ENTRY}")


def generated_path(rel: str) -> bool:
    """Whether ``rel`` (package-relative) is a file ``vendor`` writes."""
    head = rel.split("/")
    return (len(head) == 2 and head[0] in closure.SOURCES) or rel in (*LICENSE_FILES, "_compat.py")


def verify(upstream: closure.Upstream, commit: str) -> int:
    """Regenerate the engine closure and compare with ``commit``'s drop byte for byte."""
    prefix = closure.PACKAGE_RELPATH
    listing = _git_repo("ls-tree", "-r", "--name-only", commit, "--", prefix).decode().split()
    tracked = {rel for path in listing if generated_path(rel := path[len(prefix) + 1 :])}
    roots = "engines"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        cl = vendor(upstream, out, roots)
        report(cl, upstream.rev, roots)
        generated = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        assert all(map(generated_path, generated)), sorted(generated)
        mismatched = sorted(
            rel
            for rel in generated & tracked
            if (out / rel).read_bytes() != _git_repo("show", f"{commit}:{prefix}/{rel}")
        )
    only_generated = sorted(generated - tracked)
    only_tracked = sorted(tracked - generated)
    same = len(generated & tracked) - len(mismatched)
    print(f"verify {commit}: {same} identical, {len(mismatched)} differ")
    for label, items in (
        ("differ", mismatched),
        ("only in vendor output", only_generated),
        ("only in " + commit, only_tracked),
    ):
        for rel in items:
            print(f"  {label}: {rel}")
    extra = sorted(set(cl.files) - set(compute(upstream, "drivers").files))
    if extra:
        print(f"  not reached from the current drivers: {', '.join(extra)}")
    ok = not (mismatched or only_generated or only_tracked)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def diff_upstream(clone: Path, old: str, new: str) -> None:
    """Report per-file upstream churn of the closure (union of both revisions) from old to new."""
    up_old, up_new = closure.Upstream(clone, old), closure.Upstream(clone, new)
    cl_old, cl_new = compute(up_old, "drivers"), compute(up_new, "drivers")
    paths = sorted(set(cl_old.files) | set(cl_new.files))
    numstat = {}
    upstream_paths = [closure.upstream_path(rel) for rel in paths]
    out = up_new._git("diff", "--numstat", old, new, "--", *upstream_paths)
    for line in out.splitlines():
        added, deleted, path = line.split("\t")
        numstat[path] = (added, deleted)
    rows = []
    for rel, upath in zip(paths, upstream_paths):
        in_old, in_new = rel in cl_old.files, rel in cl_new.files
        if not in_new:
            state = "left closure" if upath in up_new.files else "deleted upstream"
        elif not in_old:
            state = "joined closure" if upath in up_old.files else "added upstream"
        else:
            state = "modified" if upath in numstat else "unchanged"
        added, deleted = numstat.get(upath, ("0", "0"))
        rows.append((state, rel, added, deleted))
    print(f"closure churn {old} -> {new} (driver roots; +/- are upstream lines)")
    print(f"| {'file':44} | {'state':16} | {'+':>6} | {'-':>6} |")
    print(f"|{'-' * 46}|{'-' * 18}|{'-' * 8}|{'-' * 8}|")
    changed = [row for row in rows if row[0] != "unchanged"]
    for state, rel, added, deleted in changed:
        print(f"| {rel:44} | {state:16} | {added:>6} | {deleted:>6} |")
    print(f"{len(changed)}/{len(rows)} closure files changed; the rest are byte-identical")
    if cl_old.compat != cl_new.compat:
        gone, new_names = cl_old.compat - cl_new.compat, cl_new.compat - cl_old.compat
        print(f"_compat surface: -{sorted(gone)} +{sorted(new_names)}")
    if cl_new.external:
        print(f"new closure reaches unvendorable modules: {sorted(cl_new.external)}")


def _git_repo(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(closure.repo_root()), *args], capture_output=True, check=True
    ).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--rev", help="upstream tag to vendor, e.g. v1.30.0")
    ap.add_argument(
        "--upstream",
        type=Path,
        default=os.environ.get("CUDNN_FE_UPSTREAM"),
        help="cudnn-frontend git clone (default: $CUDNN_FE_UPSTREAM)",
    )
    ap.add_argument(
        "--dest", type=Path, help="output dir (default: agent_space/cudnn_fe_vendor/<rev>)"
    )
    ap.add_argument("--verify", metavar="COMMIT", help="compare with COMMIT's vendored tree")
    ap.add_argument("--diff-upstream", nargs=2, metavar=("OLD", "NEW"), help="closure churn")
    a = ap.parse_args(argv)
    if a.upstream is None:
        ap.error("--upstream (or $CUDNN_FE_UPSTREAM) is required")
    if a.diff_upstream:
        diff_upstream(a.upstream, *a.diff_upstream)
        return 0
    if not a.rev:
        ap.error("--rev is required")
    upstream = closure.Upstream(a.upstream, a.rev)
    if a.verify:
        return verify(upstream, a.verify)
    dest = default_dest(a.rev) if a.dest is None else a.dest
    cl = vendor(upstream, dest)
    report(cl, a.rev, "driver")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
