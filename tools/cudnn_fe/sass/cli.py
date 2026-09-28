"""``python -m tools.cudnn_fe.sass {snapshot,diff}``."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from .._common import activate_tree, check_tree
from .diff import compare, render
from .sass import carve_cubins

CACHE_VARIABLES = ("CUTE_DSL_CACHE_DIR", "ATTN_GYM_CUTE_CACHE_DIR")


def snapshot(out: Path, tree: Path | None, selected: list[str]) -> int:
    """Compile the selected cases with fresh compile caches and write the snapshot to ``out``.

    Must run before CuTeDSL or ``attn_gym`` is imported: the cache directories are read at
    import time and ``tree`` is placed ahead of the editable install on ``sys.path``.
    """
    out.mkdir(parents=True, exist_ok=True)
    caches = [
        tempfile.mkdtemp(prefix=f"cudnn-fe-sass-{name.lower()}-") for name in CACHE_VARIABLES
    ]
    for name, path in zip(CACHE_VARIABLES, caches):
        os.environ[name] = path
    activate_tree(tree)
    try:
        return _snapshot(out, tree, selected, Path(caches[1]))
    finally:
        for path in caches:
            shutil.rmtree(path, ignore_errors=True)


def published_cubins_match(out: Path, cache: Path) -> tuple[int, int]:
    """Carve the cubins out of every object ``jit_cache`` published and count how many are
    byte-identical to a snapshot cubin -> ``(matched, total)``."""
    digests = {hashlib.sha256(path.read_bytes()).digest() for path in out.glob("*.cubin")}
    matched = total = 0
    for path in cache.rglob("*.o"):
        for cubin in carve_cubins(path.read_bytes()):
            total += 1
            matched += hashlib.sha256(cubin).digest() in digests
    return matched, total


def _snapshot(out: Path, tree: Path | None, selected: list[str], cache: Path) -> int:
    import cutlass
    import torch

    package = check_tree(tree)

    from .capture import Capture
    from .cases import cases

    capture = Capture(out)
    capture.install()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    meta = [
        f"attn_gym {package}",
        f"device {device.name} sms={device.multi_processor_count}",
        f"torch {torch.__version__}",
        f"cutlass {cutlass.__version__}",
        f"python {platform.python_version()}",
        f"cases {' '.join(selected) or 'all'}",
    ]
    capture.say("\n".join(meta))
    results = []
    for case in cases():
        if selected and not any(substring in case.name for substring in selected):
            continue
        capture.case = case.name
        capture.say(f"=== {case.name}")
        started = time.time()
        try:
            case.run()
            status, info = "ok", ""
        except Exception as error:  # noqa: BLE001 - one failing case must not end the run
            traceback.print_exc()
            torch.cuda.synchronize()
            status = "FAIL"
            info = (
                f"{type(error).__name__}: {str(error).splitlines()[0][:120] if str(error) else ''}"
            )
        new = sum(1 for name, _, _ in capture.artifacts if name == case.name)
        results.append((status, case.name, f"{time.time() - started:.0f}s", info))
        capture.say(f"=== {case.name}: {status} ({new} new artifacts, {results[-1][2]}) {info}")
    (out / "META.txt").write_text("".join(f"{line}\n" for line in meta))
    (out / "ARTIFACTS.txt").write_text(
        "".join(f"{case} {label} {cubins}\n" for case, label, cubins in capture.artifacts)
    )
    (out / "CASES.txt").write_text("".join(f"{' '.join(row).rstrip()}\n" for row in results))
    failed = [case for status, case, _, _ in results if status != "ok"]
    matched, total = published_cubins_match(out, cache)
    capture.say(f"{matched}/{total} cubins of the published jit_cache objects match the snapshot")
    if matched != total:
        failed.append("<published cubins differ from the snapshot>")
    capture.say(
        f"{len(capture.artifacts)} artifacts from {len(results)} cases -> {out}"
        + (f"; FAILED: {' '.join(failed)}" if failed else "")
    )
    (out / "snapshot.log").write_text("".join(f"{line}\n" for line in capture.log))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.cudnn_fe.sass",
        description="SASS equivalence gate for the vendored cudnn-frontend GDN/KDA kernels.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser(
        "snapshot", help="compile every kernel variant through the drivers and dump SASS"
    )
    snap.add_argument("out", type=Path)
    snap.add_argument(
        "--tree",
        type=Path,
        help="repository checkout whose attn_gym to snapshot (default: the importable one)",
    )
    snap.add_argument(
        "--cases", nargs="+", default=[], metavar="SUBSTR", help="run only matching case names"
    )
    cmp = commands.add_parser("diff", help="classify the kernels of two snapshots")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--strict", action="store_true", help="also fail on noise-band opcode deltas")
    cmp.add_argument(
        "--quiet", action="store_true", help="print only noise and real labels (and the summary)"
    )
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        return snapshot(args.out.resolve(), args.tree, args.cases)
    report = compare(args.a, args.b)
    print(render(report, args.a, args.b, quiet=args.quiet))
    return 1 if report.failed(args.strict) else 0
