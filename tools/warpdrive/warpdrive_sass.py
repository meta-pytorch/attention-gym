"""Dump PTX and normalized SASS for warpdrive's seven standard configurations.

    CUTE_DSL_ARCH=sm_100a python tools/warpdrive/warpdrive_sass.py dump agent_space/sass/before
    python tools/warpdrive/warpdrive_sass.py compare agent_space/sass/before agent_space/sass/after

Dumping requires CuTeDSL, flash_attn.cute, nvdisasm and an empty output directory. Compare
needs only Python. Persistent GEMM pins tile_n=256 so default-tile changes do not alter the gate.
PTX and compiler options are the primary equivalence gate; SASS changes remain visible as
assembler diagnostics. Use compare --strict-sass to require exact SASS too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def dump(output: Path, labels: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"{output}: dump needs an empty output directory")
    # Keep compare usable without the compiler or optional example dependencies.
    import cutlass
    from cutlass._mlir._mlir_libs import _cutlass_ir
    from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support
    from cutlass.base_dsl.compiler import Compiler

    import attn_gym
    from attn_gym.warpdrive import (
        ClusterLaunchControl,
        CtaPair,
        StaticPersistent,
    )
    from attn_gym.warpdrive.examples.fa4_fwd import (
        FlashAttentionForward,
        compile_fa4_fwd,
    )
    from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm
    from attn_gym.warpdrive.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    configs = {
        "gemm_oneshot": (compile_gemm, Gemm()),
        "pgemm_static_single": (
            compile_persistent_gemm,
            PersistentGemm(schedule=StaticPersistent(), tile_n=256),
        ),
        "pgemm_static_pair": (
            compile_persistent_gemm,
            PersistentGemm(schedule=StaticPersistent(), cluster=CtaPair(), tile_n=256),
        ),
        "pgemm_clc_pair": (
            compile_persistent_gemm,
            PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair(), tile_n=256),
        ),
        "fa4_single": (compile_fa4_fwd, FlashAttentionForward()),
        "fa4_clc_pair": (
            compile_fa4_fwd,
            FlashAttentionForward(schedule=ClusterLaunchControl(), cluster=CtaPair()),
        ),
        "fa4_clc_pair_splitp": (
            compile_fa4_fwd,
            FlashAttentionForward(
                schedule=ClusterLaunchControl(), cluster=CtaPair(), split_p=True
            ),
        ),
    }
    unknown = set(labels) - configs.keys()
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}; choose {list(configs)}")
    print(attn_gym.__file__, flush=True)
    compiler_identity = {
        "cutlass_version": cutlass.__version__,
        "compiler_sha256": hashlib.sha256(Path(_cutlass_ir.__file__).read_bytes()).hexdigest(),
    }

    def compile_with_ptx(build, op, label):
        original_compile = Compiler.compile
        pipelines = []

        def capture_compile(self, module, pipeline, *args, **kwargs):
            # Short artifact names avoid the filesystem limit on mangled TVM-FFI names.
            artifact_option = f"dump-ptx-path='{output.resolve() / label}'"
            clean_pipeline = re.sub(r"dump-(?:ptx|cubin)-path='[^']*'", "", pipeline)
            pipelines.append(clean_pipeline)
            pipeline = re.sub(r"dump-ptx-path='[^']*'", "", pipeline)
            pipeline = pipeline.replace("cute-to-nvvm{", f"cute-to-nvvm{{{artifact_option} ", 1)
            if "cute-to-nvvm{" not in pipeline:
                raise RuntimeError("PTX capture requires the CuTeDSL legacy compiler backend")
            return original_compile(self, module, pipeline, *args, **kwargs)

        Compiler.compile = capture_compile
        try:
            compiled = getattr(build, "__wrapped__", build)(op)  # a cached kernel has no IR
        finally:
            Compiler.compile = original_compile
        ptx_files = list(output.glob(f"{label}.sm_*.ptx"))
        if len(ptx_files) != 1 or len(pipelines) != 1:
            raise RuntimeError(f"{label}: expected one PTX artifact and one compiler pipeline")
        ptx = ptx_files[0].read_bytes().rstrip(b"\x00").decode()
        ptx = re.sub(r"cutlass_[A-Za-z0-9_]+", "KERNEL", ptx)
        (output / f"{label}.ptx").write_text(ptx)
        ptx_files[0].unlink()
        metadata = {**compiler_identity, "pipeline": pipelines[0]}
        (output / f"{label}.compiler.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return compiled

    for label in labels or configs:
        build, op = configs[label]
        compiled = compile_with_ptx(build, op, label)
        obj = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        # The first ELF is the host object; each embedded ELF is a device cubin.
        starts = [match.start() for match in re.finditer(b"\x7fELF", obj)]
        if len(starts) < 2:
            raise RuntimeError(f"{label}: compiled object contains no device cubin")
        for index, (start, end) in enumerate(zip(starts[1:], [*starts[2:], len(obj)]), 1):
            cubin = output / f"{label}_{index}.cubin"
            cubin.write_bytes(obj[start:end])
            sass = subprocess.check_output(["nvdisasm", str(cubin)], text=True)
            sass = re.sub(r"/\*[0-9a-f]{4,}\*/", "", sass)
            sass = re.sub(r"cutlass_[A-Za-z0-9_]+", "KERNEL", sass)
            cubin.with_suffix(".sass").write_text(sass)
        print(f"{label}: {len(starts) - 1} cubin(s)", flush=True)


def compare(before: Path, after: Path, strict_sass: bool = False) -> int:
    sass_before = {file.name: file for file in before.glob("*.sass")}
    sass_after = {file.name: file for file in after.glob("*.sass")}
    if not sass_before or not sass_after:
        raise ValueError("both directories must contain normalized .sass files")
    ptx_before = {file.name: file for file in before.glob("*.ptx")}
    ptx_after = {file.name: file for file in after.glob("*.ptx")}
    different = False
    labels = {re.sub(r"_\d+\.sass$", "", name) for name in sass_before.keys() | sass_after.keys()}
    for label in sorted(labels):
        for suffix in (".ptx", ".compiler.json"):
            name = label + suffix
            left, right = before / name, after / name
            equal = left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()
            print(f"{name}: {'identical' if equal else 'different or missing'}")
            different |= not equal
    expected_ptx = {label + ".ptx" for label in labels}
    if ptx_before.keys() != expected_ptx or ptx_after.keys() != expected_ptx:
        print("PTX inventory does not match SASS configuration inventory")
        different = True
    for name in sorted(sass_before.keys() | sass_after.keys()):
        if name not in sass_before:
            verdict = "only after"
            different = True
        elif name not in sass_after:
            verdict = "only before"
            different = True
        elif sass_before[name].read_bytes() != sass_after[name].read_bytes():
            verdict = "different"
            if strict_sass:
                different = True
            else:
                verdict += " (assembler diagnostic; PTX/compiler checks above remain mandatory)"
        else:
            verdict = "identical"
        print(f"{name}: {verdict}")
    return int(different)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dump_parser = commands.add_parser("dump", help="compile and disassemble standard configs")
    dump_parser.add_argument("output", type=Path)
    dump_parser.add_argument("labels", nargs="*", help="default: all seven configurations")
    compare_parser = commands.add_parser("compare", help="compare two normalized SASS dumps")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    compare_parser.add_argument(
        "--strict-sass", action="store_true", help="also fail on SASS differences"
    )
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # CuTeDSL re-parses application arguments at import/compile time.
    if args.command == "compare":
        return compare(args.before, args.after, args.strict_sass)
    dump(args.output, args.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
