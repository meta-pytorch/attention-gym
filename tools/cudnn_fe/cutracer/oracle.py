"""Bitwise determinism oracle for the vendored cuDNN GDN/KDA kernels.

``python -m tools.cudnn_fe.cutracer.oracle record --ref-dir DIR`` runs every case unperturbed
and stores outputs and input gradients under ``DIR``; ``... compare --ref-dir DIR`` (typically
under a CUTracer ``random_delay`` trace, see ``stress.py``) recomputes them and exits 1 on any
bit difference.

The cases are chosen so that the public ``chunk_gdn`` / ``chunk_kda`` cuDNN routes reach every
kernel family: uncut packed sequences with an empty interval and grouped heads, the d_v split
(GDN) / prep (KDA) plans, the exact piece chain with state cotangents, the KDA chain without
state (v1.30 routes the stateful KDA backward to the cute path), and the warmup split plans.
Only bf16 with d_k = d_v = 128 and default gate flags is exercised.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .._common import Workload, chunk_op, op_inputs

CUDNN = {"backend": "cudnn"}
SPLIT = {"backend": "cudnn", "split_forward": True, "split_backward": True}


@dataclass(frozen=True)
class Case:
    name: str
    work: Workload
    options: dict
    with_state: bool
    seed: int

    @property
    def family(self) -> str:
        return self.work.kind


CASES = (
    Case("uncut_packed", Workload("gdn", (300, 0, 129, 64), 2, 4), CUDNN, True, 100),
    Case("dv_split", Workload("gdn", (2048,), 16, 48), CUDNN, False, 101),
    Case("chain", Workload("gdn", (24576,), 16, 48), CUDNN, True, 102),
    Case("warmup_split", Workload("gdn", (8192,), 1, 2), SPLIT, False, 103),
    Case("uncut_packed", Workload("kda", (257, 0, 511, 17), 2, 2), CUDNN, True, 200),
    Case("prep", Workload("kda", (2048,), 8, 8), CUDNN, False, 201),
    Case("chain", Workload("kda", (8192,), 48, 48), CUDNN, True, 202),
    # v1.30 KDA: state routes the backward to the cute path and the forward chain needs
    # >= 8192 tokens per piece, so the exact piece chain is only reached without state.
    Case("chain_nostate", Workload("kda", (24576,), 48, 48), CUDNN, False, 203),
    Case("warmup_split", Workload("kda", (4096,), 2, 2), SPLIT, False, 204),
)


def select(family: str, names: set[str] | None) -> list[Case]:
    cases = [c for c in CASES if family in (c.family, "all") and (not names or c.name in names)]
    if not cases:
        raise SystemExit(f"no oracle case matches family={family} names={sorted(names or ())}")
    return cases


def _grads(fn, leaves, output_cotangents):
    outputs = fn(*leaves)
    live = [(o, c) for o, c in zip(outputs, output_cotangents) if o is not None]
    grads = torch.autograd.grad([o for o, _ in live], leaves, [c for _, c in live])
    return [o.detach() for o, _ in live] + list(grads)


def run_case(case: Case) -> list[torch.Tensor]:
    ops = op_inputs(case.work, seed=case.seed, state=case.with_state)
    if case.family == "gdn":
        ops.beta[..., ::7] = 0.0  # exercise the beta-free dBeta path
    leaves = [t.requires_grad_() for t in ops.leaves]
    op = chunk_op(case.family)

    def fn(*x):
        return op(
            *x,
            cu_seqlens=ops.cu_seqlens,
            output_final_state=case.with_state,
            kernel_options=case.options,
        )

    torch.manual_seed(case.seed)
    cot = [torch.randn_like(ops.v)]
    cot.append(torch.randn_like(ops.state) if case.with_state else None)
    return _grads(fn, leaves, cot)


def run(cases: list[Case]) -> dict[str, torch.Tensor]:
    results = {}
    for case in cases:
        for j, t in enumerate(run_case(case)):
            results[f"{case.family}_{case.name}_{j}"] = t.cpu()
    torch.cuda.synchronize()
    return results


def reference_path(ref_dir: Path, family: str, names: list[str]) -> Path:
    return ref_dir / f"reference_{family}_{'_'.join(sorted(set(names))) or 'all'}.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("record", "compare", "list"))
    ap.add_argument("--ref-dir", type=Path, help="directory holding reference_<family>_<cases>.pt")
    ap.add_argument("--family", default="all", choices=("gdn", "kda", "all"))
    ap.add_argument("--case", action="append", default=[], help="restrict to these case names")
    args = ap.parse_args()
    if args.mode == "list":
        for case in select(args.family, set(args.case)):
            print(
                f"{case.family}:{case.name} lengths={case.work.lengths} heads="
                f"{case.work.key_heads}/{case.work.value_heads} state={case.with_state} "
                f"options={case.options}"
            )
        return 0
    if args.ref_dir is None:
        ap.error("--ref-dir is required for record/compare")
    cases = select(args.family, set(args.case))
    reference = reference_path(args.ref_dir, args.family, args.case)
    results = run(cases)
    if args.mode == "record":
        args.ref_dir.mkdir(parents=True, exist_ok=True)
        torch.save(results, reference)
        print(f"recorded {len(results)} reference tensors -> {reference}")
        return 0
    if not reference.exists():
        print(
            f"missing reference {reference}; run `python -m tools.cudnn_fe.cutracer.oracle record` first"
        )
        return 2
    expected = torch.load(reference)
    bad = [k for k, v in results.items() if not torch.equal(v, expected[k])]
    for k in bad:
        diff = (results[k].float() - expected[k].float()).abs()
        print(f"MISMATCH {k}: max abs {diff.max().item():.3e}, {int((diff > 0).sum())} elems")
    if bad:
        return 1
    print(f"all {len(results)} tensors bitwise identical to reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
