"""Compare warpdrive and FA4 backward with interleaved, fixed-buffer CUDA graphs.

Each graph contains a full backward call: preprocess, main kernel and dQ postprocess.

* ``warpdrive``: ``fa4_bwd`` (FA4's pre/postprocess around the warpdrive 1-CTA main kernel);
* ``warpdrive_2cta``: ``fa4_bwd(..., cluster=CtaPair())``, the warpdrive 2-CTA main kernel;
* ``fa4_1cta``: FA4 with ``disable_2cta``, 128 x 128 tiles, Q_stage 2 and FA4's register split;
* ``fa4``: FA4's default (2-CTA for head_dim 128 on SM100).

Before timing, each port's dK/dV must match its FA4 counterpart bitwise when both run.
dQ allows reduce-order rounding. Each round replays a graph ``--reps`` times: CUPTI times
the main kernel; CUDA events time the whole graph. Variant order alternates across rounds
(default 7). The table reports minimum round medians in microseconds.

Run under ``gpu-run``:
``python -m attn_gym.warpdrive.examples.bench_fa4_bwd --output bwd.json``.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from flash_attn.cute import flash_bwd_sm100
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from attn_gym.warpdrive import CtaPair
from attn_gym.warpdrive.examples import fa4_bwd as port
from attn_gym.warpdrive.examples.bench_support import (
    provenance,
    replay_timed,
    settle,
)

# (batch, heads, seqlen_q, seqlen_k)
SHAPES = [(2, 16, 1024, 1024), (4, 16, 4096, 4096), (1, 16, 8192, 8192), (4, 16, 3000, 2900)]
MAIN_KERNELS = {
    "warpdrive": "warpdrive_fa4_bwd",
    "warpdrive_2cta": "warpdrive_fa4_bwd_pair",
    "fa4_1cta": "Backward",
    "fa4": "Backward",
}
COUNTERPARTS = {"warpdrive": "fa4_1cta", "warpdrive_2cta": "fa4"}


def main_kernel(names, variant):
    needle = MAIN_KERNELS[variant]
    found = [n for n in names if needle in n and "rocess" not in n]
    if len(found) != 1:
        raise RuntimeError(f"{variant}: expected one main kernel matching {needle!r}, got {names}")
    return found[0]


def run_shape(b, h, sq, sk, causal, variants, reps, rounds):
    torch.manual_seed(0)
    q = torch.randn(b, sq, h, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(b, sk, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    dout = torch.randn_like(q)
    scale = 128**-0.5
    out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)[:2]
    one_cta, default = (port.fa4_config(q, k, causal, flag) for flag in (True, False))

    def fa4(config):
        return _flash_attn_bwd(q, k, v, out, dout, lse, scale, causal, config=config)[:3]

    calls = {
        "warpdrive": lambda: port.fa4_bwd(q, k, v, out, dout, lse, scale, causal=causal),
        "warpdrive_2cta": lambda: port.fa4_bwd(
            q, k, v, out, dout, lse, scale, causal=causal, cluster=CtaPair()
        ),
        "fa4_1cta": lambda: fa4(one_cta),
        "fa4": lambda: fa4(default),
    }
    calls = {name: calls[name] for name in variants}
    for ours_name, ref_name in COUNTERPARTS.items():
        if ours_name not in calls or ref_name not in calls:
            continue
        ours, ref = calls[ours_name](), calls[ref_name]()
        dq_err = (ours[0].float() - ref[0].float()).abs().max().item()
        if not (torch.equal(ours[1], ref[1]) and torch.equal(ours[2], ref[2])):
            raise AssertionError(f"{ours_name}: dK/dV differ from {ref_name}")
        if dq_err > 2e-3 * ref[0].float().abs().max().item() + 1e-3:
            raise AssertionError(f"{ours_name}: dQ differs from {ref_name} by {dq_err}")
        print(
            (b, h, sq, sk, causal),
            f"{ours_name}: dK/dV bitwise equal to {ref_name}; dQ max abs",
            dq_err,
        )
    graphs = {}
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
    settle(graphs.values(), seconds=3.0)
    medians = {name: {"graph": [], "main": []} for name in calls}
    samples = {name: [] for name in calls}
    for r in range(rounds):
        for name in variants if r % 2 == 0 else list(reversed(variants)):
            graph_us, kernels = replay_timed(graphs[name], reps, warmup=5)
            main = kernels[main_kernel(list(kernels), name)]
            medians[name]["graph"].append(statistics.median(graph_us))
            medians[name]["main"].append(statistics.median(main))
            samples[name].append({"graph_us": graph_us, "kernels_us": kernels})
            print(
                (b, h, sq, sk, causal), r, name, medians[name]["main"][-1], "main us", flush=True
            )
    best = {name: {part: min(values) for part, values in m.items()} for name, m in medians.items()}
    print("minimum round medians (us)", (b, h, sq, sk, causal), best, flush=True)
    return {
        "shape": [b, h, sq, sk],
        "causal": causal,
        "min_median_us": best,
        "medians_us": medians,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=int, nargs="*", help="indices into SHAPES")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(MAIN_KERNELS),
        default=list(MAIN_KERNELS),
    )
    parser.add_argument("--reps", type=int, default=30, help="graph replays per round")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 1 or args.rounds < 1 or len(set(args.variants)) != len(args.variants):
        parser.error("reps and rounds must be positive and variants unique")
    sys.argv = sys.argv[:1]  # Keep application flags out of CuTeDSL's diagnostic parser.
    metadata = provenance(
        {
            "warpdrive": Path(port.__file__),
            "benchmark": Path(__file__),
            "flash_attn": Path(flash_bwd_sm100.__file__),
        },
        vars(args) | {"output": str(args.output)},
    )
    print("metadata", metadata, flush=True)
    shapes = SHAPES if not args.shapes else [SHAPES[i] for i in args.shapes]
    results = [
        run_shape(*shape, args.causal, args.variants, args.reps, args.rounds) for shape in shapes
    ]
    if args.output is not None:
        args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


if __name__ == "__main__":
    main()
