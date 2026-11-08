"""Compare warpdrive and FA4 SM100 swap-AB decode with matched splits.

FA4 uses ``FLASH_ATTENTION_SM100_BACKEND=swapab``. Capture its partials, split count and
per-sequence split table at the combine call, then pass the same counts to warpdrive.
For each shape, check partial O/LSE bitwise over owned splits and report final O error
against an fp32 reference for both kernels.

CUPTI times each decode kernel; CUDA events time the whole graph. FA4's graph includes
prepare + decode + combine; warpdrive's includes decode + combine.

Run under ``gpu-run``:
``python -m attn_gym.warpdrive.examples.bench_fa4_decode --output results.json``.
Timing uses fixed buffers and the median of seven interleaved round medians. ``--profile`` warms
both variants, then brackets matched complete calls with CUDA profiler start/stop for NCU.
``--profile-reps`` repeats alternating pairs; ``--output`` also saves profiling provenance.
``--shared-kv`` runs the port with ``shared_kv=True`` (FA4's shared K/V ring) instead of the
default split K and V rings; compare the two in separate, interleaved processes.
"""

import argparse
import json
import math
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

os.environ.setdefault("FLASH_ATTENTION_SM100_BACKEND", "swapab")

from flash_attn.cute import interface

from attn_gym.warpdrive.examples import fa4_decode_swapab
from attn_gym.warpdrive.examples.bench_support import provenance, replay_timed, settle
from attn_gym.warpdrive.examples.fa4_decode_swapab import (
    decode_partials,
    fa4_decode,
)

# (batch, kv capacity, heads, kv heads, seqused pattern): "full" or "ragged" (lengths cycle
# through 1/8, 1/2, 1 and 3/4 of the capacity, so per-sequence split counts differ).
SHAPES = [
    (1, 8192, 8, 1, "full"),
    (4, 16384, 32, 8, "full"),
    (8, 32768, 32, 8, "full"),
    (8, 65536, 8, 1, "ragged"),
    (16, 32768, 64, 8, "ragged"),
    (1, 32768, 32, 1, "full"),
    (2, 131072, 16, 1, "ragged"),
]


def lengths(batch, capacity, pattern):
    if pattern == "full":
        return None
    fractions = (0.125, 0.5, 1.0, 0.75)
    used = [max(1, int(capacity * fractions[i % 4])) for i in range(batch)]
    return torch.tensor(used, device="cuda", dtype=torch.int32)


def fa4_call(q, k, v, used):
    captured = {}
    combine = interface._flash_attn_fwd_combine

    def capture(out_partial, lse_partial, *args, **kwargs):
        table = kwargs.get("num_splits_dynamic_ptr")
        captured.update(o=out_partial.clone(), lse=lse_partial.clone(), table=table)
        return combine(out_partial, lse_partial, *args, **kwargs)

    capture.compile_cache = combine.compile_cache  # the combine reads it by global name
    interface._flash_attn_fwd_combine = capture
    try:
        out, lse, *_ = interface._flash_attn_fwd(
            q, k, v, seqused_k=used, max_seqlen_k=k.shape[1], num_splits=0, return_lse=True
        )
    finally:
        interface._flash_attn_fwd_combine = combine
    return out, lse, captured


def reference(q, k, v, used):
    b, _, h, _ = q.shape
    qpk = h // k.shape[2]
    outs = []
    for i in range(b):
        n = k.shape[1] if used is None else int(used[i])
        kk = k[i, :n].float().repeat_interleave(qpk, 1)
        vv = v[i, :n].float().repeat_interleave(qpk, 1)
        s = torch.einsum("hd,thd->ht", q[i, 0].float(), kk) / math.sqrt(q.shape[-1])
        outs.append(torch.einsum("ht,thd->hd", torch.softmax(s, -1), vv))
    return torch.stack(outs)[:, None]


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def decode_kernel(kernels, needle):
    names = [n for n in kernels if needle in n]
    if len(names) != 1:
        raise RuntimeError(f"expected one {needle!r} decode kernel, found {names}")
    return kernels[names[0]]


def write_results(output: Path | None, metadata: dict, results: list[dict]) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7, help="interleaved A/B rounds")
    parser.add_argument("--shapes", type=int, nargs="*", help="indices into SHAPES")
    parser.add_argument(
        "--output", type=Path, help="save provenance and replay samples (or profiling launches)"
    )
    parser.add_argument(
        "--profile", action="store_true", help="warm, then profile matched calls per variant"
    )
    parser.add_argument(
        "--profile-reps", type=int, default=1, help="alternating FA4/port pairs under NCU"
    )
    parser.add_argument(
        "--shared-kv", action="store_true", help="run the port with one shared K/V ring"
    )
    args = parser.parse_args()
    if args.reps < 1 or args.rounds < 1 or args.rounds % 2 == 0:
        parser.error("--reps must be positive and --rounds must be positive and odd")
    if args.profile_reps < 1:
        parser.error("--profile-reps must be positive")
    if not args.profile and args.profile_reps != 1:
        parser.error("--profile-reps requires --profile")
    sys.argv = sys.argv[:1]  # Keep application flags out of CuTeDSL's diagnostic parser.
    shape_indices = args.shapes or list(range(len(SHAPES)))
    results = []
    metadata = provenance(
        {
            "warpdrive": Path(fa4_decode_swapab.__file__),
            "benchmark": Path(__file__),
            "flash_attn": Path(interface.__file__),
        },
        vars(args) | {"output": str(args.output)},
    )
    metadata["contract"] = (
        "fixed-buffer complete calls; replay, caches and clocks controlled by external profiler"
        if args.profile
        else "fixed-buffer CUDA graph; CUPTI kernel durations; normal DVFS; capture excluded"
    )
    for shape_index in shape_indices:
        b, s, h, hkv, pattern = SHAPES[shape_index]
        torch.manual_seed(0)
        q = torch.randn(b, 1, h, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
        used = lengths(b, s, pattern)
        out_fa4, _, cap = fa4_call(q, k, v, used)
        if "o" not in cap:
            print(f"b{b} s{s} h{h}/{hkv} {pattern}: FA4 chose one split; skipped")
            continue
        splits, table = cap["o"].shape[0], cap["table"]
        port = {"splits_per_seq": table, "shared_kv": args.shared_kv}
        o, lse = decode_partials(q, k, v, splits, seqused_k=used, **port)
        out, _ = fa4_decode(q, k, v, splits, seqused_k=used, **port)
        counts = table.tolist() if table is not None else [splits] * b
        owned = torch.tensor(
            [[i < counts[j] for j in range(b)] for i in range(splits)], device="cuda"
        )
        o_fa4, lse_fa4 = cap["o"], cap["lse"][:, :, 0]  # captured transposed: (ns, b, 1, h)
        lse = lse[..., 0]
        do = (o - o_fa4)[owned].abs().max().item()
        same = torch.equal(o[owned], o_fa4[owned])
        same_lse = torch.equal(lse[owned], lse_fa4[owned])
        if not same or not same_lse:
            raise RuntimeError(f"shape {shape_index}: port partials are not bitwise equal to FA4")
        fin = owned[..., None] & torch.isfinite(lse_fa4)
        dl = (lse - lse_fa4)[fin].abs().max().item() if fin.any() else 0.0
        ref = reference(q, k, v, used)
        err_port = (out.float() - ref).abs().max().item()
        err_fa4 = (out_fa4.float() - ref).abs().max().item()
        calls = {
            "fa4": partial(
                interface._flash_attn_fwd,
                q,
                k,
                v,
                seqused_k=used,
                max_seqlen_k=s,
                num_splits=0,
                return_lse=True,
            ),
            # Allocate seqused_k outside capture so the graph contains only decode + combine.
            "port": partial(
                fa4_decode,
                q,
                k,
                v,
                splits,
                seqused_k=used if used is not None else torch.full((b,), s, device="cuda").int(),
                **port,
            ),
        }
        graphs = {name: graph_of(call) for name, call in calls.items()}
        settle(graphs.values(), seconds=3.0)
        if args.profile:
            launch_order = []
            torch.cuda.cudart().cudaProfilerStart()
            try:
                for pair in range(args.profile_reps):
                    order = ("fa4", "port") if pair % 2 == 0 else ("port", "fa4")
                    for variant in order:
                        calls[variant]()
                        launch_order.append(variant)
                torch.cuda.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
            results.append(
                {
                    "shape_index": shape_index,
                    "shape": SHAPES[shape_index],
                    "splits": splits,
                    "counts": counts,
                    "partials_bitwise_equal": same and same_lse,
                    "profile_launch_order": launch_order,
                }
            )
            write_results(args.output, metadata, results)
            print(
                f"shape {shape_index}: warmed and profiled {args.profile_reps} matched "
                f"{splits}-split decode pairs; counts {counts}",
                flush=True,
            )
            continue
        rounds = {"fa4": [], "port": []}
        for round_index in range(args.rounds):
            order = ("fa4", "port") if round_index % 2 == 0 else ("port", "fa4")
            for variant in order:
                graph_us, kernels_us = replay_timed(graphs[variant], args.reps, warmup=10)
                rounds[variant].append({"graph_us": graph_us, "kernels_us": kernels_us})
        dec_fa4, dec_port = (
            statistics.median(
                statistics.median(decode_kernel(r["kernels_us"], needle)) for r in rounds[name]
            )
            for name, needle in (("fa4", "attn_kernel"), ("port", "warpdrive"))
        )
        graph_fa4, graph_port = (
            statistics.median(statistics.median(r["graph_us"]) for r in rounds[name])
            for name in ("fa4", "port")
        )
        results.append(
            {
                "shape_index": shape_index,
                "shape": SHAPES[shape_index],
                "splits": splits,
                "counts": counts,
                "partials_bitwise_equal": same and same_lse,
                "partial_o_max_error": do,
                "partial_lse_max_error": dl,
                "final_o_max_error": {"fa4": err_fa4, "port": err_port},
                "decode_us": {"fa4": dec_fa4, "port": dec_port},
                "graph_us": {"fa4": graph_fa4, "port": graph_port},
                "rounds": rounds,
            }
        )
        write_results(args.output, metadata, results)
        print(
            f"b{b} s{s} h{h}/{hkv} {pattern} splits {splits} "
            f"{'per-seq ' + str(counts) if table is not None else 'static'}\n"
            f"  partials: max|dO| {do:.1e} bitwise {same}, max|dLSE| {dl:.1e}; "
            f"final vs fp32: port {err_port:.1e}, FA4 {err_fa4:.1e}\n"
            f"  decode kernel us: FA4 {dec_fa4:.2f}, port {dec_port:.2f} "
            f"({dec_port / dec_fa4:.3f}x); graph us: FA4 {graph_fa4:.2f}, "
            f"port {graph_port:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
