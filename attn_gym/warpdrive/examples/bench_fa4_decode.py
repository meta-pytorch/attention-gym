"""Compare the warpdrive swap-AB decode port with FA4's SM100 swap-AB decode at matched splits.

For each shape FA4 runs through its interface (``FLASH_ATTENTION_SM100_BACKEND=swapab``); its split
partials, split count and per-sequence split table are captured at the combine call. The port then
runs with the same split count and table. Reported per shape:

* partial O / LSE: bitwise equality over the splits each sequence owns (a mismatch raises);
  final O against an fp32 reference for both;
* CUDA-graph replay, per kernel (CUPTI durations, median of round medians): FA4's decode kernel
  vs the port's, plus the whole call (FA4: prepare + decode + combine; port: decode + combine)
  as graph replay time.

Run under ``gpu-run``:
``python -m attn_gym.warpdrive.examples.bench_fa4_decode --output results.json``.
Timing uses fixed buffers and the median of seven interleaved round medians. ``--profile`` warms
both variants, then brackets matched complete calls with CUDA profiler start/stop for NCU.
``--profile-reps`` repeats alternating pairs; ``--output`` also saves profiling provenance.
``--shared-kv`` runs the port with ``shared_kv=True`` (FA4's shared K/V ring) instead of the
default split K and V rings; compare the two in separate, interleaved processes.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import cutlass
import torch
import triton

os.environ.setdefault("FLASH_ATTENTION_SM100_BACKEND", "swapab")

from flash_attn.cute import interface

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


@dataclass(frozen=True)
class ReplayRound:
    graph_samples_us: list[float]
    kernel_samples_us: dict[str, list[float]]

    @property
    def graph_us(self) -> float:
        return statistics.median(self.graph_samples_us)

    @property
    def kernels_us(self) -> dict[str, float]:
        return {
            name: statistics.median(samples) for name, samples in self.kernel_samples_us.items()
        }


def replay_times(graph, reps) -> ReplayRound:
    """CUDA-event whole-graph times and CUPTI kernel times, in microseconds."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(10):
        graph.replay()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for s, e in zip(starts, ends):
            s.record()
            graph.replay()
            e.record()
        torch.cuda.synchronize()
    replay = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
    kernels = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time > 0:
            kernels.setdefault(ev.name, []).append(ev.device_time)
    return ReplayRound(replay, kernels)


def decode_kernel(kernels, needle):
    names = [n for n in kernels if needle in n]
    if len(names) != 1:
        raise RuntimeError(f"expected one {needle!r} decode kernel, found {names}")
    return kernels[names[0]]


def git_metadata(directory: Path) -> tuple[str | None, bool | None]:
    """Installed packages and source archives need not have their own Git checkout."""
    command = ["git", "-C", str(directory)]
    try:
        root = subprocess.check_output(
            [*command, "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if Path(root).resolve() != directory.resolve():
            return None, None
        commit = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output([*command, "status", "--porcelain"], text=True).strip()
        return commit, bool(dirty)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


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
    sys.argv = sys.argv[:1]  # CuTeDSL also parses application arguments during compilation.
    shape_indices = args.shapes or list(range(len(SHAPES)))
    results = []
    metadata = {}
    if args.output is not None:
        source_files = (Path(__file__), Path(decode_partials.__code__.co_filename))
        commit, dirty = git_metadata(source_files[1].parents[3])
        fa4_commit, fa4_dirty = git_metadata(Path(interface.__file__).parents[2])
        metadata = {
            "torch": torch.__version__,
            "cutlass": cutlass.__version__,
            "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": os.environ.get("GPU_RUN_GPU_UUID"),
            "source_sha256": {
                source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                for source in source_files
            },
            "warpdrive_commit": commit,
            "warpdrive_dirty": dirty,
            "fa4_commit": fa4_commit,
            "fa4_dirty": fa4_dirty,
            "reps": args.reps,
            "rounds": args.rounds,
            "profile_reps": args.profile_reps if args.profile else None,
            "shared_kv": args.shared_kv,
            "contract": (
                "fixed-buffer complete calls; replay, caches and clocks controlled by external profiler"
                if args.profile
                else "fixed-buffer CUDA graph; CUPTI kernel durations; normal DVFS; capture excluded"
            ),
        }
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
            # The port's wrapper fills seqused_k when it is None; build it outside the graph.
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
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
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
                rounds[variant].append(replay_times(graphs[variant], args.reps))
        dec_fa4 = statistics.median(
            decode_kernel(r.kernels_us, "attn_kernel") for r in rounds["fa4"]
        )
        dec_port = statistics.median(
            decode_kernel(r.kernels_us, "warpdrive") for r in rounds["port"]
        )
        graph_fa4 = statistics.median(r.graph_us for r in rounds["fa4"])
        graph_port = statistics.median(r.graph_us for r in rounds["port"])
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
                "rounds": {name: [asdict(r) for r in samples] for name, samples in rounds.items()},
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
