"""Benchmark paged_sparse_decode on Qwen QSA decode shapes, optionally against vLLM's kernel.

Each batch size uses one CUDA graph per implementation, with alternating replays to reduce
ordering bias. ``--cold`` evicts K/V from L2 before each call, then warms Q and routing metadata
to approximate a preceding indexer. With ``--vllm-root``, vLLM's native QSA Triton kernel
(``vllm/models/qwen4_exp/nvidia/ops/qsa.py``) runs on the same caches with pre-expanded logical
entry indices. Excluding vLLM's separate expansion kernel favors vLLM.

The bandwidth speed-of-light (SOL) floor counts logical K/V reads per query and KV head,
Q/O, and selected block ids. It excludes intermediate traffic, repeated metadata reads, and
launch latency. Cache reuse can exceed this HBM estimate; use ``--cold`` to reduce that effect.
B200's advertised 8 TB/s is the default peak there; pass measured bandwidth for a practical
ceiling.

Usage:
    python benchmarks/sparse/paged_sparse_decode_benchmark.py
    python benchmarks/sparse/paged_sparse_decode_benchmark.py --heads 6 --kv-heads 1
    python benchmarks/sparse/paged_sparse_decode_benchmark.py --vllm-root ~/vllm --query-len 2
    python benchmarks/sparse/paged_sparse_decode_benchmark.py --cold --measured-bandwidth-tbps 6.9
"""

import argparse
import importlib.util
import json
import sys
import types
from collections.abc import Callable
from functools import partial
from itertools import groupby
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from attn_gym.sparse import paged_sparse_decode
from attn_gym.testing.paged_decode import make_paged_sparse_decode_inputs

COLD_REPEATS = 8


def load_vllm_qsa(root: str) -> Callable[..., torch.Tensor]:
    """Load native QSA without importing the vLLM runtime; restore stubbed modules afterward."""
    stubs = {
        "vllm": {},
        "vllm.triton_utils": {"HAS_TRITON": True, "tl": tl, "triton": triton},
        "vllm.model_executor": {},
        "vllm.model_executor.warmup": {},
        "vllm.model_executor.warmup.jit_warmup_triton_helper": {
            "TritonWarmupTensor": object,
            "triton_scalar_specialization_rep": lambda value: value,
        },
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm_qsa_ops", f"{root}/vllm/models/qwen4_exp/nvidia/ops/qsa.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name)
            else:
                sys.modules[name] = previous
    return module.qsa_sparse_paged_attention


def query_positions(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sequence ids and window-source positions, assuming queries are the last tokens."""
    cu_seqlens_q = inputs["cu_seqlens_q"]
    token_to_req = torch.repeat_interleave(
        torch.arange(len(cu_seqlens_q) - 1, device=cu_seqlens_q.device, dtype=torch.int32),
        cu_seqlens_q.diff(),
    )
    requests = token_to_req.long()
    rows = torch.arange(len(token_to_req), device=cu_seqlens_q.device)
    return token_to_req, inputs["seqused_k"][requests] - (cu_seqlens_q[1:][requests] - rows)


def vllm_operands(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack this benchmark's shared-source, size-zero-window inputs for native vLLM QSA.

    Input selections must form the prefix described by ``topk_length``. Each output row holds
    expanded entries, the query's incomplete-block entries, padding, and a final valid count.
    """
    block = inputs["index_block_size"]
    token_to_req, positions = query_positions(inputs)
    selected = inputs["indices"][:, :, None]
    offsets = torch.arange(block, device=selected.device)
    tokens = torch.where(selected >= 0, selected * block + offsets, -1).flatten(1)
    tail_start = (positions + 1) // block * block
    tail = tail_start[:, None] + offsets[:-1]
    tail = torch.where(tail <= positions[:, None], tail, -1)
    counts = inputs["topk_length"] * block + (positions + 1 - tail_start)
    packed = torch.cat((tokens, torch.full_like(tail, -1), counts[:, None]), dim=1).to(torch.int32)
    # The count describes a prefix: incomplete-block entries must precede padding when a
    # short context leaves part of the sparse budget unused.
    tail_columns = inputs["topk_length"][:, None] * block + offsets[:-1]
    packed.scatter_(1, tail_columns.long(), tail.to(torch.int32))
    return packed, token_to_req


def capture(
    function: Callable[[], object], repeats: int, before: Callable[[], object] | None = None
) -> torch.cuda.CUDAGraph:
    """Keep compilation and host launches outside replay timing by capturing warmed calls.

    ``before`` runs inside the graph before every call, allowing per-call cache eviction.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
        function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            if before is not None:
                before()
            function()
    return graph


def alternating_medians(
    graphs: list[torch.cuda.CUDAGraph],
    rounds: int,
    sample: Callable[[torch.cuda.CUDAGraph], list[float]],
) -> list[float]:
    """Reduce replay-order bias by rotating the first graph; return each graph's sample median."""
    samples = [[] for _ in graphs]
    for round_index in range(rounds):
        for i in range(len(graphs)):
            index = (i + round_index) % len(graphs)
            samples[index].extend(sample(graphs[index]))
    return [median(timings) for timings in samples]


def replay_times(functions: list[Callable[[], object]], replays: int, rounds: int) -> list[float]:
    """Return median microseconds per call, amortizing event timing over a warm graph replay."""
    graphs = [capture(function, replays) for function in functions]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def sample(graph: torch.cuda.CUDAGraph) -> list[float]:
        graph.replay()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) * 1000 / replays]

    return alternating_medians(graphs, rounds, sample)


def cold_replay_times(
    functions: list[Callable[[], object]],
    metadata: list[torch.Tensor],
    rounds: int,
) -> list[float]:
    """Return median attention spans in microseconds, excluding cache eviction and warmup reads.

    Each invocation follows a 512 MiB eviction read, then Q and routing-metadata reads. CUPTI
    timestamps avoid subtracting two large timings to recover microsecond-scale attention
    latency. First kernel start to last kernel end includes split/merge dependencies even when
    programmatic dependent launch (PDL) overlaps their lifetimes.
    """
    flush = torch.empty(1 << 27, device="cuda", dtype=torch.float32)
    flush_out = torch.empty(1 << 10, device="cuda", dtype=torch.float32)

    def evict() -> None:
        torch.sum(flush.view(1 << 10, -1), dim=1, out=flush_out)
        for tensor in metadata:
            tensor.sum()

    def is_harness(event) -> bool:
        # Only eviction/warmup uses native PyTorch kernels here; attention kernels are Triton.
        return event.name.startswith(("void at::native::", "Memset"))

    def sample(graph: torch.cuda.CUDAGraph) -> list[float]:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        kernels = sorted(
            (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start,
        )
        spans = []
        for harness, group in groupby(kernels, key=is_harness):
            if not harness:
                run = list(group)
                spans.append(max(e.time_range.end for e in run) - run[0].time_range.start)
        if len(spans) != COLD_REPEATS:
            raise RuntimeError(f"Expected {COLD_REPEATS} attention spans, found {len(spans)}.")
        return spans

    graphs = [capture(function, COLD_REPEATS, before=evict) for function in functions]
    return alternating_medians(graphs, rounds, sample)


def logical_bytes(inputs: dict) -> int:
    """Total logical bytes per decode call for this benchmark's equal-width K/V inputs.

    Count every query's selected entries and incomplete-block tail without discounting reuse.
    """
    query = inputs["query"]
    kv = inputs["kv"]
    _, positions = query_positions(inputs)
    block = inputs["index_block_size"]
    selected = int(inputs["topk_length"].sum())
    keys = selected * block + int(((positions + 1) % block).sum())
    kv_bytes = (
        keys
        * kv.key_cache.shape[2]
        * (kv.key_cache.shape[-1] + kv.value_cache.shape[-1])
        * query.element_size()
    )
    return kv_bytes + 2 * query.numel() * query.element_size() + selected * 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", default="1,4,16,32,64,128,256")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--query-len", type=int, default=1, help="Queries per request (MTP)")
    parser.add_argument("--heads", type=int, default=24, help="Query heads per rank")
    parser.add_argument("--kv-heads", type=int, default=2, help="KV heads per rank")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--block-topk", type=int, default=512, help="Selected index blocks")
    parser.add_argument("--index-block-size", type=int, default=4)
    parser.add_argument("--vllm-root", help="vLLM checkout providing the native QSA kernel")
    parser.add_argument(
        "--replays",
        type=int,
        default=50,
        help=f"Calls per warm graph; ignored with --cold, which captures {COLD_REPEATS}",
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--cold", action="store_true", help="Cold K/V, warm routing metadata")
    parser.add_argument("--peak-bandwidth-tbps", type=float, help="Advertised HBM bandwidth")
    parser.add_argument(
        "--measured-bandwidth-tbps", type=float, help="Measured HBM read bandwidth"
    )
    parser.add_argument("--json", type=Path, help="Write timings, rooflines, and run metadata")
    args = parser.parse_args()
    for name in ("replays", "rounds", "peak_bandwidth_tbps", "measured_bandwidth_tbps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    vllm_attention = load_vllm_qsa(args.vllm_root) if args.vllm_root else None

    device_name = torch.cuda.get_device_name()
    peak_bandwidth = args.peak_bandwidth_tbps
    if peak_bandwidth is None and "B200" in device_name:
        peak_bandwidth = 8.0
    print(f"device: {device_name}; {'cold K/V, warm metadata' if args.cold else 'warm L2'}")
    print(
        f"shape: H={args.heads} KV={args.kv_heads} D={args.head_dim} context={args.context} "
        f"queries/request={args.query_len} page={args.page_size} "
        f"topk={args.block_topk}x{args.index_block_size}"
    )
    header = f"{'batch':>5} {'ours us':>9} {'TB/s':>6}"
    if vllm_attention:
        header += f" {'vllm us':>9} {'speedup':>8}"
    if peak_bandwidth:
        header += f" {'SOL us':>8} {'%peak':>7}"
    if args.measured_bandwidth_tbps:
        header += f" {'read SOL':>8} {'%read':>7}"
    print(header, flush=True)
    results = []
    for batch in map(int, args.batches.split(",")):
        inputs = make_paged_sparse_decode_inputs(
            [args.context] * batch,
            [args.query_len] * batch,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            topk=args.block_topk,
            index_block_size=args.index_block_size,
        )
        functions = [partial(paged_sparse_decode, **inputs)]
        metadata = [
            inputs["query"],
            inputs["indices"],
            inputs["kv"].block_table,
            inputs["cu_seqlens_q"],
            inputs["seqused_k"],
            inputs["topk_length"],
        ]
        if vllm_attention:
            packed, token_to_req = vllm_operands(inputs)
            metadata.extend((packed, token_to_req))
            out = torch.empty_like(inputs["query"])
            kv = inputs["kv"]
            functions.append(
                partial(
                    vllm_attention,
                    inputs["query"],
                    kv.key_cache,
                    kv.value_cache,
                    packed,
                    kv.block_table,
                    token_to_req,
                    False,
                    out,
                )
            )
            functions[1]()
            expected = paged_sparse_decode(**inputs)
            torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        times = (
            cold_replay_times(functions, metadata, args.rounds)
            if args.cold
            else replay_times(functions, args.replays, args.rounds)
        )
        traffic = logical_bytes(inputs)
        bandwidth = traffic / times[0] / 1e6
        row = {"batch": batch, "ours_us": times[0], "logical_bytes": traffic, "tbps": bandwidth}
        line = f"{batch:>5} {times[0]:>9.2f} {bandwidth:>6.2f}"
        if vllm_attention:
            row.update(vllm_us=times[1], speedup=times[1] / times[0])
            line += f" {times[1]:>9.2f} {row['speedup']:>7.2f}x"
        for name, ceiling in (
            ("peak", peak_bandwidth),
            ("read", args.measured_bandwidth_tbps),
        ):
            if ceiling:
                floor = traffic / ceiling / 1e6
                percent = 100 * floor / times[0]
                row.update({f"{name}_sol_us": floor, f"{name}_percent": percent})
                line += f" {floor:>8.2f} {percent:>6.1f}%"
        results.append(row)
        print(line, flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "device": device_name,
                    "torch_version": torch.__version__,
                    "triton_version": triton.__version__,
                    "cuda_version": torch.version.cuda,
                    "peak_bandwidth_tbps": peak_bandwidth,
                    "arguments": {**vars(args), "json": str(args.json)},
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
