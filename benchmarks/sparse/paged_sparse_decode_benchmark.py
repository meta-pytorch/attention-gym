"""Benchmark paged_sparse_decode on Qwen QSA decode shapes, optionally against vLLM's kernel.

Each batch captures a CUDA graph per implementation and alternates their replays. ``--cold``
evicts K/V from L2 before each call and warms the routing metadata, as an indexer would. With
``--vllm-root``, vLLM's native QSA
Triton kernel (``vllm/models/qwen4_exp/nvidia/ops/qsa.py``) runs in this process on the same
caches, given the expanded token indices it consumes; the time of vLLM's separate expansion
kernel is excluded, which favors vLLM.

The bandwidth speed-of-light (SOL) floor counts logical K/V reads per query and KV head,
Q/O, and selected indices. It excludes intermediate traffic and launch latency. Warm-cache
results can exceed an HBM roofline; use ``--cold`` for that comparison. The B200's advertised
8 TB/s is the default peak there; pass measured bandwidth explicitly for a practical ceiling.

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
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl

from attn_gym.sparse import paged_sparse_decode
from attn_gym.testing.paged_decode import make_paged_sparse_decode_inputs


def load_vllm_qsa(root: str) -> Callable[..., torch.Tensor]:
    """Import vLLM's QSA kernel module, stubbing the two vLLM helpers it imports."""
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


def vllm_operands(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand block ids to vLLM's packed token buffer: tokens, the open block, a count column."""
    block = inputs["index_block_size"]
    query_lens = inputs["cu_seqlens_q"].diff()
    token_to_req = torch.repeat_interleave(
        torch.arange(len(query_lens), device=query_lens.device, dtype=torch.int32), query_lens
    )
    rows = inputs["indices"].shape[0]
    first_row = inputs["cu_seqlens_q"][token_to_req.long()]
    positions = (
        inputs["seqused_k"][token_to_req.long()]
        - query_lens[token_to_req.long()]
        + torch.arange(rows, device=first_row.device)
        - first_row
    )
    selected = inputs["indices"]
    tokens = (selected[:, :, None] * block + torch.arange(block, device=selected.device)).flatten(
        1
    )
    tokens = torch.where(selected.repeat_interleave(block, dim=1) >= 0, tokens, -1)
    tail_start = (positions + 1) // block * block
    tail = tail_start[:, None] + torch.arange(block - 1, device=selected.device)
    tail = torch.where(tail <= positions[:, None], tail, -1)
    counts = inputs["topk_length"] * block + (positions + 1 - tail_start)
    packed = torch.cat((tokens, torch.full_like(tail, -1), counts[:, None]), dim=1).to(torch.int32)
    # vLLM stops after the valid-count tile. Keep the open block next to the valid
    # selections, including when a short context does not fill the sparse budget.
    tail_columns = inputs["topk_length"][:, None] * block + torch.arange(
        block - 1, device=selected.device
    )
    packed.scatter_(1, tail_columns.long(), tail.to(torch.int32))
    return packed, token_to_req


def replay_times(functions: list[Callable[[], object]], replays: int, rounds: int) -> list[float]:
    """Median microseconds per call over alternating CUDA graph replays."""
    graphs = []
    for function in functions:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            function()
            function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(replays):
                function()
        graphs.append(graph)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [[] for _ in functions]
    for round_index in range(rounds):
        for i in range(len(graphs)):
            index = (i + round_index) % len(graphs)
            graph, timings = graphs[index], samples[index]
            graph.replay()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / replays)
    return [median(timings) for timings in samples]


def cold_replay_times(
    functions: list[Callable[[], object]],
    metadata: list[torch.Tensor],
    rounds: int,
) -> list[float]:
    """Median GPU span from first kernel start to last kernel end, excluding cache eviction.

    Each captured invocation follows a 512 MiB read and metadata reads. CUPTI timestamps avoid
    subtracting two large timings to recover a microsecond-scale attention duration. The span
    includes the dependency gap between split and merge kernels, including overlapping PDL.
    """
    from torch.profiler import ProfilerActivity, profile

    flush = torch.empty(1 << 27, device="cuda", dtype=torch.float32)
    flush_out = torch.empty(1 << 10, device="cuda", dtype=torch.float32)
    graphs = []
    repeats = 8
    for function in functions:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            function()
            function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                torch.sum(flush.view(1 << 10, -1), dim=1, out=flush_out)
                for tensor in metadata:
                    tensor.sum()
                function()
        graphs.append(graph)

    samples = [[] for _ in functions]
    for round_index in range(rounds):
        for i in range(len(graphs)):
            index = (i + round_index) % len(graphs)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                graphs[index].replay()
                torch.cuda.synchronize()
            kernels = sorted(
                (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                key=lambda e: e.time_range.start,
            )
            spans = []
            span = None
            for event in kernels:
                # The only native PyTorch kernels in these graphs are the harness's reads.
                if event.name.startswith(("void at::native::", "Memset")):
                    if span is not None:
                        spans.append(span[1] - span[0])
                    span = None
                elif span is None:
                    span = [event.time_range.start, event.time_range.end]
                else:
                    span[1] = max(span[1], event.time_range.end)
            if span is not None:
                spans.append(span[1] - span[0])
            if len(spans) != repeats:
                raise RuntimeError(f"Expected {repeats} attention spans, found {len(spans)}.")
            samples[index].extend(spans)
    return [median(timings) for timings in samples]


def logical_bytes(inputs: dict) -> int:
    """Per-query logical traffic, including the causal tail of an incomplete index block."""
    query = inputs["query"]
    kv = inputs["kv"]
    _, token_to_req = vllm_operands(inputs)
    query_lens = inputs["cu_seqlens_q"].diff()
    positions = (
        inputs["seqused_k"][token_to_req.long()]
        - query_lens[token_to_req.long()]
        + torch.arange(query.shape[0], device=query.device)
        - inputs["cu_seqlens_q"][token_to_req.long()]
    )
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
    parser.add_argument("--replays", type=int, default=50)
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
