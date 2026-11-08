"""Benchmark paged_sparse_decode on Qwen QSA decode shapes, optionally against vLLM's kernel.

Each batch captures one CUDA graph per implementation and alternates their replays, so clock
and thermal drift affect every implementation equally. With ``--vllm-root``, vLLM's native QSA
Triton kernel (``vllm/models/qwen4_exp/nvidia/ops/qsa.py``) runs in this process on the same
caches, given the expanded token indices it consumes; the time of vLLM's separate expansion
kernel is excluded, which favors vLLM.

Usage:
    python benchmarks/sparse/paged_sparse_decode_benchmark.py
    python benchmarks/sparse/paged_sparse_decode_benchmark.py --heads 6 --kv-heads 1
    python benchmarks/sparse/paged_sparse_decode_benchmark.py --vllm-root ~/vllm --query-len 2
"""

import argparse
import importlib.util
import sys
import types
from collections.abc import Callable
from functools import partial

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
    packed = torch.cat((tokens, tail, counts[:, None]), dim=1).to(torch.int32)
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
            function()
        graphs.append(graph)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [[] for _ in functions]
    for _ in range(rounds):
        for graph, timings in zip(graphs, samples):
            graph.replay()
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / replays)
    return [sorted(timings)[len(timings) // 2] for timings in samples]


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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    vllm_attention = load_vllm_qsa(args.vllm_root) if args.vllm_root else None

    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"shape: H={args.heads} KV={args.kv_heads} D={args.head_dim} context={args.context} "
        f"queries/request={args.query_len} page={args.page_size} "
        f"topk={args.block_topk}x{args.index_block_size}"
    )
    header = f"{'batch':>5} {'ours us':>9} {'TB/s':>6}"
    print(header + (f" {'vllm us':>9} {'speedup':>8}" if vllm_attention else ""))
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
        if vllm_attention:
            packed, token_to_req = vllm_operands(inputs)
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
        times = replay_times(functions, args.replays, args.rounds)
        # Bytes of K and V rows gathered once per (query, KV head).
        keys = int(inputs["topk_length"].sum()) * args.index_block_size
        gathered = keys * args.kv_heads * args.head_dim * 2 * inputs["query"].element_size()
        line = f"{batch:>5} {times[0]:>9.1f} {gathered / times[0] / 1e6:>6.2f}"
        if vllm_attention:
            line += f" {times[1]:>9.1f} {times[1] / times[0]:>7.2f}x"
        print(line)


if __name__ == "__main__":
    main()
