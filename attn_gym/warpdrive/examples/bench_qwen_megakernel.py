"""Qwen3 batch-one decode step: eager, compiled and megakernel timings at a fixed position.

Timings exclude setup and reuse buffers. ``bytes_per_step`` is logical traffic (every
checkpoint tensor once, populated KV reads, the new KV write, logits), not measured DRAM
traffic; eager's expanded GQA and intermediates move more. The HBM rooflines are a copy
(read + write) and a reduction (read-dominated); neither bounds every kernel. Default DVFS.
"""

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples import qwen_megakernel_trace
from attn_gym.warpdrive.examples.bench_support import (
    CudaBenchmarkStats,
    benchmark_cuda_graph_stats,
    provenance,
    settle,
)
from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    QwenWeights,
    compare_logits,
    decode_step_ref,
    load_qwen3_weights,
    make_decode_state,
    rms_norm,
    rope,
    rope_angles,
)


@dataclass(frozen=True)
class FastQwenWeights:
    """Fused projections and RoPE tables for ``decode_step_fast``. ``gemm_rows`` is the row
    duplication of ``baseline_linear``, not a batch."""

    weights: QwenWeights
    qkv: torch.Tensor
    gate_up: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    positions: torch.Tensor
    gemm_rows: int = 16


@torch.no_grad()
def prepare_fast_weights(
    weights: QwenWeights, max_len: int, gemm_rows: int = 16
) -> FastQwenWeights:
    positions = torch.arange(max_len, device=weights.embed.device)
    angles = rope_angles(weights.config, positions.float())
    return FastQwenWeights(
        weights,
        torch.cat((weights.q_proj, weights.k_proj, weights.v_proj), dim=1),
        torch.cat((weights.gate, weights.up), dim=-2),  # MoE: per expert
        angles.cos().to(torch.bfloat16),
        angles.sin().to(torch.bfloat16),
        positions,
        gemm_rows,
    )


def baseline_linear(x: torch.Tensor, weight: torch.Tensor, rows: int) -> torch.Tensor:
    # Duplicating the row makes cuBLAS pick a tensor-core GEMM over its slow single-row GEMV;
    # only the first output row is used.
    return F.linear(x.expand(rows, -1).contiguous(), weight)[:1]


def moe_mlp_fast(config, packed: FastQwenWeights, layer: int, norm: torch.Tensor) -> torch.Tensor:
    """One token's routed MLP as batched matmuls over the top-k experts' gathered weights
    (compile- and CUDA-graph-friendly; the gather copies those experts' weights)."""
    weights = packed.weights
    probs = F.softmax(F.linear(norm, weights.router[layer]), dim=-1, dtype=torch.float32)
    weight, expert = probs.topk(config.experts_per_token, dim=-1)
    if config.norm_topk_prob:
        weight = weight / weight.sum(dim=-1, keepdim=True)
    weight = weight.to(norm.dtype)[0]
    expert = expert[0]
    k = config.experts_per_token
    gate_up = torch.bmm(packed.gate_up[layer][expert], norm.expand(k, -1)[..., None])[..., 0]
    gate, up = gate_up.chunk(2, dim=-1)
    y = torch.bmm(weights.down[layer][expert], (F.silu(gate) * up)[..., None])[..., 0]
    return (y * weight[:, None]).sum(dim=0, keepdim=True)


@torch.no_grad()
def decode_step_fast(
    packed: FastQwenWeights,
    token: torch.Tensor,
    pos: int,
    k_cache: tuple[torch.Tensor, ...],
    v_cache: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """SDPA baseline over per-layer views of the reference's (layers, kv heads, max_len, head
    dim) caches.

    The views and the ``index_copy_`` keep Inductor from copying whole caches: only the new
    position is written, under compile too.
    """
    weights, config = packed.weights, packed.weights.config
    x = F.embedding(token.reshape(1), weights.embed)
    cos, sin = packed.cos[pos][None, None], packed.sin[pos][None, None]
    q_heads, kv_heads, head_dim = config.num_q_heads, config.num_kv_heads, config.head_dim
    new_position = packed.positions[pos : pos + 1]
    for layer in range(config.num_layers):
        norm = rms_norm(x, weights.input_ln[layer], config.rms_eps)
        qkv = baseline_linear(norm, packed.qkv[layer], packed.gemm_rows)
        q, k, v = qkv.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), -1)
        q = rms_norm(q.reshape(1, q_heads, head_dim), weights.q_norm[layer], config.rms_eps)
        k = rms_norm(k.reshape(1, kv_heads, head_dim), weights.k_norm[layer], config.rms_eps)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        k_cache[layer].index_copy_(1, new_position, k.transpose(0, 1))
        v_cache[layer].index_copy_(
            1, new_position, v.reshape(1, kv_heads, head_dim).transpose(0, 1)
        )
        attention = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k_cache[layer][:, : pos + 1][None],
            v_cache[layer][:, : pos + 1][None],
            enable_gqa=True,
        ).reshape(1, q_heads * head_dim)
        x = x + baseline_linear(attention, weights.o_proj[layer], packed.gemm_rows)
        norm = rms_norm(x, weights.post_ln[layer], config.rms_eps)
        if config.is_moe:
            x = x + moe_mlp_fast(config, packed, layer, norm)
        else:
            gate_up = baseline_linear(norm, packed.gate_up[layer], packed.gemm_rows)
            gate, up = gate_up.chunk(2, dim=-1)
            x = x + baseline_linear(F.silu(gate) * up, weights.down[layer], packed.gemm_rows)
    x = rms_norm(x, weights.final_norm, config.rms_eps)
    return baseline_linear(x, weights.lm_head, packed.gemm_rows).reshape(-1).float()


@torch.no_grad()
def profile_step(fn: Callable[[], torch.Tensor]) -> dict:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    kernels = {}
    categories = {"gemm_gemv": 0.0, "attention": 0.0, "elementwise_other": 0.0}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA or event.device_time <= 0:
            continue
        kernels[event.name] = kernels.get(event.name, 0.0) + event.device_time
        name = event.name.lower()
        category = (
            "attention"
            if any(word in name for word in ("native_sdpa", "flash", "fmha", "lean_reduction"))
            else "gemm_gemv"
            if any(word in name for word in ("gemm", "gemv", "matmul", "nvjet", "_mm_"))
            else "elementwise_other"
        )
        categories[category] += event.device_time
    return {
        "category_us": categories,
        "top_kernel_us": sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15],
        "kernel_count": sum(
            1
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA and event.device_time > 0
        ),
    }


@torch.no_grad()
def time_step(
    fn: Callable[[], torch.Tensor],
    *,
    cuda_graph: bool = False,
    warmup: int = 20,
    iters: int = 100,
    settle_seconds: float = 1.0,
) -> CudaBenchmarkStats:
    """Time one fixed-input step in us; non-graph calls include host gaps.

    ``fn`` must be capture-safe when ``cuda_graph`` is set. Compile, capture and warmup are
    excluded. Every replay writes the cache at the same position; this is not a growing
    decode.
    """
    if warmup < 1:
        raise ValueError("warmup must be positive (preflight is required before timing)")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            expected = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if cuda_graph:
        expected = expected.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected)
        settle([graph], settle_seconds)
        return benchmark_cuda_graph_stats(graph, num_iters=iters, warmup_iters=warmup)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return CudaBenchmarkStats.from_samples(
        [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]
    )


def bytes_per_step(weights: QwenWeights, ctx_len: int) -> int:
    """Logical bytes one decode step must read: every weight once (a MoE layer only its
    top-k experts), the populated KV cache and the new KV row, and the fp32 logits."""
    config = weights.config
    tensors = weights.tensors()
    weight_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if config.is_moe:
        experts = sum(
            tensors[n].numel() * tensors[n].element_size() for n in ("gate", "up", "down")
        )
        weight_bytes -= experts * (1 - config.experts_per_token / config.num_experts)
    kv_bytes = 2 * config.num_layers * config.num_kv_heads * config.head_dim * (ctx_len + 2) * 2
    return weight_bytes + kv_bytes + config.vocab_size * 4


@torch.no_grad()
def measure_hbm_rooflines(buffer_gb: float = 2.0, iters: int = 100) -> dict[str, float]:
    elements = int(buffer_gb * 1e9) // 2
    source = torch.empty(elements, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1)
    destination = torch.empty_like(source)
    copy = time_step(lambda: destination.copy_(source), cuda_graph=True, iters=iters)
    read = time_step(lambda: source.sum(dtype=torch.float32), cuda_graph=True, iters=iters)
    return {
        "buffer_bytes": elements * 2,
        "copy_read_write_GB_s": elements * 4 / copy.median_us / 1e3,
        "read_reduction_GB_s": elements * 2 / read.median_us / 1e3,
    }


@torch.no_grad()
def trace_megakernel(weights, max_len: int, token, pos: int, caches, path: Path) -> dict:
    """One warmed megakernel step traced with globaltimer (``qwen_megakernel_trace``): per-layer
    phase spans and median input waits in us, and the Perfetto trace at ``path``."""
    runner = QwenMegakernel(weights, max_len, trace="globaltimer")
    runner(weights, token, pos, *caches)  # compile, warm, and set the step's token and position
    path.parent.mkdir(parents=True, exist_ok=True)
    with qwen_megakernel_trace.trace_session(runner, trace_path=str(path)) as session:
        runner.launch(*caches)
    return {
        "trace": str(path),
        "phase_span_us": qwen_megakernel_trace.phase_spans(session.ranges, runner.table),
        "input_wait_us": qwen_megakernel_trace.input_waits(session.ranges, runner.table),
    }


def check_budget(backend: str, metrics: dict) -> None:
    """Abort on non-finite logits; only warn when one step misses the error budget. Eager
    bf16's own error varies about 5x across prompts on deep models (Qwen3-32B at ctx 1024,
    six prompts: rms 0.024 to 0.126 against fp32), so a single-step miss is often noise; the
    megakernel tests own correctness."""
    if not math.isfinite(metrics["test_fp32_max_err"]):
        raise AssertionError(f"{backend} logits are not finite")
    if not metrics["passed"]:
        print(f"warning: {backend} misses this step's error budget: {metrics}", flush=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096, 8192])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-gb", type=float, default=2.0)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--compile-mode", default="default", choices=["default", "max-autotune-no-cudagraphs"]
    )
    parser.add_argument("--gemm-rows", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-roofline", action="store_true")
    parser.add_argument(
        "--skip-megakernel", action="store_true", help="skip the warpdrive decode megakernel"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="also trace one megakernel step per context with the globaltimer backend: write "
        "qwen_megakernel_ctx<N>.pftrace here and report per-phase spans and input waits",
    )
    args = parser.parse_args()
    weights = load_qwen3_weights(args.model)
    packed = prepare_fast_weights(weights, max(args.contexts) + 1, args.gemm_rows)
    report = {
        "metadata": provenance(
            {
                "bench": Path(__file__),
                "reference": Path(__file__).with_name("qwen_megakernel_ref.py"),
                "megakernel": Path(__file__).with_name("qwen_megakernel.py"),
            },
            vars(args),
        ),
        "rows": [],
    }
    print(json.dumps(report["metadata"], default=str), flush=True)
    if not args.skip_roofline:
        report["rooflines"] = measure_hbm_rooflines(args.buffer_gb, args.iters)
        print(json.dumps(report["rooflines"]), flush=True)
    print("ctx backend median_us p05_us p95_us bytes_step logical_GB_s", flush=True)
    for ctx_len in args.contexts:
        # The megakernel streams 64-position KV blocks; the capacity must be a multiple.
        max_len = math.ceil((ctx_len + 1) / 64) * 64
        caches, token, pos = make_decode_state(weights, ctx_len, max_len, args.seed)
        eager = partial(decode_step_ref, weights, token, pos, *caches)
        reference = eager().clone()
        # The fp32 oracle runs on the same bf16 caches, so the budget covers this step only.
        high = decode_step_ref(weights, token, pos, *caches, dtype=torch.float32)
        error = compare_logits(reference, reference, high)
        print(f"ctx={ctx_len} error_budget={json.dumps(error)}", flush=True)
        layer_caches = tuple(tuple(cache.unbind(0)) for cache in caches)
        fast = partial(decode_step_fast, packed, token, pos, *layer_caches)
        metrics = compare_logits(fast(), reference, high)
        report.setdefault("correctness", {})[str(ctx_len)] = {"reference": error, "fast": metrics}
        print(f"ctx={ctx_len} fast_correctness={json.dumps(metrics)}", flush=True)
        check_budget("fast", metrics)
        variants = [("eager", eager, False), ("fast_graph", fast, True)]
        if not weights.config.is_moe:  # the MoE reference syncs to loop over its experts
            variants.insert(1, ("eager_graph", eager, True))
        if args.profile:
            profiles = report.setdefault("profiles", {}).setdefault(str(ctx_len), {})
            if weights.config.is_moe:  # profile_step captures a graph; the reference cannot
                print(f"ctx={ctx_len} reference_profile skipped: the MoE reference syncs")
            else:
                profiles["reference"] = profile_step(eager)
                print(
                    f"ctx={ctx_len} reference_profile={json.dumps(profiles['reference'])}",
                    flush=True,
                )
        if not args.skip_compile:
            compiled = torch.compile(decode_step_fast, mode=args.compile_mode, fullgraph=True)
            compiled_step = partial(compiled, packed, token, pos, *layer_caches)
            for cache in caches:
                cache[:, :, pos].fill_(float("nan"))
            actual = compiled_step()
            torch.cuda.synchronize()
            metrics = compare_logits(actual, reference, high)
            report["correctness"][str(ctx_len)]["compiled"] = metrics
            print(f"ctx={ctx_len} compiled_correctness={json.dumps(metrics)}", flush=True)
            check_budget("compiled", metrics)
            variants.extend(
                [("compiled", compiled_step, False), ("compiled_graph", compiled_step, True)]
            )
        if args.profile:
            profiles["fast"] = profile_step(fast if args.skip_compile else compiled_step)
            print(f"ctx={ctx_len} fast_profile={json.dumps(profiles['fast'])}", flush=True)
        if not args.skip_megakernel:
            runner = QwenMegakernel(weights, max_len)
            for cache in caches:
                cache[:, :, pos].fill_(float("nan"))
            metrics = compare_logits(runner(weights, token, pos, *caches), reference, high)
            report["correctness"][str(ctx_len)]["megakernel"] = metrics
            print(f"ctx={ctx_len} megakernel_correctness={json.dumps(metrics)}", flush=True)
            check_budget("megakernel", metrics)

            def megakernel_step(runner=runner, caches=caches):
                runner.launch(*caches)  # step_info still holds (token, pos)
                return runner.logits

            variants.append(("megakernel_graph", megakernel_step, True))
            if args.trace_dir:
                report.setdefault("traces", {})[str(ctx_len)] = trace_megakernel(
                    weights,
                    max_len,
                    token,
                    pos,
                    caches,
                    args.trace_dir / f"qwen_megakernel_ctx{ctx_len}.pftrace",
                )
                print(
                    f"ctx={ctx_len} megakernel_trace={json.dumps(report['traces'][str(ctx_len)])}",
                    flush=True,
                )
        for name, fn, graph in variants:
            stats = time_step(fn, cuda_graph=graph, iters=args.iters, warmup=args.warmup)
            moved = bytes_per_step(weights, ctx_len)
            row = {
                "ctx": ctx_len,
                "backend": name,
                "median_us": stats.median_us,
                "p05_us": stats.p05_us,
                "p95_us": stats.p95_us,
                "bytes_step": moved,
                "logical_GB_s": moved / stats.median_us / 1e3,
                "samples_us": stats.samples_us,
                "error_budget": error,
            }
            report["rows"].append(row)
            print(
                f"{ctx_len} {name} {stats.median_us:.2f} {stats.p05_us:.2f} {stats.p95_us:.2f} "
                f"{moved} {row['logical_GB_s']:.2f}",
                flush=True,
            )
            if args.output:
                args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
