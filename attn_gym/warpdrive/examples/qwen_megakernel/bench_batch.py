"""Batched decoding sweep of the Qwen3 decode megakernel (tcgen05 engine) against torch.compile +
CUDA graph of the same batched step (``decode_step_fast_batch``: one weight read per step, SDPA
with per-sequence length masks, F.grouped_mm experts): step time and tokens per second against
the batch, with the CUDA-core engine at batch one for reference.

Every sequence decodes at the same context. CUDA-graph replay medians (`time_step`)::

    python -m attn_gym.warpdrive.examples.qwen_megakernel.bench_batch \
        --model Qwen/Qwen3-0.6B --batches 1 2 4 8 16 32 64 --contexts 256 1024 --plot sweep.png

Megakernel batches whose GEMV inputs do not fit in shared memory, and baseline batches that run
out of device memory, are reported and skipped.
"""

import argparse
import json
from pathlib import Path

import torch

from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel
from attn_gym.warpdrive.examples.qwen_megakernel.bench import (
    decode_step_fast_batch,
    prepare_fast_weights,
    time_step,
)
from attn_gym.warpdrive.examples.qwen_megakernel.ref import load_qwen3_weights, make_decode_state


def time_runner(runner, caches, iters: int) -> float:
    def step():
        runner.launch(*caches)
        return runner.all_logits

    return time_step(step, cuda_graph=True, iters=iters, settle_seconds=0.2).median_us


@torch.no_grad()
def time_compiled(weights, caches, token, pos, batch, max_len, iters):
    """torch.compile + CUDA graph of ``decode_step_fast_batch`` for ``batch`` copies of the state.
    Cache rows past the position are zeroed: masked SDPA reads them (see the function)."""
    packed = prepare_fast_weights(weights, max_len)
    layers = weights.config.num_layers
    k, v = (c.expand(batch, *c.shape).contiguous() for c in caches)
    k[..., pos:, :] = 0
    v[..., pos:, :] = 0
    tokens = token.reshape(1).repeat(batch)
    positions = torch.full((batch,), pos, device=token.device)
    views = (
        tuple(k[:, layer] for layer in range(layers)),
        tuple(v[:, layer] for layer in range(layers)),
    )
    step = torch.compile(decode_step_fast_batch)

    def run():
        return step(packed, tokens, positions, *views)

    return time_step(run, cuda_graph=True, iters=iters, settle_seconds=0.2).median_us


@torch.no_grad()
def sweep(
    weights, batches: list[int], ctx: int, iters: int, compiled: bool = True, stream: bool = False
) -> list[dict]:
    """One row per engine and batch: step time in us and tokens per second."""
    max_len = ctx + 64
    caches, token, pos = make_decode_state(weights, ctx, max_len, seed=0)
    runner = QwenMegakernel(weights, max_len)
    runner(weights, token, pos, *[c.clone() for c in caches])
    us = time_runner(runner, [c.clone() for c in caches], iters)
    rows = [{"engine": "cuda_core", "batch": 1, "ctx": ctx, "us": us, "tokens_per_s": 1e6 / us}]
    for batch in batches:
        if compiled:
            try:
                us = time_compiled(weights, caches, token, pos, batch, max_len, iters)
                rows.append(
                    {"engine": "compiled_graph", "batch": batch, "ctx": ctx, "us": us,
                     "tokens_per_s": batch * 1e6 / us}
                )  # fmt: skip
                print(json.dumps(rows[-1]), flush=True)
            except torch.OutOfMemoryError:
                print(
                    f"ctx={ctx} batch={batch} compiled baseline skipped: out of memory", flush=True
                )
            torch.cuda.empty_cache()
        for streams in (False, True) if stream else (False,):
            engine = "tcgen05_stream" if streams else "tcgen05"
            try:
                runner = QwenMegakernel(
                    weights, max_len, engine="tcgen05", batch=batch, stream_inputs=streams
                )
            except ValueError as error:  # the batch's GEMV inputs leave too few ring stages
                print(f"ctx={ctx} batch={batch} {engine} skipped: {error}", flush=True)
                continue
            batched = [
                c.expand(batch, *c.shape).contiguous() if batch > 1 else c.clone() for c in caches
            ]
            runner.decode(token.reshape(1).repeat(batch), [pos] * batch, *batched)
            us = time_runner(runner, batched, iters)
            rows.append(
                {"engine": engine, "batch": batch, "ctx": ctx, "us": us,
                 "tokens_per_s": batch * 1e6 / us, "stages": runner.op.stages}
            )  # fmt: skip
            print(json.dumps(rows[-1]), flush=True)
            del runner, batched
            torch.cuda.empty_cache()
    return rows


def plot(rows: list[dict], title: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, ctx in enumerate(sorted({r["ctx"] for r in rows})):
        color = f"C{i}"
        cc = [r for r in rows if r["ctx"] == ctx and r["engine"] == "cuda_core"]
        series = (("tcgen05", "-o", "megakernel"), ("tcgen05_stream", ":^", "streamed inputs"))
        for engine, style, name in series:
            tc = [r for r in rows if r["ctx"] == ctx and r["engine"] == engine]
            if not tc:
                continue
            batches = [r["batch"] for r in tc]
            label = f"{name}, ctx {ctx}"
            left.plot(batches, [r["tokens_per_s"] for r in tc], style, color=color, label=label)
            right.plot(batches, [r["us"] / 1e3 for r in tc], style, color=color, label=label)
        base = [r for r in rows if r["ctx"] == ctx and r["engine"] == "compiled_graph"]
        if base:
            bb = [r["batch"] for r in base]
            left.plot(
                bb,
                [r["tokens_per_s"] for r in base],
                "--s",
                color=color,
                alpha=0.7,
                label=f"torch.compile + graph, ctx {ctx}",
            )
            right.plot(bb, [r["us"] / 1e3 for r in base], "--s", color=color, alpha=0.7)
        for ax, key in ((left, "tokens_per_s"), (right, "us")):
            value = cc[0][key] / (1e3 if key == "us" else 1)
            ax.plot([1], [value], "*", color=color, markersize=13)
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch (sequences per step)")
        ax.grid(True, which="both", alpha=0.3)
    left.set_yscale("log")
    left.set_ylabel("tokens per second")
    right.set_ylabel("step time (ms)")
    left.plot([], [], "*", color="gray", markersize=11, label="CUDA-core engine, batch 1")
    left.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--contexts", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--skip-compile", action="store_true", help="skip the torch.compile + CUDA graph baseline"
    )
    parser.add_argument(
        "--stream-inputs",
        action="store_true",
        help="also time the megakernel with streamed GEMV inputs (QwenMegakernel(stream_inputs=True))",
    )
    parser.add_argument("--output", type=Path, help="write the rows as JSON")
    parser.add_argument("--plot", type=Path, help="write a throughput / step-time plot")
    args = parser.parse_args()
    weights = load_qwen3_weights(args.model)
    # One compile per (batch, context): every shape gets its own graph.
    torch._dynamo.config.recompile_limit = 4 * len(args.batches) * len(args.contexts)
    compiled = not args.skip_compile
    rows = [
        row
        for ctx in args.contexts
        for row in sweep(weights, args.batches, ctx, args.iters, compiled, args.stream_inputs)
    ]
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1))
    if args.plot:
        name = Path(args.model).name
        gpu = torch.cuda.get_device_name()
        plot(rows, f"{name} decode megakernel, tcgen05 engine, {gpu}", args.plot)
        print(args.plot)


if __name__ == "__main__":
    main()
