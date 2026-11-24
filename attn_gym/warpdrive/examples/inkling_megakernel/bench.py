"""Inkling 2 Mini batch-one decode timings in CUDA graphs: each launch kind of the megakernel
decode step (``megakernel.InklingDecoder``) and the whole step, with the checkpoint's experts
as stored (``quant.load_quantized_inkling_weights``) and, in the same process, dequantized to
bf16 (``--bf16``, the default; also times the torch step ``fast.decode_step`` on them).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.bench --context 1100

The cache is prefilled with ``context`` tokens by the oracle (``0 < context < max_len``);
every timed call decodes at that position with the inputs a real megakernel step produced
there, rewriting the same cache rows (a fixed-position, reused-buffer workload: weights larger
than L2 stream from DRAM, the caches and activations may stay in L2). Launch kinds are split
by attention kind, message gates and experts format; a kind's graph replays ``reps`` calls of
its first launch of the step back to back (with that launch's own message state, snapshotted
when the step ran it) and reports the time per call. The breakdown multiplies each kind's time
by its launches per step; the residual (step minus that sum) holds the gaps between launches,
whatever a repeated launch gains over its first run inside a step (a layer's ~54 MB of
attention weights may stay partly in L2 across repeats despite evict-first loads) and the
differences between a kind's launches. Graphs are timed in ``rounds`` interleaved rounds
(rotating order) and each row reports the median of the round medians. Default DVFS.

Every launch of a visit occupies the whole GPU (a cooperative grid of one CTA per SM): run the
benchmark alone on its GPU.
"""

import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Annotated

import torch
import typer

from attn_gym.warpdrive.examples.bench_support import capture, provenance, time_graphs
from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    GateArgs,
    InklingDecoder,
    check_block_tokens,
    experts_format,
)
from attn_gym.warpdrive.examples.megakernel.constants import BLOCK


def snapshot(value):
    """``value`` with the message state that later launches of the step overwrite copied: a
    ``GateArgs``' messages and message halves (``msg_kernel`` rewrites them every pass). The
    other arguments (weights, per-visit states, caches, position) keep their contents through
    the step."""
    if isinstance(value, GateArgs):
        return value._replace(mhalf=value.mhalf.clone(), messages=value.messages.clone())
    return value


class Recorder:
    """A runner whose calls also append themselves to ``calls``, ready to replay with the
    operands they read (``snapshot``)."""

    def __init__(self, runner, calls: list):
        self.runner, self.calls = runner, calls

    def __call__(self, *args, **kwargs):
        frozen = [snapshot(a) for a in args]
        self.calls.append(
            partial(self.runner, *frozen, **{k: snapshot(v) for k, v in kwargs.items()})
        )
        return self.runner(*args, **kwargs)


def launch_kinds(weights: ref.InklingWeights) -> dict[str, tuple[str, int, int]]:
    """``{kind: (runner, call, launches per step)}``: each launch kind of a step of
    ``weights`` (visits by attention kind, message gates and experts format), the runner and
    the index of the call that represents it (the first of the kind)."""
    config = weights.config
    kinds = {}

    def add(kind, runner, call):
        first, count = kinds.get(kind, (runner, call, 0))[1:]
        kinds[kind] = (runner, first, count + 1)

    for visit, layer, p in config.visits():
        layer_kind = "full" if config.is_full(layer) else "sliding"
        messages = " + messages" if config.is_loop(layer) and p > 0 else ""
        fmt = experts_format(weights.layers[layer])
        add(f"visit{messages}, {layer_kind}, {fmt}", "visit", visit)
    for p in range(1, config.num_passes):
        add("messages", "msg", p - 1)
    add("head", "head", 0)
    return kinds


def record_calls(decoder: InklingDecoder, tokens, pos, cache, block=False) -> dict[str, list]:
    """Every launch of one ``decoder`` step of ``tokens`` (``block``: a ``decode_block`` step),
    by runner (``visit``, ``msg``, ``head``), ready to replay."""
    runners = decoder.block_runners if block else decoder.token_runners
    calls = {name: [] for name in ("visit", "msg", "head")}
    recorders = {name: Recorder(getattr(runners, name), calls[name]) for name in calls}
    decoder.step(dataclasses.replace(runners, **recorders), tokens.reshape(-1), pos, cache)
    return calls


def union_sizes(decoder: InklingDecoder, tokens, pos, cache) -> list[int]:
    """The expert union's size at each visit of one ``decode_block`` step of ``tokens``
    (eager, synchronizing after every visit)."""
    sizes = []
    runner = decoder.block_runners.visit

    def visit(*args, **kwargs):
        out = runner(*args, **kwargs)
        sizes.append(int(runner.union_size().item()))
        return out

    decoder.step(dataclasses.replace(decoder.block_runners, visit=visit), tokens, pos, cache)
    return sizes


@torch.no_grad()
def main(
    context: Annotated[int, typer.Option(help="prefilled tokens; the decoded position")] = 1100,
    max_len: Annotated[int, typer.Option(help="cache capacity (a multiple of 64)")] = 2048,
    reps: Annotated[int, typer.Option(help="calls per launch graph replay")] = 20,
    rounds: Annotated[int, typer.Option(help="interleaved timing rounds (odd)")] = 5,
    iters: Annotated[int, typer.Option(help="replays per graph per round")] = 50,
    bf16: Annotated[bool, typer.Option(help="also time bf16 experts and the torch step")] = True,
    block_tokens: Annotated[
        int, typer.Option(help="also time decode_block steps of this many tokens (0: none)")
    ] = 8,
    checkpoint: str = "thinkingmachines/Inkling-2-Mini",
    json_out: Annotated[Path | None, typer.Option(help="write results, provenance")] = None,
):
    # Checked before the weights load: the capacity, the block size and the positions written.
    if max_len % BLOCK:
        raise typer.BadParameter(f"--max-len must be a multiple of {BLOCK}, got {max_len}")
    if block_tokens:
        check_block_tokens(block_tokens)
    if not 0 < context <= max_len - max(block_tokens, 1):
        raise typer.BadParameter(
            f"need 0 < context and context + max(block_tokens, 1) <= max_len, got {context}, "
            f"{block_tokens} and {max_len}"
        )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    weights = quant.load_quantized_inkling_weights(checkpoint)
    memory = {"quantized": torch.cuda.memory_allocated() - before}
    config = weights.config
    cache = ref.InklingCache.empty(config, max_len)
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(1000, 100000, (context,), device="cuda", generator=gen)
    token = quant.forward_tokens(weights, prompt, 0, cache).argmax().reshape(1)
    pos = torch.tensor([context], device="cuda", dtype=torch.int32)
    block = None
    if block_tokens:
        # The block: the megakernel's greedy continuation (a fully accepted verify block),
        # decoded on a scratch copy of the cache.
        scratch = ref.InklingCache(**{k: v.clone() for k, v in vars(cache).items()})
        greedy = InklingDecoder(weights)
        block, step_pos = [token], pos.clone()
        for _ in range(block_tokens - 1):
            block.append(greedy(block[-1], step_pos, scratch).argmax().reshape(1))
            step_pos += 1
        block = torch.cat(block)
        del scratch, greedy  # untimed: their memory goes to the bf16 model
    models = {"quantized": weights}
    if bf16:
        allocated = torch.cuda.memory_allocated()
        models["bf16"] = dense = quant.dequantized_weights(weights)
        # The bf16 model shares every tensor but the experts with the quantized one.
        stored = sum(e.nbytes for w in weights.layers for e in (w.w13, w.w2))
        memory["bf16"] = memory["quantized"] - stored + torch.cuda.memory_allocated() - allocated

    kinds = {name: launch_kinds(model) for name, model in models.items()}
    graphs = {}
    unions = {}
    for name, model in models.items():
        decoder = InklingDecoder(model, block_tokens=block_tokens or None)
        calls = record_calls(decoder, token, pos, cache)
        for kind, (runner, call, _) in kinds[name].items():
            graphs[f"{name}: {kind}"] = capture(calls[runner][call], reps)
        graphs[f"decode step: megakernel, {name}"] = capture(
            partial(decoder, token, pos, cache), 1
        )
        if block_tokens:
            calls = record_calls(decoder, block, pos, cache, block=True)
            for kind, (runner, call, _) in kinds[name].items():
                graphs[f"{name}, T={block_tokens}: {kind}"] = capture(calls[runner][call], reps)
            graphs[f"block step: megakernel, {name}"] = capture(
                partial(decoder.decode_block, block, pos, cache), 1
            )
            unions[name] = union_sizes(decoder, block, pos, cache)
    if bf16:
        graphs["decode step: torch, bf16"] = capture(
            partial(decode_step, dense, token, pos, cache), 1
        )
    results = time_graphs(graphs, rounds, iters)
    for name in models:
        header = f"{name + ' experts: launch kind':40s} {'us/launch':>10s} {'per step':>9s}"
        if block_tokens:
            header += f" {'T=' + str(block_tokens):>10s} {'ratio':>6s}"
        print(header)
        launches = blocks = 0.0
        for kind, (_, _, count) in kinds[name].items():
            us = results[f"{name}: {kind}"]
            launches += count * us
            line = f"{kind:40s} {us:10.1f} {count:9d}"
            if block_tokens:
                block_us = results[f"{name}, T={block_tokens}: {kind}"]
                blocks += count * block_us
                line += f" {block_us:10.1f} {block_us / us:6.2f}"
            print(line)
        step = results[f"decode step: megakernel, {name}"]
        print(f"{'launches (sum, us/step)':40s} {launches:10.1f} {'':9s} {blocks:10.1f}")
        if block_tokens:
            block_step = results[f"block step: megakernel, {name}"]
            print(f"{'step (us)':40s} {step:10.1f} {'':9s} {block_step:10.1f} "
                  f"{block_step / step:6.2f}")  # fmt: skip
            sizes = unions[name]
            print(f"{'expert union per visit':40s} mean {sum(sizes) / len(sizes):.1f}, min "
                  f"{min(sizes)}, max {max(sizes)} of {block_tokens} x top-k: {sizes}")  # fmt: skip
        print(f"{'model weights (GB)':40s} {memory[name] / 1e9:10.1f}")
    for name, us in results.items():
        if name.startswith(("decode step", "block step")):
            print(f"{name:40s} {us:10.1f} us")
    if json_out is not None:
        here = Path(__file__).parent
        sources = {p.stem: p for p in sorted(here.glob("*.py"))}
        arguments = {"context": context, "max_len": max_len, "reps": reps, "rounds": rounds,
                     "iters": iters, "bf16": bf16, "block_tokens": block_tokens}  # fmt: skip
        json_out.write_text(
            json.dumps(
                {
                    "results_us": results,
                    "weight_bytes": memory,
                    "union_sizes": unions,
                    **provenance(sources, arguments),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    typer.run(main)
