"""DSpark draft-step timings in CUDA graphs: the megakernel launches (``draft``) against the
torch drafter (``dspark.DSparkDrafter``) on Inkling 2 Mini's drafter, and the step's bandwidth
floor.

    python -m attn_gym.warpdrive.examples.inkling_megakernel.draft_bench --context 1100

A draft step commits ``commit`` (default 3) target feature rows and proposes ``drafts``
(default 7) tokens from an anchor at position ``context``: the megakernel's 3 launches (context
K/V, backbone, head) or the static torch drafter's ``commit`` + ``propose`` of as many drafts
at that fixed anchor and position. Each graph replays ``reps`` steps back to back (fixed
buffers: the drafter's weights stream from DRAM, its cache and activations may stay in L2);
rows report the median of ``rounds`` interleaved round medians, in us per step. The floor
divides the bytes a step must read (the drafter's matrices, the LM head once and the fold weight
once per Markov step) by the bandwidth a cuBLAS GEMV over a 1 GiB matrix reaches in the same
process. Weights: the checkpoint's drafter (with the target's
embedding, its norm and LM head), or random ones of its shapes (``--random``). Each launch
occupies the whole GPU: run alone on its GPU.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import torch
import typer

from attn_gym.warpdrive.examples.bench_support import (
    benchmark_cuda_graph_stats,
    capture,
    provenance,
    time_graphs,
)
from attn_gym.warpdrive.examples.inkling_megakernel.draft import DSparkMegakernel
from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkConfig,
    DSparkDrafter,
    DSparkWeights,
    load_dspark_weights,
    random_weights,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import checkpoint_dir

VOCAB, PADDED_VOCAB = 200058, 200064  # Inkling 2 Mini's LM head and embedding rows


def shared_target_tensors(path: Path):
    """The target's embedding, embedding norm and (unpadded) LM head, which the drafter
    shares, read from the checkpoint directory ``path`` without loading the rest of the
    target."""
    from safetensors import safe_open

    vocab = json.loads((path / "config.json").read_text())["text_config"]["unpadded_vocab_size"]
    names = {
        "embed": "embed.weight",
        "embed_norm": "embed_norm.weight",
        "lm_head": "unembed.weight",
    }
    found = {}
    for shard in sorted(path.glob("model*.safetensors")):
        with safe_open(str(shard), "pt", device="cuda") as handle:
            keys = set(handle.keys())
            for field, name in names.items():
                if f"model.llm.{name}" in keys:
                    found[field] = handle.get_tensor(f"model.llm.{name}").bfloat16()
    found["lm_head"] = found["lm_head"][:vocab].contiguous()
    return SimpleNamespace(**found)


def step_bytes(weights: DSparkWeights, drafts: int) -> int:
    """Bytes a draft step of ``drafts`` drafts reads at least once: every drafter matrix, the
    LM head once and the fold weight once per Markov step."""
    matrices = sum(
        t.nbytes
        for layer in weights.layers
        for t in (layer.wq, layer.wk, layer.wv, layer.wk_ctx, layer.wv_ctx, layer.wo,
                  layer.w_gate, layer.w_up, layer.w_down)
    )  # fmt: skip
    head = weights.lm_head.nbytes + weights.rnn_proj.nbytes
    fold = drafts * weights.fold_weight.nbytes
    return matrices + head + fold


def gemv_bandwidth(reps: int = 20) -> float:
    """Bytes per second of a cuBLAS bf16 GEMV over a 1 GiB matrix (a graph of ``reps``)."""
    matrix = torch.randn(2**17, 4096, device="cuda").bfloat16()
    vector = torch.randn(4096, 1, device="cuda").bfloat16()
    graph = capture(lambda: matrix @ vector, reps).graph
    seconds = benchmark_cuda_graph_stats(graph, num_iters=10).median_us * 1e-6 / reps
    return matrix.numel() * 2 / seconds


@torch.no_grad()
def main(
    context: Annotated[int, typer.Option(help="anchor position (committed tokens)")] = 1100,
    commit: Annotated[int, typer.Option(help="committed rows per step (1-8)")] = 3,
    drafts: Annotated[int, typer.Option(help="megakernel drafts per step (1-7)")] = 7,
    reps: int = 10,
    rounds: int = 5,
    random: Annotated[bool, typer.Option(help="random weights instead of the checkpoint")] = False,
    checkpoint: str = "thinkingmachines/Inkling-2-Mini",
    json_out: Path | None = None,
):
    # The drafters bound commit; the committed rows must also be in the history.
    if not 1 <= commit <= context:
        raise typer.BadParameter(f"need 1 <= commit <= context, got {commit} and {context}")
    if random:
        config = DSparkConfig()
    else:
        path = checkpoint_dir(checkpoint, drafter=True)  # target shards and drafter, one snapshot
        config = DSparkConfig.from_checkpoint(path)
    config.checked_drafts(drafts)  # before the weights load
    if random:
        weights = random_weights(config, VOCAB, PADDED_VOCAB)
    else:
        weights = load_dspark_weights(shared_target_tensors(path), path)
    max_len = context + 64
    taps = len(config.target_visits)
    generator = torch.Generator(device="cuda").manual_seed(0)
    history = torch.randn(context, taps, config.hidden_size, device="cuda", generator=generator)
    history = history.bfloat16()
    anchor = 1234

    kernels = DSparkMegakernel(weights, max_len, num_drafts=drafts)
    kernels.commit_rows(history, 0)
    rows = history[-commit:].contiguous()
    positions = torch.arange(context - commit, context, device="cuda")
    pos = torch.tensor([context], device="cuda")
    anchor_t = torch.tensor([anchor], device="cuda")

    # Static: device positions, capturable.
    reference = DSparkDrafter(weights, max_len, num_drafts=drafts, static=True)
    reference.commit(history, torch.arange(context, device="cuda"))
    hidden = reference.block_states(anchor_t, pos)[1 : 1 + drafts]

    steps = {
        "megakernel context K/V": lambda: kernels.commit(rows, positions),
        "megakernel backbone": lambda: kernels.block_states(anchor_t, pos),
        "megakernel head": lambda: kernels.markov(anchor_t),
        "megakernel draft step": lambda: kernels.draft(anchor_t, pos, rows, positions),
        "torch context K/V": lambda: reference.commit(rows, positions),
        "torch backbone": lambda: reference.block_states(anchor_t, pos),
        "torch head": lambda: reference.markov_drafts(hidden, anchor_t),
        "torch draft step": lambda: (
            reference.commit(rows, positions),
            reference.propose(anchor_t, pos),
        ),
    }
    graphs = {name: capture(fn, reps) for name, fn in steps.items()}
    times = time_graphs(graphs, rounds, iters=20)
    bandwidth = gemv_bandwidth()
    floor_us = step_bytes(weights, drafts) / bandwidth * 1e6
    for name, us in times.items():
        print(f"{name:24s} {us:8.1f} us")
    print(
        f"{drafts}-draft step reads {step_bytes(weights, drafts) / 1e9:.2f} GB: floor "
        f"{floor_us:.1f} us at the GEMV's {bandwidth / 1e12:.2f} TB/s; megakernel "
        f"{times['megakernel draft step']:.1f} us "
        f"({times['torch draft step'] / times['megakernel draft step']:.2f}x the torch step's speed)"
    )
    if json_out is not None:
        sources = {"draft": Path(__file__).with_name("draft.py")}
        record = {"times_us": times, "floor_us": floor_us, "bandwidth": bandwidth}
        record["provenance"] = provenance(
            sources, {"context": context, "commit": commit, "drafts": drafts, "random": random}
        )
        json_out.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    typer.run(main)
