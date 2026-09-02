"""Train a complete packed KDA attention module with context parallelism.

This reuses the transformer-style module from ``kda_training.py`` and distributes both stateful
operations through the reference recipe in ``attn_gym.linear.context_parallel``: a halo
exchange supplies the short convolution's finite history, then affine state summaries supply the
KDA recurrence's full-prefix state. Each rank owns a list of global token ranges chosen here in
plain Python (``token_ranges``), so the same code validates contiguous shards and zig-zag load
balancing. Launch with:

    torchrun --standalone --nproc-per-node=2 examples/kda_context_parallel.py

Add ``--partition zigzag`` to give each rank two mirrored blocks, ``--compute-dtype=float16``
to validate the FP16 route, ``--cuda-graph`` to capture forward and backward together and validate
a replay with changed inputs, and ``--profile`` to export a merged native Perfetto trace using
transformer-nuggets. The example requires one Hopper or datacenter Blackwell GPU per rank.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import torch.nn.functional as F
import typer

from attn_gym.linear.context_parallel import ContextParallelPlan, context_parallel_conv_history
from attn_gym.linear.kda.context_parallel import context_parallel_kda
from attn_gym.testing import kernel_stage, record_distributed_profile
from examples.kda_training import ComputeDTypeOption, KDAAttention, KDAAttentionOutput


class PartitionOption(str, Enum):
    CONTIGUOUS = "contiguous"
    ZIGZAG = "zigzag"


def token_ranges(
    tokens: int, world_size: int, partition: PartitionOption
) -> list[list[tuple[int, int]]]:
    """Choose which global token ranges each rank owns, in its local layout order.

    Contiguous gives rank ``r`` block ``r`` of ``W`` equal blocks. Zig-zag gives it blocks ``r``
    and ``2W - 1 - r`` of ``2W``: the load-balanced layout ring softmax attention uses for causal
    masks, which a hybrid model's KDA layers inherit. Any other assignment is just a different
    list; the plan cuts ranges at sequence boundaries itself.
    """
    blocks = world_size if partition is PartitionOption.CONTIGUOUS else 2 * world_size
    if tokens % blocks:
        raise ValueError(f"{tokens} tokens do not split into {blocks} equal blocks")
    size = tokens // blocks
    owned = [
        [rank] if partition is PartitionOption.CONTIGUOUS else [rank, blocks - 1 - rank]
        for rank in range(world_size)
    ]
    return [[(block * size, (block + 1) * size) for block in blocks_of] for blocks_of in owned]


@dataclass(frozen=True)
class PackedTrainingBatch:
    """Global reference tensors and this rank's fragments of them.

    ``token_ids`` maps local positions to global token ids; ``terminal_index`` lists the local
    fragments that end their sequence and ``terminal_sequences`` the matching global sequence
    ids, so endpoint states can be compared with the unsharded run.
    """

    global_hidden: torch.Tensor
    global_target: torch.Tensor
    global_offsets: torch.Tensor
    local_hidden: torch.Tensor
    local_target: torch.Tensor
    local_offsets: torch.Tensor
    token_ids: torch.Tensor
    terminal_index: torch.Tensor
    terminal_sequences: torch.Tensor
    loss_scale: float


class ContextParallelKDAAttention(KDAAttention):
    """The training example's complete KDA module with distributed state plumbing."""

    def __init__(
        self,
        *args,
        group: dist.ProcessGroup,
        plan: ContextParallelPlan,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cp_group = group
        self.plan = plan

    def short_convolution(
        self,
        qkv: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if initial_state is not None:
            raise ValueError("context parallelism constructs the short-convolution state")
        with kernel_stage("cp/conv/halo", self.enable_graph_annotations):
            initial_state = context_parallel_conv_history(
                qkv, self.plan, self.cp_group, self.qkv_conv1d.kernel_size[0] - 1
            )
        return super().short_convolution(
            qkv,
            initial_state,
            cu_seqlens=cu_seqlens,
            return_final_state=return_final_state,
        )

    def kda_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        *,
        cu_seqlens: torch.Tensor | None = None,
        return_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if initial_state is not None:
            raise ValueError("context parallelism constructs the KDA recurrent state")
        if cu_seqlens is None:
            raise ValueError("context parallel KDA requires local packed sequence boundaries")
        output, final_state = context_parallel_kda(
            q,
            k,
            v,
            gate,
            beta,
            cu_seqlens=cu_seqlens,
            plan=self.plan,
            group=self.cp_group,
            fastmath=self.fastmath,
        )
        return output, final_state if return_final_state else None


def run_training_step(
    module: KDAAttention,
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_index: torch.Tensor,
    loss_scale: float,
    *,
    annotate: bool = False,
) -> tuple[KDAAttentionOutput, tuple[torch.Tensor, ...]]:
    """Run the complete module and differentiate its token and endpoint losses."""
    module.enable_graph_annotations = annotate
    result = module(
        hidden_states,
        cu_seqlens=cu_seqlens,
        return_final_state=True,
    )
    assert result.final_state is not None
    assert result.final_conv_state is not None
    loss = F.mse_loss(result.hidden_states.float(), target, reduction="sum") * loss_scale
    # Only true sequence ends carry a loss; intermediate fragment states are handed downstream.
    loss = loss + 1e-4 * result.final_state[state_index].square().sum()
    loss = loss + 1e-4 * result.final_conv_state[state_index].float().square().sum()
    inputs = (hidden_states, *tuple(module.parameters()))
    with kernel_stage("cp/bwd", annotate, backward=False):
        gradients = torch.autograd.grad(loss, inputs)
    return result, gradients


def validate_against_reference(
    model: ContextParallelKDAAttention,
    reference_model: KDAAttention,
    batch: PackedTrainingBatch,
) -> None:
    """Compare one sharded full-module backward with the unsharded module."""
    result, gradients = run_training_step(
        model,
        batch.local_hidden,
        batch.local_target,
        batch.local_offsets,
        batch.terminal_index,
        batch.loss_scale,
    )
    reference_hidden = batch.global_hidden.detach().clone().requires_grad_()
    reference_result, reference_gradients = run_training_step(
        reference_model,
        reference_hidden,
        batch.global_target,
        batch.global_offsets,
        torch.arange(batch.global_offsets.shape[0] - 1, device=reference_hidden.device),
        batch.loss_scale,
    )

    # Rank partitioning changes accumulation order, so compare at input-precision accuracy.
    assert_close = partial(
        torch.testing.assert_close,
        atol=torch.finfo(model.compute_dtype).eps,
        rtol=torch.finfo(model.compute_dtype).eps,
    )
    assert_close(result.hidden_states, reference_result.hidden_states[:, batch.token_ids])
    assert_close(gradients[0], reference_gradients[0][:, batch.token_ids])
    assert_close(
        result.final_state[batch.terminal_index],
        reference_result.final_state[batch.terminal_sequences],
    )
    assert_close(
        result.final_conv_state[batch.terminal_index],
        reference_result.final_conv_state[batch.terminal_sequences],
    )
    for actual, expected in zip(gradients[1:], reference_gradients[1:], strict=True):
        reduced = actual.detach().clone()
        dist.all_reduce(reduced)
        assert_close(reduced, expected)


def validate_cuda_graph(
    make_model: Callable[[], ContextParallelKDAAttention],
    eager_model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
    annotations: bool,
    profile_path: Path | None,
    device: torch.device,
) -> None:
    """Capture the full distributed backward and compare a changed-input replay."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        # Parameter AccumulateGrad nodes retain their first stream, so the captured path needs
        # a fresh model whose warmup and eager replay oracle both run on this stream.
        model = make_model()
        model.load_state_dict(eager_model.state_dict())
        capture_hidden = batch.local_hidden.detach().clone().requires_grad_()
        run_training_step(
            model,
            capture_hidden.detach().clone().requires_grad_(),
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        )
    warmup_stream.synchronize()
    gc.collect()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=warmup_stream, enable_annotations=annotations):
            graph_result, graph_gradients = run_training_step(
                model,
                capture_hidden,
                batch.local_target,
                batch.local_offsets,
                batch.terminal_index,
                batch.loss_scale,
                annotate=annotations,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        captured_output = graph_result.hidden_states.clone()
        torch.cuda.synchronize(device)

        with torch.no_grad():
            capture_hidden.mul_(0.75)
        with torch.cuda.stream(warmup_stream):
            eager_hidden = capture_hidden.detach().clone().requires_grad_()
            eager_result, eager_gradients = run_training_step(
                model,
                eager_hidden,
                batch.local_target,
                batch.local_offsets,
                batch.terminal_index,
                batch.loss_scale,
            )
        warmup_stream.synchronize()
        graph.replay()
        torch.cuda.synchronize(device)
        if torch.equal(graph_result.hidden_states, captured_output):
            raise AssertionError("CUDA Graph replay did not observe changed inputs")
        torch.testing.assert_close(graph_result, eager_result)
        torch.testing.assert_close(graph_gradients, eager_gradients)
        if profile_path is not None:
            merged_path = record_distributed_profile(
                graph.replay,
                profile_path,
                "cuda_graph_replay",
                device,
            )
            if merged_path is not None:
                print(f"profile={merged_path}", flush=True)
    finally:
        torch.cuda.synchronize(device)
        graph.reset()
        gc.collect()


def profile_eager_step(
    model: ContextParallelKDAAttention,
    batch: PackedTrainingBatch,
    profile_path: Path,
    device: torch.device,
) -> None:
    """Profile one complete eager context-parallel forward and backward."""
    hidden_states = batch.local_hidden.detach().clone().requires_grad_()
    merged_path = record_distributed_profile(
        lambda: run_training_step(
            model,
            hidden_states,
            batch.local_target,
            batch.local_offsets,
            batch.terminal_index,
            batch.loss_scale,
        ),
        profile_path,
        "iteration",
        device,
    )
    if merged_path is not None:
        print(f"profile={merged_path}", flush=True)


def main(
    sequence_lengths: Annotated[
        str,
        typer.Option(help="Comma-separated nonempty packed sequence lengths."),
    ] = "256,1280,512",
    partition: Annotated[
        PartitionOption,
        typer.Option(help="How ranks own fragments of the packed stream."),
    ] = PartitionOption.CONTIGUOUS,
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 256,
    heads: Annotated[int, typer.Option(min=1, help="Number of KDA heads.")] = 2,
    compute_dtype: Annotated[
        ComputeDTypeOption,
        typer.Option(help="Use float16 or bfloat16 projection and kernel inputs."),
    ] = ComputeDTypeOption.BFLOAT16,
    short_conv_kernel_size: Annotated[
        int,
        typer.Option(min=1, help="Causal Q/K/V convolution width."),
    ] = 4,
    cuda_graph: Annotated[
        bool,
        typer.Option(help="Capture forward and backward and validate a changed-input replay."),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(help="Export a merged native Perfetto trace with transformer-nuggets."),
    ] = False,
) -> None:
    """Run and validate the complete packed context-parallel KDA module."""
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    graph_annotations_available = False
    if cuda_graph:
        try:
            from torch.cuda.graph_annotations import is_available
        except ImportError:
            pass
        else:
            graph_annotations_available = is_available()
        if rank == 0 and not graph_annotations_available:
            print(
                "CUDA Graph capture is available, but kernel labels require PyTorch graph "
                "annotations and CUDA >=13.1 tools-ID driver support.",
                flush=True,
            )

    lengths = tuple(int(length) for length in sequence_lengths.split(","))
    global_cu_seqlens = tuple(accumulate(lengths, initial=0))
    tokens = global_cu_seqlens[-1]
    plan = ContextParallelPlan.from_token_ranges(
        global_cu_seqlens, token_ranges(tokens, world_size, partition), rank
    )
    token_ids = plan.global_token_ids(device)

    torch.manual_seed(0)
    model_options = {
        "hidden_size": hidden_size,
        "num_heads": heads,
        "head_dim": 128,
        "short_conv_kernel_size": short_conv_kernel_size,
        "backend": "fused",
        "compute_dtype": getattr(torch, compute_dtype.value),
        "device": device,
    }
    make_model = partial(
        ContextParallelKDAAttention, **model_options, group=dist.group.WORLD, plan=plan
    )
    model = make_model()
    reference_model = KDAAttention(**model_options)
    reference_model.load_state_dict(model.state_dict())

    torch.manual_seed(123)
    global_hidden = torch.randn(1, tokens, hidden_size, device=device)
    global_target = torch.randn_like(global_hidden)
    batch = PackedTrainingBatch(
        global_hidden=global_hidden,
        global_target=global_target,
        global_offsets=torch.tensor(global_cu_seqlens, dtype=torch.int32, device=device),
        local_hidden=global_hidden[:, token_ids].clone().requires_grad_(),
        local_target=global_target[:, token_ids],
        local_offsets=torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=device),
        token_ids=token_ids,
        terminal_index=torch.tensor(plan.terminal, dtype=torch.long, device=device),
        terminal_sequences=torch.tensor(
            [plan.fragments[index].sequence for index in plan.terminal],
            dtype=torch.long,
            device=device,
        ),
        loss_scale=1.0 / global_target.numel(),
    )
    validate_against_reference(model, reference_model, batch)
    gc.collect()

    profile_mode = "cuda_graph" if cuda_graph else "eager"
    profile_path = Path(
        "data",
        f"kda_context_parallel_{profile_mode}_{partition.value}_{compute_dtype.value}"
        f"_w{world_size}_t{tokens}_h{heads}_c{hidden_size}_conv{short_conv_kernel_size}",
    ).resolve()
    if cuda_graph:
        validate_cuda_graph(
            make_model,
            model,
            batch,
            graph_annotations_available,
            profile_path if profile else None,
            device,
        )
    elif profile:
        profile_eager_step(model, batch, profile_path, device)

    mode = " with CUDA Graph replay" if cuda_graph else ""
    print(f"rank {rank}: full packed KDA CP ({partition.value}) passed{mode}", flush=True)
    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=<world_size>")
    typer.run(main)
