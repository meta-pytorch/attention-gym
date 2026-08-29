"""Train a complete packed KDA attention module with context parallelism.

This reuses the transformer-style module from ``kda_training.py`` and distributes both
stateful operations: a halo exchange supplies the short convolution's finite history, then
native affine summaries supply the KDA recurrence's full-prefix state. Each rank owns an
equal contiguous token shard. Launch with:

    torchrun --standalone --nproc-per-node=2 examples/kda_context_parallel.py

Add ``--cuda-graph`` to capture forward and backward together and validate a replay with
changed inputs. Add ``--profile`` to export a merged native Perfetto trace using
transformer-nuggets. The example requires one Blackwell GPU per rank because the fused KDA backend
currently targets SM100 or newer.
"""

from __future__ import annotations

import bisect
import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import torch.nn.functional as F
import typer
from torch.distributed._functional_collectives import all_gather_single

from attn_gym.linear._delta_rule.cute import build_state_grad_summary, build_state_summary
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import (
    _finish_chunk_kda_bwd,
    _prepare_chunk_kda_bwd,
)
from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, chunk_capacity
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _finish_chunk_kda_fwd,
    _normalize_packed_cotangent,
    _prepare_chunk_kda_fwd,
)
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.testing import kernel_stage, record_distributed_profile
from examples.kda_training import KDAAttention, KDAAttentionOutput

# The private prepare/finish seams avoid repeating factor computation around collectives.
# TODO: Promote those seams before moving this orchestration out of the example.
CHUNK_SIZE = 64

# NOTE [One crossing sequence per contiguous rank boundary]
# A rank owns one contiguous interval of the globally ordered packed token stream. Its boundary
# is one cut point and can split at most one logical sequence: only the last local fragment can
# continue forward, and only the first can receive a reverse cotangent. Therefore every rank
# exchanges one fixed `[H,V+K,K]` summary in each direction regardless of how many complete packed
# sequences it owns. A boundary between sequences sends an unused placeholder so collective shapes
# and ordering remain uniform.


@dataclass(frozen=True)
class PackedShard:
    """Packed sequence fragments intersecting one contiguous rank shard."""

    cu_seqlens: tuple[int, ...]
    sequence_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackedTrainingBatch:
    """Global reference tensors and the corresponding rank-local packed shard."""

    global_hidden: torch.Tensor
    global_target: torch.Tensor
    global_offsets: torch.Tensor
    local_hidden: torch.Tensor
    local_target: torch.Tensor
    local_offsets: torch.Tensor
    local_slice: slice
    first_sequence: int
    completed_sequences: int
    sequence_count: int
    loss_scale: float


def partition_packed_sequences(
    global_cu_seqlens: tuple[int, ...],
    world_size: int,
) -> tuple[PackedShard, ...]:
    """Partition nonempty packed sequences into equal contiguous token shards."""
    if len(global_cu_seqlens) < 2 or global_cu_seqlens[0] != 0:
        raise ValueError("global_cu_seqlens must start at zero and contain at least one sequence")
    if any(end <= start for start, end in pairwise(global_cu_seqlens)):
        raise ValueError("this example requires strictly increasing global_cu_seqlens")
    total_tokens = global_cu_seqlens[-1]
    if total_tokens % world_size:
        raise ValueError("total tokens must be divisible by the context-parallel world size")

    shard_tokens = total_tokens // world_size
    shards = []
    for rank in range(world_size):
        rank_start = rank * shard_tokens
        rank_end = rank_start + shard_tokens
        first_sequence = bisect.bisect_right(global_cu_seqlens[1:], rank_start)
        sequence_end = bisect.bisect_left(global_cu_seqlens[:-1], rank_end)
        sequence_ids = tuple(range(first_sequence, sequence_end))
        interior_offsets = (
            offset - rank_start
            for offset in global_cu_seqlens[first_sequence + 1 : sequence_end]
            if rank_start < offset < rank_end
        )
        shards.append(
            PackedShard(
                cu_seqlens=(0, *interior_offsets, shard_tokens),
                sequence_ids=sequence_ids,
            )
        )
    return tuple(shards)


def merge_state(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Merge a packed V-first affine summary into a recurrent state."""
    value_dim = state.shape[-2]
    bias = summary[..., :value_dim, :]
    transition = summary[..., value_dim:, :]
    return state @ transition + bias


class ContextParallelKDAFunction(torch.autograd.Function):
    """Join native affine summaries, NCCL collectives, and the existing KDA core."""

    @staticmethod
    def compose_forward_initial_states(
        gathered: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
        rank: int,
        shards: tuple[PackedShard, ...],
    ) -> torch.Tensor:
        """Compose predecessor transfers into the first local sequence state."""
        shard = shards[rank]
        states = q.new_zeros(
            len(shard.sequence_ids),
            q.shape[2],
            v.shape[-1],
            q.shape[-1],
            dtype=torch.float32,
        )
        first_state = states[0]
        first_sequence = shard.sequence_ids[0]
        for predecessor, predecessor_shard in zip(gathered[:rank], shards[:rank], strict=True):
            if predecessor_shard.sequence_ids[-1] == first_sequence:
                first_state = merge_state(first_state, predecessor)
        return torch.cat((first_state.unsqueeze(0), states[1:]), dim=0)

    @staticmethod
    def compose_reverse_final_states(
        gathered: torch.Tensor,
        d_final_state: torch.Tensor | None,
        initial_state: torch.Tensor,
        rank: int,
        shards: tuple[PackedShard, ...],
    ) -> torch.Tensor:
        """Compose successor reverse transfers into the last local state cotangent."""
        incoming = torch.zeros_like(initial_state) if d_final_state is None else d_final_state
        shard = shards[rank]
        last_sequence = shard.sequence_ids[-1]
        # See NOTE [One crossing sequence per contiguous rank boundary].
        continues = rank + 1 < len(shards) and shards[rank + 1].sequence_ids[0] == last_sequence
        if not continues:
            return incoming

        last_gradient = torch.zeros_like(incoming[-1])
        for successor_rank in range(len(shards) - 1, rank, -1):
            successor = shards[successor_rank]
            if successor.sequence_ids[0] != last_sequence:
                continue
            last_gradient = merge_state(last_gradient, gathered[successor_rank])
        incoming = incoming.clone()
        incoming[-1] += last_gradient
        return incoming

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        gate,
        beta,
        local_cu_seqlens,
        group,
        shards,
        scale,
        fastmath,
        autotune,
        annotate,
    ):
        rank = dist.get_rank(group)
        metadata = prepare_ragged_chunk_metadata(local_cu_seqlens, q.shape[1], CHUNK_SIZE)
        cumulative_gate = _plain_gate_scan_op(
            gate, metadata.cu_seqlens, metadata.chunk_offsets, False
        )
        factors = _prepare_chunk_kda_fwd(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            metadata,
            scale=scale,
            autotune=autotune,
        )

        shard = shards[rank]
        # See NOTE [One crossing sequence per contiguous rank boundary].
        continues = (
            rank + 1 < len(shards) and shard.sequence_ids[-1] == shards[rank + 1].sequence_ids[0]
        )
        if continues:
            start = shard.cu_seqlens[-2]
            with kernel_stage("cp/fwd/summary", annotate):
                summary = build_state_summary(
                    factors.kg[:, start:],
                    factors.w[:, start:],
                    factors.u[:, start:],
                    cumulative_gate[:, start:],
                )
        else:
            summary = q.new_zeros(
                q.shape[2],
                v.shape[-1] + q.shape[-1],
                q.shape[-1],
                dtype=torch.float32,
            )
        with kernel_stage("cp/fwd/all_gather", annotate):
            gathered = summary.new_empty(dist.get_world_size(group), *summary.shape)
            dist.all_gather_single(gathered, summary.contiguous(), group=group)
        with kernel_stage("cp/fwd/exclusive_prefix", annotate):
            initial_state = ContextParallelKDAFunction.compose_forward_initial_states(
                gathered, q, v, rank, shards
            )
        with kernel_stage("cp/fwd/local_output", annotate):
            output, final_state = _finish_chunk_kda_fwd(
                q,
                cumulative_gate,
                factors,
                initial_state,
                None,
                None,
                metadata,
                scale=scale,
                output_final_state=True,
                autotune=autotune,
            )
        assert final_state is not None

        ctx.save_for_backward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            factors.aqk,
            factors.akk,
            initial_state,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
        )
        ctx.group = group
        ctx.shards = shards
        ctx.scale = scale
        ctx.fastmath = fastmath
        ctx.autotune = autotune
        ctx.annotate = annotate
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state):
        (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            aqk,
            akk,
            initial_state,
            cu_seqlens,
            chunk_offsets,
        ) = ctx.saved_tensors
        rank = dist.get_rank(ctx.group)
        shards = ctx.shards
        shard = shards[rank]
        metadata = RaggedChunkMetadata(
            cu_seqlens,
            chunk_offsets,
            chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, CHUNK_SIZE),
            CHUNK_SIZE,
        )
        if d_output is None:
            d_output = torch.zeros_like(v)
        else:
            d_output = _normalize_packed_cotangent(d_output)
        if d_final_state is not None:
            d_final_state = _normalize_packed_cotangent(d_final_state.float())
        prepared = _prepare_chunk_kda_bwd(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            akk,
            d_output,
            initial_state,
            metadata,
            scale=ctx.scale,
            chunk_size=CHUNK_SIZE,
            autotune=ctx.autotune,
        )

        continues_from_previous = (
            rank > 0 and shards[rank - 1].sequence_ids[-1] == shard.sequence_ids[0]
        )
        if continues_from_previous:
            stop = shard.cu_seqlens[1]
            with kernel_stage("cp/bwd/summary", ctx.annotate, backward=False):
                summary = build_state_grad_summary(
                    prepared.qg[:, :stop],
                    prepared.kg[:, :stop],
                    prepared.w[:, :stop],
                    d_output[:, :stop],
                    aqk[:, :stop],
                    cumulative_gate[:, :stop],
                    ctx.scale,
                )
            if d_final_state is not None:
                value_dim = v.shape[-1]
                bias = merge_state(d_final_state[0], summary)
                summary = torch.cat((bias, summary[..., value_dim:, :]), dim=-2)
        else:
            summary = q.new_zeros(
                q.shape[2],
                v.shape[-1] + q.shape[-1],
                q.shape[-1],
                dtype=torch.float32,
            )
        with kernel_stage("cp/bwd/all_gather", ctx.annotate, backward=False):
            gathered = summary.new_empty(dist.get_world_size(ctx.group), *summary.shape)
            dist.all_gather_single(gathered, summary.contiguous(), group=ctx.group)
        with kernel_stage("cp/bwd/exclusive_suffix", ctx.annotate, backward=False):
            incoming = ContextParallelKDAFunction.compose_reverse_final_states(
                gathered,
                d_final_state,
                initial_state,
                rank,
                shards,
            )
        with kernel_stage("cp/bwd/local", ctx.annotate, backward=False):
            dq, dk, dv, d_cumulative, db, _d_initial_state = _finish_chunk_kda_bwd(
                q,
                k,
                v,
                cumulative_gate,
                beta,
                aqk,
                akk,
                d_output,
                incoming,
                initial_state,
                metadata,
                prepared,
                scale=ctx.scale,
                chunk_size=CHUNK_SIZE,
                fastmath=ctx.fastmath,
                autotune=ctx.autotune,
            )
        d_gate = _plain_gate_scan_op(
            d_cumulative,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            True,
        )
        return dq, dk, dv, d_gate, db, None, None, None, None, None, None, None


def context_parallel_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    shards: tuple[PackedShard, ...],
    local_cu_seqlens: torch.Tensor,
    annotate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply packed KDA with native CuTeDSL forward/reverse affine summaries."""
    return ContextParallelKDAFunction.apply(
        q,
        k,
        v,
        gate.float(),
        beta.float().contiguous(),
        local_cu_seqlens,
        group,
        shards,
        q.shape[-1] ** -0.5,
        False,
        False,
        annotate,
    )


class ContextParallelKDAAttention(KDAAttention):
    """The training example's complete KDA module with distributed state plumbing."""

    def __init__(
        self,
        *args,
        group: dist.ProcessGroup,
        shards: tuple[PackedShard, ...],
        global_cu_seqlens: tuple[int, ...],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cp_group = group
        self.shards = shards
        self.global_cu_seqlens = global_cu_seqlens

    @staticmethod
    def compose_conv_initial_states(
        gathered_tails: torch.Tensor,
        shard_tokens: int,
        rank: int,
        shards: tuple[PackedShard, ...],
        global_cu_seqlens: tuple[int, ...],
    ) -> torch.Tensor:
        """Build packed short-convolution histories from fixed-size rank tails."""
        state_length = gathered_tails.shape[1]
        states = gathered_tails.new_zeros(
            len(shards[rank].sequence_ids), state_length, gathered_tails.shape[-1]
        )
        boundary = rank * shard_tokens
        sequence_start = global_cu_seqlens[shards[rank].sequence_ids[0]]
        valid_length = min(state_length, boundary - sequence_start)
        if valid_length:
            stored_per_rank = min(state_length, shard_tokens)
            predecessor_tokens = gathered_tails[:rank, -stored_per_rank:].flatten(0, 1)
            states[0, -valid_length:] = predecessor_tokens[-valid_length:]
        # Keep all ranks in the collective backward when the first sequence starts locally.
        return states + gathered_tails.flatten()[0] * 0

    @staticmethod
    def build_conv_initial_states(
        qkv: torch.Tensor,
        group: dist.ProcessGroup,
        shards: tuple[PackedShard, ...],
        global_cu_seqlens: tuple[int, ...],
        state_length: int,
        annotate: bool,
    ) -> torch.Tensor | None:
        """All-gather rank tails and construct packed short-convolution histories."""
        if state_length == 0:
            return None
        stored = min(state_length, qkv.shape[1])
        tail = F.pad(qkv[0, -stored:], (0, 0, state_length - stored, 0))
        with kernel_stage("cp/conv/halo", annotate):
            gathered = all_gather_single(tail, gather_dim=0, group=group).view(
                dist.get_world_size(group), state_length, qkv.shape[-1]
            )
        return ContextParallelKDAAttention.compose_conv_initial_states(
            gathered,
            qkv.shape[1],
            dist.get_rank(group),
            shards,
            global_cu_seqlens,
        )

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
        state_length = self.qkv_conv1d.kernel_size[0] - 1
        initial_state = self.build_conv_initial_states(
            qkv,
            self.cp_group,
            self.shards,
            self.global_cu_seqlens,
            state_length,
            self.enable_graph_annotations,
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
            group=self.cp_group,
            shards=self.shards,
            local_cu_seqlens=cu_seqlens,
            annotate=self.enable_graph_annotations,
        )
        return output, final_state if return_final_state else None


def run_training_step(
    module: KDAAttention,
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_count: int,
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
    if state_count:
        loss = loss + 1e-4 * result.final_state[:state_count].square().sum()
        loss = loss + 1e-4 * result.final_conv_state[:state_count].float().square().sum()
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
        batch.completed_sequences,
        batch.loss_scale,
    )
    reference_hidden = batch.global_hidden.detach().clone().requires_grad_()
    reference_result, reference_gradients = run_training_step(
        reference_model,
        reference_hidden,
        batch.global_target,
        batch.global_offsets,
        batch.sequence_count,
        batch.loss_scale,
    )

    # Rank partitioning changes accumulation order, so compare at input-precision accuracy.
    assert_close = partial(
        torch.testing.assert_close,
        atol=torch.finfo(model.compute_dtype).eps,
        rtol=torch.finfo(model.compute_dtype).eps,
    )
    assert_close(result.hidden_states, reference_result.hidden_states[:, batch.local_slice])
    assert_close(gradients[0], reference_gradients[0][:, batch.local_slice])
    if batch.completed_sequences:
        local_states = slice(0, batch.completed_sequences)
        global_states = slice(
            batch.first_sequence,
            batch.first_sequence + batch.completed_sequences,
        )
        assert_close(result.final_state[local_states], reference_result.final_state[global_states])
        assert_close(
            result.final_conv_state[local_states],
            reference_result.final_conv_state[global_states],
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
            batch.completed_sequences,
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
                batch.completed_sequences,
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
                batch.completed_sequences,
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
            batch.completed_sequences,
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
    hidden_size: Annotated[int, typer.Option(min=1, help="Transformer hidden size.")] = 256,
    heads: Annotated[int, typer.Option(min=1, help="Number of KDA heads.")] = 2,
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
    shards = partition_packed_sequences(global_cu_seqlens, world_size)
    tokens = global_cu_seqlens[-1]
    shard_tokens = tokens // world_size
    local_slice = slice(rank * shard_tokens, (rank + 1) * shard_tokens)
    local_shard = shards[rank]
    local_cu_seqlens = torch.tensor(
        local_shard.cu_seqlens,
        dtype=torch.int32,
        device=device,
    )
    global_offsets = torch.tensor(global_cu_seqlens, dtype=torch.int32, device=device)
    continues = (
        rank + 1 < world_size and local_shard.sequence_ids[-1] == shards[rank + 1].sequence_ids[0]
    )
    final_state_count = len(local_shard.sequence_ids) - int(continues)

    torch.manual_seed(0)
    model_options = {
        "hidden_size": hidden_size,
        "num_heads": heads,
        "head_dim": 128,
        "short_conv_kernel_size": short_conv_kernel_size,
        "backend": "fused",
        "device": device,
    }
    make_model = partial(
        ContextParallelKDAAttention,
        **model_options,
        group=dist.group.WORLD,
        shards=shards,
        global_cu_seqlens=global_cu_seqlens,
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
        global_offsets=global_offsets,
        local_hidden=global_hidden[:, local_slice].clone().requires_grad_(),
        local_target=global_target[:, local_slice],
        local_offsets=local_cu_seqlens,
        local_slice=local_slice,
        first_sequence=local_shard.sequence_ids[0],
        completed_sequences=final_state_count,
        sequence_count=len(lengths),
        loss_scale=1.0 / global_target.numel(),
    )
    validate_against_reference(model, reference_model, batch)
    gc.collect()

    profile_mode = "cuda_graph" if cuda_graph else "eager"
    profile_path = Path(
        "data",
        f"kda_context_parallel_{profile_mode}_w{world_size}_t{tokens}_h{heads}"
        f"_c{hidden_size}_conv{short_conv_kernel_size}",
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
    print(f"rank {rank}: full packed KDA CP passed{mode}", flush=True)
    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=<world_size>")
    typer.run(main)
