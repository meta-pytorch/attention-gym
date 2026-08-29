"""Run packed KDA context parallelism with native affine-summary kernels.

Each rank owns an equal contiguous token shard. Launch with:

    torchrun --standalone --nproc-per-node=2 examples/kda_context_parallel.py

Add ``--cuda-graph`` to capture forward and backward together and validate a replay with
changed inputs. The example requires one Blackwell GPU per rank because the fused KDA backend
currently targets SM100 or newer.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import accumulate, pairwise

import torch
import torch.distributed as dist
import torch.nn.functional as F

from attn_gym.linear import chunk_kda
from attn_gym.linear._delta_rule.cute import affine_summary_fwd, affine_summary_rev
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

# NOTE [Example-local CP orchestration]
# Keep the distributed autograd wrapper here while the CP contract is experimental, matching the
# repository's ring-attention example. The reusable recurrence kernels live under _delta_rule/cute;
# the private KDA prepare/finish seams prevent duplicate factor computation around collectives.
_CHUNK_SIZE = 64


@dataclass(frozen=True)
class PackedShard:
    """Packed sequence fragments intersecting one contiguous rank shard."""

    cu_seqlens: tuple[int, ...]
    sequence_ids: tuple[int, ...]


@contextmanager
def kernel_stage(name: str, annotate: bool, *, backward: bool = True):
    """Label eager profiler ranges and optionally annotate captured CUDA Graph kernels."""
    annotation = nullcontext()
    if annotate:
        from torch.cuda.graph_annotations import mark_kernels

        annotation = mark_kernels(name, backward=backward)
    with torch.profiler.record_function(name), annotation:
        yield


# NOTE [One crossing sequence per contiguous rank boundary]
# A rank owns one contiguous interval of the globally ordered packed token stream. Its boundary
# is one cut point and can split at most one logical sequence: only the last local fragment can
# continue forward, and only the first can receive a reverse cotangent. Therefore every rank
# exchanges one fixed `[H,V+K,K]` summary in each direction regardless of how many complete packed
# sequences it owns. A boundary between sequences sends an unused placeholder so collective shapes
# and ordering remain uniform.


class _AllGather(torch.autograd.Function):
    """Autograd-enabled all-gather built from public distributed collectives."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        gathered = tensor.new_empty(dist.get_world_size(group), *tensor.shape)
        dist.all_gather_single(gathered, tensor.contiguous(), group=group)
        return gathered

    @staticmethod
    def backward(ctx, grad_gathered: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_tensor = torch.empty_like(grad_gathered[0])
        dist.reduce_scatter_single(
            grad_tensor,
            grad_gathered.contiguous(),
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return grad_tensor, None


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


def _chunk_kda_with_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call public fused KDA and narrow its requested final state to a tensor."""
    output, final_state = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
        impl="fused",
    )
    assert final_state is not None
    return output, final_state


def local_affine_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    annotate: bool = False,
) -> torch.Tensor:
    """Recover each local fragment's ``H_out = H_in @ A + B`` summary."""
    sequences = cu_seqlens.shape[0] - 1
    heads, key_dim, value_dim = q.shape[2], q.shape[-1], v.shape[-1]
    if value_dim != key_dim:
        raise ValueError("the public identity-probe construction requires value_dim == key_dim")

    zero = q.new_zeros(sequences, heads, value_dim, key_dim, dtype=torch.float32)
    identity = torch.eye(key_dim, dtype=torch.float32, device=q.device).expand_as(zero)
    with kernel_stage("cp/fwd/summary_zero", annotate):
        _, bias = _chunk_kda_with_state(q, k, v, gate, beta, zero, cu_seqlens)
    with kernel_stage("cp/fwd/summary_identity", annotate):
        _, identity_final = _chunk_kda_with_state(q, k, v, gate, beta, identity, cu_seqlens)
    transition = identity_final - bias
    return torch.cat((bias, transition), dim=-2)


def apply_summary(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Apply one packed V-first affine summary to a recurrent state."""
    value_dim = state.shape[-2]
    bias = summary[..., :value_dim, :]
    transition = summary[..., value_dim:, :]
    return state @ transition + bias


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
    """Apply zero-state packed KDA to one contiguous context-parallel shard."""
    rank = dist.get_rank(group)
    shard = shards[rank]
    shard_tokens = shard.cu_seqlens[-1]
    if q.shape[0] != 1 or q.shape[1] != shard_tokens:
        raise ValueError(f"rank inputs must have shape [1,{shard_tokens},H,D]")
    if (
        local_cu_seqlens.shape != (len(shard.cu_seqlens),)
        or local_cu_seqlens.dtype != torch.int32
        or local_cu_seqlens.device != q.device
    ):
        raise ValueError("local_cu_seqlens has invalid shape, dtype, or device")

    # See NOTE [One crossing sequence per contiguous rank boundary].
    continues_on_next_rank = (
        rank + 1 < len(shards) and shard.sequence_ids[-1] == shards[rank + 1].sequence_ids[0]
    )
    if continues_on_next_rank:
        boundary_start = shard.cu_seqlens[-2]
        boundary_inputs = tuple(tensor[:, boundary_start:] for tensor in (q, k, v, gate, beta))
        boundary_cu_seqlens = local_cu_seqlens[-2:] - local_cu_seqlens[-2]
        local_summary = local_affine_summaries(
            *boundary_inputs,
            boundary_cu_seqlens,
            annotate,
        )[0]
    else:
        # Keep collective autograd ordering identical when no state crosses this boundary.
        anchor = q.reshape(-1)[0].float() * 0
        local_summary = (
            q.new_zeros(
                q.shape[2],
                v.shape[-1] + q.shape[-1],
                q.shape[-1],
                dtype=torch.float32,
            )
            + anchor
        )
    with kernel_stage("cp/fwd/all_gather", annotate):
        gathered = _AllGather.apply(local_summary, group)

    with kernel_stage("cp/fwd/exclusive_prefix", annotate):
        states = _compose_forward_initial_states(gathered, q, v, rank, shards)
        # Ranks with an empty prefix must still execute all-gather's backward collective.
        states = states + gathered[rank].reshape(-1)[0] * 0

    with kernel_stage("cp/fwd/local_output", annotate):
        return _chunk_kda_with_state(
            q,
            k,
            v,
            gate,
            beta,
            states,
            local_cu_seqlens,
        )


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather one fixed-size tensor from every rank without autograd registration."""
    gathered = tensor.new_empty(dist.get_world_size(group), *tensor.shape)
    dist.all_gather_single(gathered, tensor.contiguous(), group=group)
    return gathered


def _unused_summary(reference: torch.Tensor) -> torch.Tensor:
    """Allocate one ignored fixed-shape summary for collective ordering."""
    return torch.zeros(
        reference.shape[2],
        reference.shape[-1] * 2,
        reference.shape[-1],
        dtype=torch.float32,
        device=reference.device,
    )


def _compose_forward_initial_states(
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
            first_state = apply_summary(first_state, predecessor)
    return torch.cat((first_state.unsqueeze(0), states[1:]), dim=0)


def _compose_reverse_final_states(
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
        last_gradient = apply_summary(last_gradient, gathered[successor_rank])
    incoming = incoming.clone()
    incoming[-1] += last_gradient
    return incoming


class _CuteContextParallelKDA(torch.autograd.Function):
    """Join native affine summaries, NCCL collectives, and the existing KDA core."""

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
        metadata = prepare_ragged_chunk_metadata(local_cu_seqlens, q.shape[1], _CHUNK_SIZE)
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
                summary = affine_summary_fwd(
                    factors.kg[:, start:],
                    factors.w[:, start:],
                    factors.u[:, start:],
                    cumulative_gate[:, start:],
                )
        else:
            summary = _unused_summary(q)
        with kernel_stage("cp/fwd/all_gather", annotate):
            gathered = _all_gather_tensor(summary, group)
        with kernel_stage("cp/fwd/exclusive_prefix", annotate):
            initial_state = _compose_forward_initial_states(gathered, q, v, rank, shards)
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
            chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, _CHUNK_SIZE),
            _CHUNK_SIZE,
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
            chunk_size=_CHUNK_SIZE,
            autotune=ctx.autotune,
        )

        continues_from_previous = (
            rank > 0 and shards[rank - 1].sequence_ids[-1] == shard.sequence_ids[0]
        )
        if continues_from_previous:
            stop = shard.cu_seqlens[1]
            with kernel_stage("cp/bwd/summary", ctx.annotate, backward=False):
                summary = affine_summary_rev(
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
                bias = apply_summary(d_final_state[0], summary)
                summary = torch.cat((bias, summary[..., value_dim:, :]), dim=-2)
        else:
            summary = _unused_summary(q)
        with kernel_stage("cp/bwd/all_gather", ctx.annotate, backward=False):
            gathered = _all_gather_tensor(summary, ctx.group)
        with kernel_stage("cp/bwd/exclusive_suffix", ctx.annotate, backward=False):
            incoming = _compose_reverse_final_states(
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
                chunk_size=_CHUNK_SIZE,
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


def context_parallel_kda_cute(
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
    scale = q.shape[-1] ** -0.5
    fastmath = False
    autotune = False
    return _CuteContextParallelKDA.apply(
        q,
        k,
        v,
        gate.float(),
        beta.float().contiguous(),
        local_cu_seqlens,
        group,
        shards,
        scale,
        fastmath,
        autotune,
        annotate,
    )


def parse_args() -> argparse.Namespace:
    """Parse the example workload."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths",
        default="256,1280,512",
        help="Comma-separated nonempty packed sequence lengths.",
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--summary-backend",
        choices=("cute", "public"),
        default="cute",
        help="Use native CuTeDSL affine summaries or the public zero/identity probes.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture forward and backward and validate a changed-input replay.",
    )
    return parser.parse_args()


def make_inputs(
    tokens: int,
    heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Create stable BF16 KDA inputs for validation."""
    torch.manual_seed(123)
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    gate = F.logsigmoid(torch.randn_like(q))
    beta = torch.sigmoid(torch.randn(shape[:3], dtype=torch.bfloat16, device=device))
    return q, k, v, gate, beta


def ending_state_count(
    shard: PackedShard,
    rank: int,
    shard_tokens: int,
    global_cu_seqlens: tuple[int, ...],
) -> int:
    """Count the prefix of local fragments ending on this rank."""
    rank_end = (rank + 1) * shard_tokens
    last_sequence_ends_later = global_cu_seqlens[shard.sequence_ids[-1] + 1] > rank_end
    return len(shard.sequence_ids) - int(last_sequence_ends_later)


def main() -> None:
    """Run and validate the packed native-summary context-parallel example."""
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    graph_annotations_available = False
    if args.cuda_graph:
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

    lengths = tuple(int(length) for length in args.sequence_lengths.split(","))
    global_cu_seqlens = tuple(accumulate(lengths, initial=0))
    shards = partition_packed_sequences(global_cu_seqlens, world_size)
    tokens = global_cu_seqlens[-1]
    shard_tokens = tokens // world_size
    local_slice = slice(rank * shard_tokens, (rank + 1) * shard_tokens)
    global_inputs = make_inputs(tokens, args.heads, device)
    local_inputs = tuple(
        tensor[:, local_slice].clone().requires_grad_() for tensor in global_inputs
    )
    local_shard = shards[rank]
    local_cu_seqlens = torch.tensor(
        local_shard.cu_seqlens,
        dtype=torch.int32,
        device=device,
    )

    final_state_count = ending_state_count(
        local_shard,
        rank,
        shard_tokens,
        global_cu_seqlens,
    )

    cp_operation = (
        context_parallel_kda_cute if args.summary_backend == "cute" else context_parallel_kda
    )

    def run(inputs: tuple[torch.Tensor, ...], *, annotate: bool = False):
        output, final_state = cp_operation(
            *inputs,
            group=dist.group.WORLD,
            shards=shards,
            local_cu_seqlens=local_cu_seqlens,
            annotate=annotate,
        )
        loss = output.float().sum()
        if final_state_count:
            loss = loss + final_state[:final_state_count].sum()
        with kernel_stage("cp/bwd", annotate, backward=False):
            gradients = torch.autograd.grad(loss, inputs)
        return output, final_state, gradients

    output, final_state, gradients = run(local_inputs)
    global_reference_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in global_inputs
    )
    global_offsets = torch.tensor(global_cu_seqlens, dtype=torch.int32, device=device)
    reference_output, reference_state = chunk_kda(
        *global_reference_inputs,
        cu_seqlens=global_offsets,
        output_final_state=True,
        autotune=False,
        impl="fused",
    )
    assert reference_state is not None
    reference_loss = reference_output.float().sum() + reference_state.sum()
    reference_gradients = torch.autograd.grad(reference_loss, global_reference_inputs)

    torch.testing.assert_close(output, reference_output[:, local_slice], atol=0.15, rtol=0.15)
    if final_state_count:
        first_sequence = local_shard.sequence_ids[0]
        torch.testing.assert_close(
            final_state[:final_state_count],
            reference_state[first_sequence : first_sequence + final_state_count],
            atol=0.15,
            rtol=0.15,
        )
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected[:, local_slice], atol=0.15, rtol=0.15)

    if args.cuda_graph:
        capture_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in local_inputs)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            run(tuple(tensor.detach().clone().requires_grad_() for tensor in capture_inputs))
        warmup_stream.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(
                graph,
                stream=warmup_stream,
                enable_annotations=graph_annotations_available,
            ):
                graph_output, graph_state, graph_gradients = run(
                    capture_inputs,
                    annotate=graph_annotations_available,
                )
            torch.cuda.current_stream().wait_stream(warmup_stream)
            captured_output = graph_output.clone()
            torch.cuda.synchronize(device)

            with torch.no_grad():
                capture_inputs[0].mul_(0.75)
                capture_inputs[2].mul_(0.5)
            eager_inputs = tuple(
                tensor.detach().clone().requires_grad_() for tensor in capture_inputs
            )
            eager_output, eager_state, eager_gradients = run(eager_inputs)
            graph.replay()
            torch.cuda.synchronize(device)
            if torch.equal(graph_output, captured_output):
                raise AssertionError("CUDA Graph replay did not observe changed inputs")
            torch.testing.assert_close(graph_output, eager_output, atol=0.15, rtol=0.15)
            torch.testing.assert_close(graph_state, eager_state, atol=0.15, rtol=0.15)
            for replayed, expected in zip(graph_gradients, eager_gradients, strict=True):
                torch.testing.assert_close(replayed, expected, atol=0.15, rtol=0.15)
        finally:
            torch.cuda.synchronize(device)
            graph.reset()
            gc.collect()

    mode = " with CUDA Graph replay" if args.cuda_graph else ""
    print(f"rank {rank}: packed KDA CP passed{mode}", flush=True)
    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=<world_size>")
    main()
