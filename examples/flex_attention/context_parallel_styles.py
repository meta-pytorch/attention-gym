"""Compare context-parallel (CP) attention styles built from FlexAttention and functional collectives.

Every style starts from the same sequence-sharded Q/K/V and returns the same sequence-sharded
output, so they can be swapped inside a model and compared on equal footing:

- ``allgather``: all-gather K/V, then run one attention call over the full sequence. This is
  what PyTorch's FlexAttention CP and TorchTitan's ``KVAllGatherCPFlexInnerAttention`` do.
- ``allgather-overlap``: start the K/V all-gather, attend to the local K/V shard while it is in
  flight, then attend to the remote shards and merge the two partial results by log-sum-exp.
- ``ulysses``: all-to-all from sequence-sharded to head-sharded Q/K/V, run full-sequence
  attention on ``H / P`` heads, then all-to-all the output back (DeepSpeed-Ulysses, TorchTitan's
  ``UlyssesCPInnerAttention``).
- ``ulysses-overlap``: the same, pipelined over head groups so the all-to-all of one group
  overlaps attention on the previous group.
- ``ring``: rotate K/V around the ranks, prefetching the next shard while attending to the
  current one and merging partial results by log-sum-exp. Its backward is written by hand so the
  K/V and dK/dV rotations also overlap compute.

All styles use load-balanced "zigzag" sharding for causal attention: with ``2P`` sequence
chunks, rank ``r`` owns chunks ``r`` and ``2P - 1 - r``. Ring shards move either over NCCL or, when
every rank is on one node, as copy-engine pushes into symmetric memory (``--ring-transport``).
The other collectives come from ``torch.distributed._functional_collectives``: in eager mode they return asynchronous tensors that
wait on first use, which is what lets the overlap variants issue communication before compute.
They also have autograd formulas (all-gather backward is a reduce-scatter, all-to-all backward is
the inverse all-to-all), so the all-gather and Ulysses backwards are derived by autograd. Those
backward collectives wait as soon as they are issued, so they do not overlap compute.

Launch with:
    torchrun --standalone --nproc_per_node=4 examples/flex_attention/context_parallel_styles.py \
        --seq-len 131072 --backend FLASH --trace-dir traces/
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory as symm_mem
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from attn_gym.testing.profiling import (
    graph_annotations_available,
    kernel_stage,
    record_distributed_profile,
)

torch._dynamo.config.cache_size_limit = 64


@dataclass(frozen=True)
class CPLayout:
    """Zigzag sequence layout shared by every style."""

    rank: int
    world_size: int
    seq_len: int

    @property
    def chunk_len(self) -> int:
        return self.seq_len // (2 * self.world_size)

    @property
    def shard_len(self) -> int:
        return 2 * self.chunk_len

    def global_positions(self, idx: torch.Tensor, first_owner: int) -> torch.Tensor:
        """Map an index into a buffer of rank-ordered shards starting at ``first_owner``."""
        owner = first_owner + idx // self.shard_len
        local = idx % self.shard_len
        mirrored_chunk = 2 * self.world_size - 1 - owner
        return torch.where(
            local < self.chunk_len,
            owner * self.chunk_len + local,
            mirrored_chunk * self.chunk_len + local - self.chunk_len,
        )

    def local_token_ids(self, device: torch.device) -> torch.Tensor:
        return self.global_positions(torch.arange(self.shard_len, device=device), self.rank)


@dataclass(frozen=True)
class CPContext:
    layout: CPLayout
    group: dist.ProcessGroup
    masks: dict[str, BlockMask]
    head_groups: int
    kernel_options: dict[str, str]
    ring: "RingChannels | None"
    global_mask_mod: Callable
    annotate: bool
    """Label captured CUDA Graph kernels with ``mark_kernels`` (eager ranges are always on)."""


def flex_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
    kernel_options: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    out, aux = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        enable_gqa=True,
        kernel_options=kernel_options,
        return_aux=AuxRequest(lse=True),
    )
    return out, aux.lse


def merge_partials(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two softmax partials over disjoint key sets; rows of ``a`` must be non-empty."""
    lse = torch.logaddexp(lse_a, lse_b)
    out = out_a.float() * torch.exp(lse_a - lse).unsqueeze(-1)
    out = out + out_b.float() * torch.exp(lse_b - lse).unsqueeze(-1)
    return out.to(out_a.dtype), lse


compiled_flex_with_lse = torch.compile(flex_with_lse, fullgraph=True)
compiled_merge_partials = torch.compile(merge_partials, fullgraph=True)
compiled_create_block_mask = torch.compile(create_block_mask, fullgraph=True)


def wait(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def start_kv_all_gather(k: torch.Tensor, v: torch.Tensor, ctx: CPContext) -> torch.Tensor:
    # Gather on dim 0: any other gather_dim makes funcol concatenate, and so wait, immediately.
    return funcol.all_gather_single(torch.stack([k, v]), 0, ctx.group)


def wait_kv_all_gather(pending_kv: torch.Tensor, world_size: int) -> torch.Tensor:
    """(P * 2, 1, Hkv, S/P, D) in rank order -> (2, 1, Hkv, S, D)."""
    kv = wait(pending_kv).unflatten(0, (world_size, 2))
    return kv.permute(1, 2, 3, 0, 4, 5).flatten(3, 4)


def allgather_attention(q, k, v, ctx: CPContext) -> torch.Tensor:
    with kernel_stage("cp/allgather/kv-gather", ctx.annotate):
        kv = wait_kv_all_gather(start_kv_all_gather(k, v, ctx), ctx.layout.world_size)
    with kernel_stage("cp/allgather/attention", ctx.annotate):
        out, _ = compiled_flex_with_lse(q, kv[0], kv[1], ctx.masks["full"], ctx.kernel_options)
    return out


def allgather_overlap_attention(q, k, v, ctx: CPContext) -> torch.Tensor:
    with kernel_stage("cp/allgather-overlap/kv-gather", ctx.annotate):
        pending_kv = start_kv_all_gather(k, v, ctx)
    with kernel_stage("cp/allgather-overlap/attention-local", ctx.annotate):
        local_out, local_lse = compiled_flex_with_lse(
            q, k, v, ctx.masks["local"], ctx.kernel_options
        )
    with kernel_stage("cp/allgather-overlap/attention-remote", ctx.annotate):
        kv = wait_kv_all_gather(pending_kv, ctx.layout.world_size)
        remote_out, remote_lse = compiled_flex_with_lse(
            q, kv[0], kv[1], ctx.masks["remote"], ctx.kernel_options
        )
    with kernel_stage("cp/allgather-overlap/merge", ctx.annotate):
        out, _ = compiled_merge_partials(local_out, local_lse, remote_out, remote_lse)
    return out


def split_heads_for_ulysses(x: torch.Tensor, world_size: int, groups: int) -> list[torch.Tensor]:
    """(1, H, S/P, D) -> per group (P, H/(P*G), S/P, D), dim 0 indexed by destination rank."""
    _, heads, shard_len, dim = x.shape
    by_rank = x.view(world_size, groups, heads // (world_size * groups), shard_len, dim)
    return [by_rank[:, group].contiguous() for group in range(groups)]


def ulysses_attention(q, k, v, ctx: CPContext) -> torch.Tensor:
    world_size = ctx.layout.world_size
    groups = ctx.head_groups
    name = "cp/ulysses-overlap" if groups > 1 else "cp/ulysses"
    # Issue every input all-to-all up front; NCCL runs them in order on its stream while the
    # compute stream works through the groups as they land.
    with kernel_stage(f"{name}/a2a-in", ctx.annotate):
        pending = [
            [funcol.all_to_all_single(chunk, None, None, ctx.group) for chunk in chunks]
            for chunks in zip(*(split_heads_for_ulysses(x, world_size, groups) for x in (q, k, v)))
        ]
    pending_out = []
    for group, (group_q, group_k, group_v) in enumerate(pending):
        with kernel_stage(f"{name}/attention-g{group}", ctx.annotate):
            # (P, h, S/P, D) indexed by source rank -> (1, h, S, D) in rank-ordered zigzag layout.
            full_seq = [
                wait(x).permute(1, 0, 2, 3).reshape(1, x.shape[1], -1, x.shape[3])
                for x in (group_q, group_k, group_v)
            ]
            out, _ = compiled_flex_with_lse(*full_seq, ctx.masks["ulysses"], ctx.kernel_options)
            heads, dim = out.shape[1], out.shape[3]
            by_seq_shard = out.view(heads, world_size, -1, dim).permute(1, 0, 2, 3).contiguous()
        with kernel_stage(f"{name}/a2a-out-g{group}", ctx.annotate):
            pending_out.append(funcol.all_to_all_single(by_seq_shard, None, None, ctx.group))
    with kernel_stage(f"{name}/a2a-out-wait", ctx.annotate):
        out = torch.cat([wait(x) for x in pending_out], dim=1)
    return out.reshape(1, -1, out.shape[2], out.shape[3])


def shift_to_next_rank(x: torch.Tensor, layout: CPLayout, group: dist.ProcessGroup):
    """Send ``x`` to rank + 1 and receive from rank - 1 as a differentiable all-to-all."""
    send_sizes = [0] * layout.world_size
    recv_sizes = [0] * layout.world_size
    send_sizes[(layout.rank + 1) % layout.world_size] = x.shape[0]
    recv_sizes[(layout.rank - 1) % layout.world_size] = x.shape[0]
    return funcol.all_to_all_single(x, recv_sizes, send_sizes, group)


def identity_score_mod(score, b, h, q_idx, kv_idx):
    return score


def flex_backward(q, k, v, out, lse, grad_out, block_mask: BlockMask, kernel_options):
    """dq, dk, dv of one K/V block given the *merged* output and log-sum-exp."""
    # The backward HOP takes the forward HOP's LSE: log2-scaled except on the FLASH backend.
    if kernel_options["BACKEND"] != "FLASH":
        lse = lse / math.log(2)
    dq, dk, dv, _ = torch.ops.higher_order.flex_attention_backward(
        q,
        k,
        v,
        out,
        lse,
        grad_out,
        None,
        identity_score_mod,
        None,
        block_mask.as_tuple(),
        q.shape[-1] ** -0.5,
        kernel_options,
        (),
        (),
    )
    return dq, dk, dv


compiled_flex_backward = torch.compile(flex_backward, fullgraph=True)


def accumulate(total: torch.Tensor, part: torch.Tensor) -> None:
    total.add_(part)


# Eager mixed-dtype add_ (fp32 += bf16) is a slow unvectorized kernel; Inductor fuses the upcast.
compiled_accumulate = torch.compile(accumulate, fullgraph=True)


class NcclRingChannel:
    """One ring hop per step over NCCL point-to-point (works across nodes)."""

    def __init__(self, layout: CPLayout, group: dist.ProcessGroup):
        self.layout = layout
        self.group = group
        self.pending: torch.Tensor | None = None

    def send(self, parts: tuple[torch.Tensor, torch.Tensor], step: int) -> None:
        self.pending = shift_to_next_rank(torch.stack(parts), self.layout, self.group)

    def recv(self, step: int) -> torch.Tensor:
        return wait(self.pending)

    def release(self, step: int) -> None:
        pass

    def join(self) -> None:
        pass


class CopyEngineRingChannel:
    """One ring hop per step as a copy-engine push into the next rank's symmetric memory.

    Only valid when every rank is on one node. Copy engines move data without SMs, so the hop
    neither competes with attention for SMs nor is limited by NCCL's per-peer channel count
    (on B200 NVLink: ~640 GB/s vs ~160 GB/s for a default NCCL send/recv).

    Each rank owns two receive slots that the previous rank writes alternately. A slot moves
    between ranks with two signals on the symmetric-memory signal pad: ``FREE`` (receiver ->
    sender: you may overwrite it) and ``READY`` (sender -> receiver: the data has landed). The
    receiver frees a slot only after its compute and any onward push from that slot are done.
    """

    READY, FREE = 0, 1

    def __init__(self, layout: CPLayout, group: dist.ProcessGroup, shape, dtype, device):
        self.next_rank = (layout.rank + 1) % layout.world_size
        self.prev_rank = (layout.rank - 1) % layout.world_size
        self.slots = symm_mem.empty((2, *shape), dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.slots, group.group_name)
        self.peer_slots = self.handle.get_buffer(self.next_rank, (2, *shape), dtype)
        self.stream = torch.cuda.Stream(device)
        for slot in range(2):
            self.handle.put_signal(self.prev_rank, self.channel(slot, self.FREE))

    @staticmethod
    def channel(slot: int, signal: int) -> int:
        return 2 * slot + signal

    def send(self, parts: tuple[torch.Tensor, torch.Tensor], step: int) -> None:
        """Push ``parts`` (e.g. K and V) into the slot the next rank reads at ``step + 1``."""
        slot = (step + 1) % 2
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.handle.wait_signal(self.next_rank, self.channel(slot, self.FREE))
            for peer_part, part in zip(self.peer_slots[slot], parts):
                peer_part.copy_(part)
            self.handle.put_signal(self.next_rank, self.channel(slot, self.READY))
        for part in parts:
            part.record_stream(self.stream)

    def recv(self, step: int) -> torch.Tensor:
        slot = step % 2
        self.handle.wait_signal(self.prev_rank, self.channel(slot, self.READY))
        return self.slots[slot]

    def release(self, step: int) -> None:
        """Hand the slot read at ``step`` back once queued compute and pushes from it finish."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.handle.put_signal(self.prev_rank, self.channel(step % 2, self.FREE))

    def join(self) -> None:
        """Order the copy stream's remaining work before later compute (required by graph capture)."""
        torch.cuda.current_stream().wait_stream(self.stream)


RingChannel = NcclRingChannel | CopyEngineRingChannel


@dataclass(frozen=True)
class RingChannels:
    kv: RingChannel
    dkv: RingChannel


def make_ring_channels(transport: str, layout: CPLayout, group, shape, device) -> RingChannels:
    match transport:
        case "nccl":
            return RingChannels(NcclRingChannel(layout, group), NcclRingChannel(layout, group))
        case "copy-engine":
            return RingChannels(
                *(
                    CopyEngineRingChannel(layout, group, shape, torch.bfloat16, device)
                    for _ in range(2)
                )
            )
    raise ValueError(f"unknown ring transport {transport!r}")


class RingAttention(torch.autograd.Function):
    """Ring attention with a hand-written backward that overlaps both rotations with compute.

    Forward sends each K/V shard on while attending to it. Backward rotates K/V the same way and
    passes a dK/dV accumulator along with it: each step adds its contribution and sends the
    accumulator on, and the next step waits for its incoming accumulator only after its own
    attention backward. Only the final hop home of the accumulator is exposed.
    """

    @staticmethod
    def forward(fn_ctx, q, k, v, ctx: CPContext):
        layout, channel = ctx.layout, ctx.ring.kv
        step_k, step_v = k, v
        # Zigzag sharding makes every visit partially visible, so no step can be skipped.
        for step in range(layout.world_size):
            with kernel_stage(f"cp/ring/fwd-step{step}", ctx.annotate, backward=False):
                if step + 1 < layout.world_size:
                    channel.send((step_k, step_v), step)
                owner = (layout.rank - step) % layout.world_size
                step_out, step_lse = compiled_flex_with_lse(
                    q, step_k, step_v, ctx.masks[f"ring{owner}"], ctx.kernel_options
                )
                if step == 0:
                    out, lse = step_out, step_lse
                else:
                    out, lse = compiled_merge_partials(out, lse, step_out, step_lse)
                    channel.release(step)
                if step + 1 < layout.world_size:
                    step_k, step_v = channel.recv(step + 1)
        channel.join()
        fn_ctx.save_for_backward(q, k, v, out, lse)
        fn_ctx.cp = ctx
        return out

    @staticmethod
    def backward(fn_ctx, grad_out):
        q, k, v, out, lse = fn_ctx.saved_tensors
        ctx: CPContext = fn_ctx.cp
        layout, kv_channel, dkv_channel = ctx.layout, ctx.ring.kv, ctx.ring.dkv
        grad_out = grad_out.contiguous()
        step_k, step_v = k, v
        dq = torch.zeros_like(q, dtype=torch.float32)
        for step in range(layout.world_size):
            with kernel_stage(f"cp/ring/bwd-step{step}", ctx.annotate, backward=False):
                if step + 1 < layout.world_size:
                    kv_channel.send((step_k, step_v), step)
                owner = (layout.rank - step) % layout.world_size
                dq_step, dk_step, dv_step = compiled_flex_backward(
                    q,
                    step_k,
                    step_v,
                    out,
                    lse,
                    grad_out,
                    ctx.masks[f"ring{owner}"],
                    ctx.kernel_options,
                )
                compiled_accumulate(dq, dq_step)
                # The accumulator for shard `owner` arrives from the rank that held it last step.
                if step > 0:
                    incoming_dk, incoming_dv = dkv_channel.recv(step)
                    dk_step += incoming_dk
                    dv_step += incoming_dv
                    dkv_channel.release(step)
                    kv_channel.release(step)
                dkv_channel.send((dk_step, dv_step), step)
                if step + 1 < layout.world_size:
                    step_k, step_v = kv_channel.recv(step + 1)
        # After world_size hops each accumulator is back on the rank that owns its shard; copy it
        # out because the receive slot is reused by the next call.
        dkv = dkv_channel.recv(layout.world_size).clone()
        dkv_channel.release(layout.world_size)
        kv_channel.join()
        dkv_channel.join()
        return dq.to(q.dtype), dkv[0], dkv[1], None


def ring_attention(q, k, v, ctx: CPContext) -> torch.Tensor:
    return RingAttention.apply(q, k, v, ctx)


STYLE_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "allgather": allgather_attention,
    "allgather-overlap": allgather_overlap_attention,
    "ulysses": ulysses_attention,
    "ulysses-overlap": ulysses_attention,
    "ring": ring_attention,
}


def document_boundaries(seq_len: int, num_docs: int) -> list[int]:
    """Start positions of ``num_docs`` documents with seeded, uneven lengths (same on every rank)."""
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(num_docs, generator=generator) + 0.1
    lengths = (weights / weights.sum() * seq_len).long()
    return torch.cumsum(lengths, 0)[:-1].tolist()


def make_global_mask_mod(kind: str, seq_len: int, window: int, num_docs: int):
    """``mask_mod`` over *global* token positions; every style evaluates it on its own buffers.

    Document boundaries are Python ints rather than a captured tensor, so the same function can be
    traced into the backward op the hand-written ring calls directly.
    """
    match kind:
        case "causal":
            return lambda b, h, q_pos, kv_pos: q_pos >= kv_pos
        case "sliding-window":
            return lambda b, h, q_pos, kv_pos: (q_pos >= kv_pos) & (q_pos - kv_pos < window)
        case "document":
            starts = document_boundaries(seq_len, num_docs)

            def doc_id(pos):
                return sum((pos >= start).to(torch.int32) for start in starts)

            return lambda b, h, q_pos, kv_pos: (
                (q_pos >= kv_pos) & (doc_id(q_pos) == doc_id(kv_pos))
            )
    raise ValueError(f"unknown mask {kind!r}")


def build_masks(
    layout: CPLayout, device: torch.device, block_size: tuple[int, int], global_mask_mod
) -> dict[str, BlockMask]:
    def make(mask_mod, q_len: int, kv_len: int) -> BlockMask:
        return compiled_create_block_mask(
            mask_mod, 1, None, q_len, kv_len, device=device, BLOCK_SIZE=block_size
        )

    rank, shard_len, seq_len = layout.rank, layout.shard_len, layout.seq_len

    def from_global(kv_owner: int, q_owner: int = rank):
        def mask_mod(b, h, q_idx, kv_idx):
            q_pos = layout.global_positions(q_idx, q_owner)
            return global_mask_mod(b, h, q_pos, layout.global_positions(kv_idx, kv_owner))

        return mask_mod

    def remote_mask(b, h, q_idx, kv_idx):
        is_remote = kv_idx // shard_len != rank
        return is_remote & from_global(0)(b, h, q_idx, kv_idx)

    masks = {
        "full": make(from_global(0), shard_len, seq_len),
        "local": make(from_global(rank), shard_len, shard_len),
        "remote": make(remote_mask, shard_len, seq_len),
        "ulysses": make(from_global(0, q_owner=0), seq_len, seq_len),
    }
    for owner in range(layout.world_size):
        masks[f"ring{owner}"] = make(from_global(owner), shard_len, shard_len)
    return masks


def mask_work(masks: dict[str, BlockMask], world_size: int) -> dict[str, int]:
    """Nonzero (q, kv) tiles each sequence-sharded style computes on this rank."""

    def tiles(mask: BlockMask) -> int:
        total = mask.kv_num_blocks.sum()
        if mask.full_kv_num_blocks is not None:
            total = total + mask.full_kv_num_blocks.sum()
        return int(total)

    return {
        "allgather": tiles(masks["full"]),
        "allgather-overlap": tiles(masks["local"]) + tiles(masks["remote"]),
        "ring": sum(tiles(masks[f"ring{owner}"]) for owner in range(world_size)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=131072, help="Global sequence length.")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--backend",
        choices=["TRITON", "FLASH"],
        default="TRITON",
        help="FlexAttention backend; FLASH needs flash-attn-4 and uses 256-row Q blocks.",
    )
    parser.add_argument(
        "--mask", choices=["causal", "document", "sliding-window"], default="causal"
    )
    parser.add_argument("--window", type=int, default=8192, help="Sliding-window size.")
    parser.add_argument("--num-docs", type=int, default=12, help="Documents for --mask document.")
    parser.add_argument("--head-groups", type=int, default=2, help="Groups for ulysses-overlap.")
    parser.add_argument("--styles", nargs="+", choices=list(STYLE_FNS), default=list(STYLE_FNS))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--ring-transport",
        choices=["auto", "nccl", "copy-engine"],
        default="auto",
        help="How ring moves shards; auto uses copy engines when all ranks share one node.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Replay each step from a CUDA graph to remove host launch overhead.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check outputs and gradients against single-device attention on the full sequence.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Append one JSON line per style (slowest-rank timings and peak memory).",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Write per-rank and merged Perfetto traces (needs transformer-nuggets).",
    )
    return parser.parse_args()


def init_distributed(seq_len: int) -> tuple[torch.device, CPLayout, bool]:
    """Join (or reuse, when a driver already created it) the world group; True if created here."""
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --nproc_per_node=<world_size>.")
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="nccl", device_id=device)
    return device, CPLayout(dist.get_rank(), dist.get_world_size(), seq_len), created


def make_inputs(args, layout: CPLayout, device: torch.device, full: bool, seed: int = 0):
    seq_len = layout.seq_len if full else layout.shard_len
    generator = torch.Generator(device=device).manual_seed(seed if full else layout.rank + 1)

    def randn(heads: int) -> torch.Tensor:
        shape = (1, heads, seq_len, args.head_dim)
        return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    return randn(args.heads), randn(args.kv_heads), randn(args.kv_heads), randn(args.heads)


def validate(args, style: str, ctx: CPContext, device: torch.device) -> None:
    layout = ctx.layout
    ids = layout.local_token_ids(device)
    reference_mask = create_block_mask(
        ctx.global_mask_mod,
        1,
        None,
        layout.seq_len,
        layout.seq_len,
        device=device,
        BLOCK_SIZE=ctx.masks["full"].BLOCK_SIZE,
    )
    ref = {}

    def reference(seed: int) -> list[torch.Tensor]:
        """Single-device fwd+bwd on the full sequence; returns this rank's shards of q, k, v, dO."""
        full = make_inputs(args, layout, device, full=True, seed=seed)
        ref["inputs"] = [x.clone().requires_grad_(True) for x in full[:3]]
        ref["out"], _ = compiled_flex_with_lse(*ref["inputs"], reference_mask, ctx.kernel_options)
        ref["out"].backward(full[3])
        return [x[:, :, ids] for x in full]

    *local, local_grad = reference(seed=0)
    local = [x.clone().requires_grad_(True) for x in local]
    outputs = {}

    def step() -> None:
        for x in local:
            x.grad = None
        outputs["out"] = STYLE_FNS[style](*local, ctx)
        outputs["out"].backward(local_grad)

    def check(mode: str) -> None:
        torch.testing.assert_close(outputs["out"], ref["out"][:, :, ids], atol=2e-2, rtol=2e-2)
        for name, x, r in zip("qkv", local, ref["inputs"]):
            torch.testing.assert_close(x.grad, r.grad[:, :, ids], atol=5e-2, rtol=5e-2, msg=name)
        if layout.rank == 0:
            print(f"{style} ({mode}): output and dq/dk/dv match the reference", flush=True)

    step()
    check("eager")
    if args.cuda_graph:
        # Fresh leaves: the eager step's autograd nodes are bound to the default stream and would
        # invalidate capture.
        outputs.clear()
        local[:] = [x.detach().clone().requires_grad_(True) for x in local]
        graph = capture_cuda_graph(step)
        # Replay twice so the second replay starts from slot state the first one left behind.
        for x in local:
            x.grad.zero_()
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        check("CUDA graph replay")
        # New inputs in the captured buffers: a replay that skipped or reused a transfer would
        # still hold the previous values (symmetric-memory slots included).
        new_inputs = reference(seed=1)
        with torch.no_grad():
            for dst, src in zip([*local, local_grad], new_inputs):
                dst.copy_(src)
            for x in local:
                x.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check("CUDA graph replay, new inputs")
        graph.reset()


def time_steps(step: Callable[[], None], iters: int) -> float:
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def step_peak_gib(step: Callable[[], None]) -> float:
    """Peak memory one eager step allocates beyond what is already live (inputs, masks).

    Measured eagerly even for --cuda-graph runs, and excludes persistent buffers such as the
    copy-engine ring's symmetric-memory slots (reported separately) and NCCL channel buffers.
    """
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2**30


def capture_cuda_graph(step: Callable[[], None], annotate: bool = False) -> torch.cuda.CUDAGraph:
    """Capture one already-warmed step, including its NCCL collectives."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        step()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    # Only pass the keyword when needed: older PyTorch has no enable_annotations argument.
    with torch.cuda.graph(graph, **({"enable_annotations": True} if annotate else {})):
        step()
    return graph


def main() -> None:
    args = parse_args()
    if args.trace_dir is not None and importlib.util.find_spec("transformer_nuggets") is None:
        # Optional dependency, checked up front so a missing install fails before any GPU work.
        raise RuntimeError("--trace-dir needs transformer-nuggets installed.")
    device, layout, owns_process_group = init_distributed(args.seq_len)
    annotate = args.cuda_graph and graph_annotations_available()
    if args.cuda_graph and not annotate and layout.rank == 0:
        print("Capturing without kernel labels: graph annotations need a newer PyTorch/driver.")
    styles = list(args.styles)
    for style, groups in (("ulysses", 1), ("ulysses-overlap", args.head_groups)):
        if style in styles and args.kv_heads % (layout.world_size * groups) != 0:
            # Ulysses shards heads, so it cannot use more ranks than KV heads (per head group).
            styles.remove(style)
            if layout.rank == 0:
                print(
                    f"skipping {style}: kv_heads must be divisible by {layout.world_size * groups}"
                )
    block_size = (256, 128) if args.backend == "FLASH" else (128, 128)
    seq_len_unit = 2 * layout.world_size * block_size[0]
    if args.seq_len % seq_len_unit != 0:
        raise RuntimeError(f"seq_len must be divisible by {seq_len_unit}.")
    global_mask_mod = make_global_mask_mod(args.mask, args.seq_len, args.window, args.num_docs)
    masks = build_masks(layout, device, block_size, global_mask_mod)
    work = [None] * layout.world_size
    dist.all_gather_object(work, mask_work(masks, layout.world_size))
    if layout.rank == 0:
        print(f"mask={args.mask}: nonzero tiles per rank (max/mean = load imbalance)")
        for style in work[0]:
            counts = [w[style] for w in work]
            imbalance = max(counts) / (sum(counts) / len(counts))
            print(f"  {style:<18} {counts}  max/mean {imbalance:.2f}")
    kernel_options = {"BACKEND": args.backend}

    q, k, v, grad_out = make_inputs(args, layout, device, full=False)
    q, k, v = (x.requires_grad_(True) for x in (q, k, v))
    transport = args.ring_transport
    if transport == "auto":
        single_node = int(os.environ["LOCAL_WORLD_SIZE"]) == layout.world_size
        transport = "copy-engine" if single_node else "nccl"
    ring = None
    if "ring" in styles:
        kv_shape = (2, 1, args.kv_heads, layout.shard_len, args.head_dim)
        ring = make_ring_channels(transport, layout, dist.group.WORLD, kv_shape, device)
    slot_gib = 0.0
    if transport == "copy-engine" and ring is not None:
        slot_gib = 2 * 2 * 2 * args.kv_heads * layout.shard_len * args.head_dim * 2 / 2**30
    if layout.rank == 0:
        print(
            f"\nP={layout.world_size} S={args.seq_len} H={args.heads} H_kv={args.kv_heads} "
            f"D={args.head_dim} bf16 {args.mask} zigzag {args.backend}"
            f"{' cudagraph' if args.cuda_graph else ''} ring={transport} "
            f"(median of {args.iters}, slowest rank)"
        )
        if slot_gib:
            print(f"copy-engine ring also holds {slot_gib:.2f} GiB of symmetric-memory slots")
        print(f"{'style':<20}{'fwd ms':>10}{'fwd+bwd ms':>12}{'eager peak GiB':>16}", flush=True)
    for style in styles:
        graphs = []
        head_groups = args.head_groups if style.endswith("overlap") else 1
        ctx = CPContext(
            layout,
            dist.group.WORLD,
            masks,
            head_groups,
            kernel_options,
            ring,
            global_mask_mod,
            annotate,
        )
        if args.validate:
            validate(args, style, ctx, device)

        def forward(style=style, ctx=ctx) -> None:
            with kernel_stage(f"cp/{style}/forward", annotate, backward=False):
                STYLE_FNS[style](q, k, v, ctx)

        def train_step(style=style, ctx=ctx) -> None:
            with kernel_stage(f"cp/{style}/forward", annotate, backward=False):
                out = STYLE_FNS[style](q, k, v, ctx)
            with kernel_stage(f"cp/{style}/backward", annotate, backward=False):
                out.backward(grad_out)

        for _ in range(3):
            train_step()
        peak_gib = step_peak_gib(train_step)
        with torch.no_grad():
            for _ in range(3):
                forward()
            if args.cuda_graph:
                graphs.append(capture_cuda_graph(forward, annotate))
                forward = graphs[-1].replay
        if args.cuda_graph:
            graphs.append(capture_cuda_graph(train_step, annotate))
            train_step = graphs[-1].replay
        torch.cuda.synchronize()
        dist.barrier()
        with torch.no_grad():
            fwd_ms = time_steps(forward, args.iters)
        step_ms = time_steps(train_step, args.iters)
        # Report the slowest rank: every rank waits for it at the next collective.
        worst = torch.tensor([fwd_ms, step_ms, peak_gib], device=device)
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        fwd_ms, step_ms, peak_gib = worst.tolist()
        if layout.rank == 0:
            print(f"{style:<20}{fwd_ms:>10.2f}{step_ms:>12.2f}{peak_gib:>16.2f}", flush=True)
            if args.results_json is not None:
                row = {
                    "style": style,
                    "world_size": layout.world_size,
                    "seq_len": args.seq_len,
                    "heads": args.heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "head_groups": head_groups,
                    "mask": args.mask,
                    "window": args.window,
                    "num_docs": args.num_docs,
                    "backend": args.backend,
                    "cuda_graph": args.cuda_graph,
                    "ring_transport": transport,
                    "fwd_ms": fwd_ms,
                    "step_ms": step_ms,
                    "iters": args.iters,
                    # Eager step peak beyond live inputs; persistent ring slots are separate.
                    "peak_gib": peak_gib,
                    "ring_slot_gib": slot_gib if style == "ring" else 0.0,
                    "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                }
                with args.results_json.open("a") as results:
                    results.write(json.dumps(row) + "\n")

        if args.trace_dir is not None:
            graph_suffix = "_cudagraph" if args.cuda_graph else ""
            style_tag = f"ring-{transport}" if style == "ring" else style
            trace_name = (
                f"{style_tag}_{args.mask}_{args.backend.lower()}{graph_suffix}_seq{args.seq_len}"
            )
            merged = record_distributed_profile(
                train_step, args.trace_dir / trace_name, style, device, warmup_steps=2
            )
            if merged is not None:
                print(f"trace={merged}", flush=True)

        # Release this style's graphs (and their memory pools) before the next style. Graphs that
        # captured NCCL work must also be released before teardown, or it hangs.
        for graph in graphs:
            graph.reset()
        torch.cuda.empty_cache()

    dist.barrier()
    if owns_process_group:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
