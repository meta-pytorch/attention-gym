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
chunks, rank ``r`` owns chunks ``r`` and ``2P - 1 - r``.

``--transport`` picks how the all-gather and ring styles move K/V: NCCL, or copy-engine pushes into
symmetric memory (``attn_gym.distributed``), which need every rank in one NVLink domain and
default on when every rank is in one NVLink domain (one node, or an NVL72 rack). The copy-engine all-gather writes each shard straight into
the layout attention reads. The NCCL path and Ulysses use
``torch.distributed._functional_collectives``: in eager mode they return asynchronous tensors that
wait on first use, which is what lets the overlap variants issue communication before compute.
Their autograd formulas give the backward collectives (all-gather backward is a reduce-scatter,
all-to-all backward the inverse all-to-all); those wait as soon as they are issued.

Launch with:
    torchrun --standalone --nproc_per_node=4 examples/flex_attention/context_parallel_styles.py \
        --seq-len 131072 --backend FLASH --cuda-graph --validate
"""

import importlib.util
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import typer
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from attn_gym.distributed import (
    DEFAULT_TIMEOUT_MS,
    CopyEngineAllGather,
    CopyEngineChannel,
    same_nvlink_domain,
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
    gather: CopyEngineAllGather | None
    """Copy-engine K/V all-gather; ``None`` gathers over NCCL."""
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


def start_kv_all_gather(k: torch.Tensor, v: torch.Tensor, ctx: CPContext) -> torch.Tensor | None:
    if ctx.gather is not None:
        ctx.gather.start((k, v))
        return None
    # Gather on dim 0: any other gather_dim makes funcol concatenate, and so wait, immediately.
    return funcol.all_gather_single(torch.stack([k, v]), 0, ctx.group)


def wait_kv_all_gather(pending_kv, k: torch.Tensor, v: torch.Tensor, ctx: CPContext):
    """K/V of the full sequence as (2, 1, Hkv, S, D), shards in rank order."""
    if ctx.gather is not None:
        return ctx.gather.wait((k, v))
    # (P * 2, 1, Hkv, S/P, D) in rank order -> (2, 1, Hkv, S, D).
    kv = wait(pending_kv).unflatten(0, (ctx.layout.world_size, 2))
    return kv.permute(1, 2, 3, 0, 4, 5).flatten(3, 4)


def allgather_attention(q, k, v, ctx: CPContext) -> torch.Tensor:
    with kernel_stage("cp/allgather/kv-gather", ctx.annotate):
        kv = wait_kv_all_gather(start_kv_all_gather(k, v, ctx), k, v, ctx)
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
        kv = wait_kv_all_gather(pending_kv, k, v, ctx)
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


RingChannel = NcclRingChannel | CopyEngineChannel


@dataclass(frozen=True)
class RingChannels:
    kv: RingChannel
    dkv: RingChannel


def make_ring_channels(
    transport: "Transport", layout: CPLayout, group, part_shape, device, timeout_ms: int
) -> RingChannels:
    """K/V and dK/dV hops to the next rank; each carries two parts (K and V, or dK and dV)."""
    if transport is Transport.NCCL:
        return RingChannels(NcclRingChannel(layout, group), NcclRingChannel(layout, group))
    prev_rank = (layout.rank - 1) % layout.world_size
    next_rank = (layout.rank + 1) % layout.world_size
    return RingChannels(
        *(
            CopyEngineChannel(
                group, prev_rank, next_rank, 2, part_shape, torch.bfloat16, device, timeout_ms
            )
            for _ in range(2)
        )
    )


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


class Backend(str, Enum):
    TRITON = "TRITON"
    FLASH = "FLASH"


class Mask(str, Enum):
    CAUSAL = "causal"
    DOCUMENT = "document"
    SLIDING_WINDOW = "sliding-window"


class Style(str, Enum):
    ALLGATHER = "allgather"
    ALLGATHER_OVERLAP = "allgather-overlap"
    ULYSSES = "ulysses"
    ULYSSES_OVERLAP = "ulysses-overlap"
    RING = "ring"


class Transport(str, Enum):
    AUTO = "auto"
    NCCL = "nccl"
    COPY_ENGINE = "copy-engine"


@dataclass(frozen=True)
class Config:
    seq_len: int
    heads: int
    kv_heads: int
    head_dim: int
    backend: Backend
    mask: Mask
    window: int
    num_docs: int
    head_groups: int
    styles: list[Style]
    iters: int
    transport: Transport
    cuda_graph: bool
    validate: bool
    results_json: Path | None
    trace_dir: Path | None
    symm_timeout_ms: int


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


def make_inputs(cfg: Config, layout: CPLayout, device: torch.device, full: bool, seed: int = 0):
    seq_len = layout.seq_len if full else layout.shard_len
    generator = torch.Generator(device=device).manual_seed(seed if full else layout.rank + 1)

    def randn(heads: int) -> torch.Tensor:
        shape = (1, heads, seq_len, cfg.head_dim)
        return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    return randn(cfg.heads), randn(cfg.kv_heads), randn(cfg.kv_heads), randn(cfg.heads)


def check_against_reference(cfg: Config, style: str, ctx: CPContext, device: torch.device) -> None:
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
        full = make_inputs(cfg, layout, device, full=True, seed=seed)
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
    if cfg.cuda_graph:
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


def runnable_styles(cfg: Config, world_size: int) -> list[str]:
    """Requested styles, minus Ulysses variants whose head split does not divide the KV heads."""
    styles = []
    for style in cfg.styles:
        groups = {Style.ULYSSES: 1, Style.ULYSSES_OVERLAP: cfg.head_groups}.get(style)
        if groups and cfg.kv_heads % (world_size * groups) != 0:
            if dist.get_rank() == 0:
                print(
                    f"skipping {style.value}: kv_heads must be divisible by {world_size * groups}"
                )
            continue
        styles.append(style.value)
    return styles


def run(cfg: Config) -> None:
    if cfg.trace_dir is not None and importlib.util.find_spec("transformer_nuggets") is None:
        # Optional dependency, checked up front so a missing install fails before any GPU work.
        raise RuntimeError("--trace-dir needs transformer-nuggets installed.")
    device, layout, owns_process_group = init_distributed(cfg.seq_len)
    annotate = cfg.cuda_graph and graph_annotations_available()
    if cfg.cuda_graph and not annotate and layout.rank == 0:
        print("Capturing without kernel labels: graph annotations need a newer PyTorch/driver.")
    styles = runnable_styles(cfg, layout.world_size)
    block_size = (256, 128) if cfg.backend is Backend.FLASH else (128, 128)
    seq_len_unit = 2 * layout.world_size * block_size[0]
    if cfg.seq_len % seq_len_unit != 0:
        raise RuntimeError(f"seq_len must be divisible by {seq_len_unit}.")
    global_mask_mod = make_global_mask_mod(cfg.mask.value, cfg.seq_len, cfg.window, cfg.num_docs)
    masks = build_masks(layout, device, block_size, global_mask_mod)
    work = [None] * layout.world_size
    dist.all_gather_object(work, mask_work(masks, layout.world_size))
    if layout.rank == 0:
        print(f"mask={cfg.mask.value}: nonzero tiles per rank (max/mean = load imbalance)")
        for style in work[0]:
            counts = [w[style] for w in work]
            imbalance = max(counts) / (sum(counts) / len(counts))
            print(f"  {style:<18} {counts}  max/mean {imbalance:.2f}")
    kernel_options = {"BACKEND": cfg.backend.value}

    q, k, v, grad_out = make_inputs(cfg, layout, device, full=False)
    q, k, v = (x.requires_grad_(True) for x in (q, k, v))
    # Copy engines only reach peers in this GPU's NVLink domain; across domains, use NCCL.
    one_domain = same_nvlink_domain()
    transport = cfg.transport
    if transport is Transport.AUTO:
        transport = Transport.COPY_ENGINE if one_domain else Transport.NCCL
    elif transport is Transport.COPY_ENGINE and not one_domain:
        raise RuntimeError("--transport copy-engine needs every rank in one NVLink domain.")
    # Persistent communication buffers, built once and shared by every style that needs them.
    kv_part = (1, cfg.kv_heads, layout.shard_len, cfg.head_dim)
    ring = gather = None
    if "ring" in styles:
        ring = make_ring_channels(
            transport, layout, dist.group.WORLD, kv_part, device, cfg.symm_timeout_ms
        )
    copy_engine = transport is Transport.COPY_ENGINE
    if copy_engine and {"allgather", "allgather-overlap"} & set(styles):
        gather = CopyEngineAllGather(
            dist.group.WORLD, 2, kv_part, 2, torch.bfloat16, device, cfg.symm_timeout_ms
        )
    gib = 2 * math.prod(kv_part) * 2 / 2**30  # one K/V shard pair, bf16
    persistent_gib = {
        "ring": 4 * gib if copy_engine else 0.0,  # two channels x two slots
        "allgather": layout.world_size * gib if gather else 0.0,
    }
    persistent_gib["allgather-overlap"] = persistent_gib["allgather"]
    if layout.rank == 0:
        print(
            f"\nP={layout.world_size} S={cfg.seq_len} H={cfg.heads} H_kv={cfg.kv_heads} "
            f"D={cfg.head_dim} bf16 {cfg.mask.value} zigzag {cfg.backend.value}"
            f"{' cudagraph' if cfg.cuda_graph else ''} transport={transport.value} "
            f"(median of {cfg.iters}, slowest rank)"
        )
        print(
            f"{'style':<20}{'fwd ms':>10}{'fwd+bwd ms':>12}{'eager peak GiB':>16}{'symm GiB':>10}",
            flush=True,
        )
    for style in styles:
        head_groups = cfg.head_groups if style.endswith("overlap") else 1
        ctx = CPContext(
            layout,
            dist.group.WORLD,
            masks,
            head_groups,
            kernel_options,
            ring,
            gather,
            global_mask_mod,
            annotate,
        )
        if cfg.validate:
            check_against_reference(cfg, style, ctx, device)
        fwd_ms, step_ms, peak_gib, trace = benchmark_style(
            cfg, style, ctx, (q, k, v, grad_out), device, transport
        )
        symm_gib = persistent_gib.get(style, 0.0)
        if layout.rank == 0:
            print(
                f"{style:<20}{fwd_ms:>10.2f}{step_ms:>12.2f}{peak_gib:>16.2f}{symm_gib:>10.2f}",
                flush=True,
            )
            if trace is not None:
                print(f"trace={trace}", flush=True)
            if cfg.results_json is not None:
                row = {
                    "style": style,
                    "world_size": layout.world_size,
                    "seq_len": cfg.seq_len,
                    "heads": cfg.heads,
                    "kv_heads": cfg.kv_heads,
                    "head_dim": cfg.head_dim,
                    "head_groups": head_groups,
                    "mask": cfg.mask.value,
                    "window": cfg.window,
                    "num_docs": cfg.num_docs,
                    "backend": cfg.backend.value,
                    "cuda_graph": cfg.cuda_graph,
                    "transport": transport.value,
                    "fwd_ms": fwd_ms,
                    "step_ms": step_ms,
                    "iters": cfg.iters,
                    # Eager step peak beyond live inputs; persistent symmetric buffers separate.
                    "peak_gib": peak_gib,
                    "symm_gib": symm_gib,
                    "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                }
                with cfg.results_json.open("a") as results:
                    results.write(json.dumps(row) + "\n")

    dist.barrier()
    if owns_process_group:
        dist.destroy_process_group()


def benchmark_style(cfg: Config, style: str, ctx: CPContext, tensors, device, transport):
    """Slowest-rank fwd and fwd+bwd median ms, eager step peak GiB, and optional merged trace."""
    q, k, v, grad_out = tensors
    annotate = ctx.annotate
    graphs = []

    def forward() -> None:
        with kernel_stage(f"cp/{style}/forward", annotate, backward=False):
            STYLE_FNS[style](q, k, v, ctx)

    def train_step() -> None:
        with kernel_stage(f"cp/{style}/forward", annotate, backward=False):
            out = STYLE_FNS[style](q, k, v, ctx)
        with kernel_stage(f"cp/{style}/backward", annotate, backward=False):
            out.backward(grad_out)

    for _ in range(3):
        train_step()
    peak_gib = step_peak_gib(train_step)
    run_forward, run_step = forward, train_step
    with torch.no_grad():
        for _ in range(3):
            forward()
        if cfg.cuda_graph:
            graphs.append(capture_cuda_graph(forward, annotate))
            run_forward = graphs[-1].replay
    if cfg.cuda_graph:
        graphs.append(capture_cuda_graph(train_step, annotate))
        run_step = graphs[-1].replay
    torch.cuda.synchronize()
    dist.barrier()
    with torch.no_grad():
        fwd_ms = time_steps(run_forward, cfg.iters)
    step_ms = time_steps(run_step, cfg.iters)
    # Report the slowest rank: every rank waits for it at the next collective.
    worst = torch.tensor([fwd_ms, step_ms, peak_gib], device=device)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)

    trace = None
    if cfg.trace_dir is not None:
        # Only the all-gather and ring styles depend on the transport.
        uses_transport = style in ("allgather", "allgather-overlap", "ring")
        style_tag = f"{style}-{transport.value}" if uses_transport else style
        graph_suffix = "_cudagraph" if cfg.cuda_graph else ""
        name = f"{style_tag}_{cfg.mask.value}_{cfg.backend.value.lower()}{graph_suffix}"
        trace = record_distributed_profile(
            run_step, cfg.trace_dir / f"{name}_seq{cfg.seq_len}", style, device, warmup_steps=2
        )
    # Release this style's graphs (and their memory pools) before the next style. Graphs that
    # captured NCCL work must also be released before teardown, or it hangs.
    for graph in graphs:
        graph.reset()
    torch.cuda.empty_cache()
    return (*worst.tolist(), trace)


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    seq_len: Annotated[int, typer.Option(help="Global sequence length.")] = 131072,
    heads: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    backend: Annotated[
        Backend, typer.Option(help="FlexAttention backend; FLASH needs flash-attn-4.")
    ] = Backend.TRITON,
    mask: Mask = Mask.CAUSAL,
    window: Annotated[int, typer.Option(help="Sliding-window size.")] = 8192,
    num_docs: Annotated[int, typer.Option(help="Documents for --mask document.")] = 12,
    head_groups: Annotated[int, typer.Option(help="Head groups for ulysses-overlap.")] = 2,
    styles: Annotated[
        list[Style] | None, typer.Option("--style", help="Repeat to pick styles (default: all).")
    ] = None,
    iters: int = 10,
    transport: Annotated[
        Transport,
        typer.Option(
            help="How allgather and ring move K/V; auto uses copy engines within one NVLink domain."
        ),
    ] = Transport.AUTO,
    cuda_graph: Annotated[
        bool, typer.Option(help="Replay each step from a CUDA graph (removes launch overhead).")
    ] = False,
    validate: Annotated[
        bool, typer.Option(help="Check outputs and gradients against single-device attention.")
    ] = False,
    results_json: Annotated[
        Path | None, typer.Option(help="Append one JSON line per style.")
    ] = None,
    trace_dir: Annotated[
        Path | None, typer.Option(help="Write merged Perfetto traces (needs transformer-nuggets).")
    ] = None,
    symm_timeout_ms: Annotated[
        int,
        typer.Option(
            min=0,
            help="Copy-engine signal timeout; a stuck rank traps with a message (0 waits forever).",
        ),
    ] = DEFAULT_TIMEOUT_MS,
) -> None:
    run(
        Config(
            seq_len=seq_len,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            backend=backend,
            mask=mask,
            window=window,
            num_docs=num_docs,
            head_groups=head_groups,
            styles=styles or list(Style),
            iters=iters,
            transport=transport,
            cuda_graph=cuda_graph,
            validate=validate,
            results_json=results_json,
            trace_dir=trace_dir,
            symm_timeout_ms=symm_timeout_ms,
        )
    )


if __name__ == "__main__":
    app()
