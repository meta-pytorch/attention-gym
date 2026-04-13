"""Run a Ring Attention example by directly invoking FlexAttention primitives.

Launch with:
    torchrun --standalone --nproc_per_node=<world_size> examples/ring_attention.py --seq-len <global_seq_len>
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch._dynamo import config as dynamo_config
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention


dynamo_config.skip_guards_on_constant_func_defaults = False

LN2 = math.log(2)


@dataclass(frozen=True)
class RingConfig:
    validate: bool
    backend: Literal["FLASH", "TRITON"] | None
    block_size: tuple[int, int] | None

    @property
    def kernel_options(self) -> dict[str, object] | None:
        if self.backend is None:
            return None
        return {"BACKEND": self.backend}


@dataclass(frozen=True)
class RingShard:
    rank: int
    world_size: int
    seq_len: int
    shard_len: int

    @property
    def next_rank(self) -> int:
        return (self.rank + 1) % self.world_size

    @property
    def prev_rank(self) -> int:
        return (self.rank - 1) % self.world_size

    @property
    def q_start(self) -> int:
        return self.rank * self.shard_len

    @property
    def q_end(self) -> int:
        return self.q_start + self.shard_len

    @property
    def local_slice(self) -> slice:
        return slice(self.q_start, self.q_end)


def _merge_attention(
    out: torch.Tensor,
    lse: torch.Tensor,
    new_out: torch.Tensor,
    new_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse = lse.unsqueeze(-1)
    new_lse = new_lse.unsqueeze(-1)
    max_lse = torch.maximum(lse, new_lse)
    exp_lse = torch.exp(lse - max_lse)
    exp_new_lse = torch.exp(new_lse - max_lse)
    denom = exp_lse + exp_new_lse
    return (
        (out * exp_lse + new_out * exp_new_lse) / denom,
        (max_lse + torch.log(denom)).squeeze(-1),
    )


def _flex_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask,
    scale: float,
    kernel_options: dict[str, object] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, aux = flex_attention(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        kernel_options=kernel_options,
        return_aux=AuxRequest(lse=True),
    )
    return out, _merge_lse_from_aux(aux.lse, kernel_options)


def _merge_lse_from_aux(
    lse: torch.Tensor,
    kernel_options: dict[str, object] | None,
) -> torch.Tensor:
    if kernel_options is not None and kernel_options.get("BACKEND") == "FLASH":
        return lse / LN2
    return lse


def _backward_lse_from_merge_lse(
    lse: torch.Tensor,
    backend: Literal["FLASH", "TRITON"] | None,
) -> torch.Tensor:
    match backend:
        case "TRITON":
            return lse / LN2
        case _:
            return lse


def _identity_score_mod(score, b, h, q_idx, kv_idx):
    return score


def _flex_bw_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fwd_out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    block_mask_tuple,
    scale: float,
    kernel_options: dict[str, object] | None,
):
    return torch.ops.higher_order.flex_attention_backward(
        q,
        k,
        v,
        fwd_out,
        lse,
        grad_out,
        None,
        _identity_score_mod,
        None,
        block_mask_tuple,
        scale,
        {} if kernel_options is None else kernel_options,
        (),
        (),
    )


merge_attention = torch.compile(_merge_attention, fullgraph=True)
compiled_create_block_mask = torch.compile(create_block_mask, fullgraph=True)
compiled_flex_chunk = torch.compile(_flex_chunk, fullgraph=True)
compiled_flex_bw_chunk = torch.compile(_flex_bw_chunk, fullgraph=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=131072, help="Global sequence length.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Input dtype.",
    )
    parser.add_argument(
        "--backend",
        choices=["FLASH", "TRITON"],
        default=None,
        help="Force a FlexAttention backend.",
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default=None,
        help="Block mask size as q_block,kv_block.",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run masked-visit invariants and single-process reference checks.",
    )
    return parser.parse_args()


def parse_block_size(block_size: str | None) -> tuple[int, int] | None:
    if block_size is None:
        return None
    q_block, kv_block = block_size.split(",", maxsplit=1)
    return int(q_block), int(kv_block)


def parse_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)


def validation_tolerances(
    dtype: torch.dtype,
    backend: Literal["FLASH", "TRITON"] | None,
) -> tuple[float, float, float, float]:
    match backend:
        case "FLASH" if dtype != torch.float32:
            return 1e-3, 1e-3, 1e-2, 1e-2
        case _:
            return 1e-4, 1e-4, 1e-3, 1e-3


def init_distributed(seq_len: int) -> tuple[torch.device, RingShard]:
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Launch with `torchrun --standalone --nproc_per_node=<world_size> "
            "examples/ring_attention.py --seq-len <global_seq_len>`."
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    declared_world_size = int(os.environ["WORLD_SIZE"])
    if seq_len <= 0:
        raise RuntimeError(f"Expected seq_len > 0, got {seq_len}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
        raise RuntimeError("This example requires one visible CUDA device per local rank.")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()
    if world_size != declared_world_size:
        raise RuntimeError(
            f"WORLD_SIZE={declared_world_size} does not match initialized process group "
            f"size {world_size}."
        )
    if world_size < 2:
        raise RuntimeError(f"Expected world_size >= 2, got {world_size}.")
    if seq_len % world_size != 0:
        raise RuntimeError(
            f"Expected seq_len={seq_len} to be divisible by world_size={world_size}."
        )
    return device, RingShard(
        rank=dist.get_rank(),
        world_size=world_size,
        seq_len=seq_len,
        shard_len=seq_len // world_size,
    )


def rank_print(shard: RingShard, message: str) -> None:
    print(f"[rank {shard.rank}] {message}", flush=True)


def make_global_tensor(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def ring_exchange(
    *tensors: torch.Tensor,
    send_rank: int,
    recv_rank: int,
) -> tuple[torch.Tensor, ...]:
    received = tuple(torch.empty_like(tensor) for tensor in tensors)
    requests = dist.batch_isend_irecv(
        [
            *(dist.P2POp(dist.isend, tensor.contiguous(), send_rank) for tensor in tensors),
            *(dist.P2POp(dist.irecv, recv_tensor, recv_rank) for recv_tensor in received),
        ]
    )
    for request in requests:
        request.wait()
    return received


def create_block_mask_for_lengths(
    mask_mod,
    q: torch.Tensor,
    kv_len: int,
    block_size: tuple[int, int] | None,
):
    kwargs = {
        "B": q.shape[0],
        "H": q.shape[1],
        "Q_LEN": q.shape[2],
        "KV_LEN": kv_len,
        "device": q.device,
    }
    if block_size is None:
        return compiled_create_block_mask(mask_mod, **kwargs)
    return compiled_create_block_mask(mask_mod, **kwargs, BLOCK_SIZE=block_size)


def build_block_mask(
    q: torch.Tensor,
    shard: RingShard,
    owner_rank: int,
    block_size: tuple[int, int] | None,
):
    kv_start = owner_rank * shard.shard_len

    def causal_shard_mask(b, h, q_idx, kv_idx, _q_start=shard.q_start, _kv_start=kv_start):
        return q_idx + _q_start >= kv_idx + _kv_start

    return create_block_mask_for_lengths(causal_shard_mask, q, shard.shard_len, block_size)


def is_fully_masked_remote_visit(shard: RingShard, owner_rank: int) -> bool:
    return shard.q_end <= owner_rank * shard.shard_len


def run_ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shard: RingShard,
    scale: float,
    config: RingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_k = k
    current_v = v
    current_owner = shard.rank
    merged_out, merged_lse = compiled_flex_chunk(
        q,
        current_k,
        current_v,
        build_block_mask(q, shard, current_owner, config.block_size),
        scale,
        config.kernel_options,
    )
    merged_out = merged_out.float()

    for _ in range(shard.world_size - 1):
        current_k, current_v = ring_exchange(
            current_k,
            current_v,
            send_rank=shard.next_rank,
            recv_rank=shard.prev_rank,
        )
        current_owner = (current_owner - 1) % shard.world_size
        chunk_out, chunk_lse = compiled_flex_chunk(
            q,
            current_k,
            current_v,
            build_block_mask(q, shard, current_owner, config.block_size),
            scale,
            config.kernel_options,
        )
        previous_out = merged_out
        previous_lse = merged_lse
        merged_out, merged_lse = merge_attention(merged_out, merged_lse, chunk_out, chunk_lse)
        if config.validate and is_fully_masked_remote_visit(shard, current_owner):
            torch.testing.assert_close(merged_out, previous_out, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(merged_lse, previous_lse, atol=1e-5, rtol=1e-5)

    return merged_out.to(q.dtype), merged_lse


def run_ring_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_out: torch.Tensor,
    merged_lse: torch.Tensor,
    grad_out: torch.Tensor,
    shard: RingShard,
    scale: float,
    config: RingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    merged_lse_for_backward = _backward_lse_from_merge_lse(merged_lse, config.backend)
    dq = torch.zeros_like(q)
    current_k = k
    current_v = v
    current_dk = torch.zeros_like(k)
    current_dv = torch.zeros_like(v)
    current_owner = shard.rank

    for _ in range(shard.world_size):
        block_mask = build_block_mask(q, shard, current_owner, config.block_size)
        dq_step, dk_step, dv_step, _ = compiled_flex_bw_chunk(
            q,
            current_k,
            current_v,
            merged_out,
            merged_lse_for_backward,
            grad_out,
            block_mask.as_tuple(),
            scale,
            config.kernel_options,
        )
        dq += dq_step
        current_dk += dk_step
        current_dv += dv_step
        current_k, current_v, current_dk, current_dv = ring_exchange(
            current_k,
            current_v,
            current_dk,
            current_dv,
            send_rank=shard.next_rank,
            recv_rank=shard.prev_rank,
        )
        current_owner = (current_owner - 1) % shard.world_size

    if current_owner != shard.rank:
        raise RuntimeError(
            f"Expected gradients for rank {shard.rank} to return home, got owner {current_owner}."
        )
    return dq, current_dk, current_dv


class RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        shard: RingShard,
        config: RingConfig,
    ):
        scale = q.shape[-1] ** -0.5
        merged_out, merged_lse = run_ring_forward(q, k, v, shard, scale, config)
        ctx.save_for_backward(q, k, v, merged_out, merged_lse)
        ctx.scale = scale
        ctx.shard = shard
        ctx.config = config
        return merged_out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, merged_out, merged_lse = ctx.saved_tensors
        dq, dk, dv = run_ring_backward(
            q,
            k,
            v,
            merged_out,
            merged_lse,
            grad_out,
            ctx.shard,
            ctx.scale,
            ctx.config,
        )
        return dq, dk, dv, None, None


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shard: RingShard,
    config: RingConfig | None = None,
) -> torch.Tensor:
    if config is None:
        config = RingConfig(validate=True, backend=None, block_size=None)
    return RingAttention.apply(q, k, v, shard, config)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: tuple[int, int] | None,
    kernel_options: dict[str, object] | None = None,
) -> torch.Tensor:
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask_for_lengths(causal_mask, q, k.shape[2], block_size)

    @torch.compile(fullgraph=True)
    def run_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask,
        scale: float,
        kernel_options: dict[str, object] | None,
    ):
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            kernel_options=kernel_options,
        )

    return run_reference(q, k, v, block_mask, q.shape[-1] ** -0.5, kernel_options)


def gather_sequence_shards(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=2)


def main() -> None:
    args = parse_args()
    device, shard = init_distributed(args.seq_len)
    config = RingConfig(
        validate=args.validate,
        backend=args.backend,
        block_size=parse_block_size(args.block_size),
    )
    torch.manual_seed(0)
    dtype = parse_dtype(args.dtype)
    if config.backend == "FLASH" and dtype == torch.float32:
        raise RuntimeError("FLASH backend requires float16 or bfloat16 inputs.")
    forward_atol, forward_rtol, grad_atol, grad_rtol = validation_tolerances(dtype, config.backend)
    shape = (1, 4, shard.seq_len, 128)
    local_slice = shard.local_slice

    q_full = make_global_tensor(shape, device, dtype, seed=0)
    k_full = make_global_tensor(shape, device, dtype, seed=1)
    v_full = make_global_tensor(shape, device, dtype, seed=2)
    q = q_full[:, :, local_slice].detach().clone().requires_grad_(True)
    k = k_full[:, :, local_slice].detach().clone().requires_grad_(True)
    v = v_full[:, :, local_slice].detach().clone().requires_grad_(True)

    out = ring_attention(q, k, v, shard, config)
    grad_out = make_global_tensor(
        shape if config.validate else q.shape,
        device,
        dtype,
        seed=3,
    )

    if config.validate:
        q_ref = q_full.detach().clone().requires_grad_(True)
        k_ref = k_full.detach().clone().requires_grad_(True)
        v_ref = v_full.detach().clone().requires_grad_(True)
        ref_out = reference_attention(
            q_ref,
            k_ref,
            v_ref,
            config.block_size,
            config.kernel_options,
        )

        torch.testing.assert_close(
            out,
            ref_out[:, :, local_slice],
            atol=forward_atol,
            rtol=forward_rtol,
        )
        rank_print(shard, "local forward shard matches reference")

        gathered_out = gather_sequence_shards(out)
        if shard.rank == 0:
            torch.testing.assert_close(
                gathered_out,
                ref_out,
                atol=forward_atol,
                rtol=forward_rtol,
            )
            rank_print(shard, "gathered forward matches reference")

        out.backward(grad_out[:, :, local_slice])
        ref_out.backward(grad_out)

        torch.testing.assert_close(
            q.grad,
            q_ref.grad[:, :, local_slice],
            atol=grad_atol,
            rtol=grad_rtol,
        )
        torch.testing.assert_close(
            k.grad,
            k_ref.grad[:, :, local_slice],
            atol=grad_atol,
            rtol=grad_rtol,
        )
        torch.testing.assert_close(
            v.grad,
            v_ref.grad[:, :, local_slice],
            atol=grad_atol,
            rtol=grad_rtol,
        )
        rank_print(shard, "local dq, dk, and dv match reference")

        gathered_dq = gather_sequence_shards(q.grad)
        gathered_dk = gather_sequence_shards(k.grad)
        gathered_dv = gather_sequence_shards(v.grad)
        if shard.rank == 0:
            torch.testing.assert_close(gathered_dq, q_ref.grad, atol=grad_atol, rtol=grad_rtol)
            torch.testing.assert_close(gathered_dk, k_ref.grad, atol=grad_atol, rtol=grad_rtol)
            torch.testing.assert_close(gathered_dv, v_ref.grad, atol=grad_atol, rtol=grad_rtol)
            rank_print(shard, "gathered outputs and gradients match the single-process reference")
    else:
        rank_print(shard, "forward completed without reference validation")
        out.backward(grad_out)
        rank_print(shard, "backward completed without reference validation")

    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
