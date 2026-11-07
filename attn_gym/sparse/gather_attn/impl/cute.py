"""CuTe DSL (SM100/SM103) backend for gather attention.

Launches FlashAttention-4 sparse MLA kernels behind the backend-neutral opaque
operators. The caller owns the tensor autograd tape; FA4 owns kernel compilation,
workspace allocation, and backward orchestration.

Constraints
-----------
- latent width = 512 with independent RoPE width 0 or 64, 1 <= nheads <= 128,
  share_kv = True (fewer than
  128 heads are zero-padded to FA4's 64/128-head tiles in-kernel via TMA out-of-bounds)
- dtype = bfloat16, SM100 or SM103 (compute capability 10.0 or 10.3)
- Requires FA4 4.0.0b32+ for sparse MLA attention sinks and, with fewer than 128
  heads, sparse-MLA head padding; 4.0.0b33+ to recompute probabilities in backward
  for any head count
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from functools import cache

import torch

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


# FA4's sparse MLA kernels fix the latent (value) width and the key-only RoPE width.
VALUE_DIM = 512
ROPE_DIMS = (0, 64)


def _constraint_violation(
    query: torch.Tensor, share_kv: bool, query_rope: torch.Tensor | None
) -> Exception | None:
    """Return the error for metadata this backend cannot run, or None when it qualifies."""
    _b, h, _s, d = query.shape
    if query.device.type != "cuda":
        return ValueError("CuTe backend requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device) not in SUPPORTED_CAPABILITIES:
        return ValueError("CuTe backend requires SM100 or SM103.")
    if query.dtype != torch.bfloat16:
        return TypeError("CuTe backend requires bfloat16.")
    if not share_kv:
        return ValueError("CuTe backend requires share_kv=True.")
    if d != VALUE_DIM or (0 if query_rope is None else query_rope.shape[-1]) not in ROPE_DIMS:
        return ValueError("CuTe backend requires latent width 512 and RoPE width 0 or 64.")
    if not 0 < h <= 128:
        return ValueError(f"CuTe backend requires at most 128 query heads, got {h}.")
    return None


def _has_sparse_mla_head_padding() -> bool:
    """Detect FA4's arbitrary-head sparse MLA kernels; older FA4 requires H=128.

    The tile helper arrived in ``pack_gqa`` and later moved to ``config``.
    """
    for module in ("flash_attn.cute.config", "flash_attn.cute.pack_gqa"):
        try:
            if hasattr(importlib.import_module(module), "sparse_mla_qhead_tile"):
                return True
        except ImportError:
            continue
    return False


@cache
def _fa4_available(with_sink: bool, *, padded_heads: bool = False) -> bool:
    """Probe the optional dependency once, only after tensor metadata qualifies."""
    try:
        from flash_attn.cute.interface import flash_attn_func  # noqa: F401

        if padded_heads and not _has_sparse_mla_head_padding():
            return False

        if with_sink:
            from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

            return (
                "learnable_sink"
                in inspect.signature(FlashAttentionMLAForwardSm100.__call__).parameters
            )
    except ImportError:
        return False
    return True


@torch.compiler.assume_constant_result
def _environment_supported(device: torch.device, with_sink: bool, padded_heads: bool) -> bool:
    """Device and installed FA4 capabilities are stable for the lifetime of a compiled graph.

    Do not pass tensors or hide model sizes/determinism here: the caller guards those.
    """
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        return False
    return _fa4_available(with_sink, padded_heads=padded_heads)


def is_supported(query, attention_sink, share_kv, query_rope) -> bool:
    """Guard tensor metadata; inspect optional FA4 features without tracing its launcher."""
    if (
        query.device.type != "cuda"
        or query.dtype != torch.bfloat16
        or not share_kv
        or query.shape[-1] != VALUE_DIM
        or (0 if query_rope is None else query_rope.shape[-1]) not in ROPE_DIMS
        or not 0 < query.shape[1] <= 128
    ):
        return False
    # FA4 requires a unit leading stride. Its contiguous repair skips singleton views,
    # which PyTorch considers contiguous even when their stride is not one.
    if (
        attention_sink is not None
        and attention_sink.numel() == 1
        and attention_sink.stride(0) != 1
    ):
        return False
    # An explicit branch guards symbolic head counts and passes a concrete bool to the probe.
    padded_heads = False if query.shape[1] == 128 else True  # noqa: SIM211
    return _environment_supported(query.device, attention_sink is not None, padded_heads)


def _check_backward_mode() -> None:
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "CuTe gather attention does not support deterministic backward; "
            "use kernel_options={'backend': 'triton'} for the forward call."
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, stacklevel=2)
        else:
            raise RuntimeError(message)


def _sparse_mla_operands(query, kv, query_rope, kv_rope):
    # The exact same latent KV operand contributes both score and value gradients.
    if query_rope is not None and query_rope.shape[-1] == 0:
        query_rope = kv_rope = None
    return {"qv": query, "v": kv, "q": query_rope, "k": kv_rope}


def _sequence_major(tensor, packed):
    if tensor is None:
        return None
    native = tensor.permute(0, 2, 1, 3)
    return native.squeeze(0) if packed else native


def _prepare_inputs(
    query: torch.Tensor,
    local_kv: torch.Tensor | None,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from .indices import build_gather_indices
    from .packed_kv import launch_pack_kv

    indices = build_gather_indices(
        kv_indices,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        sparse_kv.shape[2],
        has_local=local_kv is not None,
    )
    if local_kv is None:
        if cu_seqlens is None:
            return (
                _sequence_major(query, False),
                _sequence_major(sparse_kv, False),
                indices,
                None,
                None,
            )
        # Tiny metadata copies append an isolated inactive-capacity document, not KV copies.
        cu_q = torch.cat((cu_seqlens, cu_seqlens.new_full((1,), query.shape[2])))
        cu_kv = torch.cat((cu_seqlens_k, cu_seqlens_k.new_full((1,), sparse_kv.shape[2])))
        return (
            _sequence_major(query, True),
            _sequence_major(sparse_kv, True),
            indices[0],
            cu_q,
            cu_kv,
        )
    if cu_seqlens is None:
        kv = torch.cat([local_kv, sparse_kv], dim=2).permute(0, 2, 1, 3)
        return query.permute(0, 2, 1, 3), kv, indices, None, None
    kv, cu_q, cu_kv = launch_pack_kv(local_kv, sparse_kv, cu_seqlens, cu_seqlens_k)
    return query.squeeze(0).transpose(0, 1), kv, indices[0], cu_q, cu_kv


def _launch_forward(
    query: torch.Tensor,
    local_kv: torch.Tensor | None,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    needs_backward: bool,
    bwd_recompute_p: bool,
    *,
    query_rope: torch.Tensor | None = None,
    sparse_rope: torch.Tensor | None = None,
    local_rope: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    if (error := _constraint_violation(query, share_kv, query_rope)) is not None:
        raise error
    from flash_attn.cute.interface import _flash_attn_fwd

    q, kv, indices, cu_q, cu_kv = _prepare_inputs(
        query, local_kv, sparse_kv, kv_indices, cu_seqlens, cu_seqlens_k, sliding_window_size
    )
    # FA4 decides which tape to allocate from requires_grad, not grad mode. AOT can
    # supply detached inputs to this opaque launcher; mark a private view explicitly.
    q_rope = _sequence_major(query_rope, cu_seqlens is not None)
    kv_rope = None
    if query_rope is not None and query_rope.shape[-1] != 0:
        if local_kv is None:
            kv_rope = _sequence_major(sparse_rope, cu_seqlens is not None)
        elif cu_seqlens is None:
            kv_rope = torch.cat((local_rope, sparse_rope), dim=2).permute(0, 2, 1, 3)
        else:
            from .packed_kv import launch_pack_kv

            kv_rope, _, _ = launch_pack_kv(local_rope, sparse_rope, cu_seqlens, cu_seqlens_k)
    operands = _sparse_mla_operands(q, kv, q_rope, kv_rope)
    operands = {
        name: None if tensor is None else tensor.detach() for name, tensor in operands.items()
    }
    operands["qv"].requires_grad_(needs_backward)
    out, lse, p, row_max, o_lo = _flash_attn_fwd(
        **operands,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        gather_kv_indices=indices,
        softmax_scale=scale,
        learnable_sink=None if attention_sink is None else attention_sink.detach(),
        causal=False,
        pack_gqa=True,
        return_lse=True,
        gather_bwd_recompute_p=bwd_recompute_p,
    )
    if cu_seqlens is not None:
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)
    return (
        # Preserve FA4's sequence-major storage without making the custom Function
        # return a view. The fake advertises the same strides.
        out.permute(0, 2, 1, 3).detach(),
        lse.permute(0, 2, 1).contiguous(),
        (
            None if local_kv is None else kv,
            indices,
            cu_q,
            cu_kv,
            p,
            row_max,
            o_lo,
            None if local_kv is None else kv_rope,
        ),
    )


def _launch_backward(
    query: torch.Tensor,
    local_kv: torch.Tensor | None,
    sparse_kv: torch.Tensor,
    attention_sink: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    state: tuple[torch.Tensor | None, ...],
    scale: float,
    bwd_recompute_p: bool,
    *,
    query_rope: torch.Tensor | None = None,
    sparse_rope: torch.Tensor | None = None,
    local_rope: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if local_kv is None and sparse_kv.shape[2] == 0:
        # FA4 forward returns no probability or residual state for an empty pool.
        return tuple(
            None if tensor is None else tensor.new_zeros(tensor.shape)
            for tensor in (
                query,
                sparse_kv,
                local_kv,
                query_rope,
                sparse_rope,
                local_rope,
                attention_sink,
            )
        )
    _check_backward_mode()
    from flash_attn.cute.interface import _flash_attn_bwd_sparse_mla

    from .packed_kv import launch_unpack_kv

    kv, indices, cu_q, cu_kv, p, row_max, o_lo, kv_rope = state
    if local_kv is None:
        kv = _sequence_major(sparse_kv, cu_seqlens is not None)
        kv_rope = _sequence_major(sparse_rope, cu_seqlens is not None)
    q = query.permute(0, 2, 1, 3)
    out = output.permute(0, 2, 1, 3)
    dout = grad_output.permute(0, 2, 1, 3)
    native_lse = lse.permute(0, 2, 1)
    if cu_seqlens is not None:
        q = q.squeeze(0)
        out, dout, native_lse = out.squeeze(0), dout.squeeze(0), native_lse.squeeze(0)
    # FA4 writes each query-component gradient directly into its final-layout buffer.
    grad_query = query.new_empty(query.shape)
    native_grad_query = grad_query.permute(0, 2, 1, 3)
    if cu_seqlens is not None:
        native_grad_query = native_grad_query.squeeze(0)
    grad_query_rope = None if query_rope is None else query_rope.new_empty(query_rope.shape)
    _, dk, dv, _, dsink = _flash_attn_bwd_sparse_mla(
        **_sparse_mla_operands(
            q, kv, _sequence_major(query_rope, cu_seqlens is not None), kv_rope
        ),
        out=out,
        dout=dout,
        lse=native_lse,
        p=p,
        row_max=row_max,
        o_lo=o_lo,
        gather_kv_indices=indices,
        learnable_sink=attention_sink,
        softmax_scale=scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_kv,
        recompute_p=bwd_recompute_p,
        dqv=native_grad_query,
        dq=None
        if query_rope is None or query_rope.shape[-1] == 0
        else _sequence_major(grad_query_rope, cu_seqlens is not None),
    )

    # FA4 may return FP32 scatter accumulators. The public autograd wrapper normally
    # casts these to each leaf's dtype; our opaque backward must do that explicitly.
    def split_grad(grad, local, sparse):
        if grad is None:
            return (
                None if local is None else local.new_zeros(local.shape),
                None if sparse is None else sparse.new_zeros(sparse.shape),
            )
        grad = grad.to(sparse.dtype)
        if local is None:
            native = grad.unsqueeze(0) if cu_seqlens is not None else grad
            return None, native.permute(0, 2, 1, 3).clone(memory_format=torch.contiguous_format)
        if cu_seqlens is None:
            grad = grad.permute(0, 2, 1, 3)
            return (
                grad[:, :, : local.shape[2]].clone(memory_format=torch.contiguous_format),
                grad[:, :, local.shape[2] :].clone(memory_format=torch.contiguous_format),
            )
        return launch_unpack_kv(grad, local.shape, sparse.shape, cu_seqlens, cu_seqlens_k)

    grad_local, grad_sparse = split_grad(dv, local_kv, sparse_kv)
    grad_local_rope, grad_sparse_rope = split_grad(dk, local_rope, sparse_rope)
    if dsink is not None:
        dsink = dsink.to(attention_sink.dtype).contiguous()
    return (
        grad_query,
        grad_sparse,
        grad_local,
        grad_query_rope,
        grad_sparse_rope,
        grad_local_rope,
        dsink,
    )
