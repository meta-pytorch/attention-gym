"""CuTe DSL (SM100) backend for selected attention.

Uses FlashAttention-4's MLA forward kernel with index-gather mode and the
sparse MLA backward kernels. Registered as torch.library custom_ops (kernel
launches only) inside a torch.autograd.Function for torch.compile(fullgraph=True).

Constraints: head_dim=512, share_kv=True, nheads=128, bfloat16, SM100.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.flash_bwd_mla_dq_dqv_sm100 import dQdQvGemmKernel
from flash_attn.cute.flash_bwd_mla_sm100 import FlashAttentionSparseMLABackwardSm100
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

try:
    import cuda.bindings.driver as cuda_drv
except ImportError:
    from cuda import cuda as cuda_drv

import cutlass
from cutlass import cute


def _wrap_tensor(t, align=16):
    """Wrap a torch tensor for CuTe kernel launch with dynamic layout."""
    return from_dlpack(
        t.detach() if t.requires_grad else t, assumed_align=align
    ).mark_layout_dynamic(leading_dim=t.ndim - 1)


# ---------------------------------------------------------------------------
# Kernel compilation (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _compile_fwd(topk_length: int, nheads: int):
    batch_dummy, seqlen_q_dummy, seqlen_k_dummy = 1, 128, 1024
    hdimv = 512

    Qv = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    V = torch.empty(batch_dummy, seqlen_k_dummy, 1, hdimv, dtype=torch.bfloat16, device="cuda")
    O = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.empty(batch_dummy, seqlen_q_dummy, nheads, dtype=torch.float32, device="cuda")
    idx = torch.empty(batch_dummy, seqlen_q_dummy, topk_length, dtype=torch.int32, device="cuda")
    P = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, topk_length, dtype=torch.bfloat16, device="cuda"
    )
    RowMax = torch.empty(
        batch_dummy, seqlen_q_dummy, topk_length // 128, nheads, dtype=torch.float32, device="cuda"
    )

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    return cute.compile(
        FlashAttentionMLAForwardSm100(
            is_causal=False,
            use_cpasync_load_KV=True,
            topk_length=topk_length,
            is_topk_gather=True,
            pack_gqa=True,
            qhead_per_kvhead=nheads,
            nheads_kv=1,
            disable_bitmask=False,
            has_qk=False,
        ),
        None,
        from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1),
        None,
        from_dlpack(V, assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1),
        from_dlpack(O, assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1),
        from_dlpack(lse, assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1),
        1.0,
        from_dlpack(P, assumed_align=16).mark_layout_dynamic(leading_dim=P.ndim - 1),
        from_dlpack(RowMax, assumed_align=4).mark_layout_dynamic(leading_dim=RowMax.ndim - 1),
        mIndexTopk=from_dlpack(idx, assumed_align=16).mark_layout_dynamic(
            leading_dim=idx.ndim - 1
        ),
        stream=stream,
    )


@lru_cache(maxsize=8)
def _compile_bwd(topk_length: int, nheads: int):
    batch_dummy, seqlen_q_dummy = 1, 128
    hdimv = 512
    seqlen_k_dummy = 1024

    dO = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    V = torch.empty(batch_dummy, seqlen_k_dummy, 1, hdimv, dtype=torch.bfloat16, device="cuda")
    Qv = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    P = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, topk_length, dtype=torch.bfloat16, device="cuda"
    )
    dV = torch.empty(batch_dummy, seqlen_k_dummy, 1, hdimv, dtype=torch.float32, device="cuda")
    dS = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, topk_length, dtype=torch.bfloat16, device="cuda"
    )
    idx = torch.empty(batch_dummy, seqlen_q_dummy, topk_length, dtype=torch.int32, device="cuda")
    ScaleP = torch.empty(
        batch_dummy, seqlen_q_dummy, topk_length // 128, nheads, dtype=torch.float32, device="cuda"
    )
    dPsum = torch.empty(batch_dummy, seqlen_q_dummy, nheads, dtype=torch.float32, device="cuda")

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    def w(t, align=16):
        return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=t.ndim - 1)

    return cute.compile(
        FlashAttentionSparseMLABackwardSm100(
            is_causal=False,
            topk_length=topk_length,
            qhead_per_kvhead=nheads,
            nheads_kv=1,
            hdim=hdimv,
            hdimv=hdimv,
            has_seqused_q=False,
            disable_bitmask=False,
            use_clc_scheduler=True,
        ),
        w(dO),
        w(V),
        w(Qv),
        w(P),
        w(dV),
        w(dS),
        w(idx),
        1.0,
        w(ScaleP, 4),
        w(dPsum, 4),
        stream=stream,
    )


@lru_cache(maxsize=8)
def _compile_dq(topk_length: int, nheads: int, hdimv: int):
    batch_dummy, seqlen_q_dummy, total_kv_dummy = 1, 128, 1024
    dS = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, topk_length, dtype=torch.bfloat16, device="cuda"
    )
    V_3d = torch.empty(batch_dummy, total_kv_dummy, hdimv, dtype=torch.bfloat16, device="cuda")
    dQv = torch.empty(
        batch_dummy, seqlen_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    idx = torch.empty(batch_dummy, seqlen_q_dummy, topk_length, dtype=torch.int32, device="cuda")

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    def w(t, align=16):
        return from_dlpack(t, assumed_align=align).mark_layout_dynamic(leading_dim=t.ndim - 1)

    return cute.compile(
        dQdQvGemmKernel(
            acc_dtype=cutlass.Float32,
            nheads=nheads,
            head_dim_k=None,
            head_dim_v=hdimv,
            top_k=topk_length,
        ),
        w(dS),
        None,
        w(V_3d),
        None,
        w(dQv),
        w(idx),
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Index building (fully traceable by torch.compile)
# ---------------------------------------------------------------------------


def _build_unified_topk_indices(seq_len, sliding_window_size, indices, index_offset, device):
    batch = indices.shape[0]
    q = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    w = torch.arange(sliding_window_size, device=device, dtype=torch.int32).unsqueeze(0)
    candidate = q - sliding_window_size + 1 + w
    window_idxs = torch.where(candidate >= 0, candidate, -1)
    window_idxs = window_idxs.unsqueeze(0).expand(batch, -1, -1)

    offset_indices = torch.where(indices >= 0, (indices + index_offset).int(), -1)

    unified = torch.cat([window_idxs, offset_indices], dim=-1)
    raw_length = unified.shape[-1]
    padded_length = ((raw_length + 127) // 128) * 128
    if padded_length > raw_length:
        padding = torch.full(
            (batch, seq_len, padded_length - raw_length), -1, dtype=torch.int32, device=device
        )
        unified = torch.cat([unified, padding], dim=-1)

    return unified, padded_length


# ---------------------------------------------------------------------------
# Kernel-only custom ops (opaque to the compiler)
# ---------------------------------------------------------------------------


@torch.library.custom_op("selected_attn::fwd_kernel", mutates_args=())
def _fwd_kernel_op(
    Qv: torch.Tensor,
    V: torch.Tensor,
    topk_idxs: torch.Tensor,
    padded_topk_length: int,
    nheads: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch FA4 MLA forward kernel. Inputs/outputs in BSHD layout."""
    batch, seq_len, _, hdimv = Qv.shape
    device = Qv.device

    O = torch.empty(batch, seq_len, nheads, hdimv, dtype=torch.bfloat16, device=device)
    lse = torch.empty(batch, seq_len, nheads, dtype=torch.float32, device=device)
    P = torch.empty(
        batch, seq_len, nheads, padded_topk_length, dtype=torch.bfloat16, device=device
    )
    RowMax = torch.empty(
        batch, seq_len, padded_topk_length // 128, nheads, dtype=torch.float32, device=device
    )

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    stream = cuda_drv.CUstream(stream_ptr)
    kernel = _compile_fwd(padded_topk_length, nheads)

    w = _wrap_tensor
    kernel(
        None,
        w(Qv),
        None,
        w(V),
        w(O),
        w(lse, 4),
        softmax_scale,
        w(P),
        w(RowMax, 4),
        mIndexTopk=w(topk_idxs),
        stream=stream,
    )

    return O, lse, P, RowMax


@_fwd_kernel_op.register_fake
def _fwd_kernel_op_fake(Qv, V, topk_idxs, padded_topk_length, nheads, softmax_scale):
    batch, seq_len, _, hdimv = Qv.shape
    O = Qv.new_empty((batch, seq_len, nheads, hdimv))
    lse = Qv.new_empty((batch, seq_len, nheads), dtype=torch.float32)
    P = Qv.new_empty((batch, seq_len, nheads, padded_topk_length))
    RowMax = Qv.new_empty((batch, seq_len, padded_topk_length // 128, nheads), dtype=torch.float32)
    return O, lse, P, RowMax


@torch.library.custom_op("selected_attn::bwd_kernel", mutates_args=("dV_accum",))
def _bwd_kernel_op(
    dO: torch.Tensor,
    V: torch.Tensor,
    Qv: torch.Tensor,
    P: torch.Tensor,
    dV_accum: torch.Tensor,
    topk_idxs: torch.Tensor,
    ScaleP: torch.Tensor,
    dPsum: torch.Tensor,
    padded_topk_length: int,
    nheads: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Launch FA4 sparse MLA backward kernel. Returns dS. Mutates dV_accum in-place."""
    batch, seq_len = dO.shape[0], dO.shape[1]
    device = dO.device

    dS = torch.empty(
        batch, seq_len, nheads, padded_topk_length, dtype=torch.bfloat16, device=device
    )

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    stream = cuda_drv.CUstream(stream_ptr)
    kernel = _compile_bwd(padded_topk_length, nheads)

    w = _wrap_tensor
    kernel(
        w(dO),
        w(V),
        w(Qv),
        w(P),
        w(dV_accum),
        w(dS),
        w(topk_idxs),
        softmax_scale,
        w(ScaleP, 4),
        w(dPsum, 4),
        stream=stream,
    )

    return dS


@_bwd_kernel_op.register_fake
def _bwd_kernel_op_fake(
    dO, V, Qv, P, dV_accum, topk_idxs, ScaleP, dPsum, padded_topk_length, nheads, softmax_scale
):
    batch, seq_len = dO.shape[0], dO.shape[1]
    return dO.new_empty((batch, seq_len, nheads, padded_topk_length))


@torch.library.custom_op("selected_attn::dq_kernel", mutates_args=())
def _dq_kernel_op(
    dS: torch.Tensor,
    V_3d: torch.Tensor,
    topk_idxs: torch.Tensor,
    padded_topk_length: int,
    nheads: int,
    hdimv: int,
    batch: int,
    seq_len: int,
    total_kv: int,
) -> torch.Tensor:
    """Launch dQ/dQv GEMM kernel. Returns dQv in BSHD layout."""
    device = dS.device

    dQv = torch.empty(batch, seq_len, nheads, hdimv, dtype=torch.bfloat16, device=device)

    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    stream = cuda_drv.CUstream(stream_ptr)
    kernel = _compile_dq(padded_topk_length, nheads, hdimv)

    w = _wrap_tensor
    kernel(w(dS), None, w(V_3d), None, w(dQv), w(topk_idxs), stream=stream)

    return dQv


@_dq_kernel_op.register_fake
def _dq_kernel_op_fake(
    dS, V_3d, topk_idxs, padded_topk_length, nheads, hdimv, batch, seq_len, total_kv
):
    return dS.new_empty((batch, seq_len, nheads, hdimv))


# ---------------------------------------------------------------------------
# Autograd Function (all PyTorch logic visible to the compiler)
# ---------------------------------------------------------------------------

LOG2E = math.log2(math.e)


class _SelectedAttentionCuTe(torch.autograd.Function):
    @staticmethod
    def forward(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        sliding_window_size,
        prebuilt_topk_idxs,
    ):
        _b, _h, s, d = query.shape
        local_kv_len = local_kv.shape[2]
        device = query.device
        nheads = query.shape[1]

        unified_kv = torch.cat([local_kv, sparse_kv], dim=2)

        if prebuilt_topk_idxs is not None:
            topk_idxs = prebuilt_topk_idxs.view_as(prebuilt_topk_idxs)
            padded_topk_length = topk_idxs.shape[-1]
        else:
            topk_idxs, padded_topk_length = _build_unified_topk_indices(
                s, sliding_window_size, kv_indices, index_offset=local_kv_len, device=device
            )

        Qv = query.permute(0, 2, 1, 3)
        V = unified_kv.permute(0, 2, 1, 3)

        softmax_scale = 1.0 / math.sqrt(d)

        O_bshd, lse, P, RowMax = torch.ops.selected_attn.fwd_kernel(
            Qv,
            V,
            topk_idxs,
            padded_topk_length,
            nheads,
            softmax_scale,
        )

        O = O_bshd.permute(0, 2, 1, 3)
        return O, lse, P, RowMax, topk_idxs, unified_kv

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            query,
            local_kv,
            sparse_kv,
            _kv_indices,
            _attention_sink,
            sliding_window_size,
            _prebuilt_topk_idxs,
        ) = inputs
        O, lse, P, RowMax, topk_idxs, _unified_kv = output
        ctx.save_for_backward(query, O, lse, P, RowMax, topk_idxs, local_kv, sparse_kv)
        ctx.sliding_window_size = sliding_window_size
        ctx.local_kv_len = local_kv.shape[2]

    @staticmethod
    def backward(ctx, grad_O, grad_lse, grad_P, grad_RowMax, grad_topk, grad_ukv):
        Q, O, lse, P, RowMax, topk_idxs, local_kv, sparse_kv = ctx.saved_tensors
        local_kv_len = ctx.local_kv_len
        _b, _h, _s, d = Q.shape
        nheads = Q.shape[1]
        device = Q.device

        unified_kv = torch.cat([local_kv, sparse_kv], dim=2)

        Qv = Q.permute(0, 2, 1, 3)
        V = unified_kv.permute(0, 2, 1, 3)
        O_bshd = O.permute(0, 2, 1, 3)
        dO = grad_O.permute(0, 2, 1, 3)

        batch, seq_len, _, hdimv = Qv.shape
        total_kv = V.shape[1]
        padded = topk_idxs.shape[-1]
        softmax_scale = 1.0 / math.sqrt(d)

        dPsum = (dO * O_bshd).sum(dim=-1).float()
        lse_log2 = lse * LOG2E
        ScaleP = torch.exp2(softmax_scale * LOG2E * RowMax - lse_log2.unsqueeze(2))
        ScaleP = torch.where(
            (RowMax == -float("inf")) | (lse.unsqueeze(2) == -float("inf")),
            0.0,
            ScaleP,
        )

        dV_accum = torch.zeros(batch, total_kv, 1, hdimv, dtype=torch.float32, device=device)

        dS = torch.ops.selected_attn.bwd_kernel(
            dO,
            V,
            Qv,
            P,
            dV_accum,
            topk_idxs,
            ScaleP,
            dPsum,
            padded,
            nheads,
            softmax_scale,
        )

        V_3d = unified_kv.squeeze(1)
        dQv = torch.ops.selected_attn.dq_kernel(
            dS,
            V_3d,
            topk_idxs,
            padded,
            nheads,
            hdimv,
            batch,
            seq_len,
            total_kv,
        )

        dQ = dQv.permute(0, 2, 1, 3)
        dV_bf16 = dV_accum.to(torch.bfloat16).permute(0, 2, 1, 3)
        dKV = dV_bf16[:, :, :local_kv_len, :]
        d_ikv = dV_bf16[:, :, local_kv_len:, :]

        return dQ, dKV, d_ikv, None, None, None, None


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def _validate_cute_constraints(
    query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, share_kv
):
    if query.device.type != "cuda":
        raise ValueError("CuTe backend requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device) != (10, 0):
        raise ValueError("CuTe backend requires SM100.")
    if query.dtype != torch.bfloat16:
        raise TypeError("CuTe backend requires bfloat16.")
    if not share_kv:
        raise ValueError("CuTe backend requires share_kv=True.")

    if (attention_sink > -float("inf")).any():
        raise NotImplementedError(
            "CuTe backend does not fuse sink correction; "
            "attention_sink with any value > -inf is unsupported."
        )
    _b, h, _s, d = query.shape
    if d != 512:
        raise ValueError(f"CuTe backend requires head_dim=512, got {d}.")
    if h != 128:
        raise ValueError(f"CuTe backend requires 128 query heads, got {h}.")
    if local_kv.shape[1] != 1:
        raise ValueError("CuTe backend requires KV to have 1 head.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def selected_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool = True,
) -> torch.Tensor:
    """CuTe DSL (SM100) forward for selected attention.

    torch.compile(fullgraph=True) compatible. Sink correction not applied (assumes sink ≈ 0).
    """
    if not torch.compiler.is_compiling():
        _validate_cute_constraints(
            query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, share_kv
        )

    if doc_ids is not None:
        return _selected_attention_with_doc_ids(
            query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids, sliding_window_size
        )

    result = _SelectedAttentionCuTe.apply(
        query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, None
    )
    O = result[0]
    return O


def _selected_attention_with_doc_ids(
    query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids, sliding_window_size
):
    """Handle doc_id masking then dispatch to the same kernel."""
    s = query.shape[2]
    device = query.device
    local_kv_len = local_kv.shape[2]

    q_pos = torch.arange(s, device=device, dtype=torch.int32).unsqueeze(1)
    w_off = torch.arange(sliding_window_size, device=device, dtype=torch.int32).unsqueeze(0)
    window_kv_pos = q_pos - sliding_window_size + 1 + w_off
    valid = window_kv_pos >= 0

    query_doc = doc_ids[:, :, None]
    safe_pos = window_kv_pos.clamp(0).long()
    kv_doc = doc_ids[:, safe_pos.view(-1)].view(doc_ids.shape[0], s, sliding_window_size)
    same_doc = query_doc == kv_doc

    masked_window = torch.where(
        valid.unsqueeze(0) & same_doc,
        window_kv_pos.unsqueeze(0).expand_as(same_doc).int(),
        -1,
    )

    offset_indices = torch.where(
        kv_indices >= 0,
        (kv_indices + local_kv_len).int(),
        -1,
    )

    unified = torch.cat([masked_window, offset_indices], dim=-1)
    raw_length = unified.shape[-1]
    padded_length = ((raw_length + 127) // 128) * 128
    if padded_length > raw_length:
        padding = torch.full(
            (unified.shape[0], s, padded_length - raw_length),
            -1,
            dtype=torch.int32,
            device=device,
        )
        unified = torch.cat([unified, padding], dim=-1)

    return _selected_attention_with_prebuilt_indices(
        query, local_kv, sparse_kv, kv_indices, attention_sink, unified, sliding_window_size
    )


def _selected_attention_with_prebuilt_indices(
    query, local_kv, sparse_kv, kv_indices, attention_sink, topk_idxs, sliding_window_size
):
    """Forward+backward with pre-built indices (doc_ids path)."""
    result = _SelectedAttentionCuTe.apply(
        query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, topk_idxs
    )
    O = result[0]
    return O
