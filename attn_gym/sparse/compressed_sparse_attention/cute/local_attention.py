"""Direct SM100 qv-only attention launchers for the CSA backend."""

from __future__ import annotations

import math

import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from .fa4_local import FlashAttentionMLAForwardSm100


_compile_cache: dict[tuple[object, ...], object] = {}


def _compile_local(
    query: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    window: int,
) -> object:
    key = (
        query.dtype,
        tuple(query.shape),
        tuple(query.stride()),
        tuple(value.shape),
        tuple(value.stride()),
        tuple(output.stride()),
        window,
    )
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled

    heads = query.shape[2]
    kernel = FlashAttentionMLAForwardSm100(
        is_causal=False,
        is_local=True,
        use_cpasync_load_KV=False,
        topk_length=value.shape[1],
        is_topk_gather=False,
        pack_gqa=True,
        qhead_per_kvhead=heads,
        nheads_kv=1,
        is_varlen_q=False,
        disable_bitmask=False,
        has_qk=False,
    )
    compiled = cute.compile(
        kernel,
        None,
        to_cute_tensor(query),
        None,
        to_cute_tensor(value),
        to_cute_tensor(output),
        to_cute_tensor(lse, assumed_align=4),
        1.0 / math.sqrt(query.shape[-1]),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        window - 1,
        0,
        None,
        None,
        None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    return compiled


def _compile_compressed(
    query: torch.Tensor,
    value: torch.Tensor,
    local_value: torch.Tensor | None,
    gather: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None,
    sink: torch.Tensor | None,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    head_offset: int,
    fuse_q_rope: bool,
    rope_dims: int,
    csa_topk: int,
    csa_window: int,
) -> object:
    fuse_csa_epilogue = sink is not None
    key = (
        "compressed",
        query.dtype,
        tuple(query.shape),
        tuple(query.stride()),
        tuple(value.shape),
        tuple(value.stride()),
        tuple(local_value.shape) if local_value is not None else None,
        tuple(local_value.stride()) if local_value is not None else None,
        tuple(gather.shape),
        tuple(output.stride()),
        tuple(lse.stride()) if lse is not None else None,
        fuse_csa_epilogue,
        head_offset,
        fuse_q_rope,
        rope_dims,
        csa_topk,
        csa_window,
    )
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled

    heads = query.shape[2]
    kernel = FlashAttentionMLAForwardSm100(
        is_causal=False,
        is_local=False,
        use_cpasync_load_KV=True,
        topk_length=gather.shape[-1],
        is_topk_gather=True,
        pack_gqa=True,
        qhead_per_kvhead=heads,
        nheads_kv=1,
        is_varlen_q=False,
        disable_bitmask=False,
        has_qk=False,
        fuse_csa_epilogue=fuse_csa_epilogue,
        fuse_csa_q_rope=fuse_q_rope,
        csa_head_offset=head_offset,
        csa_rope_dims=rope_dims,
        csa_topk=csa_topk,
        csa_window=csa_window,
    )
    compiled = cute.compile(
        kernel,
        None,
        to_cute_tensor(query),
        None,
        to_cute_tensor(value),
        to_cute_tensor(output),
        to_cute_tensor(lse, assumed_align=4) if lse is not None else None,
        1.0 / math.sqrt(query.shape[-1]),
        None,
        None,
        None,
        None,
        None,
        None,
        to_cute_tensor(gather),
        None,
        None,
        None,
        to_cute_tensor(sink) if sink is not None else None,
        to_cute_tensor(cos, assumed_align=4) if cos is not None else None,
        to_cute_tensor(sin, assumed_align=4) if sin is not None else None,
        to_cute_tensor(local_value) if local_value is not None else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    return compiled


def local_attention(
    query: torch.Tensor,
    value: torch.Tensor,
    window: int,
    *,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact causal sliding-window attention with one SM100 CuTe launch.

    ``query`` and ``value`` use BSHD layout. CSA uses 128-head optimized tiles,
    one shared value head, and D=512; the public wrapper pads only the final head tile.
    """
    if query.ndim != 4 or value.ndim != 4:
        raise ValueError("query and value must use BSHD layout.")
    batch, sequence, heads, dim = query.shape
    if heads not in (64, 128) or dim != 512 or value.shape != (batch, sequence, 1, 512):
        raise ValueError(
            "local CuTe attention requires Q=[B,S,H,512] with H in {64,128} "
            "and V=[B,S,1,512]."
        )
    if not 1 <= window <= sequence:
        raise ValueError("window must be in [1, sequence].")
    if not query.is_contiguous() or not value.is_contiguous():
        raise ValueError("query and value must be contiguous BSHD tensors.")
    if output is None:
        output = torch.empty_like(query)
    if lse is None:
        lse = torch.empty(batch, sequence, heads, device=query.device, dtype=torch.float32)

    compiled = _compile_local(query, value, output, lse, window)
    compiled(
        None,
        query,
        None,
        value,
        output,
        lse,
        1.0 / math.sqrt(dim),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        window - 1,
        0,
        None,
        None,
        None,
    )
    return output, lse


def compressed_attention(
    query: torch.Tensor,
    value: torch.Tensor,
    gather: torch.Tensor,
    *,
    local_value: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    head_offset: int = 0,
    fuse_q_rope: bool = False,
    rope_dims: int = 64,
    csa_topk: int = 0,
    csa_window: int = 0,
    store_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the padded top-k compressed partial with the vendored SM100 CuTe kernel."""
    if query.ndim != 4 or value.ndim != 4 or gather.ndim != 3:
        raise ValueError("query/value must use BSHD and gather must use BSK layout.")
    batch, sequence, heads, dim = query.shape
    if heads != 128 or dim != 512 or value.shape[:1] != (batch,):
        raise ValueError("compressed CuTe attention requires Q=[B,S,128,512].")
    if value.shape[2:] != (1, 512) or gather.shape[:2] != (batch, sequence):
        raise ValueError("compressed values must be [B,N,1,512] and gather [B,S,K].")
    if gather.shape[-1] % 128 or gather.dtype != torch.int32:
        raise ValueError("the CuTe gather length must be a multiple of 128 with int32 indices.")
    if (not query.is_contiguous() and not fuse_q_rope) or query.stride(-1) != 1:
        raise ValueError("query must be contiguous unless fused Q RoPE is enabled.")
    if not value.is_contiguous() or not gather.is_contiguous():
        raise ValueError("value and gather must be contiguous.")
    if local_value is not None:
        if local_value.shape != (batch, sequence, 1, 512):
            raise ValueError("local values must be [B,S,1,512].")
        if not local_value.is_contiguous() or local_value.dtype != value.dtype:
            raise ValueError("local values must be contiguous and match compressed dtype.")
        if csa_topk < 0 or csa_window < 0 or csa_topk + csa_window > gather.shape[-1]:
            raise ValueError("CSA top-k/window counts must fit within the gather width.")
    if output is None:
        output = torch.empty_like(query)
    if lse is None and store_lse:
        lse = torch.empty(batch, sequence, heads, device=query.device, dtype=torch.float32)
    if not store_lse and lse is not None:
        raise ValueError("lse must be None when store_lse=False.")

    if (sink is None) != (cos is None) or (sink is None) != (sin is None):
        raise ValueError("sink, cos, and sin must be provided together.")
    compiled = _compile_compressed(
        query,
        value,
        local_value,
        gather,
        output,
        lse,
        sink,
        cos,
        sin,
        head_offset,
        fuse_q_rope,
        rope_dims,
        csa_topk,
        csa_window,
    )
    compiled(
        None,
        query,
        None,
        value,
        output,
        lse,
        1.0 / math.sqrt(dim),
        None,
        None,
        None,
        None,
        None,
        None,
        gather,
        None,
        None,
        None,
        sink,
        cos,
        sin,
        local_value,
    )
    return output, lse


__all__ = ["compressed_attention", "local_attention"]
