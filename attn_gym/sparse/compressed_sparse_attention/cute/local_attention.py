"""Direct SM100 qv-only attention launchers for the CSA backend."""

from __future__ import annotations

from collections import OrderedDict
import math

import cutlass.cute as cute
import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor

from .fa4_local import FlashAttentionMLAForwardSm100


_COMPILE_CACHE_MAXSIZE = 64
_compile_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()


def _tensor_signature(tensor: torch.Tensor | None) -> tuple[object, ...] | None:
    if tensor is None:
        return None
    return (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))


def _cache_get(key: tuple[object, ...]) -> object | None:
    compiled = _compile_cache.get(key)
    if compiled is not None:
        _compile_cache.move_to_end(key)
    return compiled


def _cache_put(key: tuple[object, ...], compiled: object) -> None:
    _compile_cache[key] = compiled
    _compile_cache.move_to_end(key)
    if len(_compile_cache) > _COMPILE_CACHE_MAXSIZE:
        _compile_cache.popitem(last=False)


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
    compiled = _cache_get(key)
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
        None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _cache_put(key, compiled)
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
        _tensor_signature(query),
        _tensor_signature(value),
        _tensor_signature(local_value),
        _tensor_signature(gather),
        _tensor_signature(output),
        _tensor_signature(lse),
        _tensor_signature(sink),
        _tensor_signature(cos),
        _tensor_signature(sin),
        fuse_csa_epilogue,
        head_offset,
        fuse_q_rope,
        rope_dims,
        csa_topk,
        csa_window,
    )
    compiled = _cache_get(key)
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
    _cache_put(key, compiled)
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
    if query.device != value.device or query.dtype != value.dtype:
        raise ValueError("query and value must share a device and dtype.")
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
    ):
        raise ValueError("output must match query's shape, device, and dtype.")
    if lse is None:
        lse = torch.empty(batch, sequence, heads, device=query.device, dtype=torch.float32)
    elif (
        lse.shape != (batch, sequence, heads)
        or lse.device != query.device
        or lse.dtype != torch.float32
    ):
        raise ValueError("lse must be FP32 [B,S,H] on query's device.")

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
    if value.device != query.device or value.dtype != query.dtype:
        raise ValueError("query and compressed values must share a device and dtype.")
    if gather.device != query.device:
        raise ValueError("gather must be on query's device.")
    if local_value is not None:
        if local_value.shape != (batch, sequence, 1, 512):
            raise ValueError("local values must be [B,S,1,512].")
        if (
            not local_value.is_contiguous()
            or local_value.dtype != value.dtype
            or local_value.device != value.device
        ):
            raise ValueError(
                "local values must be contiguous and match compressed device/dtype."
            )
        if csa_topk < 0 or csa_window < 0 or csa_topk + csa_window > gather.shape[-1]:
            raise ValueError("CSA top-k/window counts must fit within the gather width.")
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
    ):
        raise ValueError("output must match query's shape, device, and dtype.")
    if lse is None and store_lse:
        lse = torch.empty(batch, sequence, heads, device=query.device, dtype=torch.float32)
    if not store_lse and lse is not None:
        raise ValueError("lse must be None when store_lse=False.")
    if lse is not None and (
        lse.shape != (batch, sequence, heads)
        or lse.device != query.device
        or lse.dtype != torch.float32
    ):
        raise ValueError("lse must be FP32 [B,S,H] on query's device.")

    if (sink is None) != (cos is None) or (sink is None) != (sin is None):
        raise ValueError("sink, cos, and sin must be provided together.")
    if fuse_q_rope and cos is None:
        raise ValueError("fused query RoPE requires sink, cosine, and sine tensors.")
    if sink is not None:
        if (
            sink.ndim != 1
            or sink.numel() < head_offset + heads
            or sink.device != query.device
        ):
            raise ValueError("sink must cover this head tile on query's device.")
        expected_rope_shape = (sequence, rope_dims // 2)
        for name, table in (("cos", cos), ("sin", sin)):
            if (
                table.shape != expected_rope_shape
                or table.device != query.device
                or table.dtype != torch.float32
            ):
                raise ValueError(
                    f"{name} must be FP32 with shape {expected_rope_shape}."
                )
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
