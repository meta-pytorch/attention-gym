from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import torch
from torch import Tensor

from attn_gym.types import Impl, resolve_impl


@dataclass(frozen=True, slots=True)
class AuxRequest:
    """Specifies which auxiliary outputs to return from gather_attn.

    Attributes:
        lse: If True, include the log-sum-exp tensor in the returned auxiliary data.
    """

    lse: bool = False


@dataclass(frozen=True, slots=True)
class GatherAttnAux:
    """Auxiliary outputs returned by gather_attn when requested.

    Attributes:
        lse: Log-sum-exp values with shape (batch_size, num_heads, sequence_length),
            or None if not requested. Fused implementations return nondifferentiable LSE.
    """

    lse: Tensor | None = None


def _validate_inputs(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
) -> None:
    """Validate tensor metadata without reading values or skipping compiled callers."""
    if type(sliding_window_size) is not int:
        raise TypeError(
            f"sliding_window_size must be a Python int, got {type(sliding_window_size).__name__}."
        )

    tensors = {
        "query": query,
        "local_kv": local_kv,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
    }

    if attention_sink is not None:
        tensors["attention_sink"] = attention_sink

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    if local_kv.dtype != query.dtype:
        raise ValueError(
            f"local_kv must have the same dtype as query, but got {local_kv.dtype} and {query.dtype}."
        )
    if sparse_kv.dtype != query.dtype:
        raise ValueError(
            f"sparse_kv must have the same dtype as query, but got {sparse_kv.dtype} and {query.dtype}."
        )

    if local_kv.device != query.device:
        raise ValueError(
            f"local_kv must be on the same device as query, but got {local_kv.device} and {query.device}."
        )
    if sparse_kv.device != query.device:
        raise ValueError(
            f"sparse_kv must be on the same device as query, but got {sparse_kv.device} and {query.device}."
        )

    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence_length, head_dim].")
    batch, heads, sequence_length, head_dim = query.shape
    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("query dimensions must all be positive.")
    if not query.is_floating_point():
        raise TypeError("Gather attention inputs must have a floating-point dtype.")

    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be non-negative.")

    expected_kv_heads = 1 if share_kv else heads
    if (
        sparse_kv.ndim != 4
        or sparse_kv.shape[0] != batch
        or sparse_kv.shape[1] != expected_kv_heads
        or sparse_kv.shape[3] != head_dim
    ):
        expected_h = "1" if share_kv else "heads"
        raise ValueError(
            f"sparse_kv must have shape [batch, {expected_h}, sparse_sequence_length, head_dim], "
            f"got {list(sparse_kv.shape)}."
        )
    if (
        local_kv.ndim != 4
        or local_kv.shape[0] != batch
        or local_kv.shape[1] != expected_kv_heads
        or local_kv.shape[2] != sequence_length
        or local_kv.shape[3] != head_dim
    ):
        expected_h = "1" if share_kv else "heads"
        raise ValueError(
            f"local_kv must have shape [batch, {expected_h}, sequence_length, head_dim], "
            f"got {list(local_kv.shape)}."
        )

    if (
        kv_indices.ndim != 3
        or kv_indices.shape[0] != batch
        or kv_indices.shape[1] != sequence_length
    ):
        raise ValueError(
            f"kv_indices must have shape [batch, sequence_length, num_topk], "
            f"got {list(kv_indices.shape)}."
        )
    if kv_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"kv_indices must be an integer tensor (int32 or int64), got {kv_indices.dtype}."
        )
    if kv_indices.device != query.device:
        raise ValueError(f"kv_indices must be on {query.device}, got {kv_indices.device}.")

    if attention_sink is not None:
        if attention_sink.dtype not in (query.dtype, torch.float32):
            raise ValueError(
                "attention_sink must have the same dtype as query or use torch.float32, "
                f"but got {attention_sink.dtype} and {query.dtype}."
            )
        if attention_sink.device != query.device:
            raise ValueError(
                f"attention_sink must be on the same device as query, but got {attention_sink.device} and {query.device}."
            )
        if attention_sink.shape != (heads,):
            raise ValueError(
                f"attention_sink must have shape [{heads}], got {list(attention_sink.shape)}."
            )

    if (cu_seqlens is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens and cu_seqlens_k must be supplied together")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require q to have batch size one")
        for name, offsets in (("cu_seqlens", cu_seqlens), ("cu_seqlens_k", cu_seqlens_k)):
            if not isinstance(offsets, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if offsets.ndim != 1 or offsets.shape[0] < 2:
                raise ValueError(f"{name} must have shape [num_sequences + 1]")
            if (
                offsets.dtype != torch.int32
                or not offsets.is_contiguous()
                or offsets.device != query.device
            ):
                raise ValueError(f"{name} must be contiguous int32 on q.device")
        if cu_seqlens.shape != cu_seqlens_k.shape:
            raise ValueError(
                "cu_seqlens and cu_seqlens_k must describe the same number of sequences"
            )


def _select_backend(
    query: Tensor,
    attention_sink: Tensor | None,
    share_kv: bool,
    *,
    num_keys: int,
) -> str:
    """Choose a CUDA backend after public input validation."""
    # FA4's Python launcher is eager-only, and its sparse backward does not implement
    # deterministic accumulation. Keep both contracts on the portable Triton path.
    if (
        torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or num_keys == 0
    ):
        return "triton"

    from .impl import cute as cute_backend

    return "cute" if cute_backend.is_supported(query, attention_sink, share_kv) else "triton"


@overload
def gather_attn(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None = ...,
    *,
    sliding_window_size: int = ...,
    cu_seqlens: Tensor | None = ...,
    cu_seqlens_k: Tensor | None = ...,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str] | None = None,
    scale: float | None = None,
    return_aux: None = ...,
) -> Tensor: ...


@overload
def gather_attn(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None = ...,
    *,
    sliding_window_size: int = ...,
    cu_seqlens: Tensor | None = ...,
    cu_seqlens_k: Tensor | None = ...,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str] | None = None,
    scale: float | None = None,
    return_aux: AuxRequest,
) -> tuple[Tensor, GatherAttnAux]: ...


def gather_attn(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None = None,
    *,
    sliding_window_size: int = 512,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str] | None = None,
    scale: float | None = None,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, GatherAttnAux]:
    """
    Performs gather attention.
        Each query attends to the previous sliding_window_size elements in the local_kv tensor
        as well as the positions in sparse_kv pointed to by kv_indices.
        Only one softmax is applied, covering both the sliding window and selected sparse positions.

    Args:
        query: query, shaped like (batch_size, num_heads, sequence_length, head_dim)

        local_kv: Key and Value for the sliding window branch,
            represented as a shared tensor, (batch_size, 1, sequence_length, head_dim)
            Or represented as (batch_size, num_heads, sequence_length, head_dim)

        sparse_kv: KV candidate pool for the indexing branch, shape of (batch, 1, X, head_dim)
            Or shaped as (batch_size, num_heads, X, head_dim)
            where X is any integer

        kv_indices: Integer selections shaped (batch, sequence_length, num_topk_blocks),
            shared across heads. Without packed offsets these index the batch's sparse_kv
            pool. With packed offsets they are zero-based within the query's document:
            local index j selects sparse_kv at cu_seqlens_k[document] + j.
            Negative or out-of-pool indices are ignored. Repeated indices retain their
            multiplicity. The slot count may exceed the pool length; -1 pads unused slots.
            Sparse causality is the caller's responsibility.

        attention_sink: tensor in shape of (num_heads, ), learnable per-head weight that occupies
            the denominator of softmax. It may use the query dtype or torch.float32.
            If None, no attention sink will be applied

        cu_seqlens: Packed offsets shaped [N + 1] for batch-one inputs, as contiguous
            int32 on query.device. They partition both queries and local_kv; the local
            window never crosses a document boundary. Offsets start at zero, never
            decrease, may repeat for empty sequences, and may end before sequence_length.
            Output values and token-input gradients beyond that endpoint are undefined;
            fixed-capacity callers must mask them. Offset values are a caller contract
            and are not inspected on the host. Supply both offset tensors or neither.

        cu_seqlens_k: Packed offsets shaped [N + 1] for sparse_kv, as contiguous int32
            on query.device. They describe the same documents as cu_seqlens, start at zero,
            never decrease, and end at or before the sparse pool length. Empty sparse
            documents have repeated offsets. These offsets need not match query offsets:
            for compressed KV, each document contributes only its complete compressed blocks.
            Without offsets, each batch element is a single document.

        sliding_window_size: Integer, size of sliding window

        impl: Impl.FUSED (default, or "fused") uses optimized CUDA kernels;
            Impl.REFERENCE (or "reference") uses eager PyTorch on CPU or CUDA.

        kernel_options: Fused backend override: {"backend": "cute"} or {"backend": "triton"}.
            Omit the backend to prefer supported CuTe calls, otherwise Triton.
            Compiled calls, deterministic mode, and empty attention sets use Triton
            automatically. Explicit backend requests are honored; execution failures are never
            retried on another backend. Nonempty options are invalid with Impl.REFERENCE.
            Triton shared-KV backward uses nondeterministic atomic accumulation unless
            torch.use_deterministic_algorithms is enabled.

        scale: Positive multiplier for query-key logits. Defaults to 1 / sqrt(head_dim).
            Attention sink logits are not scaled.

        return_aux: If None (default), return only the output tensor. If an AuxRequest
            instance, return a tuple of (output, GatherAttnAux) containing the
            requested auxiliary outputs (e.g. LSE when return_aux.lse is True).

    Returns:
        If return_aux is None: output tensor with shape
            (batch_size, num_heads, sequence_length, head_dim).
        If return_aux is an AuxRequest: tuple of (output, GatherAttnAux) where
            output has shape (batch_size, num_heads, sequence_length, head_dim) and
            aux.lse has shape (batch_size, num_heads, sequence_length) when requested.
        A row with no selected or local keys returns zero output, with LSE equal to its
        sink logit, or -inf when there is no sink. Its output-loss gradients are zero.
    """
    selected_impl = resolve_impl(impl)
    if selected_impl is Impl.REFERENCE and kernel_options:
        raise ValueError("kernel_options are not supported with impl='reference'")
    if kernel_options not in (None, {}, {"backend": "cute"}, {"backend": "triton"}):
        raise ValueError(f"unsupported gather_attn kernel options: {kernel_options}")

    share_kv = isinstance(sparse_kv, Tensor) and sparse_kv.ndim == 4 and sparse_kv.shape[1] == 1
    _validate_inputs(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        sliding_window_size,
        share_kv,
        cu_seqlens,
        cu_seqlens_k,
    )
    if scale is not None and not scale > 0:
        raise ValueError("scale must be greater than 0.")
    scale = query.shape[-1] ** -0.5 if scale is None else scale

    backend = (kernel_options or {}).get("backend")
    if selected_impl is Impl.REFERENCE:
        from .impl import reference as implementation
    else:
        if query.device.type != "cuda":
            raise ValueError("fused gather_attn requires CUDA tensors; use impl='reference'")
        if backend is None:
            backend = _select_backend(
                query,
                attention_sink,
                share_kv,
                num_keys=sliding_window_size + kv_indices.shape[-1],
            )
        if backend == "cute":
            from .impl import cute as implementation
        else:
            from .impl import triton as implementation

    if backend != "cute" and attention_sink is None:
        attention_sink = torch.full(
            (query.shape[1],), float("-inf"), dtype=query.dtype, device=query.device
        )
    output, lse = implementation.gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        share_kv,
        scale=scale,
    )

    if return_aux is None:
        return output

    return output, GatherAttnAux(lse=lse if return_aux.lse else None)
