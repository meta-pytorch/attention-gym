from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, overload

import torch
from torch import Tensor

from attn_gym._cu_seqlens import validate_cu_seqlens_pair
from attn_gym.types import AuxRequest, Impl, resolve_impl

from .impl import gather_attn as fused_gather_attn
from .impl.routing import require_grouped_schedule, uses_grouped_schedule


class RopePair(NamedTuple):
    """Latent and already-rotated RoPE tensors, kept as separate operands."""

    latent: Tensor
    rope: Tensor


AttentionInput = Tensor | tuple[Tensor, Tensor]


def _split_input(value: AttentionInput, name: str) -> tuple[Tensor, Tensor | None]:
    if isinstance(value, Tensor):
        return value, None
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not all(isinstance(tensor, Tensor) for tensor in value)
    ):
        raise TypeError(f"{name} must be a Tensor or a (latent, rope) tensor pair.")
    return value


@dataclass(frozen=True, slots=True)
class GatherAttnAux:
    """Auxiliary outputs returned by gather_attn when requested.

    Attributes:
        lse: Sink-inclusive log-sum-exp values with shape
            (batch_size, num_heads, sequence_length), or None if not requested.
            Fused implementations return nondifferentiable float32 LSE.
    """

    lse: Tensor | None = None


def _validate_inputs(
    query: Tensor,
    local_kv: Tensor | None,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None,
    sliding_window_size: int,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> None:
    """Validate tensor metadata without reading values or skipping compiled callers."""
    if type(sliding_window_size) is not int:
        raise TypeError(
            f"sliding_window_size must be a Python int, got {type(sliding_window_size).__name__}."
        )

    tensors = {
        "query": query,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
    }

    if local_kv is not None:
        tensors["local_kv"] = local_kv
    if attention_sink is not None:
        tensors["attention_sink"] = attention_sink
    if sparse_v is None and local_v is not None:
        raise ValueError("local_v and sparse_v must be supplied together")
    if sparse_v is not None and (local_v is None) != (local_kv is None):
        if local_kv is None:
            raise ValueError("local_kv=None requires local_v=None.")
        raise ValueError("local_v and sparse_v must be supplied together")
    if sparse_v is not None:
        tensors["sparse_v"] = sparse_v
    if local_v is not None:
        tensors["local_v"] = local_v

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    kv_tensors = {"local_kv": local_kv, "sparse_kv": sparse_kv}
    kv_tensors.update(local_v=local_v, sparse_v=sparse_v)
    for name, tensor in kv_tensors.items():
        if tensor is None:
            continue
        if tensor.dtype != query.dtype:
            raise ValueError(
                f"{name} must have the same dtype as query, but got {tensor.dtype} and {query.dtype}."
            )
        if tensor.device != query.device:
            raise ValueError(
                f"{name} must be on the same device as query, "
                f"but got {tensor.device} and {query.device}."
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
    if local_kv is None and sliding_window_size != 0:
        raise ValueError("local_kv=None requires sliding_window_size=0.")

    if sparse_kv.ndim != 4 or sparse_kv.shape[0] != batch or sparse_kv.shape[3] != head_dim:
        raise ValueError(
            "sparse_kv must have shape [batch, kv_heads, sparse_sequence_length, head_dim], "
            f"got {list(sparse_kv.shape)}."
        )
    kv_heads = sparse_kv.shape[1]
    if kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError(
            f"sparse_kv must have a KV head count that divides the {heads} query heads, "
            f"got {kv_heads}."
        )
    if local_kv is not None and (
        local_kv.ndim != 4
        or local_kv.shape[0] != batch
        or local_kv.shape[1] != kv_heads
        or local_kv.shape[2] != sequence_length
        or local_kv.shape[3] != head_dim
    ):
        raise ValueError(
            f"local_kv must have shape [batch, {kv_heads}, sequence_length, head_dim], "
            f"got {list(local_kv.shape)}."
        )
    for name, values, keys, keys_name in (
        ("local_v", local_v, local_kv, "local_kv"),
        ("sparse_v", sparse_v, sparse_kv, "sparse_kv"),
    ):
        if values is not None and values.shape != keys.shape:
            raise ValueError(
                f"{name} must have the same shape as {keys_name}, "
                f"got {list(values.shape)} and {list(keys.shape)}."
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

    validate_cu_seqlens_pair(cu_seqlens, cu_seqlens_k, batch=batch, device=query.device)


def _validate_kernel_options(kernel_options: dict[str, str | bool] | None) -> None:
    options = kernel_options or {}
    if (
        options.keys() - {"backend", "bwd_recompute_p"}
        or options.get("backend", "cute") not in ("cute", "triton")
        or type(options.get("bwd_recompute_p", False)) is not bool
    ):
        raise ValueError(f"unsupported gather_attn kernel options: {kernel_options}")


def _select_backend(
    query: Tensor,
    attention_sink: Tensor | None,
    share_kv: bool,
    *,
    num_keys: int,
    query_rope: Tensor | None = None,
) -> str:
    """Choose a CUDA backend after public input validation; paired metadata is guarded."""
    # Metadata and determinism remain guarded by Dynamo. Only environment capabilities
    # are constant for a compiled graph; tape shapes specialize to the selected backend.
    if torch.are_deterministic_algorithms_enabled() or num_keys == 0:
        return "triton"

    from .impl import cute as cute_backend

    return (
        "cute"
        if cute_backend.is_supported(query, attention_sink, share_kv, query_rope)
        else "triton"
    )


@overload
def gather_attn(
    query: AttentionInput,
    local_kv: AttentionInput | None,
    sparse_kv: AttentionInput,
    kv_indices: Tensor,
    attention_sink: Tensor | None = ...,
    *,
    sliding_window_size: int = ...,
    cu_seqlens: Tensor | None = ...,
    cu_seqlens_k: Tensor | None = ...,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str | bool] | None = None,
    scale: float | None = None,
    return_aux: None = ...,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> Tensor: ...


@overload
def gather_attn(
    query: AttentionInput,
    local_kv: AttentionInput | None,
    sparse_kv: AttentionInput,
    kv_indices: Tensor,
    attention_sink: Tensor | None = ...,
    *,
    sliding_window_size: int = ...,
    cu_seqlens: Tensor | None = ...,
    cu_seqlens_k: Tensor | None = ...,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str | bool] | None = None,
    scale: float | None = None,
    return_aux: AuxRequest,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> tuple[Tensor, GatherAttnAux]: ...


def gather_attn(
    query: AttentionInput,
    local_kv: AttentionInput | None,
    sparse_kv: AttentionInput,
    kv_indices: Tensor,
    attention_sink: Tensor | None = None,
    *,
    sliding_window_size: int = 512,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str | bool] | None = None,
    scale: float | None = None,
    return_aux: AuxRequest | None = None,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> Tensor | tuple[Tensor, GatherAttnAux]:
    """Attend to selected sparse KV and a causal local window with one softmax.

    For query position t, let M_t be the multiset union of local positions
    max(document_start, t - sliding_window_size + 1) <= j <= t and valid sparse
    selections. Every sparse slot contributes separately, including duplicates;
    local and sparse occurrences are not deduplicated. For each occurrence i::

        z_i = scale * (dot(query.latent, kv_i.latent) + dot(query.rope, kv_i.rope))
        Z = sum_{i in M_t} exp(z_i) + exp(s_h)
        output = sum_{i in M_t} exp(z_i) * v_i / Z
        LSE = log(Z)

    Bare tensors omit the RoPE dot product and use the whole KV tensor as both key
    and value (v_i = kv_i) unless local_v/sparse_v supply separate values. Paired
    inputs use only the latent component as value; the caller has already rotated
    the RoPE components. Query head h reads KV head h // (num_heads // kv_heads), so
    kv_heads = 1 is MQA, kv_heads = num_heads is MHA, and other divisors are GQA. The
    optional sink logit s_h = attention_sink[h] is unscaled and contributes only to the
    denominator and LSE, not the numerator.
    Without a sink, s_h = -inf. When Z = 0, output is defined as zero and LSE as
    -inf. A row with no real keys returns zero even when a sink is present.

    Args:
        query: A floating-point tensor shaped (batch_size, num_heads, sequence_length,
            latent_dim), or a (latent, rope) tensor tuple, optionally written as
            RopePair(latent, rope). In paired form the RoPE tensor has shape
            (batch_size, num_heads, sequence_length, rope_dim); rope_dim may be zero.
            All query latent dimensions must be positive. Every component of query
            and the KV pools must share the query latent tensor's dtype and device.

        local_kv: Keys and values for the causal local window, as a bare tensor or
            (latent, rope) pair matching query's form. The latent shape is
            (batch_size, kv_heads, sequence_length, latent_dim); the paired RoPE
            shape has the same leading dimensions and query's rope_dim. kv_heads
            must divide num_heads and match sparse_kv.
            None requires sliding_window_size=0. A supplied pool is still validated
            when the window is zero.

        sparse_kv: Candidate keys and values for the indexing branch, as a bare
            tensor or pair matching query's form. The latent shape is
            (batch_size, kv_heads, sparse_sequence_length, latent_dim); the paired
            RoPE shape has the same leading dimensions and query's rope_dim.
            sparse_sequence_length may be zero. Fused paired inputs with nonzero
            rope_dim require kv_heads=1; fused zero-width pairs also allow
            kv_heads=num_heads, and the reference allows any kv_heads dividing
            num_heads. Backend-specific dtype, shape, and layout limits are
            described in docs/sparse.md.

        kv_indices: int32 or int64 selections on query.device, shaped
            (batch_size, sequence_length, num_topk), shared across heads. Without
            packed offsets these index the batch's sparse_kv pool. With packed
            offsets they are zero-based within the query's document: local index j
            selects sparse_kv at cu_seqlens_k[document] + j. Negative or out-of-pool
            indices are ignored. Repeated indices retain their multiplicity. The
            slot count may be zero or exceed the pool length; -1 pads unused slots.
            Sparse causality is the caller's responsibility; no causal mask is
            applied to sparse selections.

        attention_sink: Optional learnable per-head sink logits, shaped (num_heads,)
            on query.device, using the query dtype or torch.float32. These unscaled
            logits occupy the softmax denominator without supplying values.
            None omits the sink.

        sliding_window_size: Non-negative Python int counting local keys, including
            the current position. Zero disables the local branch. The window is
            clipped at the beginning of the query's document.

        cu_seqlens: Packed offsets shaped [N + 1] for batch-one inputs, as contiguous
            int32 on query.device. They partition queries and every local_kv
            component; the local window never crosses a document boundary. Offsets
            start at zero, never decrease, may repeat for empty sequences, and may
            end before sequence_length. Output values and token-input gradients
            beyond that endpoint are undefined; fixed-capacity callers must mask
            them. Offset values are a caller contract and are not inspected on the
            host. Supply both offset tensors or neither.

        cu_seqlens_k: Packed offsets shaped [N + 1] for every sparse_kv component, as
            contiguous int32 on query.device. They describe the same documents as
            cu_seqlens, start at zero, never decrease, and end at or before the sparse
            pool length. Empty sparse documents have repeated offsets. These
            offsets need not match query offsets: for compressed KV, each document
            contributes only its complete compressed blocks. Without offsets, each
            batch element is a single document.

        impl: Impl.FUSED (default, or "fused") uses optimized CUDA kernels;
            Impl.REFERENCE (or "reference") uses eager PyTorch on CPU or CUDA.

        kernel_options: Fused backend options. "backend": "cute" or "triton"
            overrides the backend; omit it to prefer eligible CuTe calls, otherwise
            Triton. "bwd_recompute_p" (default True): CuTe backward recomputes
            attention probabilities; False saves them in forward instead (BF16,
            tokens x heads x keys, with backend padding). Triton always recomputes
            and ignores this option. Deterministic mode selects Triton automatically,
            but nonempty paired RoPE has no strict deterministic backward there.
            Explicit backend requests are honored; execution failures are never
            retried on another backend. Nonempty options are invalid with
            Impl.REFERENCE. Bare and zero-width paired shared-KV Triton backward, and
            the grouped schedule's sparse dK/dV for sparse selections over small heads,
            accumulate with nondeterministic atomics unless deterministic algorithms
            are enabled.

        scale: Positive multiplier for query-key logits. Bare tensors default to
            1 / sqrt(latent_dim). Paired inputs, including zero-width RoPE pairs,
            require an explicit model scale: absorbing a key projection changes
            the latent width, not the model's attention scale. Sink logits are not
            scaled.

        local_v: Values for the local window, shaped like local_kv, or None to reuse
            local_kv as the values (shared K = V, as in DeepSeek's MLA latent).
            Requires bare tensor inputs; supply it exactly when sparse_v is given
            and local_kv is not None.

        sparse_v: Values for the indexing branch, shaped like sparse_kv, or None to
            reuse sparse_kv as the values. Separate values and 1 < kv_heads < num_heads
            run on Impl.REFERENCE, or on the Triton backend with SM100 or SM103,
            bfloat16 inputs, head_dim <= 256 and divisible by 16, unit inner strides,
            and sliding_window_size <= 2048; other inputs raise. The CuTe backend
            requires shared values and one KV head. Passing the key tensor objects
            themselves as both values (local_v is local_kv and sparse_v is sparse_kv)
            runs them as shared values; other aliases, such as views, count as separate
            values. With kv_heads == num_heads, separate values run one query head per
            grouped-schedule tile, slower than the shared-value MHA kernels. CUDA graph
            capture of their Triton backward is tested only on the atomic path the default
            mode picks for small heads. With sliding_window_size=0, the grouped
            schedule reads every pool row in the selection span of a run of nearby
            queries, selected or not, so a NaN or Inf in an unselected row near selected
            ones can make those queries' outputs and gradients NaN. Keep unused pool
            capacity finite, for example zero-filled.

        return_aux: If None (default), return only the output tensor. If an AuxRequest
            instance, return (output, GatherAttnAux) containing the requested
            auxiliary outputs, such as LSE when return_aux.lse is True.

    Returns:
        If return_aux is None: output tensor with shape
        (batch_size, num_heads, sequence_length, latent_dim), in query's dtype and
        on query.device. In paired form the output contains only latent values.
        If return_aux is an AuxRequest: (output, GatherAttnAux), where aux.lse has
        shape (batch_size, num_heads, sequence_length) when requested, otherwise
        None. LSE includes the unscaled sink; fused backends return nondifferentiable
        float32 LSE.

        A row with no selected or local keys returns zero output, with LSE equal
        to its sink logit, or -inf when there is no sink. Its output-loss gradient
        contributions to all inputs are zero. Inactive packed tails remain undefined.
    """
    selected_impl = resolve_impl(impl)
    if selected_impl is Impl.REFERENCE and kernel_options:
        raise ValueError("kernel_options are not supported with impl='reference'")
    _validate_kernel_options(kernel_options)

    query_latent, query_rope = _split_input(query, "query")
    paired = query_rope is not None
    if local_kv is None:
        local_latent = local_rope = None
    else:
        local_latent, local_rope = _split_input(local_kv, "local_kv")
        if (local_rope is not None) != paired:
            raise ValueError("local_kv must use the same bare/paired form as query.")
    if sparse_kv is None:
        raise TypeError("sparse_kv must be a Tensor or tensor pair.")
    sparse_latent, sparse_rope = _split_input(sparse_kv, "sparse_kv")
    if (sparse_rope is not None) != paired:
        raise ValueError("sparse_kv must use the same bare/paired form as query.")
    if paired and sparse_v is not None:
        raise ValueError("local_v and sparse_v require bare tensor inputs.")
    _validate_inputs(
        query_latent,
        local_latent,
        sparse_latent,
        kv_indices,
        attention_sink,
        sliding_window_size,
        cu_seqlens,
        cu_seqlens_k,
        local_v,
        sparse_v,
    )
    kv_heads = sparse_latent.shape[1]
    share_kv = kv_heads == 1
    if paired:
        for name, latent, rope in (
            ("query", query_latent, query_rope),
            ("sparse_kv", sparse_latent, sparse_rope),
            ("local_kv", local_latent, local_rope),
        ):
            if latent is None:
                continue
            if rope.ndim != 4 or rope.shape[:-1] != latent.shape[:-1]:
                raise ValueError(f"{name} rope must match its latent leading dimensions.")
            if rope.shape[-1] != query_rope.shape[-1]:
                raise ValueError(f"{name} rope width must match query rope width.")
            if rope.dtype != query_latent.dtype or rope.device != query_latent.device:
                raise ValueError(f"{name} rope must match query dtype and device.")
        if scale is None:
            raise ValueError("Paired attention inputs require an explicit model scale.")
    if sparse_v is sparse_kv and local_v is local_kv:
        # Values that are the keys themselves keep the shared-value kernels.
        local_v = sparse_v = None
    if scale is not None and not scale > 0:
        raise ValueError("scale must be greater than 0.")
    scale = query_latent.shape[-1] ** -0.5 if scale is None else scale

    options = kernel_options or {}
    if selected_impl is Impl.REFERENCE:
        from .impl import reference

        if attention_sink is None:
            attention_sink = torch.full(
                (query_latent.shape[1],),
                float("-inf"),
                dtype=query_latent.dtype,
                device=query_latent.device,
            )
        output, lse = reference.gather_attn(
            query_latent,
            local_latent,
            sparse_latent,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            share_kv,
            scale=scale,
            query_rope=query_rope,
            sparse_rope=sparse_rope,
            local_rope=local_rope,
            local_v=local_v,
            sparse_v=sparse_v,
        )
    else:
        if query_latent.device.type != "cuda":
            raise ValueError("fused gather_attn requires CUDA tensors; use impl='reference'")
        if paired and query_rope.shape[-1] > 0 and not share_kv:
            raise ValueError("Fused paired attention requires a single shared KV head.")
        backend = options.get("backend")
        grouped = uses_grouped_schedule(query_latent.shape[1], kv_heads, sparse_v is not None)
        if grouped and backend == "cute":
            raise ValueError(
                "the CuTe gather_attn backend requires keys that are also the values and "
                "one KV head; use kernel_options={'backend': 'triton'} or impl='reference'"
            )
        if grouped:
            kv_tensors = (sparse_latent, local_latent, sparse_v, local_v)
            require_grouped_schedule(
                query_latent,
                [tensor for tensor in kv_tensors if tensor is not None],
                sliding_window_size=sliding_window_size,
            )
        if backend is None:
            backend = (
                "triton"
                if grouped
                else _select_backend(
                    query_latent,
                    attention_sink,
                    share_kv,
                    num_keys=sliding_window_size + kv_indices.shape[-1],
                    query_rope=query_rope,
                )
            )
        output, lse = fused_gather_attn(
            query_latent,
            local_latent,
            sparse_latent,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            share_kv,
            scale=scale,
            query_rope=query_rope,
            sparse_rope=sparse_rope,
            local_rope=local_rope,
            backend=backend,
            bwd_recompute_p=options.get("bwd_recompute_p", True),
            local_v=local_v,
            sparse_v=sparse_v,
        )

    if return_aux is None:
        return output

    return output, GatherAttnAux(lse=lse if return_aux.lse else None)
