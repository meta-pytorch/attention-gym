"""Public weighted-ReLU Top-K selection and backend-independent validation."""

import torch
from torch import Tensor

from attn_gym._cu_seqlens import validate_cu_seqlens_pair
from attn_gym.types import Impl, resolve_impl

from .ops import _indexer_op
from .validation import validate_precision


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    q_scale: Tensor | None,
    k_scale: Tensor | None,
) -> None:
    """Validate metadata without synchronizing or inspecting tensor values."""
    # --- type checks ---
    for name, tensor in {"q": q, "k": k, "weights": weights}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"topk must be a Python int, got {type(topk).__name__}.")

    if not isinstance(causal, bool):
        raise TypeError(f"causal must be a bool, got {type(causal).__name__}.")

    if not isinstance(compress_ratio, int) or isinstance(compress_ratio, bool):
        raise TypeError(
            f"compress_ratio must be a Python int, got {type(compress_ratio).__name__}."
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}.")
    if compress_ratio != 1 and not causal:
        raise ValueError("compress_ratio != 1 requires causal selection; pass causal=True.")

    # --- ndim ---
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, D], got {list(q.shape)}.")
    if k.ndim != 3:
        raise ValueError(f"k must have shape [B, S, D], got {list(k.shape)}.")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B, T, H], got {list(weights.shape)}.")

    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]

    # --- positive dimensions ---
    if min(batch, queries, heads, head_dim) <= 0:
        raise ValueError("All q dimensions must be positive.")

    # --- shape agreement ---
    if k.shape[0] != batch or k.shape[2] != head_dim:
        raise ValueError(f"k must have shape [B={batch}, S, D={head_dim}], got {list(k.shape)}.")
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {[batch, queries, heads]}, got {list(weights.shape)}."
        )

    # --- dtype and optional FP8 scales ---
    validate_precision(q, k, weights, q_scale, k_scale)

    # --- device ---
    if k.device != q.device:
        raise ValueError(f"k must be on the same device as q, but got {k.device} and {q.device}.")
    if weights.device != q.device:
        raise ValueError(
            f"weights must be on the same device as q, but got {weights.device} and {q.device}."
        )

    # --- topk range ---
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}.")

    if q.dtype == torch.float8_e4m3fn and (cu_seqlens is not None or cu_seqlens_k is not None):
        raise ValueError("MXFP8 indexing does not support packed cu_seqlens.")
    validate_cu_seqlens_pair(cu_seqlens, cu_seqlens_k, batch=batch, device=q.device)

    # Packed pools contain the sum of independently floored document lengths.
    # Their device-resident offsets and per-document counts are caller invariants.
    if cu_seqlens is None and candidates != queries // compress_ratio:
        raise ValueError(
            f"k must hold S = T // compress_ratio = {queries // compress_ratio} candidates, "
            f"got S={candidates} (T={queries}, compress_ratio={compress_ratio})."
        )


def lightning_indexer(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    *,
    causal: bool = False,
    compress_ratio: int = 1,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
    kernel_options: dict[str, str] | None = None,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> Tensor:
    """Return the Top-K candidate indices for every (batch, query) row.

    Computes a multi-head weighted ReLU score for each query-candidate pair
    and selects the topk highest-scoring candidates per query::

        dots[b, t, h, s] = q[b, t, h, :] · k[b, s, :]
        score[b, t, s]   = sum_h(w[b, t, h] * relu(dots[b, t, h, s]))
                           / sqrt(H * D)
        output[b, t, :]  = topk(score[b, t, :]).indices

    Args:
        q: Query tensor, [B, T, H, D].

        k: Key candidate pool shared across heads, [B, S, D]. Without packed
            offsets, ``S = T // compress_ratio``. With offsets, candidates from each
            document are concatenated in document order; S may include inactive capacity.

        weights: Per-head weights, [B, T, H]. May be negative. MXFP8 Q/K accept
            BF16 or FP32 weights; other inputs require matching dtypes.

        topk: Non-negative output width. Rows with fewer valid candidates are
            padded with -1, including when S is zero or topk exceeds S.

        causal: If True, query t can only select candidates
            ``s < (t + 1) // compress_ratio``, i.e. those whose covered tokens all
            lie at positions <= t (candidates ``0..t`` when ``compress_ratio=1``).

        compress_ratio: Number of consecutive tokens summarized by each
            candidate, as in DeepSeek compressed sparse attention. A trailing
            partial window forms no candidate. In packed mode grouping and the
            causal query position restart at each document boundary: document d has
            ``(cu_seqlens[d + 1] - cu_seqlens[d]) // compress_ratio`` candidates.
            Values other than 1 require ``causal=True``.

        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences, and may end before ``T``.
            Query rows beyond the endpoint return only -1. This API processes whole
            documents, not query chunks with cached history. Must be supplied together
            with cu_seqlens_k. Offset values are caller invariants, not host-validated.

        cu_seqlens_k: Packed candidate offsets shaped ``[N + 1]``, contiguous
            ``int32`` on ``q.device``. They start at zero, never decrease, and end
            at or before S. Each document's span must equal its query length divided
            by compress_ratio, rounded down. Returned indices are zero-based within
            that document's candidate pool, suitable for gather_attn with these offsets.
            Without packed offsets, indices are positions in the batch element's pool.

        q_scale: E8M0 (``torch.float8_e8m0fnu``) [B, T, H, D // 32] scales for E4M3FN
            Q; each multiplies one contiguous 32-element group of D inside the dot
            product (FP32 accumulation). Required together with k_scale for MXFP8,
            otherwise None. Values are not checked on device; non-finite scales or
            scores are outside the contract. This function does not quantize.

        k_scale: E8M0 [B, S, D // 32] scales for K; same semantics as q_scale.

        impl: Impl.REFERENCE (or ``"reference"``) uses eager PyTorch on CPU or CUDA;
            Impl.FUSED (default, or ``"fused"``) uses optimized CUDA kernels.

        kernel_options: Fused backend override: ``{"backend": "cute"}`` or
            ``{"backend": "triton"}``. Omit options to select CuTe
            on SM100/SM103 and Triton on other Hopper-or-newer NVIDIA GPUs. There is
            no fallback when the selected backend rejects a shape or layout.
            ``"selector"`` chooses the exact Top-K algorithm: ``"auto"`` (the value when
            omitted) lets CuTe use GVR2 when ``topk < 2048`` and radix otherwise,
            and keeps Triton's streaming Top-K; ``"default"`` forces CuTe radix /
            Triton streaming; ``"gvr2"`` forces GVR2 on either backend. CuTe auto
            uses radix for deterministic or packed calls; explicit GVR2 rejects
            deterministic CuTe calls and packed offsets on either backend.

    Returns:
        Contiguous [B, T, topk] INT32 indices. Order and tie-breaking are unspecified, and
        CuTe's may differ across repeated calls. Under
        ``torch.use_deterministic_algorithms(True)`` each fused backend is repeatable for
        identical inputs (results may differ between backends), and CuTe returns valid
        indices ascending, resolving equal ordered-FP32 scores toward lower indices.
        Rows with fewer than topk valid candidates contain -1 padding after the valid
        indices; topk=0 returns an empty last dimension. Packed selection excludes other
        documents before top-k, not by filtering its result.

    Fused backends support ``torch.compile(fullgraph=True)`` and CUDA Graph replay.
    Both support FP16/BF16 with T <= 2**20, and MXFP8 (E4M3FN Q/K with E8M0 scales)
    for H in {32, 64}, D=128 on SM100/SM103. MXFP8 rejects packed cu_seqlens;
    deterministic mode changes only selection, not scoring. CuTe requires
    SM100/SM103, even H, D divisible by 16, and Q/K with unit last strides and 16-byte-aligned
    bases and non-singleton outer strides. Other Q/K strides may vary independently;
    weights may have arbitrary strides and need only element alignment.
    Triton requires SM90 or newer, H <= 256 and D <= 256 divisible by 8. Its streaming
    FP16/BF16 path requires Q/K unit last strides and 16-byte-aligned bases and outer strides;
    weights may be strided, and its register-resident selection cost grows with topk.
    Triton GVR2 and MXFP8 accept arbitrary nonnegative input strides; both backends accept
    arbitrary nonnegative scale strides. Triton GVR2 uses its own tiled scorer, so its
    FP16/BF16 scores may round differently from the streaming path.
    CuTe and Triton MXFP8/GVR2 reuse a per-call FP32 score workspace capped at 32 MiB and 1024 query
    rows. For nonzero topk, candidate capacity S must be at most 2**22 so one query
    pair fits that budget. Large inputs use slabs rather than an unbounded quadratic
    score allocation.
    This workspace is additional to the returned indices and is not shared across calls.
    Packed CuTe calls also prepare an INT32 [T, 2] bounds buffer, reused by scoring and
    top-k to restrict work to each document. Triton computes those bounds in its kernel
    prologue without a separate buffer.

    Selection with NaN/Inf scores is unspecified and may differ across backends.
    Indices are nondifferentiable even when inputs require gradients. Training
    attention over the selected positions does not propagate gradients through
    selection into q, k, weights, or scales.
    """

    selected_impl = resolve_impl(impl)
    _validate_inputs(
        q, k, weights, topk, causal, compress_ratio, cu_seqlens, cu_seqlens_k, q_scale, k_scale
    )
    match selected_impl:
        case Impl.REFERENCE:
            if kernel_options:
                raise ValueError("kernel_options are not supported with impl='reference'")
            from .impl import reference

            return reference.launch(
                q,
                k,
                weights,
                topk,
                causal,
                compress_ratio,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                q_scale=q_scale,
                k_scale=k_scale,
            )
        case Impl.FUSED:
            options = {} if kernel_options is None else kernel_options
            if (
                not isinstance(options, dict)
                or options.keys() - {"backend", "selector"}
                or ("backend" in options and options["backend"] not in ("cute", "triton"))
            ):
                raise ValueError(f"unsupported lightning_indexer kernel options: {kernel_options}")
            selector = options.get("selector", "auto")
            if selector not in ("auto", "default", "gvr2"):
                raise ValueError(f"unsupported lightning_indexer kernel options: {kernel_options}")
            if not q.is_cuda:
                raise ValueError("the fused lightning_indexer requires CUDA tensors")
            return _indexer_op(
                q,
                k,
                weights,
                topk,
                causal,
                compress_ratio,
                options.get("backend", "auto"),
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                q_scale=q_scale,
                k_scale=k_scale,
                selector=selector,
            )
