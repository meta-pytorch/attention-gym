"""Public weighted-ReLU Top-K selection and backend-independent validation."""

import torch
from torch import Tensor

from .ops import _indexer_op


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
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
    if candidates <= 0:
        raise ValueError("k candidate length must be positive.")

    # --- shape agreement ---
    if k.shape[0] != batch or k.shape[2] != head_dim:
        raise ValueError(f"k must have shape [B={batch}, S, D={head_dim}], got {list(k.shape)}.")
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {[batch, queries, heads]}, got {list(weights.shape)}."
        )

    # --- dtype ---
    if not q.is_floating_point():
        raise TypeError(f"q must have a floating-point dtype, got {q.dtype}.")
    if k.dtype != q.dtype:
        raise ValueError(f"k must have the same dtype as q, but got {k.dtype} and {q.dtype}.")
    if weights.dtype != q.dtype:
        raise ValueError(
            f"weights must have the same dtype as q, but got {weights.dtype} and {q.dtype}."
        )

    # --- device ---
    if k.device != q.device:
        raise ValueError(f"k must be on the same device as q, but got {k.device} and {q.device}.")
    if weights.device != q.device:
        raise ValueError(
            f"weights must be on the same device as q, but got {weights.device} and {q.device}."
        )

    # --- topk range ---
    if topk < 0 or topk > candidates:
        raise ValueError(f"topk must be in [0, {candidates}], got {topk}.")

    # --- candidate count: one key per completed window of compress_ratio tokens ---
    if candidates != queries // compress_ratio:
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
    impl: str = "fused",
    kernel_options: dict[str, str] | None = None,
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

        k: Key candidate pool shared across heads, [B, S, D], with
            ``S = T // compress_ratio``.

        weights: Per-head weights, [B, T, H]. May be negative.

        topk: Number of candidates to select per query.  Must be in [0, S].

        causal: If True, query t can only select candidates
            ``s < (t + 1) // compress_ratio``, i.e. those whose covered tokens all
            lie at positions <= t (candidates ``0..t`` when ``compress_ratio=1``).

        compress_ratio: Number of consecutive tokens summarized by each
            candidate, as in DeepSeek compressed sparse attention. A trailing
            partial window forms no candidate, so ``S = T // compress_ratio``.
            Values other than 1 require ``causal=True``.

        impl: ``"reference"`` uses eager PyTorch on CPU or CUDA; ``"fused"`` uses
            optimized CUDA kernels. Defaults to ``"fused"``.

        kernel_options: Fused backend override: ``{"backend": "cute"}`` or
            ``{"backend": "triton"}``. Omit options to select CuTe
            on SM100 and Triton on other Hopper-or-newer NVIDIA GPUs. There is
            no fallback when the selected backend rejects a shape or layout.

    Returns:
        Contiguous [B, T, topk] INT32 indices. Order and tie-breaking are not
        guaranteed, including across repeated calls. Causal rows with fewer than topk candidates
        contain -1 padding; topk=0 returns an empty last dimension.

    Fused backends support ``torch.compile(fullgraph=True)`` and CUDA Graph replay.
    Both require FP16/BF16 inputs and T <= 2**20 and accept any topk <= S. CuTe requires
    SM100, even H, D divisible by 16, and Q/K with unit last strides and 16-byte-aligned
    bases and non-singleton outer strides. Other Q/K strides may vary independently;
    weights may have arbitrary strides and need only element alignment.
    Triton requires SM90 or newer, H <= 256, D <= 256 divisible by 8, and Q/K with
    unit last strides and 16-byte-aligned bases and outer strides; weights may be strided.
    Its register-resident selection makes per-tile cost grow with topk.
    CuTe reuses a per-call FP32 score workspace capped at 32 MiB and 1024 query
    rows. Large inputs use slabs rather than an unbounded quadratic score allocation.
    This workspace is additional to the returned indices and is not shared across calls.

    Selection with NaN/Inf scores is unspecified and may differ across backends.
    Indices are nondifferentiable even when inputs require gradients. Training
    attention over the selected positions does not propagate gradients through
    selection into q, k, or weights.
    """

    _validate_inputs(q, k, weights, topk, causal, compress_ratio)

    match impl:
        case "reference":
            if kernel_options:
                raise ValueError("kernel_options are not supported with impl='reference'")
            from .impl import reference

            return reference.launch(q, k, weights, topk, causal, compress_ratio)
        case "fused":
            if kernel_options not in (
                None,
                {},
                {"backend": "cute"},
                {"backend": "triton"},
            ):
                raise ValueError(f"unsupported lightning_indexer kernel options: {kernel_options}")
            if not q.is_cuda:
                raise ValueError("the fused lightning_indexer requires CUDA tensors")
            backend = (kernel_options or {}).get("backend", "auto")
            return _indexer_op(q, k, weights, topk, causal, compress_ratio, backend)
        case _:
            raise ValueError(f"unknown impl {impl!r}; expected 'reference' or 'fused'")
