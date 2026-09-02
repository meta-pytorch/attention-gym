import torch
from torch import Tensor


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
) -> None:
    # --- type checks ---
    for name, tensor in {"q": q, "k": k, "weights": weights}.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"topk must be a Python int, got {type(topk).__name__}.")

    if not isinstance(causal, bool):
        raise TypeError(f"causal must be a bool, got {type(causal).__name__}.")

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
        raise ValueError(
            f"k must have shape [B={batch}, S, D={head_dim}], got {list(k.shape)}."
        )
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {[batch, queries, heads]}, got {list(weights.shape)}."
        )

    # --- dtype ---
    if not q.is_floating_point():
        raise TypeError(f"q must have a floating-point dtype, got {q.dtype}.")
    if k.dtype != q.dtype:
        raise ValueError(
            f"k must have the same dtype as q, but got {k.dtype} and {q.dtype}."
        )
    if weights.dtype != q.dtype:
        raise ValueError(
            f"weights must have the same dtype as q, but got {weights.dtype} and {q.dtype}."
        )

    # --- device ---
    if k.device != q.device:
        raise ValueError(
            f"k must be on the same device as q, but got {k.device} and {q.device}."
        )
    if weights.device != q.device:
        raise ValueError(
            f"weights must be on the same device as q, "
            f"but got {weights.device} and {q.device}."
        )

    # --- topk range ---
    if topk < 0 or topk > candidates:
        raise ValueError(f"topk must be in [0, {candidates}], got {topk}.")

    # --- square constraint (nonsquare not yet supported) ---
    if candidates != queries:
        raise NotImplementedError(
            f"Nonsquare inputs are not supported yet (T={queries}, S={candidates})."
        )


def index(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool = False,
    backend: str = "eager",
    mode: str = "auto",
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

        k: Key candidate pool shared across heads, [B, S, D].
            S may differ from T (nonsquare) unless causal=True.

        weights: Per-head weights, [B, T, H].  May be negative.

        topk: Number of candidates to select per query.  Must be in [0, S].

        causal: If True, query at position t can only attend to candidates
            at positions <= t.  Requires S == T.

        backend: One of "eager", "triton", or "cute".

        mode: Currently only prefill is supported; auto defaults to prefill.

    Returns:
        [B, T, topk] INT32 tensor of selected candidate indices.
    """
    if not torch.compiler.is_compiling():
        _validate_inputs(q, k, weights, topk, causal)

    if topk == 0:
        return torch.empty((*q.shape[:2], 0), dtype=torch.int32, device=q.device)

    match mode:
        case "auto":
            pass  # auto currently dispatches to prefill
        case "prefill":
            pass
        case _:
            raise NotImplementedError(f"Mode {mode!r} is not supported.")

    match backend:
        case "eager":
            from .impl import reference

            return reference.index(q, k, weights, topk, causal)
        case "triton":
            raise NotImplementedError("Triton backend is not implemented yet.")
        case "cute":
            from .impl import cute as cute_backend

            return cute_backend.index(q, k, weights, topk, causal)
        case _:
            raise NotImplementedError(f"Backend {backend!r} is not supported.")
