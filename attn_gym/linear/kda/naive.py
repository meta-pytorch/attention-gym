"""KDA (Kimi Delta Attention) delta-rule ops."""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference O(T) KDA delta rule. State S in [K, V], per (batch, head).

    Per step t (per-channel decay a_t = 2^{g_t} in R^K):
        S <- diag(a_t) S ;  delta = beta_t * (v_t - k^T S) ;  S <- S + outer(k_t, delta) ;  o_t = q^T S

    Shapes: q, k, g (B, T, H, K); v (B, T, H, V); beta (B, T, H); state (B, H, K, V).

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        g: per-channel log2-decay, a_t = 2^{g_t} in R^K (the KDA diagonal gate)
        beta: scalar-per-head delta step size / write gate
        scale: query scale for q k^T (optional; default 1/sqrt(K))
        initial_state: initial recurrent state (B, H, K, V) (optional)
        output_final_state: also return the final state (optional)
    """
    b, t, h, k_dim = q.shape
    q = q * scale if scale else q * k_dim**-0.5
    state = initial_state if initial_state is not None else q.new_zeros(b, h, k_dim, v.shape[-1])
    outputs = []

    for i in range(t):
        state = state * g[:, i].exp2()[..., None]  # per-channel decay: diag(2^{g_t}) over V
        delta = v[:, i] - torch.einsum("bhk,bhkv->bhv", k[:, i], state)  # vt - k^T S (v_old)
        delta = delta * beta[:, i][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, i], delta)  # + outer(k, delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state))  # q^T S
    return torch.stack(outputs, dim=1), (state if output_final_state else None)


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive chunk-parallel KDA delta rule (training / prefill).

    Same delta rule as :func:`naive_recurrent_kda`, computed chunk-at-a-time: the intra-chunk
    solve builds ``(I + Akk)^-1`` and the inter-chunk scan carries the state S across chunks.
    Per-channel decay is folded into q/k as ``2^{+/-decay}`` so the intra-chunk sums stay plain
    matmuls.
    """
    b, t, h, k_dim = q.shape
    v_dim = v.shape[-1]
    orig_dtype = q.dtype
    if scale is None:
        scale = k_dim**-0.5

    # -> (B, H, T, .) in fp32
    q, k, v, g = (x.transpose(1, 2).float() for x in (q, k, v, g))
    beta = beta.transpose(1, 2).float()
    pad = (chunk_size - t % chunk_size) % chunk_size
    if pad:
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    length = q.shape[-2]
    num_chunks = length // chunk_size

    def to_chunks(x):
        return x.reshape(b, h, num_chunks, chunk_size, x.shape[-1])

    q, k, v, g = (to_chunks(x) for x in (q, k, v, g))
    beta = beta.reshape(b, h, num_chunks, chunk_size)
    decay = g.cumsum(-2)  # cumulative per-channel log2-decay within each chunk

    # fold per-channel decay into keys/queries so intra-chunk sums are plain matmuls
    q_g = q * scale * decay.exp2()  # gated (scaled) queries, 2^{+decay}
    k_gi = k * (-decay).exp2()  # gated keys, 2^{-decay}
    kb_g = k * beta[..., None] * decay.exp2()  # k * beta * 2^{+decay}

    diag_incl = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    diag_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )

    # strictly-lower Akk, then (I + Akk)^-1 via triangular solve
    raw = (kb_g @ k_gi.transpose(-1, -2)).masked_fill(diag_incl, 0)
    eye = torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = torch.linalg.solve_triangular(eye + raw, eye.expand_as(raw), upper=False)

    u = attn @ (v * beta[..., None])  # "pseudo-values" per chunk
    w = attn @ kb_g  # gated key basis for the carried-state correction

    state = q.new_zeros(b, h, k_dim, v_dim) if initial_state is None else initial_state.float()
    o = torch.zeros_like(u)
    for i in range(num_chunks):
        q_i = q_g[:, :, i]
        attn_i = (q_i @ k_gi[:, :, i].transpose(-1, -2)).masked_fill(diag_strict, 0)
        u_eff = u[:, :, i] - w[:, :, i] @ state  # correction against carried state
        o[:, :, i] = q_i @ state + attn_i @ u_eff  # inter-chunk read + intra-chunk read
        # carry state to next chunk: decayed old state + this chunk's writes
        d_last = decay[:, :, i, -1]  # total per-channel decay over the chunk
        kg = k[:, :, i] * (d_last[:, :, None] - decay[:, :, i]).exp2()
        state = state * d_last[..., None].exp2() + kg.transpose(-1, -2) @ u_eff

    o = o.reshape(b, h, length, v_dim)[:, :, :t].transpose(1, 2).to(orig_dtype)
    return o, (state.to(orig_dtype) if output_final_state else None)


def l2norm_fwd_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization over the last dim: ``y = x / sqrt(sum(x^2) + eps)``.

    Naive counterpart of ``l2norm_fwd_kernel`` / ``l2norm_fwd_kernel1``. The reduction runs
    in fp32 (or fp64 when ``x`` is already fp64) to mirror the kernels' fp32 accumulation,
    then downcasts back to ``x.dtype``. Pass an fp64 ``x`` for a golden value or a
    low-precision ``x`` to mimic the kernel.

    Args:
        x: input tensor, normalized over the last dim; shape ``(..., D)``.
        eps: variance floor added before the reciprocal sqrt.
    """
    dtype = torch.promote_types(x.dtype, torch.float32)
    xc = x.to(dtype)
    rstd = torch.rsqrt((xc * xc).sum(-1, keepdim=True) + eps)
    return (xc * rstd).to(x.dtype)


def l2norm_bwd_ref(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Gradient of the L2 norm: ``dx = rstd * (dy - y * <dy, y>)``.

    Naive counterpart of ``l2norm_bwd_kernel`` / ``l2norm_bwd_kernel1``. Runs in fp32 (or
    fp64 when ``y`` is fp64) to match the kernels' accumulation, then downcasts to ``y.dtype``.

    Args:
        y: normalized forward output; shape ``(..., D)``.
        rstd: reciprocal std saved by the forward pass, broadcastable over the last dim.
        dy: upstream gradient w.r.t. ``y``; shape ``(..., D)``.
    """
    dtype = torch.promote_types(y.dtype, torch.float32)
    yc, dyc, rc = y.to(dtype), dy.to(dtype), rstd.to(dtype)
    dot = (dyc * yc).sum(-1, keepdim=True)
    return (rc * (dyc - yc * dot)).to(y.dtype)


def chunk_cumsum_ref(
    x: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunk-local cumsum over the time axis (dim=1), optionally per document.

    Naive counterpart of ``chunk_local_cumsum_scalar_kernel`` /
    ``chunk_local_cumsum_vector_kernel``. The cumulative sum restarts at every
    ``chunk_size`` boundary (and at every document boundary, in varlen mode).

    Args:
        x: input with time on dim=1, e.g. ``(B, T, H)`` (scalar) or ``(B, T, H, S)``
            (vector); the cumsum runs over the time axis.
        chunk_size: length of each chunk; the running sum resets at each multiple.
        reverse: if True, accumulate from the end of each chunk toward the start.
        scale: optional multiplier applied to the result.
        cu_seqlens: optional int32 offsets (varlen mode); the sum also resets at each
            document boundary ``x[:, cu_seqlens[i]:cu_seqlens[i + 1]]``.
    """
    if cu_seqlens is None:
        spans = [x]
    else:
        offs = cu_seqlens.tolist()
        spans = [x[:, bos:eos] for bos, eos in pairwise(offs)]

    chunks = [c for span in spans for c in span.split(chunk_size, dim=1)]
    out = torch.cat(
        [c.flip(1).cumsum(1).flip(1) if reverse else c.cumsum(1) for c in chunks], dim=1
    )
    return out * scale if scale is not None else out


def _gate_map(z: torch.Tensor, A: torch.Tensor, lower_bound: float | None) -> torch.Tensor:
    """KDA gate ``-A*softplus(z)`` or lower-bounded ``lb*sigmoid(A*z)``; ``A = exp(A_log)``."""
    if lower_bound is None:
        return -A * F.softplus(z)
    return lower_bound * torch.sigmoid(A * z)


def gate_fwd_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    scale: float | None,
    reverse: bool,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Gate map, then chunk-local cumsum.

    Naive counterpart of ``kda_gate_chunk_cumsum_vector_kernel`` (fused gate + chunk cumsum).
    The gate map runs in fp32 (or fp64 when ``g`` is fp64) to match the kernel's fp32 math.

    Args:
        g: raw gate input; shape ``(B, T, H, S)``.
        A_log: per-head log-magnitude, ``A = exp(A_log)``; shape ``(H,)``.
        dt_bias: optional per-(head, channel) bias added to ``g``; shape ``(H, S)``.
        lower_bound: if None, gate is ``-A*softplus(g)``; else ``lower_bound*sigmoid(A*g)``.
        scale: optional multiplier applied to the cumsum output.
        reverse: reverse the chunk-local cumsum.
        chunk_size: cumsum chunk length.
        cu_seqlens: optional int32 varlen document offsets.
    """
    dtype = torch.promote_types(g.dtype, torch.float32)
    z = g.to(dtype)
    if dt_bias is not None:
        z = z + dt_bias.to(dtype)  # (H, S) broadcasts over (B, T, H, S)
    gate = _gate_map(z, A_log.to(dtype).exp().view(1, 1, -1, 1), lower_bound)
    return chunk_cumsum_ref(gate, chunk_size, reverse, scale, cu_seqlens)


def fused_gate_bwd_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    lower_bound: float,
    scale: float,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the bounded gate map and its chunk-local prefix sum.

    This is the reference for the fused KDA backward helper. The adjoint of a
    forward prefix sum is a reverse prefix sum, so ``d_cumulative`` is scanned
    in reverse before applying the pointwise gate derivative.
    """
    d_gate = chunk_cumsum_ref(
        d_cumulative,
        chunk_size,
        reverse=True,
        scale=scale,
    )
    dg, dA_log, d_bias = gate_bwd_ref(g, A_log, dt_bias, d_gate, lower_bound)
    assert d_bias is not None
    return dg, dA_log, d_bias


def gate_bwd_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dyg: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Gradients of the pointwise gate map (no cumsum) w.r.t. ``g``, ``A_log``, ``dt_bias``.

    Naive counterpart of ``kda_gate_bwd_kernel``, computed by autograd through the forward
    map in fp32 (or fp64 when ``g`` is fp64) to match the kernel's fp32 math.

    Args:
        g: raw gate input; shape ``(B, T, H, S)``.
        A_log: per-head log-magnitude, ``A = exp(A_log)``; shape ``(H,)``.
        dt_bias: optional per-(head, channel) bias added to ``g``; shape ``(H, S)``.
        dyg: upstream gradient w.r.t. the gate map output; shape ``(B, T, H, S)``.
        lower_bound: selects the gate variant (see :func:`gate_fwd_ref`).

    Returns:
        Gradients ``(dg, dA_log, dt_bias)``; the last is None when ``dt_bias`` is None.
    """
    dtype = torch.promote_types(g.dtype, torch.float32)
    gg = g.to(dtype).detach().requires_grad_()
    aa = A_log.to(dtype).detach().requires_grad_()
    bb = dt_bias.to(dtype).detach().requires_grad_() if dt_bias is not None else None
    z = gg if bb is None else gg + bb
    _gate_map(z, aa.exp().view(1, 1, -1, 1), lower_bound).backward(dyg.to(dtype))
    return gg.grad, aa.grad, (bb.grad if bb is not None else None)


__all__ = [
    "chunk_cumsum_ref",
    "fused_gate_bwd_ref",
    "gate_bwd_ref",
    "gate_fwd_ref",
    "l2norm_bwd_ref",
    "l2norm_fwd_ref",
    "naive_chunk_kda",
    "naive_recurrent_kda",
]
