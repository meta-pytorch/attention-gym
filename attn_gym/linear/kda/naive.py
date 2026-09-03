"""KDA (Kimi Delta Attention) delta-rule ops."""

from __future__ import annotations

from functools import reduce
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference O(T) KDA delta rule with V-major state storage.

    Per step t (per-channel decay a_t = 2^{g_t} in R^K):
        S <- S diag(a_t) ; delta = beta_t * (v_t - S k) ; S <- S + outer(delta, k_t) ; o_t = S q

    Shapes: q, k, g (B, T, H, K); v (B, T, H, V); beta (B, T, H); state (B, H, V, K),
    or (N, H, V, K) over the N documents of a packed row in varlen mode.

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        g: per-channel log2-decay, a_t = 2^{g_t} in R^K (the KDA diagonal gate)
        beta: scalar-per-head delta step size / write gate
        scale: query scale for q k^T (optional; default 1/sqrt(K))
        initial_state: initial recurrent state (B, H, V, K) (optional)
        output_final_state: also return the final state (optional; in the compute dtype)
        cu_seqlens: optional int32 offsets (varlen mode) with the public packed
            contract: the recurrence restarts at each document boundary, empty
            documents pass their state through, and output rows past the
            terminal offset stay zero.
    """
    b, t, h, k_dim = q.shape

    if cu_seqlens is not None:
        if b != 1:
            raise ValueError(f"varlen mode packs documents into one row, got batch {b}")
        offsets = cu_seqlens.tolist()
        num_documents = len(offsets) - 1
        compute_dtype = torch.promote_types(q.dtype, torch.float32)
        output = torch.zeros(1, t, h, v.shape[-1], dtype=q.dtype, device=q.device)
        final_state = (
            initial_state.to(compute_dtype).clone()
            if initial_state is not None
            else torch.zeros(
                num_documents, h, v.shape[-1], k_dim, dtype=compute_dtype, device=q.device
            )
        )
        for doc, (bos, eos) in enumerate(pairwise(offsets)):
            if bos == eos:
                continue
            doc_output, doc_state = naive_recurrent_kda(
                q[:, bos:eos],
                k[:, bos:eos],
                v[:, bos:eos],
                g[:, bos:eos],
                beta[:, bos:eos],
                scale,
                initial_state[doc : doc + 1] if initial_state is not None else None,
                output_final_state=True,
            )
            output[:, bos:eos] = doc_output
            final_state[doc] = doc_state[0]
        return output, (final_state if output_final_state else None)

    output_dtype = q.dtype
    optional = () if initial_state is None else (initial_state,)
    dtype = reduce(torch.promote_types, (x.dtype for x in (k, v, g, beta, *optional)), q.dtype)
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.to(dtype)

    q = q * scale if scale is not None else q * k_dim**-0.5
    state = initial_state if initial_state is not None else q.new_zeros(b, h, v.shape[-1], k_dim)
    outputs = []

    for i in range(t):
        state = state * g[:, i].exp2()[..., None, :]
        delta = v[:, i] - torch.einsum("bhvk,bhk->bhv", state, k[:, i])
        delta = delta * beta[:, i][..., None]
        state = state + torch.einsum("bhv,bhk->bhvk", delta, k[:, i])
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, i]))
    return torch.stack(outputs, dim=1).to(output_dtype), (state if output_final_state else None)


def _naive_chunk_kda_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    strict_upper: torch.Tensor,
    eye: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one stable KDA chunk and update its recurrent state."""
    relative_decay = cumulative_g[..., :, None, :] - cumulative_g[..., None, :, :]
    relative_decay.masked_fill_(strict_upper[..., None], 0).exp2_()
    key_attention = torch.einsum(
        "...ik,...jk,...ijk->...ij",
        k * beta[..., None],
        k,
        relative_decay,
    )
    query_attention = torch.einsum(
        "...ik,...jk,...ijk->...ij",
        q * scale,
        k,
        relative_decay,
    )
    attn = torch.linalg.solve_triangular(
        eye + key_attention.tril(-1),
        eye.expand_as(key_attention),
        upper=False,
    )
    decay = cumulative_g.exp2()
    u = attn @ (v * beta[..., None])
    w = attn @ (k * beta[..., None] * decay)
    u_eff = u - w @ state
    output = (q * scale * decay) @ state + query_attention.tril() @ u_eff
    d_last = cumulative_g[..., -1, :]
    kg = k * (d_last[..., None, :] - cumulative_g).exp2()
    state = state * d_last.exp2()[..., None] + kg.transpose(-1, -2) @ u_eff
    return output, state


def _naive_chunk_kda_from_cumulative(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference chunk KDA consuming inclusive chunk-local cumulative log2 decay.

    Inputs use ``q/k/cumulative_g: [B,T,H,K]``, ``v: [B,T,H,V]``, and
    ``beta: [B,T,H]``. ``cumulative_g`` must reset at each ``chunk_size``
    boundary and use the same chunk size passed here. Unlike
    :func:`naive_chunk_kda`, this function does not apply another cumulative
    sum. Internal math runs in FP32, or FP64 when ``q`` is FP64, and outputs use
    ``q.dtype``. Gradient-enabled
    execution checkpoints each chunk so the stable pairwise decay does not retain
    ``[B, H, chunks, chunk_size, chunk_size, K]`` intermediates.
    """
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {q.shape}")
    batch, tokens, heads, key_dim = q.shape
    if tokens == 0:
        raise ValueError("sequence length must be greater than zero")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {q.shape}, got {k.shape}")
    if cumulative_g.shape != q.shape:
        raise ValueError(f"cumulative_g must have shape {q.shape}, got {cumulative_g.shape}")
    if v.ndim != 4 or v.shape[:3] != (batch, tokens, heads):
        raise ValueError(f"v must have shape [B, T, H, V], got {v.shape}")
    if beta.shape != (batch, tokens, heads):
        raise ValueError(f"beta must have shape {(batch, tokens, heads)}, got {beta.shape}")
    expected_state = (batch, heads, v.shape[-1], key_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {initial_state.shape}"
        )
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")

    output_dtype = q.dtype
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    scale = key_dim**-0.5 if scale is None else scale
    q, k, v, cumulative_g = (
        tensor.transpose(1, 2).to(compute_dtype) for tensor in (q, k, v, cumulative_g)
    )
    beta = beta.transpose(1, 2).to(compute_dtype)
    pad = (-tokens) % chunk_size
    if pad:
        q, k, v = (F.pad(tensor, (0, 0, 0, pad)) for tensor in (q, k, v))
        tail = cumulative_g[:, :, -1:].expand(batch, heads, pad, key_dim)
        cumulative_g = torch.cat((cumulative_g, tail), dim=2)
        beta = F.pad(beta, (0, pad))

    length = q.shape[-2]
    chunks = length // chunk_size

    def split_chunks(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, heads, chunks, chunk_size, tensor.shape[-1])

    q, k, v, cumulative_g = map(split_chunks, (q, k, v, cumulative_g))
    beta = beta.reshape(batch, heads, chunks, chunk_size)
    strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )
    eye = torch.eye(chunk_size, dtype=compute_dtype, device=q.device)
    value_dim = v.shape[-1]
    state = (
        q.new_zeros(batch, heads, key_dim, value_dim)
        if initial_state is None
        else initial_state.transpose(-1, -2).to(compute_dtype)
    )
    outputs = []
    for index in range(chunks):
        step_args = (
            q[:, :, index],
            k[:, :, index],
            v[:, :, index],
            cumulative_g[:, :, index],
            beta[:, :, index],
            state,
            scale,
            strict_upper,
            eye,
        )
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in step_args[:6]):
            output, state = checkpoint(
                _naive_chunk_kda_step,
                *step_args,
                use_reentrant=False,
            )
        else:
            output, state = _naive_chunk_kda_step(*step_args)
        outputs.append(output)

    output = torch.stack(outputs, dim=2).reshape(batch, heads, length, value_dim)
    output = output[:, :, :tokens].transpose(1, 2).to(output_dtype)
    final_state = (
        state.transpose(-1, -2).contiguous().to(output_dtype) if output_final_state else None
    )
    return output, final_state


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
    Relative decay is evaluated as ``2 ** (cumulative_i - cumulative_j)`` so excluded positive
    exponents cannot overflow.
    """
    cumulative_g = chunk_cumsum_ref(g.float(), chunk_size)
    return _naive_chunk_kda_from_cumulative(
        q,
        k,
        v,
        cumulative_g,
        beta,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
    )


def l2norm_fwd_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization over the last dim: ``y = x / sqrt(sum(x^2) + eps)``.

    Naive counterpart of ``l2norm_fwd_kernel``. The reduction runs in fp32 (or fp64 when
    ``x`` is already fp64) to mirror the kernel's fp32 accumulation,
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


def l2norm_bwd_ref(x: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Gradient of the L2 norm: ``dx = rstd * (dy - y * <dy, y>)`` with ``y = x * rstd``.

    Naive counterpart of ``l2norm_bwd_kernel``. Recomputes ``y`` from the exact input in
    fp32 (or fp64 when ``x`` is fp64) to match the kernel, then downcasts to ``x.dtype``.

    Args:
        x: forward input; shape ``(..., D)``.
        rstd: reciprocal std saved by the forward pass, broadcastable over the last dim.
        dy: upstream gradient w.r.t. ``y``; shape ``(..., D)``.
    """
    dtype = torch.promote_types(x.dtype, torch.float32)
    xc, dyc, rc = x.to(dtype), dy.to(dtype), rstd.to(dtype)
    yc = xc * rc
    dot = (dyc * yc).sum(-1, keepdim=True)
    return (rc * (dyc - yc * dot)).to(x.dtype)


def chunk_cumsum_ref(
    x: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunk-local cumsum over the time axis (dim=1), optionally per document.

    The cumulative sum restarts at every ``chunk_size`` boundary and at every document
    boundary in variable-length mode.

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


__all__ = [
    "chunk_cumsum_ref",
    "l2norm_bwd_ref",
    "l2norm_fwd_ref",
    "naive_chunk_kda",
    "naive_recurrent_kda",
]
