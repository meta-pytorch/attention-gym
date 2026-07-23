"""Gated DeltaNet (GDN) delta-rule ops.

Runnable naive reference ops — recurrent (``naive_recurrent_gated_delta_rule``) and
chunk-parallel (``naive_chunk_gated_delta_rule``) — plus a stub for the optimized Triton
decode kernel (``fused_recurrent_gated_delta_rule``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference O(T) gated delta rule. State S in [K, V], per (batch, head).

    Per step t (scalar-per-head decay a_t = exp(g_t)):
        S <- a_t * S ;  delta = beta_t * (v_t - k^T S) ;  S <- S + outer(k_t, delta) ;  o_t = q^T S

    Shapes: q, k (B, T, H, K); v (B, T, H, V); g, beta (B, T, H); state (B, H, K, V).

    Args:
        q: query tensor
        k: key tensor
        v: value tensor
        g: scalar-per-head log-decay, a_t = exp(g_t)
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
        state = state * g[:, i].exp()[..., None, None]  # a_t = exp(g_t)
        delta = v[:, i] - torch.einsum("bhk,bhkv->bhv", k[:, i], state)  # vt - k^T S (v_old)
        delta = delta * beta[:, i][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, i], delta)  # + outer(k, delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state))  # q^T S
    return torch.stack(outputs, dim=1), (state if output_final_state else None)


def naive_chunk_gated_delta_rule(
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
    """Naive chunk-parallel gated delta rule (training / prefill).

    split into chunks
    do the pre compute A matrix thing
    """
    b, t, h, k_dim = q.shape
    v_dim = v.shape[-1]
    orig_dtype = q.dtype
    if scale is None:
        scale = k_dim**-0.5

    # -> (B, H, T, .) in fp32
    q, k, v, beta, g = (x.transpose(1, 2).float() for x in (q, k, v, beta, g))
    pad = (chunk_size - t % chunk_size) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    length = q.shape[-2]
    num_chunks = length // chunk_size

    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    def to_chunks(x):
        return x.reshape(b, h, num_chunks, chunk_size, x.shape[-1])

    q, k, v, k_beta = (to_chunks(x) for x in (q, k, v, k_beta))
    decay = g.reshape(b, h, num_chunks, chunk_size).cumsum(-1)  # cumulative log-decay per chunk

    # intra-chunk decay factor between positions i>=j; strictly-lower delta matrix
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    diag_incl = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0
    )
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(diag_incl, 0)

    # forward substitution, building (I − A)^-1 − I
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)

    # add identity matrix back
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v  # "pseudo-values" per chunk
    k_cumdecay = attn @ (k_beta * decay.exp()[..., None])
    v = u

    state = q.new_zeros(b, h, k_dim, v_dim) if initial_state is None else initial_state.float()
    o = torch.zeros_like(v)
    diag_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1
    )
    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(diag_strict, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state  # correction against carried state
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ state
        o[:, :, i] = o_inter + attn_i @ v_new  # inter-chunk read + intra-chunk read
        # carry state to next chunk (decayed) + this chunk's writes
        d_last = decay[:, :, i, -1, None]
        state = (
            state * d_last[..., None].exp()
            + (k_i * (d_last - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    o = o.reshape(b, h, length, v_dim)[:, :, :t].transpose(1, 2).to(orig_dtype)
    return o, (state.to(orig_dtype) if output_final_state else None)


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """TODO: Triton fused recurrent gated delta rule (decode)."""
    raise NotImplementedError("fused_recurrent_gated_delta_rule Triton kernel not implemented yet")
