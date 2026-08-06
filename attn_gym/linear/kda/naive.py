"""KDA (Kimi Delta Attention) delta-rule ops."""

from __future__ import annotations

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


__all__ = ["naive_chunk_kda", "naive_recurrent_kda"]
