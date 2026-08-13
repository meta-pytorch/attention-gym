"""Shared KDA test helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def cumulative_sequence_offsets(
    lengths: Sequence[int],
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build an int32 packed-sequence boundary tensor from token lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def bwd_wy_dqkg_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 oracle for ``chunk_kda_bwd_wy_dqkg_fused`` — the fused WY / ``(I-Akk)^-1``
    chunk-level backward.

    Given the recomputed forward intermediates (``A`` = Akk inverse, ``h`` = chunk-start
    hidden state, ``v_new``) and the incoming grads (``do``, ``dh``, ``dv``), produces the
    six per-token grads. All six are FRESH (write-only). Conventions: ``scale`` (=K**-0.5)
    multiplies ``dq`` only; ``beta`` enters raw; gate grads differentiate ``2^g`` directly
    (no ``ln2``); ``dA`` is masked with a STRICT-lower mask (diagonal=-1)::

        dq  = (do @ h) * 2^g * scale
        dv2 = (A^T @ dv) * beta ; dk = (v_new @ dh) * 2^{g_last-g} + (A^T @ dw) * 2^g * beta
        dA  = -A^T @ striL(beta * (dv@v^T + dw@kg^T)) @ A     (dw = -dv @ h, kg = k*2^g)

    Per 64-token chunk; ``A``/``dA`` sliced to ``[:chunk_len]``; ``h``/``dh`` indexed by
    the chunk id on the num_chunks axis.
    """
    B, T, H, K = q.shape
    V = v.shape[3]
    device = q.device
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    dq = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dk = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dv2 = torch.zeros(B, T, H, V, dtype=acc, device=device)
    db = torch.zeros(B, T, H, dtype=acc, device=device)
    dg = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dA = torch.zeros(B, T, H, chunk_size, dtype=acc, device=device)
    if chunk_indices is None:
        chunks = (T + chunk_size - 1) // chunk_size
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
        chunk_indices = torch.stack(
            (
                torch.zeros(chunks, dtype=torch.int64, device=device),
                torch.arange(chunks, device=device),
            ),
            dim=1,
        )
    cu = cu_seqlens.tolist()
    for b in range(B):
        for flat, (seq_idx, chunk_idx) in enumerate(chunk_indices.tolist()):
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            row_start = bos + chunk_idx * chunk_size
            cl = min(eos - row_start, chunk_size)
            s = slice(row_start, row_start + cl)
            q_f, k_f, v_f = q[b, s].to(acc), k[b, s].to(acc), v[b, s].to(acc)
            v_new_f, g_f = v_new[b, s].to(acc), g[b, s].to(acc)
            beta_f = beta[b, s].to(acc)
            A_f = A[b, s, :, :cl].to(acc)
            h_f, dh_f = h[b, flat].to(acc), dh[b, flat].to(acc)
            do_f, dv_f = do[b, s].to(acc), dv[b, s].to(acc)

            strict = torch.tril(torch.ones(cl, cl, device=device, dtype=torch.bool), -1)
            exp2_g = torch.exp2(g_f)
            beta_k = beta_f.unsqueeze(-1)
            kg = k_f * exp2_g  # k * 2^g
            A_t = A_f.permute(1, 2, 0)  # (H, col, row): A_f transposed on (row, col)

            rev_decay = torch.exp2(g_f[-1:].float() - g_f)  # 2^{g_last - g}, per key channel
            dq_chunk = torch.einsum("thv,hkv->thk", do_f, h_f) * exp2_g * scale
            dk_state = torch.einsum("thv,hkv->thk", v_new_f, dh_f) * rev_decay
            dw = -torch.einsum("thv,hkv->thk", dv_f, h_f)  # (row, H, K)
            dvb = torch.einsum("ths,thv->shv", A_f, dv_f)  # (A^T @ dv)  (col, H, V)
            dkgb = torch.einsum("ths,thk->shk", A_f, dw)  # (A^T @ dw)  (col, H, K)

            dv2_chunk = dvb * beta_k
            db_chunk = (dvb * v_f).sum(-1) + (dkgb * kg).sum(-1)
            dk_chunk = dk_state + dkgb * exp2_g * beta_k
            kdk_state = k_f * dk_state
            dg_chunk = q_f * dq_chunk - kdk_state + kg * dkgb * beta_k
            dg_chunk[-1] += (h_f * dh_f).sum(-1) * torch.exp2(g_f[-1]) + kdk_state.sum(0)

            dA_repr = torch.einsum("thv,shv->hts", dv_f, v_f) + torch.einsum(
                "thk,shk->hts", dw, kg
            )
            dA_repr = torch.where(strict, dA_repr, 0.0) * beta_f.transpose(0, 1).unsqueeze(1)
            dA_raw = torch.where(strict, -torch.bmm(A_t, torch.bmm(dA_repr, A_t)), 0.0)

            dq[b, s], dk[b, s] = dq_chunk, dk_chunk
            dv2[b, s], db[b, s], dg[b, s] = dv2_chunk, db_chunk, dg_chunk
            dA[b, s, :, :cl] = dA_raw.permute(1, 0, 2)
    return dq, dk, dv2, db, dg, dA


__all__ = ["bwd_wy_dqkg_reference", "cumulative_sequence_offsets"]
