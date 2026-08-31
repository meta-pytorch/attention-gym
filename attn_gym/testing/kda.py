"""Shared KDA test helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.naive import naive_recurrent_kda


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


def strided_state_pool(
    num_slots: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    *,
    prefix: int = 11,
    suffix: int = 17,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the slot-strided FP32 recurrent state view produced by vLLM's packed byte pages.

    Returns the flat backing storage plus a non-contiguous ``[num_slots, heads, V, K]`` view
    whose slots are separated by padding, matching how serving engines carve recurrent state
    out of larger per-slot pages. Keep the storage alive while the view is in use.

    The default ``prefix`` deliberately misaligns the pool base to stress the Triton paths,
    which tolerate arbitrary element offsets. CuTe backends declare ``assumed_align=16`` on
    the pool pointer, so their tests must pass ``prefix=0`` to keep the base 16-byte aligned.
    """
    state_elements = heads * key_dim * value_dim
    storage = torch.randn(
        num_slots, prefix + state_elements + suffix, device="cuda", dtype=torch.float32
    )
    state = storage[:, prefix : prefix + state_elements].view(num_slots, heads, value_dim, key_dim)
    assert not state.is_contiguous()
    assert state.stride()[1:] == (value_dim * key_dim, key_dim, 1)
    return storage, state


def make_kda_test_inputs(
    tokens: int,
    *,
    batch: int = 1,
    heads: int = 1,
    seed: int = 41,
    gate_scale: float = 1.0,
    gate_value: float | None = None,
    log_uniform_gate: bool = False,
    sigmoid_beta: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    normalize_qk: bool = False,
    value_scale: float = 0.125,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic public KDA inputs with per-token natural-log gates."""
    torch.manual_seed(seed)
    shape = (batch, tokens, heads, 128)
    if normalize_qk:
        q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
        k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
    else:
        q = torch.randn(shape, device="cuda", dtype=dtype) / 8
        k = torch.randn(shape, device="cuda", dtype=dtype) / 8
    value = torch.randn(shape, device="cuda", dtype=dtype) * value_scale
    if gate_value is not None:
        gate = torch.full(shape, gate_value, device="cuda")
    elif log_uniform_gate:
        gate = torch.empty(shape, device="cuda").uniform_(math.exp(-gate_scale), 1.0).log_()
    else:
        gate = -torch.rand(shape, device="cuda") * gate_scale
    beta = (
        torch.randn(batch, tokens, heads, device="cuda").sigmoid_()
        if sigmoid_beta
        else torch.rand(batch, tokens, heads, device="cuda")
    )
    values = (q, k, value, gate, beta)
    return tuple(value.requires_grad_(requires_grad) for value in values)


def clone_kda_inputs(
    inputs: Sequence[torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    """Clone KDA inputs into independent autograd leaves, optionally changing precision."""
    return tuple(
        value.detach()
        .to(dtype=value.dtype if dtype is None else dtype)
        .clone()
        .requires_grad_(value.requires_grad)
        for value in inputs
    )


def kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the eager recurrent KDA oracle using natural-log public gates."""
    return naive_recurrent_kda(
        q,
        k,
        value,
        gate * LOG2_E,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


def assert_matches_low_precision_reference(
    actual: torch.Tensor,
    high_precision: torch.Tensor,
    low_precision: torch.Tensor,
    name: str,
    *,
    source_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Bound kernel error by the reference error and source-operand precision."""
    high_precision = high_precision.double()
    rounding_band = torch.finfo(source_dtype).eps * high_precision.abs().max().item()
    actual_error = (actual.double() - high_precision).abs().max().item()
    reference_error = (low_precision.double() - high_precision).abs().max().item()
    budget = 2 * (reference_error + rounding_band)
    assert torch.isfinite(actual).all(), f"{name}: kernel output contains non-finite values"
    assert actual_error <= budget, (
        f"{name}: kernel error {actual_error:.3e} exceeds {budget:.3e} "
        f"(reference error {reference_error:.3e})"
    )


def bwd_daqk_reference(
    value: torch.Tensor,
    d_output: torch.Tensor,
    lengths: Sequence[int],
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute packed sequence-local dAqk in FP32."""
    result = torch.zeros(*value.shape[:-1], chunk_size, device=value.device)
    begin = 0
    for length in lengths:
        for offset in range(0, length, chunk_size):
            end = min(offset + chunk_size, length)
            size = end - offset
            token = slice(begin + offset, begin + end)
            block = torch.einsum(
                "blhv,bmhv->blhm",
                d_output[:, token].float(),
                value[:, token].float(),
            )
            causal = torch.ones(size, size, dtype=torch.bool, device=value.device).tril()
            result[:, token, :, :size] = block * scale * causal[None, :, None, :]
        begin += length
    return result


def bwd_intra_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 oracle for ``chunk_kda_bwd_intra`` (the intra-chunk backward of the K3b/K4b
    forward).

    Given the incoming grads ``dAqk``/``dAkk`` (grad of loss w.r.t. the forward Aqk/Akk),
    returns this stage's intra contributions (dq, dk, db, dg) with the running grads set to
    zero. Per 64-token chunk, indexing i=query/row, j=key/col, ``exp2 = 2^{g_i - g_j}``::

        Aqk path:  dq = sum_j dAqk*exp2*k_j        dk += sum_i dAqk*exp2*q_i
                   dg  = sum_j dAqk*exp2*q_i*k_j - sum_i (same)
        Akk path:  dk += sum_j dAkk*exp2*beta_i*k_j + sum_i dAkk*exp2*beta_i*k_i
                   db  = sum_j dAkk * <exp2*k_i, k_j>
                   dg += sum_j dAkk*exp2*beta_i*k_i*k_j - sum_i (same)

    Both dAqk and dAkk are masked with a NON-strict causal mask (i>=j, incl. diagonal) to
    match fla upstream. There is no ``scale`` because it is folded upstream into dAqk.
    The final ``dg`` writer applies ``ln(2)`` for the derivative of ``2**g``.
    """
    B, T, H, Kd = q.shape
    device = q.device
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, gf, bf = (t.to(acc) for t in (q, k, g, beta))
    daqk, dakk = dAqk.to(acc), dAkk.to(acc)
    dq = torch.zeros(B, T, H, Kd, dtype=acc, device=device)
    dk = torch.zeros_like(dq)
    dg = torch.zeros_like(dq)
    db = torch.zeros(B, T, H, dtype=acc, device=device)
    # Varlen work-list iteration (see ``bwd_wy_dqkg_reference``): a partial last chunk uses only its
    # ``valid`` rows/cols so the non-strict causal mask never straddles a document boundary.
    if chunk_indices is None:
        chunks = T // chunk_size
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
        for seq_idx, chunk_idx in chunk_indices.tolist():
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            row_start = bos + chunk_idx * chunk_size
            cl = min(eos - row_start, chunk_size)
            s = slice(row_start, row_start + cl)
            mask = torch.tril(torch.ones(cl, cl, dtype=torch.bool, device=device))[:, :, None]
            q_i, k_i, k_j = qf[b, s][:, None], kf[b, s][:, None], kf[b, s][None, :]
            beta_i = bf[b, s][:, None, :, None]
            exp2 = torch.exp2(gf[b, s][:, None] - gf[b, s][None, :])  # 2^{g_i - g_j}
            aq = torch.where(mask, daqk[b, s, :, :cl].permute(0, 2, 1), 0.0)  # (i, j, H)
            ak = torch.where(mask, dakk[b, s, :, :cl].permute(0, 2, 1), 0.0)

            aqk = aq[..., None] * exp2  # (i, j, H, K)
            t_aqk = aqk * q_i * k_j
            dq[b, s] = (aqk * k_j).sum(1)
            dk[b, s] = (aqk * q_i).sum(0)
            dg[b, s] = t_aqk.sum(1) - t_aqk.sum(0)

            akk = ak[..., None] * exp2 * beta_i  # (i, j, H, K)
            t_akk = akk * k_i * k_j
            dk[b, s] += (akk * k_j).sum(1) + (akk * k_i).sum(0)
            db[b, s] = (ak * (exp2 * k_i * k_j).sum(-1)).sum(1)
            dg[b, s] += t_akk.sum(1) - t_akk.sum(0)
    return dq, dk, db, dg * math.log(2.0)


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


__all__ = [
    "assert_matches_low_precision_reference",
    "bwd_daqk_reference",
    "bwd_intra_reference",
    "bwd_wy_dqkg_reference",
    "clone_kda_inputs",
    "cumulative_sequence_offsets",
    "kda_reference",
    "make_kda_test_inputs",
]
