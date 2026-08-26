"""Readable eager PyTorch implementation of gated delta rule attention."""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

_CHUNK_SIZE = 64


def _packed_reference(
    dense_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate packed sequences independently through a dense reference operation."""
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[-1]
    num_sequences = cu_seqlens.shape[0] - 1
    output = torch.zeros_like(v)
    final_state = None
    if output_final_state:
        final_state = (
            q.new_zeros(num_sequences, heads, key_dim, value_dim)
            if initial_state is None
            else initial_state.clone()
        )

    offsets = cu_seqlens.cpu().tolist()
    if (
        offsets[0] != 0
        or any(begin > end for begin, end in pairwise(offsets))
        or offsets[-1] > q.shape[1]
    ):
        raise ValueError(
            "cu_seqlens offsets must start at zero, be nondecreasing, and end within "
            "the physical token capacity"
        )
    for sequence, (begin, end) in enumerate(pairwise(offsets)):
        if begin == end:
            continue
        span = slice(begin, end)
        span_output, span_state = dense_op(
            q[:, span],
            k[:, span],
            v[:, span],
            log_decay[:, span],
            beta[:, span],
            scale=scale,
            initial_state=None
            if initial_state is None
            else initial_state[sequence : sequence + 1],
            output_final_state=output_final_state,
        )
        output[:, span] = span_output
        if final_state is not None:
            final_state[sequence] = span_state[0]
    return output, final_state


def reference_gdn(
    dense_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a GDN reference operation in the documented compute dtype."""
    output_dtype = q.dtype
    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    q, k, v, log_decay, beta = (tensor.to(compute_dtype) for tensor in (q, k, v, log_decay, beta))
    # Explicit casts do not stop autocast from selecting low-precision contractions.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if cu_seqlens is None:
            output, state = dense_op(
                q,
                k,
                v,
                log_decay,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        else:
            output, state = _packed_reference(
                dense_op,
                q,
                k,
                v,
                log_decay,
                beta,
                initial_state,
                cu_seqlens,
                scale,
                output_final_state,
            )
    return output.to(output_dtype), state


def recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the gated delta rule recurrence one token at a time."""
    batch, sequence, heads, key_dim = q.shape
    q = q * scale
    state = (
        q.new_zeros(batch, heads, key_dim, v.shape[-1]) if initial_state is None else initial_state
    )
    outputs = []

    for token in range(sequence):
        state = state * log_decay[:, token].exp()[..., None, None]
        residual = v[:, token] - torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        residual = residual * beta[:, token, :, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, token], residual)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, token], state))

    return torch.stack(outputs, dim=1), state if output_final_state else None


def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the gated delta rule with the fixed chunk-parallel decomposition."""
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    padding = (-sequence) % _CHUNK_SIZE
    if padding:
        q, k, v = (F.pad(tensor, (0, 0, 0, 0, 0, padding)) for tensor in (q, k, v))
        beta, log_decay = (F.pad(tensor, (0, 0, 0, padding)) for tensor in (beta, log_decay))

    padded_length = q.shape[1]
    chunk_count = padded_length // _CHUNK_SIZE
    q = q * scale
    v = v * beta[..., None]
    beta_key = k * beta[..., None]

    q, k, v, beta_key = (
        tensor.reshape(batch, chunk_count, _CHUNK_SIZE, heads, tensor.shape[-1]).permute(
            0, 3, 1, 2, 4
        )
        for tensor in (q, k, v, beta_key)
    )
    cumulative_decay = (
        log_decay.reshape(batch, chunk_count, _CHUNK_SIZE, heads).permute(0, 3, 1, 2).cumsum(-1)
    )

    decay_matrix = (
        (cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)).tril().exp().tril()
    )
    diagonal_and_upper = torch.triu(
        torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.bool, device=q.device)
    )
    transition = -((beta_key @ k.transpose(-1, -2)) * decay_matrix).masked_fill(
        diagonal_and_upper, 0
    )

    for row in range(1, _CHUNK_SIZE):
        transition[..., row, :row] = transition[..., row, :row].clone() + (
            transition[..., row, :row, None].clone() * transition[..., :row, :row].clone()
        ).sum(-2)

    transition = transition + torch.eye(_CHUNK_SIZE, dtype=q.dtype, device=q.device)
    v = transition @ v
    decayed_key = transition @ (beta_key * cumulative_decay.exp()[..., None])

    state = (
        q.new_zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state
    )
    output = torch.zeros_like(v)
    strictly_upper = torch.triu(
        torch.ones(_CHUNK_SIZE, _CHUNK_SIZE, dtype=torch.bool, device=q.device), diagonal=1
    )

    for chunk in range(chunk_count):
        chunk_query = q[:, :, chunk]
        chunk_key = k[:, :, chunk]
        chunk_value = v[:, :, chunk]
        attention = (
            (chunk_query @ chunk_key.transpose(-1, -2)) * decay_matrix[:, :, chunk]
        ).masked_fill(strictly_upper, 0)
        corrected_value = chunk_value - (decayed_key[:, :, chunk] @ state)
        prior_output = (chunk_query * cumulative_decay[:, :, chunk, :, None].exp()) @ state
        output[:, :, chunk] = prior_output + attention @ corrected_value

        final_decay = cumulative_decay[:, :, chunk, -1, None]
        state = (
            state * final_decay[..., None].exp()
            + (
                chunk_key * (final_decay - cumulative_decay[:, :, chunk]).exp()[..., None]
            ).transpose(-1, -2)
            @ corrected_value
        )

    output = output.permute(0, 2, 3, 1, 4).reshape(batch, padded_length, heads, value_dim)
    return output[:, :sequence], state if output_final_state else None


__all__ = ["chunk_forward", "recurrent_forward", "reference_gdn"]
