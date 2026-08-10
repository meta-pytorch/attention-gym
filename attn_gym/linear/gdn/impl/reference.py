"""Readable eager PyTorch implementation of gated delta rule attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor | None,
    return_final_state: bool,
    mode: str,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the selected eager gated delta rule execution form."""
    tensors = (query, key, value, log_decay, beta)
    if initial_state is not None:
        tensors += (initial_state,)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must have floating-point dtypes")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(tensor.dtype != query.dtype for tensor in tensors[1:]):
        raise ValueError("all inputs must have the same dtype")

    if mode == "recurrent":
        return recurrent_forward(
            query,
            key,
            value,
            log_decay,
            beta,
            scale=scale,
            initial_state=initial_state,
            return_final_state=return_final_state,
        )
    if mode == "chunked":
        return chunked_forward(
            query,
            key,
            value,
            log_decay,
            beta,
            scale=scale,
            initial_state=initial_state,
            return_final_state=return_final_state,
            chunk_size=chunk_size,
        )
    raise AssertionError(f"API validation allowed unexpected mode {mode!r}")


def recurrent_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor | None,
    return_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the gated delta rule recurrence one token at a time."""
    batch, heads, sequence, key_dimension = query.shape
    query = query * (key_dimension**-0.5 if scale is None else scale)
    state = (
        query.new_zeros(batch, heads, key_dimension, value.shape[-1])
        if initial_state is None
        else initial_state
    )
    outputs = []

    for token in range(sequence):
        state = state * log_decay[:, :, token].exp()[..., None, None]
        residual = value[:, :, token] - torch.einsum("bhk,bhkv->bhv", key[:, :, token], state)
        residual = residual * beta[:, :, token, None]
        state = state + torch.einsum("bhk,bhv->bhkv", key[:, :, token], residual)
        outputs.append(torch.einsum("bhk,bhkv->bhv", query[:, :, token], state))

    output = torch.stack(outputs, dim=2)
    return output, state if return_final_state else None


def chunked_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor | None,
    return_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the gated delta rule with a naive chunk-parallel decomposition."""
    batch, heads, sequence, key_dimension = query.shape
    value_dimension = value.shape[-1]
    output_dtype = query.dtype
    scale = key_dimension**-0.5 if scale is None else scale
    compute_dtype = torch.float32 if query.dtype != torch.float64 else torch.float64

    query, key, value, beta, log_decay = (
        tensor.to(compute_dtype) for tensor in (query, key, value, beta, log_decay)
    )
    padding = (-sequence) % chunk_size
    if padding:
        query, key, value = (F.pad(tensor, (0, 0, 0, padding)) for tensor in (query, key, value))
        beta, log_decay = (F.pad(tensor, (0, padding)) for tensor in (beta, log_decay))

    padded_length = query.shape[-2]
    chunk_count = padded_length // chunk_size
    query = query * scale
    value = value * beta[..., None]
    beta_key = key * beta[..., None]

    query, key, value, beta_key = (
        tensor.reshape(batch, heads, chunk_count, chunk_size, tensor.shape[-1])
        for tensor in (query, key, value, beta_key)
    )
    cumulative_decay = log_decay.reshape(batch, heads, chunk_count, chunk_size).cumsum(-1)

    decay_matrix = (
        (cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)).tril().exp().tril()
    )
    diagonal_and_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device)
    )
    transition = -((beta_key @ key.transpose(-1, -2)) * decay_matrix).masked_fill(
        diagonal_and_upper, 0
    )

    for row in range(1, chunk_size):
        transition[..., row, :row] = transition[..., row, :row].clone() + (
            transition[..., row, :row, None].clone() * transition[..., :row, :row].clone()
        ).sum(-2)

    transition = transition + torch.eye(chunk_size, dtype=torch.float, device=query.device)
    value = transition @ value
    decayed_key = transition @ (beta_key * cumulative_decay.exp()[..., None])

    state = (
        query.new_zeros(batch, heads, key_dimension, value_dimension)
        if initial_state is None
        else initial_state.to(compute_dtype)
    )
    output = torch.zeros_like(value)
    strictly_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for chunk in range(chunk_count):
        chunk_query = query[:, :, chunk]
        chunk_key = key[:, :, chunk]
        chunk_value = value[:, :, chunk]
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

    output = output.reshape(batch, heads, padded_length, value_dimension)[:, :, :sequence]
    final_state = state.to(output_dtype) if return_final_state else None
    return output.to(output_dtype), final_state
