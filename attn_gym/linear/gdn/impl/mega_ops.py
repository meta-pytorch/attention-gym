# SPDX-License-Identifier: BSD-3-Clause

"""Private registered operators around the optional scalar-GDN Mega launchers.

The opaque operators keep CuTeDSL imports and launch-time setup outside captured graphs. Public
input normalization and autograd live in :mod:`mega`; these operators own only fixed kernel ABIs.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear._delta_rule.paged_state import PagedState

torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_fwd",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "Tensor chunk_offsets, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor initial_state, "
    "Tensor cu_seqlens, Tensor chunk_offsets, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor initial_state, "
    "Tensor cu_seqlens, Tensor chunk_offsets, float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_fwd_paged",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_bwd",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, float scale) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_mega_packed_bwd_with_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor initial_state, Tensor? d_final_state, Tensor cu_seqlens, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _normalize_tma_tensor(tensor: Tensor) -> Tensor:
    """Copy only runtime tensors that cannot satisfy the raw TMA ABI."""
    return (
        tensor
        if tensor_supports_tma(tensor)
        else tensor.clone(memory_format=torch.contiguous_format)
    )


def _normalize_scalar_tensor(tensor: Tensor) -> Tensor:
    """Copy only scalar rows that cannot satisfy their contiguous-head ABI."""
    return (
        tensor
        if tensor_supports_contiguous_dim(tensor, alignment_bytes=4)
        else tensor.clone(memory_format=torch.contiguous_format)
    )


def _normalize_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    """Copy routing metadata only when its raw launcher ABI requires it."""
    if cu_seqlens.is_contiguous() and cu_seqlens.data_ptr() % 8 == 0:
        return cu_seqlens
    return cu_seqlens.clone(memory_format=torch.contiguous_format)


def _empty_with_layout(template: Tensor) -> Tensor:
    """Allocate a tensor with exactly the template's shape, dtype, device, and strides."""
    return torch.empty_strided(
        template.shape,
        template.stride(),
        dtype=template.dtype,
        device=template.device,
    )


def _copy_to_layout(tensor: Tensor, template: Tensor) -> Tensor:
    """Return a gradient/output with the fake-visible input layout."""
    if tensor.stride() == template.stride():
        return tensor
    return _empty_with_layout(template).copy_(tensor)


def _forward_backend() -> object:
    """Lazily import the optional forward launcher."""
    try:
        return importlib.import_module("attn_gym.linear._delta_rule.mega.gdn_forward")
    except ImportError as error:
        raise ImportError("the Mega GDN backend requires CUDA and attn-gym[mega]") from error


def _backward_backend() -> object:
    """Lazily import the optional checkpoint-recompute and backward launcher."""
    try:
        return importlib.import_module("attn_gym.linear._delta_rule.mega.gdn_backward")
    except ImportError as error:
        raise ImportError("the Mega GDN backend requires CUDA and attn-gym[mega]") from error


def validate_mega_available(q: Tensor) -> None:
    """Fail before caller-side setup when the optional backend cannot run."""
    _forward_backend().validate_available(q)


def _packed_fwd_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> Tensor:
    del chunk_offsets
    value_template = value
    q, k, value = (_normalize_tma_tensor(tensor) for tensor in (q, k, value))
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    output, _ = _forward_backend().run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        None,
        scale=scale,
        output_final_state=False,
    )
    return _copy_to_layout(output, value_template)


def _packed_fwd_with_initial_state_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> Tensor:
    del chunk_offsets
    value_template = value
    q, k, value, initial_state = (
        _normalize_tma_tensor(tensor) for tensor in (q, k, value, initial_state)
    )
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    output, _ = _forward_backend().run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=False,
    )
    return _copy_to_layout(output, value_template)


def _packed_fwd_with_state_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    del chunk_offsets
    value_template, state_template = value, initial_state
    q, k, value, initial_state = (
        _normalize_tma_tensor(tensor) for tensor in (q, k, value, initial_state)
    )
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    output, final_state = _forward_backend().run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=True,
    )
    assert final_state is not None
    return _copy_to_layout(output, value_template), _copy_to_layout(final_state, state_template)


def _packed_fwd_paged_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    state_cache: Tensor,
    state_indices: Tensor,
    has_initial_state: Tensor | None,
    cu_seqlens: Tensor,
    scale: float,
) -> Tensor:
    """Advance selected pool slots in place; the pool itself is never copied."""
    value_template = value
    q, k, value = (_normalize_tma_tensor(tensor) for tensor in (q, k, value))
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    backend = _forward_backend()
    paged_state = PagedState.validate(
        state_cache,
        state_indices,
        has_initial_state,
        num_sequences=cu_seqlens.shape[0] - 1,
        heads=value.shape[2],
        value_dim=value.shape[3],
        key_dim=q.shape[3],
        device=q.device,
        read_only_inputs=(q, k, value, gate, beta, cu_seqlens),
    ).require_alignment(16)
    output, _ = backend.run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        paged_state,
        scale=scale,
        output_final_state=False,
    )
    return _copy_to_layout(output, value_template)


def _packed_bwd_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    d_output: Tensor,
    cu_seqlens: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    templates = (q, k, value, gate, beta)
    q, k, value, d_output = (_normalize_tma_tensor(tensor) for tensor in (q, k, value, d_output))
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    dq, dk, dv, dgate, dbeta, d_initial_state = _backward_backend().chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        scale=scale,
    )
    assert d_initial_state is None
    return tuple(
        _copy_to_layout(gradient, template)
        for gradient, template in zip((dq, dk, dv, dgate, dbeta), templates, strict=True)
    )


def _packed_bwd_with_state_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    d_output: Tensor,
    initial_state: Tensor,
    d_final_state: Tensor | None,
    cu_seqlens: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    templates = (q, k, value, gate, beta, initial_state)
    q, k, value, d_output, initial_state = (
        _normalize_tma_tensor(tensor) for tensor in (q, k, value, d_output, initial_state)
    )
    if d_final_state is not None:
        d_final_state = _normalize_tma_tensor(d_final_state)
    gate, beta = (_normalize_scalar_tensor(tensor) for tensor in (gate, beta))
    cu_seqlens = _normalize_cu_seqlens(cu_seqlens)
    dq, dk, dv, dgate, dbeta, d_initial_state = _backward_backend().chunk_gdn_bwd_mega_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        initial_state,
        d_final_state,
        scale=scale,
    )
    assert d_initial_state is not None
    return tuple(
        _copy_to_layout(gradient, template)
        for gradient, template in zip(
            (dq, dk, dv, dgate, dbeta, d_initial_state), templates, strict=True
        )
    )


torch.library.impl("attn_gym::gdn_chunk_mega_packed_fwd", "CUDA", _packed_fwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_mega_packed_fwd_with_initial_state",
    "CUDA",
    _packed_fwd_with_initial_state_cuda,
)
torch.library.impl(
    "attn_gym::gdn_chunk_mega_packed_fwd_with_state",
    "CUDA",
    _packed_fwd_with_state_cuda,
)
torch.library.impl("attn_gym::gdn_chunk_mega_packed_fwd_paged", "CUDA", _packed_fwd_paged_cuda)
torch.library.impl("attn_gym::gdn_chunk_mega_packed_bwd", "CUDA", _packed_bwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_mega_packed_bwd_with_state",
    "CUDA",
    _packed_bwd_with_state_cuda,
)


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_fwd")
def _packed_fwd_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> Tensor:
    """Describe the output allocation made by the no-state forward launcher."""
    return _empty_with_layout(value)


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> Tensor:
    """Describe the output-only stateful forward launcher."""
    return _empty_with_layout(value)


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_fwd_with_state")
def _packed_fwd_with_state_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Describe the forward output and launcher-cloned final state."""
    return _empty_with_layout(value), _empty_with_layout(initial_state)


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_fwd_paged")
def _packed_fwd_paged_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    state_cache: Tensor,
    state_indices: Tensor,
    has_initial_state: Tensor | None,
    cu_seqlens: Tensor,
    scale: float,
) -> Tensor:
    """Describe the paged forward launcher; the pool mutation is declared in the schema."""
    return _empty_with_layout(value)


def _gradient_allocation(tensor: Tensor) -> Tensor:
    """Match the CUDA wrapper's exact-layout gradient allocation."""
    return _empty_with_layout(tensor)


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_bwd")
def _packed_bwd_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    d_output: Tensor,
    cu_seqlens: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Describe the five no-state backward gradients."""
    return tuple(_gradient_allocation(tensor) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::gdn_chunk_mega_packed_bwd_with_state")
def _packed_bwd_with_state_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    d_output: Tensor,
    initial_state: Tensor,
    d_final_state: Tensor | None,
    cu_seqlens: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Describe stateful backward gradients, including the initial-state cotangent."""
    return (
        *(_gradient_allocation(tensor) for tensor in (q, k, value, gate, beta)),
        _empty_with_layout(initial_state),
    )


chunk_gdn_mega_packed_fwd_op = torch.ops.attn_gym.gdn_chunk_mega_packed_fwd.default
chunk_gdn_mega_packed_fwd_with_initial_state_op = (
    torch.ops.attn_gym.gdn_chunk_mega_packed_fwd_with_initial_state.default
)
chunk_gdn_mega_packed_fwd_with_state_op = (
    torch.ops.attn_gym.gdn_chunk_mega_packed_fwd_with_state.default
)
chunk_gdn_mega_packed_fwd_paged_op = torch.ops.attn_gym.gdn_chunk_mega_packed_fwd_paged.default
chunk_gdn_mega_packed_bwd_op = torch.ops.attn_gym.gdn_chunk_mega_packed_bwd.default
chunk_gdn_mega_packed_bwd_with_state_op = (
    torch.ops.attn_gym.gdn_chunk_mega_packed_bwd_with_state.default
)


__all__ = [
    "chunk_gdn_mega_packed_bwd_op",
    "chunk_gdn_mega_packed_bwd_with_state_op",
    "chunk_gdn_mega_packed_fwd_op",
    "chunk_gdn_mega_packed_fwd_paged_op",
    "chunk_gdn_mega_packed_fwd_with_initial_state_op",
    "chunk_gdn_mega_packed_fwd_with_state_op",
    "validate_mega_available",
]
