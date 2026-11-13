# SPDX-License-Identifier: BSD-3-Clause

"""Private registered operators around the optional scalar-GDN warpdrive launchers.

The opaque operators keep CuTeDSL imports and launch-time setup outside captured graphs. Public
input normalization and autograd live in :mod:`attn_gym.linear.gdn.impl.warpdrive`; these operators
own fixed kernel ABIs and device-side boundary validation.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from attn_gym._backends.cute import normalize_tma_tensor
from attn_gym.linear._delta_rule.operand_layout import (
    normalize_cu_seqlens,
    normalize_scalar_rows,
)
from attn_gym.linear._delta_rule.paged_state import PagedState
from attn_gym.linear._delta_rule.validation import validate_packed_offsets_tensor

# The registered op names keep the backend's former ``cudnn`` name so that serialized graphs
# and exported programs that reference them still load.
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_fwd",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "bool split, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor initial_state, "
    "Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor initial_state, "
    "Tensor cu_seqlens, float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_fwd_paged",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_bwd",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, bool split, float scale) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_cudnn_packed_bwd_with_state",
    "(Tensor q, Tensor k, Tensor value, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor initial_state, Tensor? d_final_state, Tensor cu_seqlens, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _empty_with_layout(template: Tensor) -> Tensor:
    """Allocate with the template's strides, or contiguously if it broadcasts (stride 0)."""
    if 0 in template.stride():
        return template.new_empty(template.shape)
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
        return importlib.import_module(
            "attn_gym.linear._delta_rule.warpdrive.launchers.gdn_forward"
        )
    except ImportError as error:
        raise ImportError(
            "the warpdrive GDN backend requires CUDA and attn-gym[linear]"
        ) from error


def _backward_backend() -> object:
    """Lazily import the optional checkpoint-recompute and backward launcher."""
    try:
        return importlib.import_module(
            "attn_gym.linear._delta_rule.warpdrive.launchers.gdn_backward"
        )
    except ImportError as error:
        raise ImportError(
            "the warpdrive GDN backend requires CUDA and attn-gym[linear]"
        ) from error


def validate_warpdrive_available(q: Tensor) -> None:
    """Fail before caller-side setup when the optional backend cannot run."""
    _forward_backend().validate_available(q)


def _packed_fwd_cuda(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    split: bool,
    scale: float,
) -> Tensor:
    """Run the fixed-arity forward launcher; its prologue kernel checks the packed boundaries."""
    value_template = value
    q, k, value = (normalize_tma_tensor(tensor) for tensor in (q, k, value))
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
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
        split=split,
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
    scale: float,
) -> Tensor:
    """Run the fixed-arity forward launcher; its prologue kernel checks the packed boundaries."""
    value_template = value
    q, k, value, initial_state = (
        normalize_tma_tensor(tensor) for tensor in (q, k, value, initial_state)
    )
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
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
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Run the fixed-arity forward launcher; its prologue kernel checks the packed boundaries."""
    value_template, state_template = value, initial_state
    q, k, value, initial_state = (
        normalize_tma_tensor(tensor) for tensor in (q, k, value, initial_state)
    )
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
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
    q, k, value = (normalize_tma_tensor(tensor) for tensor in (q, k, value))
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
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
    )
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
    split: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    templates = (q, k, value, gate, beta)
    q, k, value, d_output = (normalize_tma_tensor(tensor) for tensor in (q, k, value, d_output))
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    dq, dk, dv, dgate, dbeta, d_initial_state = _backward_backend().chunk_gdn_bwd_warpdrive_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        scale=scale,
        split=split,
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
        normalize_tma_tensor(tensor) for tensor in (q, k, value, d_output, initial_state)
    )
    if d_final_state is not None:
        d_final_state = normalize_tma_tensor(d_final_state)
    gate, beta = (normalize_scalar_rows(tensor) for tensor in (gate, beta))
    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    dq, dk, dv, dgate, dbeta, d_initial_state = _backward_backend().chunk_gdn_bwd_warpdrive_packed(
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


torch.library.impl("attn_gym::gdn_chunk_cudnn_packed_fwd", "CUDA", _packed_fwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_cudnn_packed_fwd_with_initial_state",
    "CUDA",
    _packed_fwd_with_initial_state_cuda,
)
torch.library.impl(
    "attn_gym::gdn_chunk_cudnn_packed_fwd_with_state",
    "CUDA",
    _packed_fwd_with_state_cuda,
)
torch.library.impl("attn_gym::gdn_chunk_cudnn_packed_fwd_paged", "CUDA", _packed_fwd_paged_cuda)
torch.library.impl("attn_gym::gdn_chunk_cudnn_packed_bwd", "CUDA", _packed_bwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_cudnn_packed_bwd_with_state",
    "CUDA",
    _packed_bwd_with_state_cuda,
)


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_fwd")
def _packed_fwd_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    cu_seqlens: Tensor,
    split: bool,
    scale: float,
) -> Tensor:
    """Describe the output allocation made by the no-state forward launcher."""
    return _empty_with_layout(value)


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    scale: float,
) -> Tensor:
    """Describe the output-only stateful forward launcher."""
    return _empty_with_layout(value)


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_fwd_with_state")
def _packed_fwd_with_state_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Describe the forward output and launcher-cloned final state."""
    return _empty_with_layout(value), _empty_with_layout(initial_state)


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_fwd_paged")
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


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_bwd")
def _packed_bwd_fake(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    d_output: Tensor,
    cu_seqlens: Tensor,
    split: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Describe the five no-state backward gradients."""
    return tuple(_gradient_allocation(tensor) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::gdn_chunk_cudnn_packed_bwd_with_state")
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


chunk_gdn_warpdrive_packed_fwd_op = torch.ops.attn_gym.gdn_chunk_cudnn_packed_fwd.default
chunk_gdn_warpdrive_packed_fwd_with_initial_state_op = (
    torch.ops.attn_gym.gdn_chunk_cudnn_packed_fwd_with_initial_state.default
)
chunk_gdn_warpdrive_packed_fwd_with_state_op = (
    torch.ops.attn_gym.gdn_chunk_cudnn_packed_fwd_with_state.default
)
chunk_gdn_warpdrive_packed_fwd_paged_op = (
    torch.ops.attn_gym.gdn_chunk_cudnn_packed_fwd_paged.default
)
chunk_gdn_warpdrive_packed_bwd_op = torch.ops.attn_gym.gdn_chunk_cudnn_packed_bwd.default
chunk_gdn_warpdrive_packed_bwd_with_state_op = (
    torch.ops.attn_gym.gdn_chunk_cudnn_packed_bwd_with_state.default
)


__all__ = [
    "chunk_gdn_warpdrive_packed_bwd_op",
    "chunk_gdn_warpdrive_packed_bwd_with_state_op",
    "chunk_gdn_warpdrive_packed_fwd_op",
    "chunk_gdn_warpdrive_packed_fwd_paged_op",
    "chunk_gdn_warpdrive_packed_fwd_with_initial_state_op",
    "chunk_gdn_warpdrive_packed_fwd_with_state_op",
    "validate_warpdrive_available",
]
