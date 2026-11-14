# SPDX-License-Identifier: BSD-3-Clause

"""Private KDA operators around the shared warpdrive launchers."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch

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
    "attn_gym::kda_chunk_cudnn_packed_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "bool split, float scale, bool fastmath=True) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale, bool fastmath=True) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale, bool fastmath=True) "
    "-> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_local_bwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, bool split, float scale, bool fastmath=True) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
# Fixed-arity pair over one launcher: only the with-state backward returns the entry cotangent.
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor initial_state, Tensor? d_final_state, float scale, "
    "bool fastmath=True) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor? d_final_state, float scale, bool fastmath=True) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _backend(q: torch.Tensor):
    """Import and preflight the optional backend before launching asynchronous work."""
    try:
        backend = importlib.import_module(
            "attn_gym.linear._delta_rule.warpdrive.launchers.forward"
        )
    except ImportError as error:
        raise ImportError(
            "the warpdrive KDA backend requires CUDA and attn-gym[linear]"
        ) from error
    backend.validate_available(q)
    return backend


def validate_warpdrive_available(q: torch.Tensor) -> None:
    """Fail before caller-side setup when the optional backend cannot run."""
    _backend(q)


def _normalize_operands(q, k, value, gate, beta, cu_seqlens, *states):
    """Copy only the operands whose layout the launchers reject; ``states`` may hold ``None``."""
    return (
        *(normalize_tma_tensor(tensor) for tensor in (q, k, value, gate)),
        normalize_scalar_rows(beta),
        normalize_cu_seqlens(cu_seqlens),
        *(None if state is None else normalize_tma_tensor(state) for state in states),
    )


def _empty_like_token_operand(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate a ``[1, T, ...]`` result as the launchers do: like the operand's token rows."""
    return torch.empty_like(tensor[0]).unsqueeze(0)


def _restore_layout(
    result: torch.Tensor,
    normalized: torch.Tensor,
    template: torch.Tensor,
    allocate: Callable[[torch.Tensor], torch.Tensor] = _empty_like_token_operand,
) -> torch.Tensor:
    """Give a result computed from a normalized operand the layout ``allocate(template)`` that
    the fake declares. The launchers allocate results like their operand, so a result whose
    operand was not copied already has it. The expected strides come from a meta-device
    allocation, so storage is allocated only when the layout differs and a copy is needed."""
    if normalized is template:
        return result
    expected = allocate(template.to("meta")).stride()
    if result.stride() == expected:
        return result
    out = torch.empty_strided(result.shape, expected, dtype=result.dtype, device=result.device)
    return out.copy_(result)


# The dispatcher drops trailing arguments left at their schema default, so every Python impl and
# fake repeats ``fastmath=True``.
def _packed_fwd_cuda(q, k, value, gate, beta, cu_seqlens, split, scale, fastmath=True):
    template = value
    q, k, value, gate, beta, cu_seqlens = _normalize_operands(q, k, value, gate, beta, cu_seqlens)
    output, _ = _backend(q).run_forward(
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
        fastmath=fastmath,
    )
    return _restore_layout(output, value, template)


def _packed_fwd_with_initial_state_cuda(
    q, k, value, gate, beta, initial_state, cu_seqlens, scale, fastmath=True
):
    template = value
    q, k, value, gate, beta, cu_seqlens, initial_state = _normalize_operands(
        q, k, value, gate, beta, cu_seqlens, initial_state
    )
    output, _ = _backend(q).run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=False,
        fastmath=fastmath,
    )
    return _restore_layout(output, value, template)


def _packed_fwd_with_state_cuda(
    q, k, value, gate, beta, initial_state, cu_seqlens, scale, fastmath=True
):
    template, state_template = value, initial_state
    q, k, value, gate, beta, cu_seqlens, initial_state = _normalize_operands(
        q, k, value, gate, beta, cu_seqlens, initial_state
    )
    output, final_state = _backend(q).run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=True,
        fastmath=fastmath,
    )
    assert final_state is not None
    return (
        _restore_layout(output, value, template),
        _restore_layout(final_state, initial_state, state_template, torch.empty_like),
    )


def _packed_fwd_paged_cuda(
    q,
    k,
    value,
    gate,
    beta,
    state_cache,
    state_indices,
    has_initial_state,
    cu_seqlens,
    scale,
):
    template = value
    # The pool is advanced in place, so it is validated rather than copied.
    q, k, value, gate, beta, cu_seqlens = _normalize_operands(q, k, value, gate, beta, cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
    backend = _backend(q)
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
    return _restore_layout(output, value, template)


def _packed_bwd_cuda(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    initial_state,
    d_final_state,
    scale,
    split,
    fastmath,
):
    from attn_gym.linear._delta_rule.warpdrive.launchers.backward import (
        chunk_delta_rule_bwd_warpdrive_packed,
    )

    templates = (q, k, value, gate, beta)
    *normalized, cu_seqlens, d_output, initial_state, d_final_state = _normalize_operands(
        q, k, value, gate, beta, cu_seqlens, d_output, initial_state, d_final_state
    )
    *gradients, d_initial_state = chunk_delta_rule_bwd_warpdrive_packed(
        *normalized,
        d_output,
        cu_seqlens,
        scale=scale,
        split=split,
        initial_state=initial_state,
        d_final_state=d_final_state,
        fastmath=fastmath,
    )
    return (
        *(
            _restore_layout(*operand)
            for operand in zip(gradients, normalized, templates, strict=True)
        ),
        d_initial_state,
    )


def _packed_local_bwd_cuda(
    q, k, value, gate, beta, d_output, cu_seqlens, split, scale, fastmath=True
):
    return _packed_bwd_cuda(
        q, k, value, gate, beta, d_output, cu_seqlens, None, None, scale, split, fastmath
    )[:5]


def _packed_bwd_with_state_cuda(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    initial_state,
    d_final_state,
    scale,
    fastmath=True,
):
    return _packed_bwd_cuda(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        initial_state,
        d_final_state,
        scale,
        False,
        fastmath,
    )


def _packed_bwd_with_exit_cotangent_cuda(
    q, k, value, gate, beta, d_output, cu_seqlens, d_final_state, scale, fastmath=True
):
    return _packed_bwd_cuda(
        q, k, value, gate, beta, d_output, cu_seqlens, None, d_final_state, scale, False, fastmath
    )[:5]


torch.library.impl("attn_gym::kda_chunk_cudnn_packed_fwd", "CUDA", _packed_fwd_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state",
    "CUDA",
    _packed_fwd_with_initial_state_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_state", "CUDA", _packed_fwd_with_state_cuda
)
torch.library.impl("attn_gym::kda_chunk_cudnn_packed_fwd_paged", "CUDA", _packed_fwd_paged_cuda)
torch.library.impl("attn_gym::kda_chunk_cudnn_packed_local_bwd", "CUDA", _packed_local_bwd_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_state", "CUDA", _packed_bwd_with_state_cuda
)
torch.library.impl(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent",
    "CUDA",
    _packed_bwd_with_exit_cotangent_cuda,
)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd")
def _packed_fwd_fake(q, k, value, gate, beta, cu_seqlens, split, scale, fastmath=True):
    del q, k, gate, beta, cu_seqlens, split, scale, fastmath
    return _empty_like_token_operand(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(
    q, k, value, gate, beta, initial_state, cu_seqlens, scale, fastmath=True
):
    del q, k, gate, beta, initial_state, cu_seqlens, scale, fastmath
    return _empty_like_token_operand(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_state")
def _packed_fwd_with_state_fake(
    q, k, value, gate, beta, initial_state, cu_seqlens, scale, fastmath=True
):
    del q, k, gate, beta, cu_seqlens, scale, fastmath
    return _empty_like_token_operand(value), torch.empty_like(initial_state)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_paged")
def _packed_fwd_paged_fake(
    q,
    k,
    value,
    gate,
    beta,
    state_cache,
    state_indices,
    has_initial_state,
    cu_seqlens,
    scale,
):
    return _empty_like_token_operand(value)


def _token_gradients_fake(q, k, value, gate, beta):
    """One gradient per token operand, as the backward launchers allocate them."""
    return tuple(_empty_like_token_operand(tensor) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_local_bwd")
def _packed_local_bwd_fake(
    q, k, value, gate, beta, d_output, cu_seqlens, split, scale, fastmath=True
):
    return _token_gradients_fake(q, k, value, gate, beta)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_bwd_with_state")
def _packed_bwd_with_state_fake(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    initial_state,
    d_final_state,
    scale,
    fastmath=True,
):
    # The launcher allocates a compact FP32 cotangent whatever the entry state's outer strides.
    return (
        *_token_gradients_fake(q, k, value, gate, beta),
        initial_state.new_empty(initial_state.shape),
    )


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent")
def _packed_bwd_with_exit_cotangent_fake(
    q, k, value, gate, beta, d_output, cu_seqlens, d_final_state, scale, fastmath=True
):
    return _token_gradients_fake(q, k, value, gate, beta)


chunk_warpdrive_packed_fwd_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd.default
chunk_warpdrive_packed_fwd_paged_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_paged.default
chunk_warpdrive_packed_fwd_with_initial_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_with_initial_state.default
)
chunk_warpdrive_packed_fwd_with_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_with_state.default
)
chunk_warpdrive_packed_local_bwd_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_local_bwd.default
chunk_warpdrive_packed_bwd_with_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_bwd_with_state.default
)
chunk_warpdrive_packed_bwd_with_exit_cotangent_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_bwd_with_exit_cotangent.default
)


__all__ = [
    "chunk_warpdrive_packed_bwd_with_exit_cotangent_op",
    "chunk_warpdrive_packed_bwd_with_state_op",
    "chunk_warpdrive_packed_fwd_op",
    "chunk_warpdrive_packed_fwd_paged_op",
    "chunk_warpdrive_packed_fwd_with_initial_state_op",
    "chunk_warpdrive_packed_fwd_with_state_op",
    "chunk_warpdrive_packed_local_bwd_op",
    "validate_warpdrive_available",
]
