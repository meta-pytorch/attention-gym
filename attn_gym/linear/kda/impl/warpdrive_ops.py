# SPDX-License-Identifier: BSD-3-Clause

"""Private KDA operators around the shared warpdrive launchers."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch

from attn_gym._backends.cute import normalize_compact_tensor, normalize_tma_tensor
from attn_gym.linear._delta_rule.operand_layout import (
    normalize_cu_seqlens,
    normalize_scalar_rows,
)
from attn_gym.linear._delta_rule.paged_state import PagedState
from attn_gym.linear._delta_rule.state_checkpoints import (
    state_checkpoints_from_op_args,
    validate_state_checkpoints,
)
from attn_gym.linear._delta_rule.validation import validate_packed_offsets_tensor

# The registered op names keep the backend's former ``cudnn`` name so that serialized graphs
# and exported programs that reference them still load. Append new arguments so old positional
# calls still bind. Defaults keep the original operand precision for the Q/K/V dtype and the
# shape-dependent work plans, not the historical rounding of backward intermediates.
# ``exact_recompute=False`` lets small packs share the prep, as those graphs did.
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    'bool split, float scale, bool fastmath=True, str intra_chunk_dtypes="bf16", '
    "bool batch_invariant=False, bool exact_recompute=False) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale, bool fastmath=True, "
    'str intra_chunk_dtypes="bf16", bool batch_invariant=False, bool exact_recompute=False) '
    "-> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale, bool fastmath=True, "
    'bool nonempty=False, str intra_chunk_dtypes="bf16", bool batch_invariant=False, '
    "bool exact_recompute=False) -> (Tensor, Tensor)",
)
# The exact-recompute forward of a backward that reads its piece entry states (NOTE [Piece entry
# states] in the warpdrive KDA driver): ``entry_rows`` sizes them (``entry_state_rows``); an
# absent ``initial_state`` is zero, and without ``output_final_state`` the final state is empty.
torch.library.define(
    "attn_gym::kda_chunk_warpdrive_packed_fwd_entry_states",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor? initial_state, "
    "Tensor cu_seqlens, float scale, bool fastmath, bool nonempty, bool output_final_state, "
    "str intra_chunk_dtypes, SymInt entry_rows) -> (Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor cu_seqlens, float scale, "
    'str intra_chunk_dtypes="bf16", bool batch_invariant=False, Tensor(b!)? checkpoint_states=None, '
    "Tensor? checkpoint_positions=None, Tensor? checkpoint_slots=None) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_local_bwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, bool split, float scale, bool fastmath=True, "
    'str intra_chunk_dtypes="bf16", bool batch_invariant=False, Tensor? entry_states=None) '
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
# Fixed-arity pair over one launcher: only the with-state backward returns the entry cotangent.
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor initial_state, Tensor? d_final_state, float scale, "
    'bool fastmath=True, str intra_chunk_dtypes="bf16", bool batch_invariant=False, '
    "Tensor? entry_states=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor? d_final_state, float scale, bool fastmath=True, "
    'str intra_chunk_dtypes="bf16", bool batch_invariant=False, Tensor? entry_states=None) '
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
# fake repeats the schema defaults.
def _packed_fwd_cuda(
    q,
    k,
    value,
    gate,
    beta,
    cu_seqlens,
    split,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
):
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
        intra_chunk_dtypes=intra_chunk_dtypes,
        batch_invariant=batch_invariant,
        exact_recompute=exact_recompute,
    )
    return _restore_layout(output, value, template)


def _packed_fwd_with_initial_state_cuda(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
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
        intra_chunk_dtypes=intra_chunk_dtypes,
        batch_invariant=batch_invariant,
        exact_recompute=exact_recompute,
    )
    return _restore_layout(output, value, template)


def _packed_fwd_with_state_cuda(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath=True,
    nonempty=False,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
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
        intra_chunk_dtypes=intra_chunk_dtypes,
        nonempty=nonempty,
        batch_invariant=batch_invariant,
        exact_recompute=exact_recompute,
    )
    assert final_state is not None
    return (
        _restore_layout(output, value, template),
        _restore_layout(final_state, initial_state, state_template, torch.empty_like),
    )


def _entry_state_shape(q, value, rows):
    # Lazy like the launchers: ``import attn_gym.linear`` loads no warpdrive module. This one
    # imports nothing else, so fake implementations work without CuTeDSL.
    from attn_gym.linear._delta_rule.warpdrive.drivers.entry_states import entry_state_shape

    return entry_state_shape(rows, value.shape[2], value.shape[3], q.shape[3])


def _packed_fwd_entry_states_cuda(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath,
    nonempty,
    output_final_state,
    intra_chunk_dtypes,
    entry_rows,
):
    if output_final_state and initial_state is None:
        shape = (cu_seqlens.shape[0] - 1, value.shape[2], value.shape[3], q.shape[3])
        initial_state = torch.zeros(shape, dtype=torch.float32, device=q.device)
    template, state_template = value, initial_state
    q, k, value, gate, beta, cu_seqlens, initial_state = _normalize_operands(
        q, k, value, gate, beta, cu_seqlens, initial_state
    )
    shape = _entry_state_shape(q, value, entry_rows)
    entry_states = torch.empty(shape, dtype=torch.float32, device=q.device)
    output, final_state = _backend(q).run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=output_final_state,
        fastmath=fastmath,
        intra_chunk_dtypes=intra_chunk_dtypes,
        nonempty=nonempty,
        exact_recompute=True,
        entry_states=entry_states,
    )
    if final_state is None:
        final_state = q.new_empty((0,), dtype=torch.float32)
    else:
        final_state = _restore_layout(final_state, initial_state, state_template, torch.empty_like)
    return _restore_layout(output, value, template), final_state, entry_states


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
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    checkpoint_states=None,
    checkpoint_positions=None,
    checkpoint_slots=None,
):
    template = value
    # The pool is advanced in place, so it is validated rather than copied.
    q, k, value, gate, beta, cu_seqlens = _normalize_operands(q, k, value, gate, beta, cu_seqlens)
    validate_packed_offsets_tensor(cu_seqlens, q.device)
    backend = _backend(q)
    checkpoints = state_checkpoints_from_op_args(
        checkpoint_states, checkpoint_positions, checkpoint_slots
    )
    validate_state_checkpoints(checkpoints, q, value, cu_seqlens)
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
        checkpoints=checkpoints,
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
        intra_chunk_dtypes=intra_chunk_dtypes,
        batch_invariant=batch_invariant,
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
    intra_chunk_dtypes,
    batch_invariant,
    entry_states=None,
):
    from attn_gym.linear._delta_rule.warpdrive.launchers.backward import (
        chunk_delta_rule_bwd_warpdrive_packed,
    )

    templates = (q, k, value, gate, beta)
    *normalized, cu_seqlens, d_output, initial_state, d_final_state = _normalize_operands(
        q, k, value, gate, beta, cu_seqlens, d_output, initial_state, d_final_state
    )
    if entry_states is not None:
        # The kernels read compact rows; a saved-tensor hook may restore other strides.
        entry_states = normalize_compact_tensor(entry_states, alignment_bytes=16)
    *gradients, d_initial_state = chunk_delta_rule_bwd_warpdrive_packed(
        *normalized,
        d_output,
        cu_seqlens,
        scale=scale,
        split=split,
        initial_state=initial_state,
        d_final_state=d_final_state,
        fastmath=fastmath,
        intra_chunk_dtypes=intra_chunk_dtypes,
        batch_invariant=batch_invariant,
        entry_states=entry_states,
    )
    return (
        *(
            _restore_layout(*operand)
            for operand in zip(gradients, normalized, templates, strict=True)
        ),
        d_initial_state,
    )


def _packed_local_bwd_cuda(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    split,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
):
    return _packed_bwd_cuda(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        None,
        None,
        scale,
        split,
        fastmath,
        intra_chunk_dtypes,
        batch_invariant,
        entry_states,
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
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
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
        intra_chunk_dtypes,
        batch_invariant,
        entry_states,
    )


def _packed_bwd_with_exit_cotangent_cuda(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    d_final_state,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
):
    return _packed_bwd_cuda(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        None,
        d_final_state,
        scale,
        False,
        fastmath,
        intra_chunk_dtypes,
        batch_invariant,
        entry_states,
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
torch.library.impl(
    "attn_gym::kda_chunk_warpdrive_packed_fwd_entry_states", "CUDA", _packed_fwd_entry_states_cuda
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
def _packed_fwd_fake(
    q,
    k,
    value,
    gate,
    beta,
    cu_seqlens,
    split,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
):
    del q, k, gate, beta, cu_seqlens, split, scale, intra_chunk_dtypes, fastmath, batch_invariant
    del exact_recompute
    return _empty_like_token_operand(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
):
    del q, k, gate, beta, initial_state, cu_seqlens, scale, intra_chunk_dtypes, fastmath
    del batch_invariant, exact_recompute
    return _empty_like_token_operand(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_state")
def _packed_fwd_with_state_fake(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath=True,
    nonempty=False,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    exact_recompute=False,
):
    del q, k, gate, beta, cu_seqlens, scale, intra_chunk_dtypes, fastmath, nonempty
    del batch_invariant, exact_recompute
    return _empty_like_token_operand(value), torch.empty_like(initial_state)


@torch.library.register_fake("attn_gym::kda_chunk_warpdrive_packed_fwd_entry_states")
def _packed_fwd_entry_states_fake(
    q,
    k,
    value,
    gate,
    beta,
    initial_state,
    cu_seqlens,
    scale,
    fastmath,
    nonempty,
    output_final_state,
    intra_chunk_dtypes,
    entry_rows,
):
    final_state = q.new_empty((0,), dtype=torch.float32)
    if output_final_state:
        state = (cu_seqlens.shape[0] - 1, value.shape[2], value.shape[3], q.shape[3])
        final_state = (
            q.new_empty(state, dtype=torch.float32)
            if initial_state is None
            else torch.empty_like(initial_state)
        )
    entry_states = q.new_empty(_entry_state_shape(q, value, entry_rows), dtype=torch.float32)
    return _empty_like_token_operand(value), final_state, entry_states


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
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    checkpoint_states=None,
    checkpoint_positions=None,
    checkpoint_slots=None,
):
    return _empty_like_token_operand(value)


def _token_gradients_fake(q, k, value, gate, beta):
    """One gradient per token operand, as the backward launchers allocate them."""
    return tuple(_empty_like_token_operand(tensor) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_local_bwd")
def _packed_local_bwd_fake(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    split,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
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
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
):
    # The launcher allocates a compact FP32 cotangent whatever the entry state's outer strides.
    return (
        *_token_gradients_fake(q, k, value, gate, beta),
        initial_state.new_empty(initial_state.shape),
    )


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent")
def _packed_bwd_with_exit_cotangent_fake(
    q,
    k,
    value,
    gate,
    beta,
    d_output,
    cu_seqlens,
    d_final_state,
    scale,
    fastmath=True,
    intra_chunk_dtypes="bf16",
    batch_invariant=False,
    entry_states=None,
):
    return _token_gradients_fake(q, k, value, gate, beta)


chunk_warpdrive_packed_fwd_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd.default
chunk_warpdrive_packed_fwd_paged_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_paged.default
chunk_warpdrive_packed_fwd_entry_states_op = (
    torch.ops.attn_gym.kda_chunk_warpdrive_packed_fwd_entry_states.default
)
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
    "chunk_warpdrive_packed_fwd_entry_states_op",
    "chunk_warpdrive_packed_fwd_op",
    "chunk_warpdrive_packed_fwd_paged_op",
    "chunk_warpdrive_packed_fwd_with_initial_state_op",
    "chunk_warpdrive_packed_fwd_with_state_op",
    "chunk_warpdrive_packed_local_bwd_op",
    "validate_warpdrive_available",
]
