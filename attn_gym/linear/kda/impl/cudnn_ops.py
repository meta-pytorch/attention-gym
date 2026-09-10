# SPDX-License-Identifier: BSD-3-Clause

"""Private KDA operators around the shared CuTeDSL 4.7 cuDNN launchers."""

from __future__ import annotations

import importlib

import torch

from attn_gym.linear._delta_rule.paged_state import PagedState

torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "bool split, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_local_bwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, bool split, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
# Fixed-arity pair over one launcher: only the with-state backward returns the entry cotangent.
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor initial_state, Tensor? d_final_state, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, Tensor? d_final_state, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _backend(q: torch.Tensor):
    """Import and preflight the optional backend before launching asynchronous work."""
    try:
        backend = importlib.import_module("attn_gym.linear._delta_rule.cudnn.forward")
    except ImportError as error:
        raise ImportError("the cuDNN KDA backend requires CUDA and attn-gym[cudnn]") from error
    backend.validate_available(q)
    return backend


def validate_cudnn_available(q: torch.Tensor) -> None:
    """Fail before caller-side setup when the optional backend cannot run."""
    _backend(q)


def _packed_fwd_cuda(q, k, value, gate, beta, cu_seqlens, split, scale):
    backend = _backend(q)
    return backend.chunk_delta_rule_fwd_cudnn(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        scale,
        split=split,
    )


def _packed_fwd_with_initial_state_cuda(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    backend = _backend(q)
    return backend.chunk_delta_rule_fwd_cudnn_unsplit_with_initial_state(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
    )


def _packed_fwd_with_state_cuda(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    backend = _backend(q)
    return backend.chunk_delta_rule_fwd_cudnn_unsplit_with_state(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
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
    return output


def _packed_local_bwd_cuda(q, k, value, gate, beta, d_output, cu_seqlens, split, scale):
    from attn_gym.linear._delta_rule.cudnn.backward import chunk_delta_rule_bwd_cudnn_packed

    return chunk_delta_rule_bwd_cudnn_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        scale=scale,
        split=split,
    )[:5]


def _packed_bwd_with_state_cuda(
    q, k, value, gate, beta, d_output, cu_seqlens, initial_state, d_final_state, scale
):
    from attn_gym.linear._delta_rule.cudnn.backward import chunk_delta_rule_bwd_cudnn_packed

    return chunk_delta_rule_bwd_cudnn_packed(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        scale=scale,
        initial_state=initial_state,
        d_final_state=d_final_state,
    )


def _packed_bwd_with_exit_cotangent_cuda(
    q, k, value, gate, beta, d_output, cu_seqlens, d_final_state, scale
):
    return _packed_bwd_with_state_cuda(
        q, k, value, gate, beta, d_output, cu_seqlens, None, d_final_state, scale
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
def _packed_fwd_fake(q, k, value, gate, beta, cu_seqlens, split, scale):
    del q, k, gate, beta, cu_seqlens, split, scale
    return torch.empty_like(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    del q, k, gate, beta, initial_state, cu_seqlens, scale
    return torch.empty_like(value)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_fwd_with_state")
def _packed_fwd_with_state_fake(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    del q, k, gate, beta, cu_seqlens, scale
    return torch.empty_like(value), torch.empty_like(initial_state)


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
    return torch.empty_like(value)


def _token_gradients_fake(q, k, value, gate, beta):
    """One compact gradient per token operand, as the backward launchers allocate them."""
    return tuple(torch.empty_like(tensor[0]).unsqueeze(0) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_local_bwd")
def _packed_local_bwd_fake(q, k, value, gate, beta, d_output, cu_seqlens, split, scale):
    return _token_gradients_fake(q, k, value, gate, beta)


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_bwd_with_state")
def _packed_bwd_with_state_fake(
    q, k, value, gate, beta, d_output, cu_seqlens, initial_state, d_final_state, scale
):
    # The launcher allocates a compact FP32 cotangent whatever the entry state's outer strides.
    return (
        *_token_gradients_fake(q, k, value, gate, beta),
        initial_state.new_empty(initial_state.shape),
    )


@torch.library.register_fake("attn_gym::kda_chunk_cudnn_packed_bwd_with_exit_cotangent")
def _packed_bwd_with_exit_cotangent_fake(
    q, k, value, gate, beta, d_output, cu_seqlens, d_final_state, scale
):
    return _token_gradients_fake(q, k, value, gate, beta)


chunk_cudnn_packed_fwd_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd.default
chunk_cudnn_packed_fwd_paged_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_paged.default
chunk_cudnn_packed_fwd_with_initial_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_with_initial_state.default
)
chunk_cudnn_packed_fwd_with_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_fwd_with_state.default
)
chunk_cudnn_packed_local_bwd_op = torch.ops.attn_gym.kda_chunk_cudnn_packed_local_bwd.default
chunk_cudnn_packed_bwd_with_state_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_bwd_with_state.default
)
chunk_cudnn_packed_bwd_with_exit_cotangent_op = (
    torch.ops.attn_gym.kda_chunk_cudnn_packed_bwd_with_exit_cotangent.default
)


__all__ = [
    "chunk_cudnn_packed_bwd_with_exit_cotangent_op",
    "chunk_cudnn_packed_bwd_with_state_op",
    "chunk_cudnn_packed_fwd_op",
    "chunk_cudnn_packed_fwd_paged_op",
    "chunk_cudnn_packed_fwd_with_initial_state_op",
    "chunk_cudnn_packed_fwd_with_state_op",
    "chunk_cudnn_packed_local_bwd_op",
    "validate_cudnn_available",
]
