# SPDX-License-Identifier: BSD-3-Clause

"""Private KDA operators around the shared CuTeDSL 4.7 Mega launchers."""

from __future__ import annotations

import importlib

import torch

from attn_gym.utils import fork_join_streams

torch.library.define(
    "attn_gym::kda_chunk_mega_packed_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_mega_dense_training_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor cu_seqlens, "
    "float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_mega_packed_training_fwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor cu_seqlens, Tensor chunk_offsets, float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_mega_packed_fwd_with_initial_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_mega_packed_fwd_with_state",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, float scale) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_mega_packed_local_bwd",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor d_output, "
    "Tensor cu_seqlens, bool split, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_plain_gate_bwd_dense_cute",
    "(Tensor d_cumulative) -> Tensor",
)

_GATE_STREAMS: dict[int, torch.cuda.Stream] = {}
_MAIN_STREAMS: dict[int, torch.cuda.Stream] = {}


def _backend(q: torch.Tensor):
    """Import and preflight the optional backend before launching asynchronous work."""
    try:
        backend = importlib.import_module("attn_gym.linear._delta_rule.mega.forward")
    except ImportError as error:
        raise ImportError("the Mega KDA backend requires CUDA and attn-gym[mega]") from error
    backend.validate_available(q)
    return backend


def validate_mega_available(q: torch.Tensor) -> None:
    """Fail before caller-side setup when the optional backend cannot run."""
    _backend(q)


def _stream(cache: dict[int, torch.cuda.Stream], device: torch.device, priority: int = 0):
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = cache.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=index, priority=priority)
        cache[index] = stream
    return stream


def _packed_fwd_cuda(q, k, value, gate, beta, cu_seqlens, scale, *, backend=None):
    backend = _backend(q) if backend is None else backend
    return backend.chunk_delta_rule_fwd_mega_unsplit(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        scale,
    )


def _dense_training_fwd_cuda(q, k, value, gate, beta, cu_seqlens, scale):
    from attn_gym.linear.kda.fwd.triton.plain_gate import _plain_gate_scan_cuda

    backend = _backend(q)
    # Prioritize the persistent one-CTA-per-SM Mega grid; the scan can fill its drain tail.
    cumulative_gate, output = fork_join_streams(
        torch.cuda.current_stream(q.device),
        _stream(_MAIN_STREAMS, q.device, priority=-1),
        lambda: _packed_fwd_cuda(
            q,
            k,
            value,
            gate,
            beta,
            cu_seqlens,
            scale,
            backend=backend,
        ),
        lambda: _plain_gate_scan_cuda(gate, None, None, False),
    )
    return output, cumulative_gate


def _packed_training_fwd_cuda(q, k, value, gate, beta, cu_seqlens, chunk_offsets, scale):
    from attn_gym.linear.kda.fwd.triton.plain_gate import _plain_gate_scan_cuda

    backend = _backend(q)
    output, cumulative_gate = fork_join_streams(
        torch.cuda.current_stream(q.device),
        _stream(_GATE_STREAMS, q.device),
        lambda: _plain_gate_scan_cuda(gate, cu_seqlens, chunk_offsets, False),
        lambda: _packed_fwd_cuda(
            q,
            k,
            value,
            gate,
            beta,
            cu_seqlens,
            scale,
            backend=backend,
        ),
    )
    return output, cumulative_gate


def _packed_fwd_with_initial_state_cuda(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    backend = _backend(q)
    return backend.chunk_delta_rule_fwd_mega_unsplit_with_initial_state(
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
    return backend.chunk_delta_rule_fwd_mega_unsplit_with_state(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
    )


def _packed_local_bwd_cuda(q, k, value, gate, beta, d_output, cu_seqlens, split, scale):
    from attn_gym.linear._delta_rule.mega.backward import chunk_delta_rule_bwd_mega_packed

    return chunk_delta_rule_bwd_mega_packed(
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


def _plain_gate_bwd_cuda(d_cumulative):
    from attn_gym.linear._delta_rule.mega.kernels.kda_plain_gate_bwd import (
        plain_gate_cumsum_dense_bwd_cute,
    )

    return plain_gate_cumsum_dense_bwd_cute(d_cumulative)


torch.library.impl("attn_gym::kda_chunk_mega_packed_fwd", "CUDA", _packed_fwd_cuda)
torch.library.impl("attn_gym::kda_chunk_mega_dense_training_fwd", "CUDA", _dense_training_fwd_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_mega_packed_training_fwd", "CUDA", _packed_training_fwd_cuda
)
torch.library.impl(
    "attn_gym::kda_chunk_mega_packed_fwd_with_initial_state",
    "CUDA",
    _packed_fwd_with_initial_state_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_mega_packed_fwd_with_state", "CUDA", _packed_fwd_with_state_cuda
)
torch.library.impl("attn_gym::kda_chunk_mega_packed_local_bwd", "CUDA", _packed_local_bwd_cuda)
torch.library.impl("attn_gym::kda_plain_gate_bwd_dense_cute", "CUDA", _plain_gate_bwd_cuda)


@torch.library.register_fake("attn_gym::kda_chunk_mega_packed_fwd")
def _packed_fwd_fake(q, k, value, gate, beta, cu_seqlens, scale):
    del q, k, gate, beta, cu_seqlens, scale
    return torch.empty_like(value)


@torch.library.register_fake("attn_gym::kda_chunk_mega_dense_training_fwd")
def _dense_training_fwd_fake(q, k, value, gate, beta, cu_seqlens, scale):
    del q, k, beta, cu_seqlens, scale
    return torch.empty_like(value), gate.new_empty(gate.shape)


@torch.library.register_fake("attn_gym::kda_chunk_mega_packed_training_fwd")
def _packed_training_fwd_fake(q, k, value, gate, beta, cu_seqlens, chunk_offsets, scale):
    del q, k, beta, cu_seqlens, chunk_offsets, scale
    return torch.empty_like(value), gate.new_empty(gate.shape)


@torch.library.register_fake("attn_gym::kda_chunk_mega_packed_fwd_with_initial_state")
def _packed_fwd_with_initial_state_fake(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    del q, k, gate, beta, initial_state, cu_seqlens, scale
    return torch.empty_like(value)


@torch.library.register_fake("attn_gym::kda_chunk_mega_packed_fwd_with_state")
def _packed_fwd_with_state_fake(q, k, value, gate, beta, initial_state, cu_seqlens, scale):
    del q, k, gate, beta, cu_seqlens, scale
    return torch.empty_like(value), torch.empty_like(initial_state)


@torch.library.register_fake("attn_gym::kda_chunk_mega_packed_local_bwd")
def _packed_local_bwd_fake(q, k, value, gate, beta, d_output, cu_seqlens, split, scale):
    del d_output, cu_seqlens, split, scale
    return tuple(torch.empty_like(tensor[0]).unsqueeze(0) for tensor in (q, k, value, gate, beta))


@torch.library.register_fake("attn_gym::kda_plain_gate_bwd_dense_cute")
def _plain_gate_bwd_fake(d_cumulative):
    return torch.empty_like(d_cumulative)


chunk_mega_packed_fwd_op = torch.ops.attn_gym.kda_chunk_mega_packed_fwd.default
chunk_mega_dense_training_fwd_op = torch.ops.attn_gym.kda_chunk_mega_dense_training_fwd.default
chunk_mega_packed_training_fwd_op = torch.ops.attn_gym.kda_chunk_mega_packed_training_fwd.default
chunk_mega_packed_fwd_with_initial_state_op = (
    torch.ops.attn_gym.kda_chunk_mega_packed_fwd_with_initial_state.default
)
chunk_mega_packed_fwd_with_state_op = (
    torch.ops.attn_gym.kda_chunk_mega_packed_fwd_with_state.default
)
chunk_mega_packed_local_bwd_op = torch.ops.attn_gym.kda_chunk_mega_packed_local_bwd.default
plain_gate_bwd_dense_cute_op = torch.ops.attn_gym.kda_plain_gate_bwd_dense_cute.default


__all__ = [
    "chunk_mega_dense_training_fwd_op",
    "chunk_mega_packed_fwd_op",
    "chunk_mega_packed_fwd_with_initial_state_op",
    "chunk_mega_packed_fwd_with_state_op",
    "chunk_mega_packed_local_bwd_op",
    "chunk_mega_packed_training_fwd_op",
    "plain_gate_bwd_dense_cute_op",
    "validate_mega_available",
]
