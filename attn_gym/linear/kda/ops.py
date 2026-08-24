"""Torch-only private operator contracts for fused KDA backends.

Schemas, fake implementations, and dispatch registrations live here so they
exist before graph capture. CUDA implementations import their optional backend
only when the dispatcher executes the operator.
"""

from __future__ import annotations

import importlib

import torch

from attn_gym.utils import ceildiv

_BOUND_GATE_TILE_TOKENS = 32
_CHUNK_SIZE = 64


# Fixed-arity schema pairs avoid optional outputs on hot paths.
_CHUNK_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor? initial_state,"
    " bool autotune)"
)
torch.library.define(
    "attn_gym::kda_chunk_fwd",
    f"{_CHUNK_FWD_ARGS} -> (Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_with_state",
    f"{_CHUNK_FWD_ARGS} -> (Tensor, Tensor, Tensor, Tensor)",
)

_CHUNK_RAGGED_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? initial_state, Tensor cu_seqlens, Tensor chunk_offsets, bool autotune)"
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged",
    f"{_CHUNK_RAGGED_FWD_ARGS} -> (Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged_with_state",
    f"{_CHUNK_RAGGED_FWD_ARGS} -> (Tensor, Tensor, Tensor, Tensor)",
)

_CHUNK_BWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor Aqk, "
    "Tensor Akk, Tensor? cu_seqlens, Tensor? chunk_offsets, Tensor? d_output, "
    "Tensor? d_final_state, {initial_state}, bool fastmath, bool autotune)"
)
torch.library.define(
    "attn_gym::kda_chunk_bwd",
    _CHUNK_BWD_ARGS.format(initial_state="Tensor? initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_bwd_with_state_grad",
    _CHUNK_BWD_ARGS.format(initial_state="Tensor initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)

_CHUNK_BWD_RECOMPUTE_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? cu_seqlens, Tensor? chunk_offsets, Tensor? d_output, "
    "Tensor? d_final_state, {initial_state}, bool fastmath, bool autotune)"
)
torch.library.define(
    "attn_gym::kda_chunk_bwd_recompute_factors",
    _CHUNK_BWD_RECOMPUTE_ARGS.format(initial_state="Tensor? initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_bwd_recompute_factors_with_state_grad",
    _CHUNK_BWD_RECOMPUTE_ARGS.format(initial_state="Tensor initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)

torch.library.define(
    "attn_gym::_kda_plain_gate_scan",
    "(Tensor values, Tensor? cu_seqlens, Tensor? chunk_offsets, bool reverse) -> Tensor",
)
torch.library.define(
    "attn_gym::_kda_bound_gate_fwd",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, float lower_bound, bool fastmath) -> Tensor",
)
torch.library.define(
    "attn_gym::_kda_bound_gate_bwd",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor d_gate, "
    "float lower_bound, bool fastmath) -> (Tensor, Tensor, Tensor)",
)

_RECURRENT_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta,"
    " Tensor? initial_state, Tensor? cu_seqlens, bool autotune)"
)
torch.library.define("attn_gym::kda_recurrent_fwd", _RECURRENT_FWD_ARGS + " -> (Tensor, Tensor)")
torch.library.define("attn_gym::kda_recurrent_fwd_no_state", _RECURRENT_FWD_ARGS + " -> Tensor")
# Separate schema: the paged variant advances the state pool in place, so the final state
# is not an output and the alias annotation has to declare the mutation.
torch.library.define(
    "attn_gym::kda_recurrent_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache,"
    " Tensor state_indices, Tensor? cu_seqlens, bool autotune) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_prepare_chunk_offsets",
    "(Tensor cu_seqlens, SymInt tokens, int chunk_size) -> Tensor",
)

_DELTA_H_ARGS = (
    "(Tensor k, Tensor w, Tensor u, Tensor gk, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_offsets, SymInt capacity)"
)
torch.library.define(
    "attn_gym::kda_delta_h",
    f"{_DELTA_H_ARGS} -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_delta_h_with_state",
    f"{_DELTA_H_ARGS} -> (Tensor, Tensor, Tensor)",
)

torch.library.define(
    "attn_gym::_cute_short_conv_fwd",
    "(Tensor x, Tensor weight, Tensor? cu_seqlens=None, Tensor? initial_state=None,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_decode",
    "(Tensor x, Tensor weight, Tensor(a!) state, Tensor? state_indices,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_decode",
    "(Tensor x, Tensor weight, Tensor(a!) state, Tensor? state_indices,"
    " int forward_threads, int forward_channels, int forward_times,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_bwd",
    "(Tensor x, Tensor weight, Tensor grad_output, Tensor? cu_seqlens=None,"
    " *, str? activation=None) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_fwd",
    "(Tensor x, Tensor weight, Tensor? cu_seqlens, Tensor? initial_state,"
    " int forward_threads, int forward_channels, int forward_times,"
    " int input_threads, int input_channels, int input_times,"
    " int weight_threads, int weight_channels, int weight_times,"
    " *, str? activation=None) -> Tensor",
)

_SHORT_CONV_CONFIGURED_BWD_ARGS = (
    "(Tensor x, Tensor weight, Tensor grad_output, Tensor? cu_seqlens, {initial_state},"
    " int input_threads, int input_channels, int input_times,"
    " int weight_threads, int weight_channels, int weight_times,"
    " bool persistent_tma_input_gradient, *, str? activation=None)"
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_bwd",
    _SHORT_CONV_CONFIGURED_BWD_ARGS.format(initial_state="Tensor? initial_state")
    + " -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_bwd_with_state_grad",
    _SHORT_CONV_CONFIGURED_BWD_ARGS.format(initial_state="Tensor initial_state")
    + " -> (Tensor, Tensor, Tensor)",
)


def _chunk_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.cute.chunk_kda_fwd")
    except ImportError as error:
        raise ImportError(
            "chunk_kda(impl='fused') requires the optional CuTeDSL backend: "
            "pip install attn-gym[linear]"
        ) from error


def _bound_gate_fwd_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.cute.gate_fwd")
    except ImportError as error:
        raise ImportError(
            "bound_gate(impl='fused') requires the optional CuTeDSL backend: "
            "pip install attn-gym[linear]"
        ) from error


def _bound_gate_bwd_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.bwd.cute.gate_bwd")
    except ImportError as error:
        raise ImportError(
            "bound_gate(impl='fused') requires the optional CuTeDSL backend: "
            "pip install attn-gym[linear]"
        ) from error


def _plain_gate_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.triton.plain_gate")
    except ImportError as error:
        raise ImportError("chunk_kda(impl='fused') requires CUDA with Triton support") from error


def _recurrent_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.triton.recurrent")
    except ImportError as error:
        raise ImportError(
            "recurrent_kda(impl='fused') requires CUDA with Triton support"
        ) from error


def _delta_h_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.triton.chunk_delta_h")
    except ImportError as error:
        raise ImportError("chunk_kda(impl='fused') requires CUDA with Triton support") from error


def _chunk_fwd_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_cuda(*args)


def _chunk_fwd_with_state_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_with_state_cuda(*args)


def _chunk_fwd_ragged_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_ragged_cuda(*args)


def _chunk_fwd_ragged_with_state_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_ragged_with_state_cuda(*args)


def _chunk_bwd_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_cuda(*args)


def _chunk_bwd_with_state_grad_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_with_state_grad_cuda(*args)


def _chunk_bwd_recompute_factors_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_recompute_factors_cuda(*args)


def _chunk_bwd_recompute_factors_with_state_grad_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_recompute_factors_with_state_grad_cuda(*args)


def _plain_gate_scan_cuda(*args):
    return _plain_gate_backend()._plain_gate_scan_cuda(*args)


def _bound_gate_fwd_cuda(*args):
    return _bound_gate_fwd_backend()._bound_gate_fwd_cuda(*args)


def _bound_gate_bwd_cuda(*args):
    return _bound_gate_bwd_backend()._bound_gate_bwd_cuda(*args)


def _recurrent_fwd_cuda(*args):
    return _recurrent_backend()._kda_recurrent_fwd_cuda(*args)


def _recurrent_fwd_no_state_cuda(*args):
    return _recurrent_backend()._kda_recurrent_fwd_no_state_cuda(*args)


def _recurrent_fwd_paged_cuda(*args):
    return _recurrent_backend()._kda_recurrent_fwd_paged_cuda(*args)


def _delta_h_cuda(*args):
    return _delta_h_backend()._delta_h_cuda(*args)


def _delta_h_with_state_cuda(*args):
    return _delta_h_backend()._delta_h_with_state_cuda(*args)


def _prepare_chunk_offsets_cuda(*args):
    from attn_gym.linear.kda.chunk_scheduler import _prepare_ragged_chunk_offsets

    return _prepare_ragged_chunk_offsets(*args)


torch.library.impl("attn_gym::kda_chunk_fwd", "CUDA", _chunk_fwd_cuda)
torch.library.impl("attn_gym::kda_chunk_fwd_with_state", "CUDA", _chunk_fwd_with_state_cuda)
torch.library.impl("attn_gym::kda_chunk_fwd_ragged", "CUDA", _chunk_fwd_ragged_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_fwd_ragged_with_state",
    "CUDA",
    _chunk_fwd_ragged_with_state_cuda,
)
torch.library.impl("attn_gym::kda_chunk_bwd", "CUDA", _chunk_bwd_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_bwd_with_state_grad",
    "CUDA",
    _chunk_bwd_with_state_grad_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_bwd_recompute_factors",
    "CUDA",
    _chunk_bwd_recompute_factors_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_bwd_recompute_factors_with_state_grad",
    "CUDA",
    _chunk_bwd_recompute_factors_with_state_grad_cuda,
)
torch.library.impl("attn_gym::_kda_plain_gate_scan", "CUDA", _plain_gate_scan_cuda)
torch.library.impl("attn_gym::_kda_bound_gate_fwd", "CUDA", _bound_gate_fwd_cuda)
torch.library.impl("attn_gym::_kda_bound_gate_bwd", "CUDA", _bound_gate_bwd_cuda)
torch.library.impl("attn_gym::kda_recurrent_fwd", "CUDA", _recurrent_fwd_cuda)
torch.library.impl(
    "attn_gym::kda_recurrent_fwd_no_state",
    "CUDA",
    _recurrent_fwd_no_state_cuda,
)
torch.library.impl(
    "attn_gym::kda_recurrent_fwd_paged",
    "CUDA",
    _recurrent_fwd_paged_cuda,
)
torch.library.impl(
    "attn_gym::kda_prepare_chunk_offsets",
    "CUDA",
    _prepare_chunk_offsets_cuda,
)
torch.library.impl("attn_gym::kda_delta_h", "CUDA", _delta_h_cuda)
torch.library.impl("attn_gym::kda_delta_h_with_state", "CUDA", _delta_h_with_state_cuda)


def _chunk_fwd_fake_common(q: torch.Tensor, v: torch.Tensor):
    factor_shape = (q.shape[0], q.shape[1], q.shape[2], _CHUNK_SIZE)
    return v.new_empty(v.shape), q.new_empty(factor_shape), q.new_empty(factor_shape)


@torch.library.register_fake("attn_gym::kda_chunk_fwd")
def _chunk_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    autotune: bool,
):
    del k, cumulative_gate, beta, initial_state, autotune
    return _chunk_fwd_fake_common(q, v)


@torch.library.register_fake("attn_gym::kda_chunk_fwd_with_state")
def _chunk_fwd_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    autotune: bool,
):
    del k, cumulative_gate, beta, initial_state, autotune
    output, aqk, akk = _chunk_fwd_fake_common(q, v)
    state = q.new_empty(
        (q.shape[0], q.shape[2], q.shape[3], v.shape[-1]),
        dtype=torch.float32,
    )
    return output, state, aqk, akk


@torch.library.register_fake("attn_gym::kda_chunk_fwd_ragged")
def _chunk_fwd_ragged_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
):
    del k, cumulative_gate, beta, initial_state, cu_seqlens, chunk_offsets, autotune
    return _chunk_fwd_fake_common(q, v)


@torch.library.register_fake("attn_gym::kda_chunk_fwd_ragged_with_state")
def _chunk_fwd_ragged_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
):
    del k, cumulative_gate, beta, initial_state, chunk_offsets, autotune
    output, aqk, akk = _chunk_fwd_fake_common(q, v)
    state = q.new_empty(
        (cu_seqlens.shape[0] - 1, q.shape[2], q.shape[3], v.shape[-1]),
        dtype=torch.float32,
    )
    return output, state, aqk, akk


def _chunk_bwd_fake_common(q, k, v, cumulative_gate, beta):
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
        cumulative_gate.new_empty(cumulative_gate.shape),
        beta.new_empty(beta.shape),
    )


@torch.library.register_fake("attn_gym::kda_chunk_bwd")
def _chunk_bwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    fastmath: bool,
    autotune: bool,
):
    del aqk, akk, cu_seqlens, chunk_offsets, d_output, d_final_state, initial_state
    del fastmath, autotune
    return _chunk_bwd_fake_common(q, k, v, cumulative_gate, beta)


@torch.library.register_fake("attn_gym::kda_chunk_bwd_with_state_grad")
def _chunk_bwd_with_state_grad_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    aqk: torch.Tensor,
    akk: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor,
    fastmath: bool,
    autotune: bool,
):
    del aqk, akk, cu_seqlens, chunk_offsets, d_output, d_final_state, fastmath, autotune
    return (
        *_chunk_bwd_fake_common(q, k, v, cumulative_gate, beta),
        torch.empty_like(initial_state),
    )


@torch.library.register_fake("attn_gym::kda_chunk_bwd_recompute_factors")
def _chunk_bwd_recompute_factors_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    fastmath: bool,
    autotune: bool,
):
    del cu_seqlens, chunk_offsets, d_output, d_final_state, initial_state, fastmath, autotune
    return _chunk_bwd_fake_common(q, k, v, cumulative_gate, beta)


@torch.library.register_fake("attn_gym::kda_chunk_bwd_recompute_factors_with_state_grad")
def _chunk_bwd_recompute_factors_with_state_grad_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor,
    fastmath: bool,
    autotune: bool,
):
    del cu_seqlens, chunk_offsets, d_output, d_final_state, fastmath, autotune
    return (
        *_chunk_bwd_fake_common(q, k, v, cumulative_gate, beta),
        torch.empty_like(initial_state),
    )


@torch.library.register_fake("attn_gym::_kda_plain_gate_scan")
def _plain_gate_scan_fake(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    reverse: bool,
) -> torch.Tensor:
    """Describe the compact internal gate scan output."""
    del cu_seqlens, chunk_offsets, reverse
    return torch.empty_like(values, memory_format=torch.contiguous_format)


@torch.library.register_fake("attn_gym::_kda_bound_gate_fwd")
def _bound_gate_fwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    fastmath: bool,
) -> torch.Tensor:
    """Describe the compact FP32 gate output."""
    del A_log, dt_bias, lower_bound, fastmath
    return torch.empty(raw_gate.shape, device=raw_gate.device, dtype=torch.float32)


@torch.library.register_fake("attn_gym::_kda_bound_gate_bwd")
def _bound_gate_bwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_gate: torch.Tensor,
    lower_bound: float,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe raw and reduced parameter-gradient metadata."""
    del A_log, lower_bound, fastmath
    partial_shape = (
        raw_gate.shape[0],
        ceildiv(raw_gate.shape[1], _BOUND_GATE_TILE_TOKENS),
        raw_gate.shape[2],
    )
    return (
        raw_gate.new_empty(raw_gate.shape),
        d_gate.new_empty(partial_shape),
        dt_bias.new_empty(dt_bias.shape),
    )


@torch.library.register_fake("attn_gym::kda_recurrent_fwd")
def _recurrent_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, gate, beta, initial_state, autotune
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = q.new_empty(
        num_sequences, q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float32
    )
    return torch.empty_like(v, dtype=q.dtype), final_state


@torch.library.register_fake("attn_gym::kda_recurrent_fwd_no_state")
def _recurrent_fwd_no_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> torch.Tensor:
    del k, gate, beta, initial_state, cu_seqlens, autotune
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.register_fake("attn_gym::kda_recurrent_fwd_paged")
def _recurrent_fwd_paged_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> torch.Tensor:
    del k, gate, beta, state_cache, state_indices, cu_seqlens, autotune
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.register_fake("attn_gym::kda_prepare_chunk_offsets")
def _prepare_chunk_offsets_fake(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> torch.Tensor:
    del tokens, chunk_size
    return torch.empty_like(cu_seqlens)


def _delta_h_fake_common(
    k: torch.Tensor, u: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor]:
    h = k.new_empty(k.shape[0], capacity, k.shape[2], k.shape[3], u.shape[-1])
    return h, u.new_empty(u.shape)


@torch.library.register_fake("attn_gym::kda_delta_h")
def _delta_h_fake(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del w, gk, initial_state, cu_seqlens, chunk_offsets
    return _delta_h_fake_common(k, u, capacity)


@torch.library.register_fake("attn_gym::kda_delta_h_with_state")
def _delta_h_with_state_fake(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del w, gk, initial_state, chunk_offsets
    h, v_new = _delta_h_fake_common(k, u, capacity)
    state_batch = k.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = k.new_empty(
        (state_batch, k.shape[2], k.shape[3], u.shape[-1]), dtype=torch.float32
    )
    return h, v_new, final_state


chunk_fwd_op = torch.ops.attn_gym.kda_chunk_fwd.default
chunk_fwd_with_state_op = torch.ops.attn_gym.kda_chunk_fwd_with_state.default
chunk_fwd_ragged_op = torch.ops.attn_gym.kda_chunk_fwd_ragged.default
chunk_fwd_ragged_with_state_op = torch.ops.attn_gym.kda_chunk_fwd_ragged_with_state.default
chunk_bwd_op = torch.ops.attn_gym.kda_chunk_bwd.default
chunk_bwd_with_state_grad_op = torch.ops.attn_gym.kda_chunk_bwd_with_state_grad.default
chunk_bwd_recompute_factors_op = torch.ops.attn_gym.kda_chunk_bwd_recompute_factors.default
chunk_bwd_recompute_factors_with_state_grad_op = (
    torch.ops.attn_gym.kda_chunk_bwd_recompute_factors_with_state_grad.default
)
_plain_gate_scan_op = torch.ops.attn_gym._kda_plain_gate_scan.default
_bound_gate_fwd_op = torch.ops.attn_gym._kda_bound_gate_fwd.default
_bound_gate_bwd_op = torch.ops.attn_gym._kda_bound_gate_bwd.default
recurrent_fwd_op = torch.ops.attn_gym.kda_recurrent_fwd.default
recurrent_fwd_no_state_op = torch.ops.attn_gym.kda_recurrent_fwd_no_state.default
recurrent_fwd_paged_op = torch.ops.attn_gym.kda_recurrent_fwd_paged.default
prepare_chunk_offsets_op = torch.ops.attn_gym.kda_prepare_chunk_offsets.default
delta_h_op = torch.ops.attn_gym.kda_delta_h.default
delta_h_with_state_op = torch.ops.attn_gym.kda_delta_h_with_state.default
short_conv_forward_op = torch.ops.attn_gym._cute_short_conv_fwd.default
short_conv_backward_op = torch.ops.attn_gym._cute_short_conv_bwd.default
short_conv_decode_op = torch.ops.attn_gym._cute_short_conv_decode.default
short_conv_configured_forward_op = torch.ops.attn_gym._cute_short_conv_configured_fwd.default
short_conv_configured_backward_op = torch.ops.attn_gym._cute_short_conv_configured_bwd.default
short_conv_configured_backward_with_state_grad_op = (
    torch.ops.attn_gym._cute_short_conv_configured_bwd_with_state_grad.default
)
short_conv_configured_decode_op = torch.ops.attn_gym._cute_short_conv_configured_decode.default


def recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    autotune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Validate and invoke the lazily loaded fused recurrent implementation."""
    if q.shape[-1] > 256:
        raise ValueError(f"recurrent_kda requires K in [1, 256], got {q.shape[-1]}")
    if not q.is_cuda:
        raise ValueError("the fused recurrent scan requires CUDA tensors")
    data_tensors = (q, k, v, gate, beta)
    if initial_state is not None:
        data_tensors += (initial_state,)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in data_tensors):
        raise RuntimeError(
            "recurrent_kda is inference-only and has no backward; use chunk_kda for "
            "training or call under torch.no_grad() / torch.inference_mode()"
        )

    q, k, v, gate, beta = (tensor.contiguous() for tensor in (q, k, v, gate, beta))
    if state_indices is not None:
        # `.contiguous()` on the pool would copy and silently drop the in-place advance.
        assert initial_state is not None
        return recurrent_fwd_paged_op(
            q, k, v, gate, beta, initial_state, state_indices, cu_seqlens, autotune
        ), None
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if output_final_state:
        return recurrent_fwd_op(q, k, v, gate, beta, initial_state, cu_seqlens, autotune)
    return recurrent_fwd_no_state_op(
        q, k, v, gate, beta, initial_state, cu_seqlens, autotune
    ), None


__all__ = [
    "chunk_bwd_op",
    "chunk_bwd_recompute_factors_op",
    "chunk_bwd_recompute_factors_with_state_grad_op",
    "chunk_bwd_with_state_grad_op",
    "chunk_fwd_op",
    "chunk_fwd_ragged_op",
    "chunk_fwd_ragged_with_state_op",
    "chunk_fwd_with_state_op",
    "delta_h_op",
    "delta_h_with_state_op",
    "prepare_chunk_offsets_op",
    "recurrent_forward",
    "recurrent_fwd_no_state_op",
    "recurrent_fwd_op",
    "recurrent_fwd_paged_op",
    "short_conv_backward_op",
    "short_conv_configured_backward_op",
    "short_conv_configured_backward_with_state_grad_op",
    "short_conv_configured_decode_op",
    "short_conv_configured_forward_op",
    "short_conv_decode_op",
    "short_conv_forward_op",
]
