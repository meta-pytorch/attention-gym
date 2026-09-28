"""Torch-only private operator contracts for fused KDA backends.

Schemas, fake implementations, and dispatch registrations live here so they
exist before graph capture. CUDA implementations import their optional backend
only when the dispatcher executes the operator.
"""

from __future__ import annotations

import importlib

import torch

_CHUNK_SIZE = 64


# Fixed-arity schema pairs avoid optional outputs on hot paths.
_CHUNK_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor? initial_state,"
    " float scale, bool autotune, str schedule, bool fastmath)"
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
    "Tensor? initial_state, Tensor cu_seqlens, Tensor chunk_offsets, float scale, "
    "bool autotune, str schedule, bool fastmath)"
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged",
    f"{_CHUNK_RAGGED_FWD_ARGS} -> (Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged_with_state",
    f"{_CHUNK_RAGGED_FWD_ARGS} -> (Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor(a!) state_cache, Tensor state_indices, Tensor? has_initial_state, "
    "Tensor cu_seqlens, "
    "Tensor chunk_offsets, bool autotune, str schedule) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_replay_prepare",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor state_cache,"
    " Tensor(a!) replay_q, Tensor(b!) replay_k, Tensor(c!) replay_v,"
    " Tensor(d!) replay_gate, Tensor(e!) replay_beta, Tensor replay_count,"
    " Tensor state_indices, Tensor? has_initial_state)"
    " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_replay_commit",
    "(Tensor final_state, Tensor state_indices, Tensor? has_initial_state,"
    " Tensor active_counts, Tensor(a!) state_cache, Tensor(b!) replay_count) -> ()",
)
torch.library.define(
    "attn_gym::kda_chunk_replay_prefill_prepare",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor state_cache,"
    " Tensor replay_q, Tensor replay_k, Tensor replay_v, Tensor replay_gate,"
    " Tensor replay_beta, Tensor replay_count, Tensor state_indices,"
    " Tensor? has_initial_state, Tensor input_cu_seqlens)"
    " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,"
    " Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_replay_state_gather",
    "(Tensor state_cache, Tensor state_indices) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_chunk_replay_prefill_commit",
    "(Tensor prefix_output, Tensor tail_output, Tensor output_route,"
    " Tensor tail_q, Tensor tail_k, Tensor tail_v, Tensor tail_gate, Tensor tail_beta,"
    " Tensor tail_counts, Tensor state_indices, Tensor(a!) replay_q, Tensor(b!) replay_k,"
    " Tensor(c!) replay_v, Tensor(d!) replay_gate, Tensor(e!) replay_beta,"
    " Tensor(f!) replay_count, Tensor output_template) -> Tensor",
)

_CHUNK_BWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor Aqk, "
    "Tensor Akk, Tensor? cu_seqlens, Tensor? chunk_offsets, Tensor? d_output, "
    "Tensor? d_final_state, {initial_state}, float scale, bool fastmath, bool autotune, "
    "str schedule)"
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
    "Tensor? d_final_state, {initial_state}, float scale, bool fastmath, bool autotune, "
    "str schedule)"
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

_RECURRENT_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta,"
    " Tensor? initial_state, Tensor? cu_seqlens, float scale, bool autotune)"
)
torch.library.define("attn_gym::kda_recurrent_fwd", _RECURRENT_FWD_ARGS + " -> (Tensor, Tensor)")
torch.library.define("attn_gym::kda_recurrent_fwd_no_state", _RECURRENT_FWD_ARGS + " -> Tensor")
# Separate schema: the paged variant advances the state pool in place, so the final state
# is not an output and the alias annotation has to declare the mutation.
torch.library.define(
    "attn_gym::kda_recurrent_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache,"
    " Tensor state_indices, Tensor? has_initial_state, Tensor? cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_recurrent_decode",
    "(Tensor packed_qkv, Tensor raw_gate, Tensor raw_beta, Tensor A_log, Tensor dt_bias,"
    " Tensor(a!) state_cache, Tensor state_indices, Tensor? has_initial_state, Tensor(b!) out,"
    " float lower_bound, bool use_lower_bound, float scale) -> ()",
)

_DELTA_H_ARGS = (
    "(Tensor k, Tensor w, Tensor u, Tensor gk, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_offsets, SymInt capacity, bool fastmath)"
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
    "attn_gym::kda_delta_h_paged",
    "(Tensor k, Tensor w, Tensor u, Tensor gk, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor? cu_seqlens, "
    "Tensor? chunk_offsets, "
    "SymInt capacity, bool fastmath) -> (Tensor, Tensor)",
)


def _chunk_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.cute.chunk_kda_fwd")
    except ImportError as error:
        raise ImportError(
            "chunk_kda(impl='fused') requires the optional CuTeDSL backend: "
            "pip install attn-gym[linear]"
        ) from error


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


def _replay_backend():
    try:
        return importlib.import_module("attn_gym.linear.kda.fwd.triton.paged_replay")
    except ImportError as error:
        raise ImportError("paged_chunk_kda replay requires CUDA with Triton support") from error


def _chunk_fwd_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_cuda(*args)


def _chunk_fwd_with_state_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_with_state_cuda(*args)


def _chunk_fwd_ragged_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_ragged_cuda(*args)


def _chunk_fwd_ragged_with_state_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_ragged_with_state_cuda(*args)


def _chunk_fwd_ragged_paged_cuda(*args):
    return _chunk_backend()._chunk_kda_fwd_ragged_paged_cuda(*args)


def _chunk_replay_prepare_cuda(*args):
    return _replay_backend().prepare_paged_chunk_replay(*args)


def _chunk_replay_commit_cuda(*args):
    return _replay_backend().commit_paged_chunk_replay(*args)


def _chunk_replay_prefill_prepare_cuda(*args):
    return _replay_backend().prepare_paged_chunk_prefill(*args)


def _chunk_replay_state_gather_cuda(*args):
    return _replay_backend().gather_paged_chunk_state(*args)


def _chunk_replay_prefill_commit_cuda(*args):
    return _replay_backend().commit_paged_chunk_prefill(*args)


def _chunk_bwd_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_cuda(*args)


def _chunk_bwd_with_state_grad_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_with_state_grad_cuda(*args)


def _chunk_bwd_recompute_factors_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_recompute_factors_cuda(*args)


def _chunk_bwd_recompute_factors_with_state_grad_cuda(*args):
    return _chunk_backend()._chunk_kda_bwd_recompute_factors_with_state_grad_cuda(*args)


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


def _recurrent_decode_cuda(*args):
    return _recurrent_backend()._kda_recurrent_decode_cuda(*args)


def _delta_h_paged_cuda(*args):
    return _delta_h_backend()._delta_h_paged_cuda(*args)


torch.library.impl("attn_gym::kda_chunk_fwd", "CUDA", _chunk_fwd_cuda)
torch.library.impl("attn_gym::kda_chunk_fwd_with_state", "CUDA", _chunk_fwd_with_state_cuda)
torch.library.impl("attn_gym::kda_chunk_fwd_ragged", "CUDA", _chunk_fwd_ragged_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_fwd_ragged_with_state",
    "CUDA",
    _chunk_fwd_ragged_with_state_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_fwd_ragged_paged",
    "CUDA",
    _chunk_fwd_ragged_paged_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_replay_prepare",
    "CUDA",
    _chunk_replay_prepare_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_replay_commit",
    "CUDA",
    _chunk_replay_commit_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_replay_prefill_prepare",
    "CUDA",
    _chunk_replay_prefill_prepare_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_replay_state_gather",
    "CUDA",
    _chunk_replay_state_gather_cuda,
)
torch.library.impl(
    "attn_gym::kda_chunk_replay_prefill_commit",
    "CUDA",
    _chunk_replay_prefill_commit_cuda,
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
    "attn_gym::kda_recurrent_decode",
    "CUDA",
    _recurrent_decode_cuda,
)
torch.library.impl("attn_gym::kda_delta_h", "CUDA", _delta_h_cuda)
torch.library.impl("attn_gym::kda_delta_h_with_state", "CUDA", _delta_h_with_state_cuda)
torch.library.impl("attn_gym::kda_delta_h_paged", "CUDA", _delta_h_paged_cuda)


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
    scale: float,
    autotune: bool,
    schedule: str,
    fastmath: bool,
):
    del k, cumulative_gate, beta, initial_state, scale, autotune, schedule, fastmath
    return _chunk_fwd_fake_common(q, v)


@torch.library.register_fake("attn_gym::kda_chunk_fwd_with_state")
def _chunk_fwd_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
    autotune: bool,
    schedule: str,
    fastmath: bool,
):
    del k, cumulative_gate, beta, initial_state, scale, autotune, schedule, fastmath
    output, aqk, akk = _chunk_fwd_fake_common(q, v)
    state = q.new_empty(
        (q.shape[0], q.shape[2], v.shape[-1], q.shape[3]),
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
    scale: float,
    autotune: bool,
    schedule: str,
    fastmath: bool,
):
    del (
        k,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        scale,
        autotune,
        schedule,
        fastmath,
    )
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
    scale: float,
    autotune: bool,
    schedule: str,
    fastmath: bool,
):
    del k, cumulative_gate, beta, initial_state, chunk_offsets, scale, autotune, schedule
    del fastmath
    output, aqk, akk = _chunk_fwd_fake_common(q, v)
    state = q.new_empty(
        (cu_seqlens.shape[0] - 1, q.shape[2], v.shape[-1], q.shape[3]),
        dtype=torch.float32,
    )
    return output, state, aqk, akk


@torch.library.register_fake("attn_gym::kda_chunk_fwd_ragged_paged")
def _chunk_fwd_ragged_paged_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
    schedule: str,
) -> torch.Tensor:
    del (
        k,
        cumulative_gate,
        beta,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_offsets,
        autotune,
        schedule,
    )
    return v.new_empty(v.shape, dtype=q.dtype)


@torch.library.register_fake("attn_gym::kda_chunk_replay_prepare")
def _chunk_replay_prepare_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    del k, v, gate, beta, replay_count, has_initial_state
    num_active = q.shape[0]
    return (
        replay_q.new_empty((num_active, *replay_q.shape[1:])),
        replay_k.new_empty((num_active, *replay_k.shape[1:])),
        replay_v.new_empty((num_active, *replay_v.shape[1:])),
        replay_gate.new_empty((num_active, *replay_gate.shape[1:])),
        replay_beta.new_empty((num_active, *replay_beta.shape[1:])),
        state_cache.new_empty((num_active, *state_cache.shape[1:])),
        state_indices.new_empty((num_active,)),
    )


@torch.library.register_fake("attn_gym::kda_chunk_replay_commit")
def _chunk_replay_commit_fake(
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    active_counts: torch.Tensor,
    state_cache: torch.Tensor,
    replay_count: torch.Tensor,
) -> None:
    del final_state, state_indices, has_initial_state, active_counts, state_cache, replay_count


@torch.library.register_fake("attn_gym::kda_chunk_replay_prefill_prepare")
def _chunk_replay_prefill_prepare_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    input_cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    del k, v, gate, beta, state_cache, has_initial_state, input_cu_seqlens
    num_sequences = state_indices.shape[0]
    prefix_capacity = (
        (q.shape[0] * q.shape[1] + (_CHUNK_SIZE - 1) * num_sequences + _CHUNK_SIZE - 1)
        // _CHUNK_SIZE
        * _CHUNK_SIZE
    )
    return (
        *(
            cache.new_empty((1, prefix_capacity, *cache.shape[2:]))
            for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
        ),
        replay_count.new_empty((num_sequences + 1,)),
        q.new_empty((q.shape[0] * q.shape[1],), dtype=torch.int64),
        *(
            cache.new_empty((num_sequences, _CHUNK_SIZE, *cache.shape[2:]))
            for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
        ),
        replay_count.new_empty((num_sequences,)),
    )


@torch.library.register_fake("attn_gym::kda_chunk_replay_state_gather")
def _chunk_replay_state_gather_fake(
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    return state_cache.new_empty((state_indices.shape[0], *state_cache.shape[1:]))


@torch.library.register_fake("attn_gym::kda_chunk_replay_prefill_commit")
def _chunk_replay_prefill_commit_fake(
    prefix_output: torch.Tensor,
    tail_output: torch.Tensor,
    output_route: torch.Tensor,
    tail_q: torch.Tensor,
    tail_k: torch.Tensor,
    tail_v: torch.Tensor,
    tail_gate: torch.Tensor,
    tail_beta: torch.Tensor,
    tail_counts: torch.Tensor,
    state_indices: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    output_template: torch.Tensor,
) -> torch.Tensor:
    del (
        prefix_output,
        tail_output,
        output_route,
        tail_q,
        tail_k,
        tail_v,
        tail_gate,
        tail_beta,
        tail_counts,
        state_indices,
        replay_q,
        replay_k,
        replay_v,
        replay_gate,
        replay_beta,
        replay_count,
    )
    return output_template.new_empty(output_template.shape)


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
    scale: float,
    fastmath: bool,
    autotune: bool,
    schedule: str,
):
    del aqk, akk, cu_seqlens, chunk_offsets, d_output, d_final_state, initial_state
    del scale, fastmath, autotune, schedule
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
    scale: float,
    fastmath: bool,
    autotune: bool,
    schedule: str,
):
    del (
        aqk,
        akk,
        cu_seqlens,
        chunk_offsets,
        d_output,
        d_final_state,
        scale,
        fastmath,
        autotune,
        schedule,
    )
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
    scale: float,
    fastmath: bool,
    autotune: bool,
    schedule: str,
):
    del cu_seqlens, chunk_offsets, d_output, d_final_state, initial_state
    del scale, fastmath, autotune, schedule
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
    scale: float,
    fastmath: bool,
    autotune: bool,
    schedule: str,
):
    del cu_seqlens, chunk_offsets, d_output, d_final_state, scale, fastmath, autotune, schedule
    return (
        *_chunk_bwd_fake_common(q, k, v, cumulative_gate, beta),
        torch.empty_like(initial_state),
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
    scale: float,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, gate, beta, initial_state, scale, autotune
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    # The state carries one [V, K] slab per value head; grouped callers have v.shape[2] > HK.
    final_state = q.new_empty(
        num_sequences, v.shape[2], v.shape[-1], q.shape[3], dtype=torch.float32
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
    scale: float,
    autotune: bool,
) -> torch.Tensor:
    del k, gate, beta, initial_state, cu_seqlens, scale, autotune
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
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    del k, gate, beta, state_cache, state_indices, has_initial_state, cu_seqlens, scale
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.register_fake("attn_gym::kda_recurrent_decode")
def _recurrent_decode_fake(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    out: torch.Tensor,
    lower_bound: float,
    use_lower_bound: bool,
    scale: float,
) -> None:
    del (
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        out,
        lower_bound,
        use_lower_bound,
        scale,
    )


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
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del w, gk, initial_state, cu_seqlens, chunk_offsets, fastmath
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
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del w, gk, initial_state, chunk_offsets, fastmath
    h, v_new = _delta_h_fake_common(k, u, capacity)
    state_batch = k.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = k.new_empty(
        (state_batch, k.shape[2], k.shape[3], u.shape[-1]), dtype=torch.float32
    )
    return h, v_new, final_state


@torch.library.register_fake("attn_gym::kda_delta_h_paged")
def _delta_h_paged_fake(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del w, gk, state_cache, state_indices, has_initial_state, cu_seqlens, chunk_offsets
    del fastmath
    return _delta_h_fake_common(k, u, capacity)


chunk_fwd_op = torch.ops.attn_gym.kda_chunk_fwd.default
chunk_fwd_with_state_op = torch.ops.attn_gym.kda_chunk_fwd_with_state.default
chunk_fwd_ragged_op = torch.ops.attn_gym.kda_chunk_fwd_ragged.default
chunk_fwd_ragged_with_state_op = torch.ops.attn_gym.kda_chunk_fwd_ragged_with_state.default
chunk_fwd_ragged_paged_op = torch.ops.attn_gym.kda_chunk_fwd_ragged_paged.default
chunk_replay_prepare_op = torch.ops.attn_gym.kda_chunk_replay_prepare.default
chunk_replay_commit_op = torch.ops.attn_gym.kda_chunk_replay_commit.default
chunk_replay_prefill_prepare_op = torch.ops.attn_gym.kda_chunk_replay_prefill_prepare.default
chunk_replay_state_gather_op = torch.ops.attn_gym.kda_chunk_replay_state_gather.default
chunk_replay_prefill_commit_op = torch.ops.attn_gym.kda_chunk_replay_prefill_commit.default
chunk_bwd_op = torch.ops.attn_gym.kda_chunk_bwd.default
chunk_bwd_with_state_grad_op = torch.ops.attn_gym.kda_chunk_bwd_with_state_grad.default
chunk_bwd_recompute_factors_op = torch.ops.attn_gym.kda_chunk_bwd_recompute_factors.default
chunk_bwd_recompute_factors_with_state_grad_op = (
    torch.ops.attn_gym.kda_chunk_bwd_recompute_factors_with_state_grad.default
)
recurrent_fwd_op = torch.ops.attn_gym.kda_recurrent_fwd.default
recurrent_fwd_no_state_op = torch.ops.attn_gym.kda_recurrent_fwd_no_state.default
recurrent_fwd_paged_op = torch.ops.attn_gym.kda_recurrent_fwd_paged.default
recurrent_decode_op = torch.ops.attn_gym.kda_recurrent_decode.default
delta_h_op = torch.ops.attn_gym.kda_delta_h.default
delta_h_with_state_op = torch.ops.attn_gym.kda_delta_h_with_state.default
delta_h_paged_op = torch.ops.attn_gym.kda_delta_h_paged.default


def recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
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

    q, k, v, beta = (tensor.contiguous() for tensor in (q, k, v, beta))
    # FP32 gate loads measured faster than bf16 in the latency-bound scan loop.
    gate = gate.float().contiguous()
    if state_indices is not None:
        # `.contiguous()` on the pool would copy and silently drop the in-place advance.
        assert initial_state is not None
        return recurrent_fwd_paged_op(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            scale,
        ), None
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    args = (q, k, v, gate, beta, initial_state, cu_seqlens, scale, autotune)
    if output_final_state:
        return recurrent_fwd_op(*args)
    return recurrent_fwd_no_state_op(*args), None


def recurrent_decode_forward(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    out: torch.Tensor,
    lower_bound: float,
    use_lower_bound: bool,
    scale: float,
) -> torch.Tensor:
    """Invoke the lazily loaded fused decode implementation."""
    if not packed_qkv.is_cuda:
        raise ValueError("recurrent_kda_decode requires CUDA tensors")
    data_tensors = (packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, out)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in data_tensors):
        raise RuntimeError(
            "recurrent_kda_decode is inference-only and has no backward; "
            "call under torch.no_grad() / torch.inference_mode()"
        )
    recurrent_decode_op(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        out,
        lower_bound,
        use_lower_bound,
        scale,
    )
    return out


__all__ = [
    "chunk_bwd_op",
    "chunk_bwd_recompute_factors_op",
    "chunk_bwd_recompute_factors_with_state_grad_op",
    "chunk_bwd_with_state_grad_op",
    "chunk_fwd_op",
    "chunk_fwd_ragged_op",
    "chunk_fwd_ragged_paged_op",
    "chunk_fwd_ragged_with_state_op",
    "chunk_fwd_with_state_op",
    "chunk_replay_commit_op",
    "chunk_replay_prefill_commit_op",
    "chunk_replay_prefill_prepare_op",
    "chunk_replay_prepare_op",
    "chunk_replay_state_gather_op",
    "delta_h_op",
    "delta_h_paged_op",
    "delta_h_with_state_op",
    "recurrent_decode_forward",
    "recurrent_decode_op",
    "recurrent_forward",
    "recurrent_fwd_no_state_op",
    "recurrent_fwd_op",
    "recurrent_fwd_paged_op",
]
