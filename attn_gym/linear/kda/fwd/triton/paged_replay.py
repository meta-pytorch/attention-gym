# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active-slot cache movement for streaming paged chunk KDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets

_CHUNK_SIZE = tl.constexpr(64)


def _int64_offset_heuristic(*tensor_names: str):
    return {
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            *(args[name] for name in tensor_names)
        )
    }


def _validate_replay_aliases(
    replay_tensors: tuple[tuple[str, torch.Tensor], ...],
    read_only_tensors: tuple[tuple[str, torch.Tensor | None], ...],
) -> None:
    """Reject storage aliasing that would make replay updates order-dependent."""
    for index, (name, tensor) in enumerate(replay_tensors):
        for other_name, other_tensor in replay_tensors[index + 1 :]:
            if torch._C._overlaps(tensor, other_tensor):
                raise ValueError(f"replay_state {name} must not alias replay_state {other_name}")
        for other_name, other_tensor in read_only_tensors:
            if other_tensor is not None and torch._C._overlaps(tensor, other_tensor):
                raise ValueError(f"replay_state {name} must not alias {other_name}")


@triton.heuristics(
    _int64_offset_heuristic(
        "replay_count",
        "state_indices",
        "has_initial_state",
        "cu_seqlens",
        "tail_counts",
        "prefix_cu_seqlens",
    )
)
@triton.jit
def _plan_prefill_kernel(
    replay_count,
    state_indices,
    has_initial_state,
    cu_seqlens,
    tail_counts,
    prefix_cu_seqlens,
    count_stride_s: tl.constexpr,
    num_sequences: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    total = 0
    tl.store(prefix_cu_seqlens, 0)
    for sequence in range(num_sequences):
        sequence_offset = sequence
        if USE_INT64_OFFSETS:
            sequence_offset = sequence_offset.to(tl.int64)
        slot = tl.load(state_indices + sequence_offset).to(tl.int64)
        count = tl.load(replay_count + slot * count_stride_s, mask=slot > 0, other=0)
        if use_has_initial_state:
            count = tl.where(tl.load(has_initial_state + sequence_offset), count, 0)
        input_length = tl.load(cu_seqlens + sequence_offset + 1) - tl.load(
            cu_seqlens + sequence_offset
        )
        complete_length = tl.where(slot > 0, count + input_length, 0) // _CHUNK_SIZE * _CHUNK_SIZE
        tl.store(
            tail_counts + sequence_offset,
            tl.where(slot > 0, count + input_length, 0) - complete_length,
        )
        total += complete_length
        tl.store(prefix_cu_seqlens + sequence_offset + 1, total)


@triton.heuristics(
    _int64_offset_heuristic(
        "source",
        "replay_cache",
        "state_indices",
        "input_cu_seqlens",
        "prefix_cu_seqlens",
        "tail_counts",
        "prefix",
        "tail",
        "prefix_map",
        "tail_map",
    )
)
@triton.jit
def _pack_prefill_kernel(
    source,
    replay_cache,
    state_indices,
    input_cu_seqlens,
    prefix_cu_seqlens,
    tail_counts,
    prefix,
    tail,
    prefix_map,
    tail_map,
    source_strides: tl.constexpr,
    replay_strides: tl.constexpr,
    prefix_strides: tl.constexpr,
    tail_strides: tl.constexpr,
    num_sequences: tl.constexpr,
    max_logical_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    is_packed: tl.constexpr,
    write_output_map: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        sequence = sequence.to(tl.int64)
        offset = offset.to(tl.int64)
    feature_size = num_heads * head_dim
    feature = offset % feature_size
    logical_token = offset // feature_size
    head = feature // head_dim
    dim = feature % head_dim
    slot = tl.load(state_indices + sequence).to(tl.int64)
    input_start = tl.load(input_cu_seqlens + sequence).to(tl.int64)
    input_length = tl.load(input_cu_seqlens + sequence + 1) - input_start
    prefix_start = tl.load(prefix_cu_seqlens + sequence).to(tl.int64)
    complete_length = tl.load(prefix_cu_seqlens + sequence + 1) - prefix_start
    tail_count = tl.load(tail_counts + sequence)
    count = tl.where(slot > 0, complete_length + tail_count - input_length, 0)
    active = (
        (sequence < num_sequences)
        & (logical_token < max_logical_tokens)
        & (slot > 0)
        & (logical_token < complete_length + tail_count)
    )

    from_replay = active & (logical_token < count)
    input_token = input_start + logical_token - count
    value = tl.where(
        from_replay,
        tl.load(
            replay_cache + ptr_offset((slot, logical_token, head, dim), replay_strides),
            mask=from_replay,
            other=0.0,
        ),
        tl.load(
            source
            + ptr_offset(
                (
                    0 if is_packed else sequence,
                    input_token if is_packed else input_token - input_start,
                    head,
                    dim,
                ),
                source_strides,
            ),
            mask=active & ~from_replay,
            other=0.0,
        ),
    )

    in_prefix = active & (logical_token < complete_length)
    tl.store(
        prefix
        + ptr_offset(
            (
                0,
                prefix_start + logical_token,
                head,
                dim,
            ),
            prefix_strides,
        ),
        value,
        mask=in_prefix,
    )
    tl.store(
        tail
        + ptr_offset(
            (sequence, logical_token - complete_length, head, dim),
            tail_strides,
        ),
        value,
        mask=active & ~in_prefix,
    )
    if write_output_map:
        output_token = tl.where(
            active & (logical_token >= count),
            input_start + logical_token - count,
            -1,
        )
        first_feature = feature == 0
        tl.store(
            prefix_map + prefix_start + logical_token,
            output_token,
            mask=in_prefix & first_feature,
        )
        tl.store(
            tail_map + sequence * _CHUNK_SIZE + logical_token - complete_length,
            output_token,
            mask=active & ~in_prefix & first_feature,
        )


@triton.heuristics(_int64_offset_heuristic("source", "output_map", "output"))
@triton.jit
def _scatter_output_kernel(
    source,
    output_map,
    output,
    source_tokens: tl.constexpr,
    feature_size: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    source_token = offset // feature_size
    output_token = tl.load(output_map + source_token, mask=in_bounds, other=-1).to(tl.int64)
    active = in_bounds & (source_token < source_tokens) & (output_token >= 0)
    tl.store(
        output + output_token * feature_size + offset % feature_size,
        tl.load(source + offset, mask=active, other=0.0),
        mask=active,
    )


@triton.heuristics(
    _int64_offset_heuristic("tail", "tail_counts", "state_indices", "replay_cache", "replay_count")
)
@triton.jit
def _commit_tail_kernel(
    tail,
    tail_counts,
    state_indices,
    replay_cache,
    replay_count,
    tail_strides: tl.constexpr,
    replay_strides: tl.constexpr,
    count_stride_s: tl.constexpr,
    num_sequences: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    write_count: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        sequence = sequence.to(tl.int64)
        offset = offset.to(tl.int64)
    feature_size = num_heads * head_dim
    feature = offset % feature_size
    token = offset // feature_size
    head = feature // head_dim
    dim = feature % head_dim
    slot = tl.load(state_indices + sequence).to(tl.int64)
    tail_count = tl.load(tail_counts + sequence)
    active = (sequence < num_sequences) & (slot > 0) & (token < tail_count)
    tl.store(
        replay_cache + ptr_offset((slot, token, head, dim), replay_strides),
        tl.load(
            tail + ptr_offset((sequence, token, head, dim), tail_strides),
            mask=active,
            other=0.0,
        ),
        mask=active,
    )
    if write_count:
        tl.store(
            replay_count + slot * count_stride_s + offset * 0,
            tail_count,
            mask=(offset == 0) & (slot > 0),
        )


@triton.heuristics(
    _int64_offset_heuristic("replay_count", "state_indices", "has_initial_state", "active_counts")
)
@triton.jit
def _gather_counts_kernel(
    replay_count,
    state_indices,
    has_initial_state,
    active_counts,
    count_stride_s: tl.constexpr,
    num_active: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    active_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        active_index = active_index.to(tl.int64)
    in_bounds = active_index < num_active
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    count = tl.load(
        replay_count + slot * count_stride_s,
        mask=in_bounds & (slot > 0),
        other=0,
    )
    if use_has_initial_state:
        count = tl.where(
            tl.load(has_initial_state + active_index, mask=in_bounds, other=0),
            count,
            0,
        )
    tl.store(active_counts + active_index, count, mask=in_bounds)


@triton.heuristics(
    _int64_offset_heuristic(
        "replay_cache", "current", "state_indices", "active_counts", "active_cache"
    )
)
@triton.jit
def _update_and_gather_kernel(
    replay_cache,
    current,
    state_indices,
    active_counts,
    active_cache,
    cache_strides: tl.constexpr,
    current_strides: tl.constexpr,
    active_strides: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens: tl.constexpr,
    feature_size: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    feature = offset % feature_size
    remaining = offset // feature_size
    token = remaining % num_tokens
    active_index = remaining // num_tokens

    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    count = tl.load(active_counts + active_index, mask=in_bounds, other=0)
    head = feature // head_dim
    dim = feature % head_dim
    cache_offset = ptr_offset((slot, token, head, dim), cache_strides)
    current_offset = ptr_offset((active_index, 0, head, dim), current_strides)
    active_offset = ptr_offset((active_index, token, head, dim), active_strides)
    cache_ptr = replay_cache + cache_offset
    is_current = active & (token == count)
    value = tl.load(cache_ptr, mask=active & (token < count), other=0.0)
    value = tl.where(
        is_current,
        tl.load(current + current_offset, mask=is_current, other=0.0),
        value,
    )
    tl.store(cache_ptr, value, mask=is_current)
    tl.store(active_cache + active_offset, value, mask=in_bounds)


@triton.heuristics(
    _int64_offset_heuristic("state_cache", "state_indices", "has_initial_state", "active_state")
)
@triton.jit
def _gather_state_kernel(
    state_cache,
    state_indices,
    has_initial_state,
    active_state,
    state_stride_s: tl.constexpr,
    active_stride_b: tl.constexpr,
    state_size: tl.constexpr,
    num_elements: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    state_offset = offset % state_size
    active_index = offset // state_size
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    if use_has_initial_state:
        active &= tl.load(has_initial_state + active_index, mask=in_bounds, other=0)
    tl.store(
        active_state + active_index * active_stride_b + state_offset,
        tl.load(
            state_cache + slot * state_stride_s + state_offset,
            mask=active,
            other=0.0,
        ),
        mask=in_bounds,
    )


@triton.heuristics(
    _int64_offset_heuristic(
        "final_state",
        "state_indices",
        "has_initial_state",
        "active_counts",
        "state_cache",
        "replay_count",
    )
)
@triton.jit
def _commit_state_kernel(
    final_state,
    state_indices,
    has_initial_state,
    active_counts,
    state_cache,
    replay_count,
    final_stride_b: tl.constexpr,
    state_stride_s: tl.constexpr,
    count_stride_s: tl.constexpr,
    state_size: tl.constexpr,
    chunk_size: tl.constexpr,
    num_elements: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    state_offset = offset % state_size
    active_index = offset // state_size
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    count = tl.load(active_counts + active_index, mask=in_bounds, other=0)
    completes_chunk = count == chunk_size - 1
    fresh_slot = False
    if use_has_initial_state:
        fresh_slot = ~tl.load(has_initial_state + active_index, mask=in_bounds, other=0)
    tl.store(
        state_cache + slot * state_stride_s + state_offset,
        tl.where(
            completes_chunk,
            tl.load(
                final_state + active_index * final_stride_b + state_offset,
                mask=active & completes_chunk,
                other=0.0,
            ),
            0.0,
        ),
        mask=active & (completes_chunk | fresh_slot),
    )
    tl.store(
        replay_count + slot * count_stride_s,
        tl.where(count == chunk_size - 1, 0, count + 1),
        mask=active & (state_offset == 0),
    )


def prepare_paged_chunk_replay(
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
    """Append one token and gather active replay windows and boundary states."""
    _validate_replay_aliases(
        (
            ("q", replay_q),
            ("k", replay_k),
            ("v", replay_v),
            ("gate", replay_gate),
            ("beta", replay_beta),
            ("count", replay_count),
        ),
        (
            ("q", q),
            ("k", k),
            ("v", v),
            ("gate", gate),
            ("beta", beta),
            ("state_cache", state_cache),
            ("state_indices", state_indices),
            ("has_initial_state", has_initial_state),
        ),
    )
    num_active = q.shape[0]
    active_counts = replay_count.new_empty((num_active,))
    active_state = state_cache.new_empty((num_active, *state_cache.shape[1:]))
    active_caches = tuple(
        cache.new_empty((num_active, *cache.shape[1:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    count_block = 128
    _gather_counts_kernel[(triton.cdiv(num_active, count_block),)](
        replay_count,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_counts,
        count_stride_s=replay_count.stride(0),
        num_active=num_active,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=count_block,
    )

    block = 256
    for cache, current, active_cache in zip(
        (replay_q, replay_k, replay_v, replay_gate, replay_beta),
        (q, k, v, gate, beta),
        active_caches,
        strict=True,
    ):
        feature_size = current[0, 0].numel()
        num_elements = num_active * cache.shape[1] * feature_size
        _update_and_gather_kernel[(triton.cdiv(num_elements, block),)](
            cache,
            current,
            state_indices,
            active_counts,
            active_cache,
            cache_strides=(cache.stride() if current.ndim == 4 else (*cache.stride(), 0)),
            current_strides=(current.stride() if current.ndim == 4 else (*current.stride(), 0)),
            active_strides=(
                active_cache.stride() if current.ndim == 4 else (*active_cache.stride(), 0)
            ),
            head_dim=current.shape[-1] if current.ndim == 4 else 1,
            num_tokens=cache.shape[1],
            feature_size=feature_size,
            num_elements=num_elements,
            BLOCK=block,
        )

    state_size = state_cache[0].numel()
    num_state_elements = num_active * state_size
    _gather_state_kernel[(triton.cdiv(num_state_elements, block),)](
        state_cache,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_state,
        state_stride_s=state_cache.stride(0),
        active_stride_b=active_state.stride(0),
        state_size=state_size,
        num_elements=num_state_elements,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=block,
    )
    return (*active_caches, active_state, active_counts)


def commit_paged_chunk_replay(
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    active_counts: torch.Tensor,
    state_cache: torch.Tensor,
    replay_count: torch.Tensor,
) -> None:
    """Commit completed boundaries and advance active replay counts."""
    num_active = final_state.shape[0]
    state_size = final_state[0].numel()
    num_elements = num_active * state_size
    block = 256
    _commit_state_kernel[(triton.cdiv(num_elements, block),)](
        final_state,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_counts,
        state_cache,
        replay_count,
        final_stride_b=final_state.stride(0),
        state_stride_s=state_cache.stride(0),
        count_stride_s=replay_count.stride(0),
        state_size=state_size,
        chunk_size=_CHUNK_SIZE,
        num_elements=num_elements,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=block,
    )


def prepare_paged_chunk_prefill(
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
    """Pack complete chunks and tails for graph-safe two-pass replay prefill."""
    _validate_replay_aliases(
        (
            ("q", replay_q),
            ("k", replay_k),
            ("v", replay_v),
            ("gate", replay_gate),
            ("beta", replay_beta),
            ("count", replay_count),
        ),
        (
            ("q", q),
            ("k", k),
            ("v", v),
            ("gate", gate),
            ("beta", beta),
            ("state_cache", state_cache),
            ("state_indices", state_indices),
            ("has_initial_state", has_initial_state),
            ("cu_seqlens", input_cu_seqlens),
        ),
    )
    num_sequences = state_indices.shape[0]
    prefix_capacity = (
        triton.cdiv(
            q.shape[0] * q.shape[1] + (_CHUNK_SIZE - 1) * num_sequences,
            _CHUNK_SIZE,
        )
        * _CHUNK_SIZE
    )
    max_logical_tokens = q.shape[1] + _CHUNK_SIZE - 1
    tail_counts = replay_count.new_empty((num_sequences,))
    prefix_cu_seqlens = replay_count.new_empty((num_sequences + 1,))
    prefix_map = replay_count.new_full((prefix_capacity,), -1)
    tail_map = replay_count.new_full((num_sequences, _CHUNK_SIZE), -1)
    prefix_caches = tuple(
        cache.new_zeros((1, prefix_capacity, *cache.shape[2:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    tail_caches = tuple(
        cache.new_zeros((num_sequences, _CHUNK_SIZE, *cache.shape[2:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    _plan_prefill_kernel[(1,)](
        replay_count,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        input_cu_seqlens,
        tail_counts,
        prefix_cu_seqlens,
        count_stride_s=replay_count.stride(0),
        num_sequences=num_sequences,
        use_has_initial_state=has_initial_state is not None,
    )

    is_packed = q.shape[0] == 1
    pack_block = 256
    for index, (source, replay_cache, prefix, tail) in enumerate(
        zip(
            (q, k, v, gate),
            (replay_q, replay_k, replay_v, replay_gate),
            prefix_caches[:4],
            tail_caches[:4],
            strict=True,
        )
    ):
        num_heads, head_dim = source.shape[2:]
        _pack_prefill_kernel[
            (
                num_sequences,
                triton.cdiv(max_logical_tokens * num_heads * head_dim, pack_block),
            )
        ](
            source,
            replay_cache,
            state_indices,
            input_cu_seqlens,
            prefix_cu_seqlens,
            tail_counts,
            prefix,
            tail,
            prefix_map,
            tail_map,
            source_strides=source.stride(),
            replay_strides=replay_cache.stride(),
            prefix_strides=prefix.stride(),
            tail_strides=tail.stride(),
            num_sequences=num_sequences,
            max_logical_tokens=max_logical_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            is_packed=is_packed,
            write_output_map=index == 0,
            BLOCK=pack_block,
        )

    prefix_beta = prefix_caches[4]
    tail_beta = tail_caches[4]
    _pack_prefill_kernel[
        (
            num_sequences,
            triton.cdiv(max_logical_tokens * beta.shape[2], pack_block),
        )
    ](
        beta,
        replay_beta,
        state_indices,
        input_cu_seqlens,
        prefix_cu_seqlens,
        tail_counts,
        prefix_beta,
        tail_beta,
        prefix_map,
        tail_map,
        source_strides=(*beta.stride(), 0),
        replay_strides=(*replay_beta.stride(), 0),
        prefix_strides=(*prefix_beta.stride(), 0),
        tail_strides=(*tail_beta.stride(), 0),
        num_sequences=num_sequences,
        max_logical_tokens=max_logical_tokens,
        num_heads=beta.shape[2],
        head_dim=1,
        is_packed=is_packed,
        write_output_map=False,
        BLOCK=pack_block,
    )
    return (
        *prefix_caches,
        prefix_cu_seqlens,
        prefix_map,
        *tail_caches,
        tail_map,
        tail_counts,
    )


def gather_paged_chunk_state(
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather post-prefix boundary states for the fixed tail pass."""
    num_sequences = state_indices.shape[0]
    active_state = state_cache.new_empty((num_sequences, *state_cache.shape[1:]))
    state_size = state_cache[0].numel()
    num_elements = num_sequences * state_size
    block = 256
    _gather_state_kernel[(triton.cdiv(num_elements, block),)](
        state_cache,
        state_indices,
        state_indices,
        active_state,
        state_stride_s=state_cache.stride(0),
        active_stride_b=active_state.stride(0),
        state_size=state_size,
        num_elements=num_elements,
        use_has_initial_state=False,
        BLOCK=block,
    )
    return active_state


def commit_paged_chunk_prefill(
    prefix_output: torch.Tensor,
    tail_output: torch.Tensor,
    prefix_map: torch.Tensor,
    tail_map: torch.Tensor,
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
    """Scatter new-token outputs and retain the final incomplete chunks."""
    output = output_template.new_zeros(output_template.shape)
    feature_size = output[0, 0].numel()
    block = 256
    for source, output_map in (
        (prefix_output, prefix_map),
        (tail_output, tail_map),
    ):
        num_elements = source.shape[0] * source.shape[1] * feature_size
        _scatter_output_kernel[(triton.cdiv(num_elements, block),)](
            source,
            output_map,
            output,
            source_tokens=source.shape[0] * source.shape[1],
            feature_size=feature_size,
            num_elements=num_elements,
            BLOCK=block,
        )

    num_sequences = state_indices.shape[0]
    for index, (tail, replay_cache) in enumerate(
        zip(
            (tail_q, tail_k, tail_v, tail_gate),
            (replay_q, replay_k, replay_v, replay_gate),
            strict=True,
        )
    ):
        num_heads, head_dim = tail.shape[2:]
        _commit_tail_kernel[
            (
                num_sequences,
                triton.cdiv(_CHUNK_SIZE * num_heads * head_dim, block),
            )
        ](
            tail,
            tail_counts,
            state_indices,
            replay_cache,
            replay_count,
            tail_strides=tail.stride(),
            replay_strides=replay_cache.stride(),
            count_stride_s=replay_count.stride(0),
            num_sequences=num_sequences,
            num_heads=num_heads,
            head_dim=head_dim,
            write_count=index == 0,
            BLOCK=block,
        )
    _commit_tail_kernel[
        (
            num_sequences,
            triton.cdiv(_CHUNK_SIZE * tail_beta.shape[2], block),
        )
    ](
        tail_beta,
        tail_counts,
        state_indices,
        replay_beta,
        replay_count,
        tail_strides=(*tail_beta.stride(), 0),
        replay_strides=(*replay_beta.stride(), 0),
        count_stride_s=replay_count.stride(0),
        num_sequences=num_sequences,
        num_heads=tail_beta.shape[2],
        head_dim=1,
        write_count=False,
        BLOCK=block,
    )
    return output


__all__ = [
    "commit_paged_chunk_prefill",
    "commit_paged_chunk_replay",
    "gather_paged_chunk_state",
    "prepare_paged_chunk_prefill",
    "prepare_paged_chunk_replay",
]
