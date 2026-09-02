"""Two-queries-per-CTA tensor-core DSA/CSA prefill indexer for SM100.

Each CTA owns two adjacent query rows, reuses every staged key tile across both
queries for each 64-head tile, and overlaps the tensor-core producer with two
independent CUDA-warpgroup Top-K consumers.  Both final Top-K lists remain
entirely in shared memory.  There are no global partial lists and no merge
kernel.  The public operation is a direct CuTeDSL launch guarded by an
in-process compile cache keyed on the static shape/dtype contract (dtype,
batch, queries, heads, head_dim, topk, causal); it has no dispatcher
registration or fallback implementation.
"""

import math

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, cute, pipeline, utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

_TILE_CANDIDATES = 128
_TILE_HEADS = 64
_TILE_D = 64
_HEAD_DIM_GRANULARITY = 16
_MMA_INSTRUCTION = (128, 64, 16)
_MMA_TILE = (_TILE_CANDIDATES, _TILE_HEADS, _TILE_D)
_QUERIES_PER_CTA = 2
_SELECTION_WARPS = 4
_SELECTION_THREADS = _SELECTION_WARPS * 32
_CONSUMER_WARPS = _QUERIES_PER_CTA * _SELECTION_WARPS
_K_STAGES = 2
_Q_STAGES = 4
_MAILBOX_STAGES = 4
# Each M128xN64 FP32 accumulator occupies 64 TMEM columns.  Four stages per
# query hide reducer latency while remaining within the SM100 512-column TMEM
# allocation.  Reducer and selector warps are permanently disjoint.
_ACC_STAGES = 4
_MIN_BLOCKS_PER_MP = 1
_SELECTOR_WARP_BASE = _CONSUMER_WARPS
_MMA_WARP = _CONSUMER_WARPS * 2
_LOAD_WARP = _MMA_WARP + 1
_THREADS = (_LOAD_WARP + 1) * 32
_ACC_TMEM_STAGES = _QUERIES_PER_CTA * _ACC_STAGES
_ALIGNMENT = 16
_MAX_SEQUENCE = 1 << 20
_INVALID_KEY = -(1 << 63)
_INVALID_ORDINAL = -(1 << 31)
_WARP_SIZE = 32


@cute.struct
class _SharedStorage:
    k_barriers: cute.struct.MemRange[Int64, _K_STAGES * 2]
    q_barriers: cute.struct.MemRange[Int64, _Q_STAGES * 2]
    acc0_barriers: cute.struct.MemRange[Int64, _ACC_STAGES * 2]
    acc1_barriers: cute.struct.MemRange[Int64, _ACC_STAGES * 2]
    mailbox0_barriers: cute.struct.MemRange[Int64, _MAILBOX_STAGES * 2]
    mailbox1_barriers: cute.struct.MemRange[Int64, _MAILBOX_STAGES * 2]
    tmem_holding: Int32


@dsl_user_op
def _bitcast_f32_to_i32(value, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(value).ir_value()))


@cute.jit
def _make_score_ordinal(score: Float32) -> Int32:
    """Map an FP32 score to the signed ordering used by packed keys."""
    bits = _bitcast_f32_to_i32(score)
    magnitude = bits & Int32(0x7FFFFFFF)
    bits = Int32(0) if magnitude == Int32(0) else bits
    ordinal = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    return Int32(0x7FFFFFFF) if magnitude > Int32(0x7F800000) else ordinal


@cute.jit
def _pack_key(ordinal: Int32, index: Int32) -> Int64:
    """Append the ascending-index tie break to a signed score ordinal."""
    inverse_index = Int64(~index) & Int64(0xFFFFFFFF)
    return ((Int64(ordinal) & Int64(0xFFFFFFFF)) << Int64(32)) | inverse_index


@cute.jit
def _bitonic_lane_value(
    value: Int64,
    other: Int64,
    lower: cutlass.Boolean,
    descending: cutlass.Boolean,
) -> Int64:
    """Select this lane's value for one descending bitonic comparator."""
    larger = value if value > other else other  # noqa: FURB136
    smaller = other if value > other else value  # noqa: FURB136
    keep_larger = lower == descending
    return larger if keep_larger else smaller


@cute.jit
def _sort_warp_quarter_desc(
    keys: cute.Tensor,
    run_base: Int32,
    lane: Int32,
):
    """Sort one 128-key run using one warp and four keys per lane."""
    values = cute.make_rmem_tensor(cute.make_layout((4,)), Int64)
    for item in cutlass.range_constexpr(4):
        values[item] = Int64(keys[run_base + lane + Int32(item * _WARP_SIZE)])

    for level in cutlass.range_constexpr(7):
        network_size = 1 << (level + 1)
        for reverse_stage in cutlass.range_constexpr(level + 1):
            distance = 1 << (level - reverse_stage)
            if cutlass.const_expr(distance < _WARP_SIZE):
                for item in cutlass.range_constexpr(4):
                    local_index = lane + Int32(item * _WARP_SIZE)
                    other = Int64(cute.arch.shuffle_sync_bfly(values[item], distance))
                    lower = (local_index & Int32(distance)) == Int32(0)
                    descending = (local_index & Int32(network_size)) == Int32(0)
                    values[item] = _bitonic_lane_value(
                        values[item],
                        other,
                        lower,
                        descending,
                    )
            else:
                register_distance = distance // _WARP_SIZE
                for item in cutlass.range_constexpr(4):
                    partner_item = item ^ register_distance
                    if cutlass.const_expr(partner_item > item):
                        local_index = lane + Int32(item * _WARP_SIZE)
                        value = Int64(values[item])
                        other = Int64(values[partner_item])
                        descending = (local_index & Int32(network_size)) == Int32(0)
                        larger = value if value > other else other  # noqa: FURB136
                        smaller = other if value > other else value  # noqa: FURB136
                        values[item] = larger if descending else smaller
                        values[partner_item] = smaller if descending else larger

    for item in cutlass.range_constexpr(4):
        keys[run_base + lane + Int32(item * _WARP_SIZE)] = values[item]


@cute.jit
def _merge_two_sorted_128_at_rank(
    keys: cute.Tensor,
    a_base: Int32,
    b_base: Int32,
    rank: Int32,
) -> Int64:
    """Return one rank of a stable descending merge of two 128-key runs."""
    diagonal = rank + Int32(1)
    lower = Int32(0)
    upper = diagonal

    # Find how many A elements occur in the first diagonal outputs.
    # A wins equal-key ties.  Eight fixed iterations cover the full [0, 128]
    # partition interval and avoid a divergent data-dependent loop backedge.
    for _iteration in cutlass.range_constexpr(8):
        if lower < upper:
            a_count = (lower + upper) >> Int32(1)
            b_count = diagonal - a_count
            move_right = cutlass.Boolean(False)
            if a_count < Int32(_TILE_CANDIDATES) and b_count > Int32(0):
                move_right = Int64(keys[a_base + a_count]) >= Int64(
                    keys[b_base + b_count - Int32(1)]
                )
            if move_right:
                lower = a_count + Int32(1)
            else:
                upper = a_count

    a_count = lower
    b_count = diagonal - a_count
    result = Int64(_INVALID_KEY)
    if a_count == Int32(0):
        result = Int64(keys[b_base + b_count - Int32(1)])
    elif b_count == Int32(0):
        result = Int64(keys[a_base + a_count - Int32(1)])
    else:
        a_value = Int64(keys[a_base + a_count - Int32(1)])
        b_value = Int64(keys[b_base + b_count - Int32(1)])
        result = a_value if a_value < b_value else b_value  # noqa: FURB136
    return result


@cute.jit
def _drain_hierarchical_512_128(
    keys: cute.Tensor,
    buffer_counts: cute.Tensor,
    query_base: Int32,
    logical_tid: Int32,
    query_slot: cutlass.Constexpr,
    selection_barrier,
    sort_span: cutlass.Constexpr,
):
    """Retain Top-128 from 512 keys with warp-local runs and two merge levels."""
    warp = logical_tid >> Int32(5)
    lane = logical_tid & Int32(_WARP_SIZE - 1)
    run_base = query_base + warp * Int32(_TILE_CANDIDATES)
    _sort_warp_quarter_desc(keys, run_base, lane)
    selection_barrier.arrive_and_wait()

    # Two pairs merge concurrently.  Each 64-thread pair group emits two
    # ranks per thread into a disjoint 128-key scratch run.
    pair = logical_tid >> Int32(6)
    pair_tid = logical_tid & Int32(63)
    pair_a_base = query_base + pair * Int32(2 * _TILE_CANDIDATES)
    pair_b_base = pair_a_base + Int32(_TILE_CANDIDATES)
    scratch_base = Int32(_QUERIES_PER_CTA * sort_span) + Int32(query_slot * 2 * _TILE_CANDIDATES)
    pair_output_base = scratch_base + pair * Int32(_TILE_CANDIDATES)
    for output_round in cutlass.range_constexpr(2):
        rank = pair_tid + Int32(output_round * 64)
        keys[pair_output_base + rank] = _merge_two_sorted_128_at_rank(
            keys,
            pair_a_base,
            pair_b_base,
            rank,
        )
    selection_barrier.arrive_and_wait()

    # Merge the pair results directly into the persistent long-term buffer.
    # Clearing all 384 tail slots makes it safe to discard every loser rather
    # than persisting a fully sorted 512-key permutation.
    final_value = _merge_two_sorted_128_at_rank(
        keys,
        scratch_base,
        scratch_base + Int32(_TILE_CANDIDATES),
        logical_tid,
    )
    keys[query_base + logical_tid] = final_value
    for item in cutlass.range_constexpr(3):
        keys[
            query_base + Int32(_TILE_CANDIDATES) + logical_tid + Int32(item * _SELECTION_THREADS)
        ] = Int64(_INVALID_KEY)
    if logical_tid == Int32(0):
        buffer_counts[query_slot] = Int32(0)
    selection_barrier.arrive_and_wait()


@cute.jit
def _drain_long_term_buffer(
    keys: cute.Tensor,
    buffer_counts: cute.Tensor,
    query_base: Int32,
    logical_tid: Int32,
    query_slot: cutlass.Constexpr,
    selection_barrier,
    run_size: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
):
    """Sort the long-term-buffer/buffer2 union and retain its first run_size keys.

    ``run_size`` is the padded persistent-run size (a power of two, >= 128;
    equal to the caller's requested topk only when topk is itself such a
    power of two). Global indices are striped across 128 threads. Distances
    below a warp use shuffles, distances of 32/64 exchange through shared
    memory, and distances of at least 128 compare register pairs owned by
    the same thread.
    """
    if cutlass.const_expr(run_size == 128 and sort_span == 512):
        _drain_hierarchical_512_128(
            keys,
            buffer_counts,
            query_base,
            logical_tid,
            query_slot,
            selection_barrier,
            sort_span,
        )
        return

    items_per_thread = sort_span // _SELECTION_THREADS
    levels = int(math.log2(sort_span))
    values = cute.make_rmem_tensor(cute.make_layout((items_per_thread,)), Int64)
    for item in cutlass.range_constexpr(items_per_thread):
        global_index = logical_tid + Int32(item * _SELECTION_THREADS)
        values[item] = Int64(keys[query_base + global_index])

    for level in cutlass.range_constexpr(levels):
        network_size = 1 << (level + 1)
        for reverse_stage in cutlass.range_constexpr(level + 1):
            distance = 1 << (level - reverse_stage)
            if cutlass.const_expr(distance < _WARP_SIZE):
                for item in cutlass.range_constexpr(items_per_thread):
                    global_index = logical_tid + Int32(item * _SELECTION_THREADS)
                    other = Int64(cute.arch.shuffle_sync_bfly(values[item], distance))
                    lower = (global_index & Int32(distance)) == Int32(0)
                    descending = (global_index & Int32(network_size)) == Int32(0)
                    values[item] = _bitonic_lane_value(
                        values[item],
                        other,
                        lower,
                        descending,
                    )
            elif cutlass.const_expr(distance < _SELECTION_THREADS):
                for item in cutlass.range_constexpr(items_per_thread):
                    global_index = logical_tid + Int32(item * _SELECTION_THREADS)
                    keys[query_base + global_index] = values[item]
                selection_barrier.arrive_and_wait()
                for item in cutlass.range_constexpr(items_per_thread):
                    global_index = logical_tid + Int32(item * _SELECTION_THREADS)
                    other_index = global_index ^ Int32(distance)
                    other = Int64(keys[query_base + other_index])
                    lower = (global_index & Int32(distance)) == Int32(0)
                    descending = (global_index & Int32(network_size)) == Int32(0)
                    values[item] = _bitonic_lane_value(
                        values[item],
                        other,
                        lower,
                        descending,
                    )
                selection_barrier.arrive_and_wait()
            else:
                register_distance = distance // _SELECTION_THREADS
                for item in cutlass.range_constexpr(items_per_thread):
                    partner_item = item ^ register_distance
                    if cutlass.const_expr(partner_item > item):
                        global_index = logical_tid + Int32(item * _SELECTION_THREADS)
                        value = Int64(values[item])
                        other = Int64(values[partner_item])
                        descending = (global_index & Int32(network_size)) == Int32(0)
                        larger = value if value > other else other  # noqa: FURB136
                        smaller = other if value > other else value  # noqa: FURB136
                        values[item] = larger if descending else smaller
                        values[partner_item] = smaller if descending else larger

    # Persist the whole final permutation.  In particular, buffer2 must hold
    # only true losers after a drain; leaving an intermediate permutation can
    # duplicate a retained key in the next union.
    for item in cutlass.range_constexpr(items_per_thread):
        global_index = logical_tid + Int32(item * _SELECTION_THREADS)
        keys[query_base + global_index] = values[item]
    if logical_tid == Int32(0):
        buffer_counts[query_slot] = Int32(0)
    selection_barrier.arrive_and_wait()


@cute.jit
def _gemm_query_tile(
    tiled_mma: cute.TiledMma,
    accumulator: cute.Tensor,
    accumulator_stage: Int32,
    fragment_k: cute.Tensor,
    k_stage: Int32,
    fragment_q: cute.Tensor,
    q_stage: Int32,
    d_tile: Int32,
):
    """Accumulate one 128-candidate by 64-head tensor-core tile."""
    for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
        # TiledMma.set mutates its Python trait wrapper.  Clone it so an
        # SSA value emitted in the MMA-warp branch cannot leak into another
        # warp-role branch during CuTeDSL IR construction.
        issue_mma = tiled_mma.with_()
        issue_mma.set(
            tcgen05.Field.ACCUMULATE,
            cutlass.Boolean(d_tile != Int32(0) or d_block != 0),
        )
        cute.gemm(
            issue_mma,
            accumulator[(None, None, None, accumulator_stage)],
            fragment_k[(None, None, d_block, k_stage)],
            fragment_q[(None, None, d_block, q_stage)],
            accumulator[(None, None, None, accumulator_stage)],
        )


@cute.jit
def _run_paired_load(
    tiled_mma,
    tma_atom_k,
    tma_atom_q,
    mK_sd,
    mQ0_hd,
    mQ1_hd,
    sK,
    sQ,
    candidate_tiles,
    head_tiles,
    k_producer,
    q_producer,
):
    """TMA warp: reuse each staged K tile across two H64 tiles."""
    mma_zero = tiled_mma.get_slice(0)
    head_pairs = head_tiles // Int32(2)
    for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        for head_pair in cutlass.range(head_pairs, unroll=0):
            head0 = head_pair * Int32(2)
            head1 = head0 + Int32(1)
            k_coord = (candidate_tile, head0, None)
            q0_coord = (candidate_tile, head0, None)
            q1_coord = (candidate_tile, head1, None)
            gK = cute.local_tile(mK_sd, _MMA_TILE, k_coord, proj=(1, None, 1))
            gQ0_h0 = cute.local_tile(mQ0_hd, _MMA_TILE, q0_coord, proj=(None, 1, 1))
            gQ1_h0 = cute.local_tile(mQ1_hd, _MMA_TILE, q0_coord, proj=(None, 1, 1))
            gQ0_h1 = cute.local_tile(mQ0_hd, _MMA_TILE, q1_coord, proj=(None, 1, 1))
            gQ1_h1 = cute.local_tile(mQ1_hd, _MMA_TILE, q1_coord, proj=(None, 1, 1))

            mma_k = mma_zero.partition_A(gK)
            mma_q0_h0 = mma_zero.partition_B(gQ0_h0)
            mma_q1_h0 = mma_zero.partition_B(gQ1_h0)
            mma_q0_h1 = mma_zero.partition_B(gQ0_h1)
            mma_q1_h1 = mma_zero.partition_B(gQ1_h1)
            part_s_k, part_g_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(mma_k, 0, 3),
            )
            part_s_q, part_g_q0_h0 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q0_h0, 0, 3),
            )
            _, part_g_q1_h0 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q1_h0, 0, 3),
            )
            _, part_g_q0_h1 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q0_h1, 0, 3),
            )
            _, part_g_q1_h1 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q1_h1, 0, 3),
            )

            for d_tile in cutlass.range(cute.size(gK, mode=[2]), unroll=0):
                k_empty = k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    part_g_k[(None, d_tile)],
                    part_s_k[(None, k_empty.index)],
                    tma_bar_ptr=k_empty.barrier,
                )
                q0_h0_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q0_h0[(None, d_tile)],
                    part_s_q[(None, q0_h0_empty.index)],
                    tma_bar_ptr=q0_h0_empty.barrier,
                )
                q1_h0_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q1_h0[(None, d_tile)],
                    part_s_q[(None, q1_h0_empty.index)],
                    tma_bar_ptr=q1_h0_empty.barrier,
                )
                q0_h1_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q0_h1[(None, d_tile)],
                    part_s_q[(None, q0_h1_empty.index)],
                    tma_bar_ptr=q0_h1_empty.barrier,
                )
                q1_h1_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q1_h1[(None, d_tile)],
                    part_s_q[(None, q1_h1_empty.index)],
                    tma_bar_ptr=q1_h1_empty.barrier,
                )

        # An odd H64 tail keeps the original singleton K/Q stage semantics.
        if (head_tiles & Int32(1)) != Int32(0):
            head_tile = head_pairs * Int32(2)
            mma_coord = (candidate_tile, head_tile, None)
            gK = cute.local_tile(mK_sd, _MMA_TILE, mma_coord, proj=(1, None, 1))
            gQ0 = cute.local_tile(mQ0_hd, _MMA_TILE, mma_coord, proj=(None, 1, 1))
            gQ1 = cute.local_tile(mQ1_hd, _MMA_TILE, mma_coord, proj=(None, 1, 1))
            mma_k = mma_zero.partition_A(gK)
            mma_q0 = mma_zero.partition_B(gQ0)
            mma_q1 = mma_zero.partition_B(gQ1)
            part_s_k, part_g_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(mma_k, 0, 3),
            )
            part_s_q, part_g_q0 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q0, 0, 3),
            )
            _, part_g_q1 = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q1, 0, 3),
            )
            for d_tile in cutlass.range(cute.size(gK, mode=[2]), unroll=0):
                k_empty = k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    part_g_k[(None, d_tile)],
                    part_s_k[(None, k_empty.index)],
                    tma_bar_ptr=k_empty.barrier,
                )
                q0_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q0[(None, d_tile)],
                    part_s_q[(None, q0_empty.index)],
                    tma_bar_ptr=q0_empty.barrier,
                )
                q1_empty = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    part_g_q1[(None, d_tile)],
                    part_s_q[(None, q1_empty.index)],
                    tma_bar_ptr=q1_empty.barrier,
                )
    k_producer.tail()
    q_producer.tail()


@cute.jit
def _run_paired_mma(
    tiled_mma,
    accumulator,
    fragment_k,
    fragment_q,
    candidate_tiles,
    head_tiles,
    d_tiles,
    k_consumer,
    q_consumer,
    acc0_producer,
    acc1_producer,
):
    """UMMA warp: reuse each staged K tile across adjacent H64 tiles."""
    head_pairs = head_tiles // Int32(2)
    for _candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        for _head_pair in cutlass.range(head_pairs, unroll=0):
            q0_acc_h0 = acc0_producer.acquire_and_advance()
            q1_acc_h0 = acc1_producer.acquire_and_advance()
            q0_acc_h1 = acc0_producer.acquire_and_advance()
            q1_acc_h1 = acc1_producer.acquire_and_advance()

            for d_tile in cutlass.range(d_tiles, unroll=0):
                k_full = k_consumer.wait_and_advance()

                q0_h0_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    q0_acc_h0.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q0_h0_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q0_acc_h0.commit()
                q0_h0_full.release()

                q1_h0_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    Int32(_ACC_STAGES) + q1_acc_h0.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q1_h0_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q1_acc_h0.commit()
                q1_h0_full.release()

                q0_h1_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    q0_acc_h1.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q0_h1_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q0_acc_h1.commit()
                q0_h1_full.release()

                q1_h1_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    Int32(_ACC_STAGES) + q1_acc_h1.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q1_h1_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q1_acc_h1.commit()
                q1_h1_full.release()
                k_full.release()

        # An odd H64 tail publishes one accumulator per query, as before.
        if (head_tiles & Int32(1)) != Int32(0):
            q0_acc = acc0_producer.acquire_and_advance()
            q1_acc = acc1_producer.acquire_and_advance()
            for d_tile in cutlass.range(d_tiles, unroll=0):
                k_full = k_consumer.wait_and_advance()
                q0_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    q0_acc.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q0_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q0_acc.commit()
                q0_full.release()

                q1_full = q_consumer.wait_and_advance()
                _gemm_query_tile(
                    tiled_mma,
                    accumulator,
                    Int32(_ACC_STAGES) + q1_acc.index,
                    fragment_k,
                    k_full.index,
                    fragment_q,
                    q1_full.index,
                    d_tile,
                )
                if d_tile == d_tiles - Int32(1):
                    q1_acc.commit()
                q1_full.release()
                k_full.release()
    acc0_producer.tail()
    acc1_producer.tail()



@cute.jit
def _run_query_reducer_mailbox(
    accumulator,
    acc_consumer,
    mailbox_ordinals,
    mailbox_producer,
    mW_bth,
    tidx,
    warp_idx,
    lane,
    batch,
    query_index,
    query_active,
    query_slot: cutlass.Constexpr,
    num_candidates,
    num_heads,
    candidate_tiles,
    head_tiles,
    score_scale,
    causal: cutlass.Constexpr,
):
    """Reduce every score tile and publish one ordinal per reducer thread."""
    accumulator_flat = accumulator[((None, None), 0, 0, None)]
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x8, tcgen05.Pack.NONE),
        Float32,
    )
    tmem_copy = tcgen05.make_tmem_copy(
        tmem_load_atom,
        accumulator_flat[(None, None, 0)],
    )
    tmem_thread = tmem_copy.get_slice(tidx)
    tmem_source = tmem_thread.partition_S(accumulator_flat)
    coord_fragment = tmem_thread.partition_D(cute.make_identity_tensor(_MMA_TILE[:2]))
    head_logits = cute.make_rmem_tensor(coord_fragment.shape, Float32)
    lane_in_group = lane & Int32(3)
    candidate_group = lane >> Int32(2)
    candidate_offset = warp_idx * Int32(_WARP_SIZE) + candidate_group + (lane_in_group << Int32(3))
    stage_offset = Int32(query_slot * _ACC_STAGES)
    mailbox_query_base = Int32(query_slot * _MAILBOX_STAGES * _TILE_CANDIDATES)

    for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        score0 = Float32(0.0)
        score1 = Float32(0.0)
        score2 = Float32(0.0)
        score3 = Float32(0.0)

        for head_tile in cutlass.range(head_tiles, unroll=0):
            acc_full = acc_consumer.wait_and_advance()
            cute.copy(
                tmem_copy,
                tmem_source[(None, None, None, stage_offset + acc_full.index)],
                head_logits,
            )
            cute.arch.fence_view_async_tmem_load()
            # Release TMEM before any mailbox acquire can backpressure this
            # reducer group, so the MMA producer can immediately reuse a stage.
            acc_full.release()

            for head_block in cutlass.range_constexpr(8):
                for head_pair in cutlass.range_constexpr(2):
                    local_head = (
                        Int32(8 * head_block) + (lane_in_group << Int32(1)) + Int32(head_pair)
                    )
                    global_head = head_tile * Int32(_TILE_HEADS) + local_head
                    if global_head < num_heads:
                        weight = Float32(mW_bth[batch, query_index, global_head])
                        item0 = head_block * 4 + head_pair
                        item1 = head_block * 4 + 2 + head_pair
                        item2 = 32 + head_block * 4 + head_pair
                        item3 = 32 + head_block * 4 + 2 + head_pair
                        logit0 = Float32(head_logits[item0])
                        logit1 = Float32(head_logits[item1])
                        logit2 = Float32(head_logits[item2])
                        logit3 = Float32(head_logits[item3])
                        logit0 = logit0 if logit0 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                        logit1 = logit1 if logit1 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                        logit2 = logit2 if logit2 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                        logit3 = logit3 if logit3 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                        score0 = score0 + weight * logit0
                        score1 = score1 + weight * logit1
                        score2 = score2 + weight * logit2
                        score3 = score3 + weight * logit3

        for shuffle_stage in cutlass.range_constexpr(2):
            shuffle_mask = 1 << shuffle_stage
            score0 = score0 + Float32(cute.arch.shuffle_sync_bfly(score0, shuffle_mask))
            score1 = score1 + Float32(cute.arch.shuffle_sync_bfly(score1, shuffle_mask))
            score2 = score2 + Float32(cute.arch.shuffle_sync_bfly(score2, shuffle_mask))
            score3 = score3 + Float32(cute.arch.shuffle_sync_bfly(score3, shuffle_mask))
        reduced_score = score0
        if lane_in_group == Int32(1):
            reduced_score = score1
        elif lane_in_group == Int32(2):
            reduced_score = score2
        elif lane_in_group == Int32(3):
            reduced_score = score3

        candidate_index = candidate_tile * Int32(_TILE_CANDIDATES) + candidate_offset
        candidate_is_valid = (
            candidate_index <= query_index if causal else candidate_index < num_candidates
        ) and query_active
        candidate_ordinal = Int32(_INVALID_ORDINAL)
        if candidate_is_valid:
            candidate_ordinal = _make_score_ordinal(reduced_score * score_scale)

        mailbox_empty = mailbox_producer.acquire_and_advance()
        mailbox_stage_base = (
            mailbox_query_base + mailbox_empty.index * Int32(_TILE_CANDIDATES)
        )
        mailbox_ordinals[mailbox_stage_base + tidx] = candidate_ordinal
        cute.arch.fence_view_async_shared()
        mailbox_empty.commit()

    mailbox_producer.tail()


@cute.jit
def _run_query_selector_mailbox(
    mailbox_ordinals,
    mailbox_consumer,
    selection_keys,
    buffer_counts,
    tidx,
    warp_idx,
    lane,
    query_slot: cutlass.Constexpr,
    candidate_tiles,
    topk: cutlass.Constexpr,
    run_size: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
):
    """Consume reduced ordinals and maintain the exact shared-memory Top-run_size.

    ``run_size`` is the padded persistent-run size; the caller truncates the
    final sorted run to its requested ``topk`` when writing output. The
    unsorted direct-seed shortcut below is only safe when the caller reads
    back the *entire* persistent run (topk == run_size); otherwise the
    output truncation would read an arbitrary, unsorted subset.
    """
    selection_barrier_id = 3
    if cutlass.const_expr(query_slot == 1):
        selection_barrier_id = 4
    selection_barrier = pipeline.NamedBarrier(
        barrier_id=selection_barrier_id,
        num_threads=_SELECTION_THREADS,
    )
    query_base = Int32(query_slot * sort_span)
    items_per_thread = sort_span // _SELECTION_THREADS
    for item in cutlass.range_constexpr(items_per_thread):
        if cutlass.const_expr(
            not (topk == run_size and run_size == 128 and sort_span == 512 and item == 0)
        ):
            selection_keys[query_base + tidx + Int32(item * _SELECTION_THREADS)] = Int64(
                _INVALID_KEY
            )
    if tidx == Int32(0):
        buffer_counts[query_slot] = Int32(0)
    if cutlass.const_expr(not (topk == run_size and run_size == 128 and sort_span == 512)):
        selection_barrier.arrive_and_wait()

    cutoff_key = Int64(_INVALID_KEY)
    cutoff_ordinal = Int32(_INVALID_ORDINAL)
    buffer_capacity = sort_span - run_size
    drain_threshold = buffer_capacity - _TILE_CANDIDATES
    lane_in_group = lane & Int32(3)
    candidate_group = lane >> Int32(2)
    candidate_offset = warp_idx * Int32(_WARP_SIZE) + candidate_group + (lane_in_group << Int32(3))
    mailbox_query_base = Int32(query_slot * _MAILBOX_STAGES * _TILE_CANDIDATES)

    for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        mailbox_full = mailbox_consumer.wait_and_advance()
        mailbox_stage_base = mailbox_query_base + mailbox_full.index * Int32(_TILE_CANDIDATES)
        candidate_ordinal = Int32(mailbox_ordinals[mailbox_stage_base + tidx])
        # Once every lane has copied its ordinal to a register, this stage may
        # be overwritten while the selector group performs Top-K maintenance.
        mailbox_full.release()

        candidate_index = candidate_tile * Int32(_TILE_CANDIDATES) + candidate_offset
        candidate_is_valid = candidate_ordinal != Int32(_INVALID_ORDINAL)
        candidate_key = Int64(_INVALID_KEY)
        accept = cutlass.Boolean(False)
        might_beat_cutoff = candidate_is_valid and candidate_ordinal >= cutoff_ordinal
        if might_beat_cutoff:
            candidate_key = _pack_key(candidate_ordinal, candidate_index)
            accept = candidate_key > cutoff_key

        direct_seed = cutlass.Boolean(False)
        if cutlass.const_expr(topk == run_size and run_size == 128 and sort_span == 512):
            direct_seed = candidate_tile == Int32(0)

        if direct_seed:
            # The first 128 candidates are already a complete Top-K set.  Put
            # them directly in the long-term buffer and compute its cutoff;
            # otherwise two full tiles would be appended before the first
            # drain merely to rediscover the same initial set.
            selection_keys[query_base + tidx] = candidate_key
            seed_min = candidate_key
            for shuffle_stage in cutlass.range_constexpr(5):
                other = Int64(
                    cute.arch.shuffle_sync_bfly(seed_min, 1 << shuffle_stage)
                )
                seed_min = other if other < seed_min else seed_min  # noqa: FURB136
            seed_scratch_base = Int32(
                _QUERIES_PER_CTA * sort_span
                + query_slot * 2 * _TILE_CANDIDATES
            )
            if lane == Int32(0):
                selection_keys[seed_scratch_base + warp_idx] = seed_min
            selection_barrier.arrive_and_wait()
            cutoff_key = Int64(_INVALID_KEY)
            if lane < Int32(_SELECTION_WARPS):
                cutoff_key = Int64(selection_keys[seed_scratch_base + lane])
            for shuffle_stage in cutlass.range_constexpr(2):
                other = Int64(
                    cute.arch.shuffle_sync_bfly(cutoff_key, 1 << shuffle_stage)
                )
                cutoff_key = other if other < cutoff_key else cutoff_key  # noqa: FURB136
            cutoff_key = Int64(cute.arch.shuffle_sync(cutoff_key, 0))
            cutoff_ordinal = Int32(cutoff_key >> Int64(32))
        else:
            accepted_mask = cutlass.Uint32(cute.arch.vote_ballot_sync(accept))
            warp_accepted = Int32(cute.arch.popc(accepted_mask))
            lane_rank = Int32(cute.arch.popc(accepted_mask & cute.arch.lanemask_lt()))
            warp_base = Int32(0)
            if lane == Int32(0) and warp_accepted > Int32(0):
                count_ptr = buffer_counts.iterator + buffer_counts.layout(query_slot)
                warp_base = Int32(
                    cute.arch.atomic_add(
                        count_ptr,
                        warp_accepted,
                        sem="relaxed",
                        scope="cta",
                    )
                )
            warp_base = Int32(cute.arch.shuffle_sync(warp_base, 0))
            if accept:
                selection_keys[query_base + Int32(run_size) + warp_base + lane_rank] = candidate_key

            selection_barrier.arrive_and_wait()
            occupied = Int32(buffer_counts[query_slot])
            if occupied > Int32(0) and occupied >= Int32(drain_threshold):
                _drain_long_term_buffer(
                    selection_keys,
                    buffer_counts,
                    query_base,
                    tidx,
                    query_slot,
                    selection_barrier,
                    run_size,
                    sort_span,
                )
                cutoff_key = Int64(selection_keys[query_base + Int32(run_size - 1)])
                cutoff_ordinal = Int32(cutoff_key >> Int64(32))

    occupied = Int32(buffer_counts[query_slot])
    if occupied > Int32(0):
        _drain_long_term_buffer(
            selection_keys,
            buffer_counts,
            query_base,
            tidx,
            query_slot,
            selection_barrier,
            run_size,
            sort_span,
        )


@cute.kernel
def _prefill_index_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_k: cute.CopyAtom,
    mK_sdb: cute.Tensor,
    tma_atom_q: cute.CopyAtom,
    mQ_hdtb: cute.Tensor,
    mW_bth: cute.Tensor,
    mOut_btk: cute.Tensor,
    k_smem_layout: cute.ComposedLayout,
    q_smem_layout: cute.ComposedLayout,
    io_dtype: cutlass.Constexpr,
    score_scale: Float32,
    topk: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
    causal: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx & Int32(_WARP_SIZE - 1)
    query_pair, batch, _ = cute.arch.block_idx()
    run_size = sort_span // 4

    num_candidates = cute.size(mK_sdb.shape[0])
    num_heads = cute.size(mQ_hdtb.shape[0])
    num_queries = cute.size(mQ_hdtb.shape[2])
    query0 = query_pair * Int32(_QUERIES_PER_CTA)
    query1 = query0 + Int32(1)
    query1_active = query1 < num_queries
    query1_load = query1 if query1_active else query0
    mK_sd = mK_sdb[None, None, batch]
    mQ0_hd = mQ_hdtb[None, None, query0, batch]
    mQ1_hd = mQ_hdtb[None, None, query1_load, batch]

    smem = utils.SmemAllocator()
    storage = smem.allocate(_SharedStorage)
    sK = smem.allocate_tensor(
        element_type=io_dtype,
        layout=k_smem_layout.outer,
        byte_alignment=128,
        swizzle=k_smem_layout.inner,
    )
    sQ = smem.allocate_tensor(
        element_type=io_dtype,
        layout=q_smem_layout.outer,
        byte_alignment=128,
        swizzle=q_smem_layout.inner,
    )
    selection_keys = smem.allocate_tensor(
        Int64,
        cute.make_layout(
            (
                _QUERIES_PER_CTA * sort_span
                + _QUERIES_PER_CTA * 2 * _TILE_CANDIDATES,
            )
        ),
        byte_alignment=128,
    )
    buffer_counts = smem.allocate_tensor(
        Int32,
        cute.make_layout((_QUERIES_PER_CTA,)),
        byte_alignment=16,
    )
    mailbox_ordinals = smem.allocate_tensor(
        Int32,
        cute.make_layout(
            (_QUERIES_PER_CTA * _MAILBOX_STAGES * _TILE_CANDIDATES,)
        ),
        byte_alignment=128,
    )

    if warp_idx == Int32(_LOAD_WARP):
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)

    k_bytes_per_stage = cute.size_in_bytes(
        io_dtype,
        cute.select(k_smem_layout, mode=[0, 1, 2]),
    )
    q_bytes_per_stage = cute.size_in_bytes(
        io_dtype,
        cute.select(q_smem_layout, mode=[0, 1, 2]),
    )
    k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_K_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=k_bytes_per_stage,
        barrier_storage=storage.k_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()
    q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_Q_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=q_bytes_per_stage,
        barrier_storage=storage.q_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()
    acc0_producer, acc0_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _SELECTION_THREADS),
        barrier_storage=storage.acc0_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()
    acc1_producer, acc1_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _SELECTION_THREADS),
        barrier_storage=storage.acc1_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()
    mailbox0_producer, mailbox0_consumer = pipeline.PipelineAsync.create(
        num_stages=_MAILBOX_STAGES,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            _SELECTION_THREADS,
        ),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            _SELECTION_THREADS,
        ),
        barrier_storage=storage.mailbox0_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()
    mailbox1_producer, mailbox1_consumer = pipeline.PipelineAsync.create(
        num_stages=_MAILBOX_STAGES,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            _SELECTION_THREADS,
        ),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            _SELECTION_THREADS,
        ),
        barrier_storage=storage.mailbox1_barriers.data_ptr(),
        defer_sync=True,
    ).make_participants()

    # All pipeline barriers are now initialized.  One fence and CTA rendezvous
    # replaces the otherwise redundant synchronization performed by every
    # individual pipeline constructor.
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    fragment_k = tiled_mma.make_fragment_A(sK)
    fragment_q = tiled_mma.make_fragment_B(sQ)
    accumulator_shape = tiled_mma.partition_shape_C(_MMA_TILE[:2])
    accumulator_template = tiled_mma.make_fragment_C(
        cute.append(accumulator_shape, _ACC_TMEM_STAGES)
    )

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=_THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr,
        barrier_for_retrieve=tmem_barrier,
        allocator_warp_id=_MMA_WARP,
    )
    tmem.allocate(utils.get_num_tmem_alloc_cols(accumulator_template))
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    accumulator = cute.make_tensor(tmem_ptr, accumulator_template.layout)

    candidate_tiles = (
        cute.ceil_div(query1_load + Int32(1), _TILE_CANDIDATES)
        if causal
        else cute.ceil_div(num_candidates, _TILE_CANDIDATES)
    )
    head_tiles = cute.ceil_div(num_heads, _TILE_HEADS)

    if warp_idx < Int32(_SELECTION_WARPS):
        _run_query_reducer_mailbox(
            accumulator,
            acc0_consumer,
            mailbox_ordinals,
            mailbox0_producer,
            mW_bth,
            tidx,
            warp_idx,
            lane,
            batch,
            query0,
            query0 < num_queries,
            0,
            num_candidates,
            num_heads,
            candidate_tiles,
            head_tiles,
            score_scale,
            causal,
        )
    elif warp_idx < Int32(_CONSUMER_WARPS):
        _run_query_reducer_mailbox(
            accumulator,
            acc1_consumer,
            mailbox_ordinals,
            mailbox1_producer,
            mW_bth,
            tidx - Int32(_SELECTION_THREADS),
            warp_idx - Int32(_SELECTION_WARPS),
            lane,
            batch,
            query1_load,
            query1_active,
            1,
            num_candidates,
            num_heads,
            candidate_tiles,
            head_tiles,
            score_scale,
            causal,
        )
    elif warp_idx < Int32(_SELECTOR_WARP_BASE + _SELECTION_WARPS):
        _run_query_selector_mailbox(
            mailbox_ordinals,
            mailbox0_consumer,
            selection_keys,
            buffer_counts,
            tidx - Int32(_SELECTOR_WARP_BASE * _WARP_SIZE),
            warp_idx - Int32(_SELECTOR_WARP_BASE),
            lane,
            0,
            candidate_tiles,
            topk,
            run_size,
            sort_span,
        )
    elif warp_idx < Int32(_SELECTOR_WARP_BASE + _CONSUMER_WARPS):
        _run_query_selector_mailbox(
            mailbox_ordinals,
            mailbox1_consumer,
            selection_keys,
            buffer_counts,
            tidx - Int32((_SELECTOR_WARP_BASE + _SELECTION_WARPS) * _WARP_SIZE),
            warp_idx - Int32(_SELECTOR_WARP_BASE + _SELECTION_WARPS),
            lane,
            1,
            candidate_tiles,
            topk,
            run_size,
            sort_span,
        )

    if warp_idx == Int32(_LOAD_WARP):
        _run_paired_load(
            tiled_mma,
            tma_atom_k,
            tma_atom_q,
            mK_sd,
            mQ0_hd,
            mQ1_hd,
            sK,
            sQ,
            candidate_tiles,
            head_tiles,
            k_producer,
            q_producer,
        )
    elif warp_idx == Int32(_MMA_WARP):
        _run_paired_mma(
            tiled_mma,
            accumulator,
            fragment_k,
            fragment_q,
            candidate_tiles,
            head_tiles,
            cute.ceil_div(cute.size(mK_sd, mode=[1]), _TILE_D),
            k_consumer,
            q_consumer,
            acc0_producer,
            acc1_producer,
        )
    # Every selection group has already performed its mandatory final drain,
    # so it may write its disjoint output rows without waiting for other roles.
    selection_query = (warp_idx - Int32(_SELECTOR_WARP_BASE)) // Int32(_SELECTION_WARPS)
    selection_tid = (tidx - Int32(_SELECTOR_WARP_BASE * _WARP_SIZE)) & Int32(
        _SELECTION_THREADS - 1
    )
    selection_base = selection_query * Int32(sort_span)
    query_index = query0 if selection_query == Int32(0) else query1
    query_active = selection_query == Int32(0) or query1_active
    is_selection_warp = warp_idx >= Int32(_SELECTOR_WARP_BASE) and warp_idx < Int32(
        _SELECTOR_WARP_BASE + _CONSUMER_WARPS
    )
    if is_selection_warp:
        for write_round in cutlass.range_constexpr(cute.ceil_div(topk, _SELECTION_THREADS)):
            slot = selection_tid + Int32(write_round * _SELECTION_THREADS)
            if query_active and slot < Int32(topk):
                key = Int64(selection_keys[selection_base + slot])
                mOut_btk[batch, query_index, slot] = ~Int32(key & Int64(0xFFFFFFFF))
    tmem.relinquish_alloc_permit()
    tmem_free_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=_THREADS)
    tmem_free_barrier.arrive_and_wait()
    tmem.free(tmem_ptr)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    weights: cute.Tensor,
    output: cute.Tensor,
    score_scale: Float32,
    topk: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
    causal: cutlass.Constexpr,
):
    """Build the SM100 TMA/MMA objects and launch exactly one kernel."""
    q_hdtb = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 3, 1, 0]))
    k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))

    mma_op = tcgen05.MmaF16BF16Op(
        q.element_type,
        Float32,
        _MMA_INSTRUCTION,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    k_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        _MMA_TILE,
        k.element_type,
        _K_STAGES,
    )
    q_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        _MMA_TILE,
        q.element_type,
        _Q_STAGES,
    )

    tma_load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_k, k_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load,
        k_sdb,
        cute.select(k_smem_layout, mode=[0, 1, 2]),
        _MMA_TILE,
        tiled_mma,
    )
    tma_atom_q, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load,
        q_hdtb,
        cute.select(q_smem_layout, mode=[0, 1, 2]),
        _MMA_TILE,
        tiled_mma,
    )
    _prefill_index_kernel(
        tiled_mma,
        tma_atom_k,
        k_tma_tensor,
        tma_atom_q,
        q_tma_tensor,
        weights,
        output,
        k_smem_layout,
        q_smem_layout,
        q.element_type,
        score_scale,
        topk,
        sort_span,
        causal,
    ).launch(
        grid=(cute.ceil_div(q.shape[1], _QUERIES_PER_CTA), q.shape[0], 1),
        block=(_THREADS, 1, 1),
        min_blocks_per_mp=_MIN_BLOCKS_PER_MP,
    )


_MAX_SUPPORTED_TOPK = 512


def _selection_run_size(topk: int) -> int:
    """Return the padded power-of-two persistent-run size for one query's Top-K.

    Floored at 128 (the incoming candidate-tile granularity) and capped at
    512. topk <= 128 always floors to 128, so it reuses the exact same
    specialized fast path as topk == 128 -- there is no separate "small topk"
    case.
    """
    if topk > _MAX_SUPPORTED_TOPK:
        raise NotImplementedError(
            f"topk > {_MAX_SUPPORTED_TOPK} is not supported by the CuTeDSL indexer, got {topk}."
        )
    padded = 1 << (max(topk, 1) - 1).bit_length()
    return max(_TILE_CANDIDATES, padded)


def _selection_sort_span(topk: int) -> int:
    """Return each query's compile-time shared selection-workspace stride.

    The workspace holds four runs of the padded run size: one persistent
    run plus three buffer runs (raw accept buffer, plus the two scratch runs
    a drain needs to sort and merge that buffer against the persistent run).
    """
    return 4 * _selection_run_size(topk)


def _validate(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, topk: int) -> None:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,D], got {tuple(q.shape)}")
    if k.ndim != 3:
        raise ValueError(f"k must have shape [B,T,D], got {tuple(k.shape)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B,T,H], got {tuple(weights.shape)}")

    batch, queries, heads, head_dim = q.shape
    if tuple(k.shape) != (batch, queries, head_dim):
        raise ValueError(
            "prefill requires equal query/key lengths and matching B,D: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {(batch, queries, heads)}, got {tuple(weights.shape)}"
        )
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if queries <= 0 or queries > _MAX_SEQUENCE:
        raise ValueError(f"sequence length must be in [1, {_MAX_SEQUENCE}], got {queries}")
    if heads <= 0 or heads % 2:
        raise ValueError(f"number of heads must be a positive multiple of 2, got {heads}")
    if head_dim <= 0 or head_dim % _HEAD_DIM_GRANULARITY:
        raise ValueError(
            "head dimension must be positive and divisible by "
            f"{_HEAD_DIM_GRANULARITY}, got {head_dim}"
        )
    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"topk must be an int, got {type(topk).__name__}")
    if topk < 0 or topk > queries:
        raise ValueError(f"topk must be in [0, {queries}], got {topk}")
    if topk > 512:
        raise ValueError(f"topk must be <= 512 for the CuTeDSL indexer, got {topk}")

    tensors = (q, k, weights)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("q, k, and weights must all be CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("q, k, and weights must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or weights.dtype != q.dtype:
        raise TypeError(
            f"q, k, and weights must have one dtype, got {q.dtype}, {k.dtype}, {weights.dtype}"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("q, k, and weights must be contiguous")
    if any(tensor.data_ptr() % _ALIGNMENT for tensor in tensors):
        raise ValueError(f"q, k, and weights must be {_ALIGNMENT}-byte aligned")
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise RuntimeError("this tcgen05 kernel requires an SM100 GPU")


_compile_cache: dict[tuple, object] = {}


def index(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool = False,
) -> torch.Tensor:
    """Return the shared-key Top-K for every full-sequence query.

    Inputs are contiguous BF16/FP16 tensors with layouts q[B,T,H,D],
    k[B,T,D], and weights[B,T,H].  The INT32 result has shape
    [B,T,topk].  Invalid contracts, compilation errors, shared-memory
    exhaustion, and launch errors propagate; there is no fallback.

    A compiled kernel is cached per (dtype, batch, queries, heads, head_dim,
    topk, causal, device compute capability); the same shape/dtype/topk/
    causal/capability combination reuses the already-compiled kernel instead
    of retracing on every call. A new sequence length recompiles, and so
    does a device whose compute capability differs from a cached entry's
    (e.g. a heterogeneous multi-GPU host) -- the generated PTX/SASS bakes in
    the compiling device's architecture, so a capability mismatch across
    devices in the same process must never share a cache entry.
    """
    _validate(q, k, weights, topk)
    if topk == 0:
        return torch.empty((*q.shape[:2], 0), dtype=torch.int32, device=q.device)

    output = torch.empty((*q.shape[:2], topk), dtype=torch.int32, device=q.device)
    batch, queries, heads, head_dim = q.shape
    capability = torch.cuda.get_device_capability(q.device)
    compile_key = (q.dtype, batch, queries, heads, head_dim, topk, causal, capability)
    score_scale = 1.0 / math.sqrt(heads * head_dim)
    q_c = from_dlpack(q, assumed_align=_ALIGNMENT)
    k_c = from_dlpack(k, assumed_align=_ALIGNMENT)
    weights_c = from_dlpack(weights, assumed_align=_ALIGNMENT)
    output_c = from_dlpack(output, assumed_align=_ALIGNMENT)
    compiled = _compile_cache.get(compile_key)
    if compiled is None:
        compiled = cute.compile(
            _launch,
            q_c,
            k_c,
            weights_c,
            output_c,
            score_scale,
            topk,
            _selection_sort_span(topk),
            causal,
        )
        _compile_cache[compile_key] = compiled
    compiled(q_c, k_c, weights_c, output_c, score_scale)
    return output


__all__ = ["index"]

