"""Two-queries-per-CTA tensor-core DSA/CSA prefill indexer for SM100.

Each CTA owns two adjacent query rows, reuses every staged key tile across both
queries for each 64-head tile, and overlaps the tensor-core producer with two
independent CUDA-warpgroup Top-K consumers.  Both final Top-K lists remain
entirely in shared memory.  There are no global partial lists and no merge
kernel.  The public operation is deliberately a direct CuTeDSL launch: it has
no dispatcher registration, compilation cache, or fallback implementation.
"""

import math
import os

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
_MMA_WARP = _CONSUMER_WARPS
_LOAD_WARP = _MMA_WARP + 1
_THREADS = (_LOAD_WARP + 1) * 32
_K_STAGES = 2
_Q_STAGES = 4
# Each M128xN64 FP32 accumulator occupies 64 TMEM columns.  The environment
# switches are intentionally import-time-only so experiments compile distinct
# kernels in fresh processes without adding runtime dispatch or cache machinery.
_ACC_STAGES = int(os.environ.get("ATTN_GYM_PREFILL_ACC_STAGES", "3"))
_MIN_BLOCKS_PER_MP = int(os.environ.get("ATTN_GYM_PREFILL_MIN_BLOCKS_PER_MP", "1"))
if _ACC_STAGES not in (2, 3, 4):
    raise ValueError("ATTN_GYM_PREFILL_ACC_STAGES must be one of 2, 3, or 4")
if _MIN_BLOCKS_PER_MP not in (1, 2):
    raise ValueError("ATTN_GYM_PREFILL_MIN_BLOCKS_PER_MP must be 1 or 2")
_ACC_TMEM_STAGES = _QUERIES_PER_CTA * _ACC_STAGES
_MIN_SORT_SPAN = 512
_ALIGNMENT = 16
_MAX_SEQUENCE = 1 << 20
_INVALID_KEY = -(1 << 63)
_WARP_SIZE = 32


@cute.struct
class _SharedStorage:
    k_barriers: cute.struct.MemRange[Int64, _K_STAGES * 2]
    q_barriers: cute.struct.MemRange[Int64, _Q_STAGES * 2]
    acc0_barriers: cute.struct.MemRange[Int64, _ACC_STAGES * 2]
    acc1_barriers: cute.struct.MemRange[Int64, _ACC_STAGES * 2]
    tmem_holding: Int32


@dsl_user_op
def _bitcast_f32_to_i32(value, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), Float32(value).ir_value()))


@cute.jit
def _make_key(score: Float32, index: Int32) -> Int64:
    """Pack ``(score descending, index ascending)`` as one sortable key."""
    bits = _bitcast_f32_to_i32(score)
    magnitude = bits & Int32(0x7FFFFFFF)
    bits = Int32(0) if magnitude == Int32(0) else bits
    ordinal = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    ordinal = Int32(0x7FFFFFFF) if magnitude > Int32(0x7F800000) else ordinal
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
def _drain_long_term_buffer(
    keys: cute.Tensor,
    buffer_counts: cute.Tensor,
    query_base: Int32,
    logical_tid: Int32,
    query_slot: cutlass.Constexpr,
    selection_barrier,
    topk: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
):
    """Sort the long-term-buffer/buffer2 union and retain its first K keys.

    Global indices are striped across 128 threads.  Distances below a warp use
    shuffles, distances of 32/64 exchange through shared memory, and distances
    of at least 128 compare register pairs owned by the same thread.
    """
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
        # ``TiledMma.set`` mutates its Python trait wrapper.  Clone it so an
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
    """TMA warp: stream one K and two Q tiles per head tile."""
    mma_zero = tiled_mma.get_slice(0)
    for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        for head_tile in cutlass.range(head_tiles, unroll=0):
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

            d_tiles = cute.size(gK, mode=[2])
            for d_tile in cutlass.range(d_tiles, unroll=0):
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
    """UMMA warp: reuse each staged K tile across the two query results."""
    for _candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        for _head_tile in cutlass.range(head_tiles, unroll=0):
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
def _run_query_selection(
    tiled_mma,
    accumulator,
    acc_consumer,
    mW_bth,
    selection_keys,
    buffer_counts,
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
    topk: cutlass.Constexpr,
    sort_span: cutlass.Constexpr,
    causal: cutlass.Constexpr,
):
    """Reduce heads, buffer cutoff winners, and periodically drain into Top-K."""
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
        selection_keys[query_base + tidx + Int32(item * _SELECTION_THREADS)] = Int64(_INVALID_KEY)
    if tidx == Int32(0):
        buffer_counts[query_slot] = Int32(0)
    selection_barrier.arrive_and_wait()

    buffer_capacity = sort_span - topk
    drain_threshold = buffer_capacity - _TILE_CANDIDATES

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
            acc_full.release()

            # Four aligned lanes own disjoint quarters of the 64 heads for
            # the same four candidate rows.  A weight is loaded once and
            # reused across all four rows before the subgroup reduction.
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

        # XOR 1 and 2 stay inside each four-lane subgroup and combine its
        # four disjoint 16-head contributions.  Lane p retains candidate p.
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
        candidate_key = Int64(_INVALID_KEY)
        if candidate_is_valid:
            candidate_key = _make_key(reduced_score * score_scale, candidate_index)

        cutoff = Int64(selection_keys[query_base + Int32(topk - 1)])
        accept = candidate_is_valid and candidate_key > cutoff
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
            selection_keys[query_base + Int32(topk) + warp_base + lane_rank] = candidate_key

        # The barrier publishes all four warp reservations and writes.  Any
        # prior occupancy at or above the threshold was drained, so pre-tile
        # occupancy is below it; adding at most 128 cannot overflow buffer2.
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
                topk,
                sort_span,
            )

    occupied = Int32(buffer_counts[query_slot])
    if occupied > Int32(0):
        _drain_long_term_buffer(
            selection_keys,
            buffer_counts,
            query_base,
            tidx,
            query_slot,
            selection_barrier,
            topk,
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
        cute.make_layout((_QUERIES_PER_CTA * sort_span,)),
        byte_alignment=128,
    )
    buffer_counts = smem.allocate_tensor(
        Int32,
        cute.make_layout((_QUERIES_PER_CTA,)),
        byte_alignment=16,
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
    ).make_participants()
    q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_Q_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=q_bytes_per_stage,
        barrier_storage=storage.q_barriers.data_ptr(),
    ).make_participants()
    acc0_producer, acc0_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _SELECTION_THREADS),
        barrier_storage=storage.acc0_barriers.data_ptr(),
    ).make_participants()
    acc1_producer, acc1_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _SELECTION_THREADS),
        barrier_storage=storage.acc1_barriers.data_ptr(),
    ).make_participants()

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
        _run_query_selection(
            tiled_mma,
            accumulator,
            acc0_consumer,
            mW_bth,
            selection_keys,
            buffer_counts,
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
            topk,
            sort_span,
            causal,
        )
    elif warp_idx < Int32(_CONSUMER_WARPS):
        _run_query_selection(
            tiled_mma,
            accumulator,
            acc1_consumer,
            mW_bth,
            selection_keys,
            buffer_counts,
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
            topk,
            sort_span,
            causal,
        )
    elif warp_idx == Int32(_LOAD_WARP):
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
    # Every selection group has already performed its mandatory final drain.
    cute.arch.barrier()
    selection_query = warp_idx // Int32(_SELECTION_WARPS)
    selection_tid = tidx & Int32(_SELECTION_THREADS - 1)
    selection_base = selection_query * Int32(sort_span)
    query_index = query0 if selection_query == Int32(0) else query1
    query_active = selection_query == Int32(0) or query1_active
    if warp_idx < Int32(_CONSUMER_WARPS):
        for write_round in cutlass.range_constexpr(cute.ceil_div(topk, _SELECTION_THREADS)):
            slot = selection_tid + Int32(write_round * _SELECTION_THREADS)
            if query_active and slot < Int32(topk):
                key = Int64(selection_keys[selection_base + slot])
                mOut_btk[batch, query_index, slot] = ~Int32(key & Int64(0xFFFFFFFF))
    cute.arch.barrier()

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


def _selection_sort_span(topk: int, minimum: int = _MIN_SORT_SPAN) -> int:
    """Return a power-of-two span with one threshold tile and one slack tile."""
    span = minimum
    required = topk + 2 * _TILE_CANDIDATES
    while span < required:
        span *= 2
    return span


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
    if batch <= 0 or batch % 2:
        raise ValueError(f"batch must be a positive multiple of 2, got {batch}")
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


def index(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool = False,
) -> torch.Tensor:
    """Return the shared-key Top-K for every full-sequence query.

    Inputs are contiguous BF16/FP16 tensors with layouts ``q[B,T,H,D]``,
    ``k[B,T,D]``, and ``weights[B,T,H]``.  The INT32 result has shape
    ``[B,T,topk]``.  Invalid contracts, compilation errors, shared-memory
    exhaustion, and launch errors propagate; there is no fallback.
    """
    if not isinstance(causal, bool):
        raise TypeError(f"causal must be a bool, got {type(causal).__name__}")
    _validate(q, k, weights, topk)
    if topk == 0:
        return torch.empty((*q.shape[:2], 0), dtype=torch.int32, device=q.device)

    output = torch.empty((*q.shape[:2], topk), dtype=torch.int32, device=q.device)
    _launch(
        from_dlpack(q, assumed_align=_ALIGNMENT),
        from_dlpack(k, assumed_align=_ALIGNMENT),
        from_dlpack(weights, assumed_align=_ALIGNMENT),
        from_dlpack(output, assumed_align=_ALIGNMENT),
        1.0 / math.sqrt(q.shape[2] * q.shape[3]),
        topk,
        _selection_sort_span(topk),
        causal,
    )
    return output


__all__ = ["index"]
