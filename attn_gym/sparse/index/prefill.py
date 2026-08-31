"""One-CTA-per-query tensor-core DSA/CSA prefill indexer for SM100.

For each ``(batch, query)`` row, one CTA scans the complete key sequence,
reduces every indexer head into one score per key, and retains its final Top-K
entirely in shared memory.  There are no global partial lists and no merge
kernel.  The public operation is deliberately a direct CuTeDSL launch: it has
no dispatcher registration, compilation cache, or fallback implementation.
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
_THREADS = 128
_AB_STAGES = 2
_ACC_STAGES = 1
_ALIGNMENT = 16
_MAX_SEQUENCE = 1 << 20
_INVALID_KEY = -(1 << 63)
_WARP_SIZE = 32
_WARPS = _THREADS // _WARP_SIZE


@cute.struct
class _SharedStorage:
    ab_barriers: cute.struct.MemRange[Int64, _AB_STAGES * 2]
    acc_barriers: cute.struct.MemRange[Int64, _ACC_STAGES * 2]
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
def _warp_sort_32_desc(value: Int64, lane: Int32) -> Int64:
    """Sort one incoming candidate per lane, descending, using shuffles."""
    for stage in cutlass.range_constexpr(5):
        for substage_reverse in cutlass.range_constexpr(stage + 1):
            step = 1 << (stage - substage_reverse)
            other = Int64(cute.arch.shuffle_sync_bfly(value, step))
            descending = ((lane >> Int32(stage + 1)) & Int32(1)) == Int32(0)
            lower = (lane & Int32(step)) == Int32(0)
            if descending:
                value = other if (value < other if lower else value > other) else value
            else:
                value = other if (value > other if lower else value < other) else value
    return value


@cute.jit
def _merge_local_topk(
    local_keys: cute.Tensor,
    read_base: Int32,
    write_base: Int32,
    incoming: Int64,
    lane: Int32,
    topk: cutlass.Constexpr,
):
    """Co-rank merge sorted K and sorted 32, retaining the first K."""
    output_items = cute.ceil_div(topk, _WARP_SIZE)
    first_rank = lane * Int32(output_items)
    low = first_rank - Int32(_WARP_SIZE) if first_rank > Int32(_WARP_SIZE) else Int32(0)
    high = first_rank if first_rank < Int32(topk) else Int32(topk)  # noqa: FURB136

    # Every lane executes every shuffle, including the padded lanes for an
    # arbitrary non-multiple-of-32 Top-K, so the full-warp mask stays valid.
    active = cutlass.Boolean(True)
    for _ in cutlass.range_constexpr(8):
        old_count = (low + high) >> Int32(1)
        incoming_count = first_rank - old_count
        old_before_index = old_count - Int32(1) if old_count > Int32(0) else Int32(0)
        old_next_index = old_count if old_count < Int32(topk) else Int32(topk - 1)
        incoming_before_lane = incoming_count - Int32(1) if incoming_count > Int32(0) else Int32(0)
        incoming_next_lane = (
            incoming_count if incoming_count < Int32(_WARP_SIZE) else Int32(_WARP_SIZE - 1)
        )

        old_before = Int64(local_keys[read_base + old_before_index])
        old_next = Int64(local_keys[read_base + old_next_index])
        incoming_before = Int64(cute.arch.shuffle_sync(incoming, incoming_before_lane))
        incoming_next = Int64(cute.arch.shuffle_sync(incoming, incoming_next_lane))
        too_many_old = (
            old_count > Int32(0)
            and incoming_count < Int32(_WARP_SIZE)
            and old_before < incoming_next
        )
        too_many_incoming = (
            incoming_count > Int32(0) and old_count < Int32(topk) and incoming_before < old_next
        )
        if active:
            if too_many_old:
                high = old_count - Int32(1)
            elif too_many_incoming:
                low = old_count + Int32(1)
            else:
                low = old_count
                high = old_count - Int32(1)
                active = cutlass.Boolean(False)

    old_count = low
    incoming_count = first_rank - old_count
    for item in cutlass.range_constexpr(output_items):
        rank = first_rank + Int32(item)
        old_next_index = old_count if old_count < Int32(topk) else Int32(topk - 1)
        incoming_next_lane = (
            incoming_count if incoming_count < Int32(_WARP_SIZE) else Int32(_WARP_SIZE - 1)
        )
        old_next = Int64(local_keys[read_base + old_next_index])
        incoming_next = Int64(cute.arch.shuffle_sync(incoming, incoming_next_lane))
        old_next = old_next if old_count < Int32(topk) else Int64(_INVALID_KEY)
        incoming_next = (
            incoming_next if incoming_count < Int32(_WARP_SIZE) else Int64(_INVALID_KEY)
        )
        take_old = old_next > incoming_next
        merged = old_next if take_old else incoming_next
        if rank < Int32(topk):
            local_keys[write_base + rank] = merged
        if take_old:
            old_count = old_count + Int32(1)
        else:
            incoming_count = incoming_count + Int32(1)


@cute.jit
def _merge_shared_topk(
    local_keys: cute.Tensor,
    left_base: Int32,
    right_base: Int32,
    write_base: Int32,
    lane: Int32,
    topk: cutlass.Constexpr,
):
    """Merge two shared-memory sorted K-lists into one sorted K-list."""
    output_items = cute.ceil_div(topk, _WARP_SIZE)
    first_rank = lane * Int32(output_items)
    if first_rank < Int32(topk):
        low = Int32(0)
        high = first_rank
        merge_iterations = math.ceil(math.log2(topk + 1)) + 2
        active = cutlass.Boolean(True)
        for _ in cutlass.range_constexpr(merge_iterations):
            left_count = (low + high) >> Int32(1)
            right_count = first_rank - left_count
            left_before_index = left_count - Int32(1) if left_count > Int32(0) else Int32(0)
            left_next_index = left_count if left_count < Int32(topk) else Int32(topk - 1)
            right_before_index = right_count - Int32(1) if right_count > Int32(0) else Int32(0)
            right_next_index = right_count if right_count < Int32(topk) else Int32(topk - 1)
            left_before = Int64(local_keys[left_base + left_before_index])
            left_next = Int64(local_keys[left_base + left_next_index])
            right_before = Int64(local_keys[right_base + right_before_index])
            right_next = Int64(local_keys[right_base + right_next_index])
            too_many_left = (
                left_count > Int32(0) and right_count < Int32(topk) and left_before < right_next
            )
            too_many_right = (
                right_count > Int32(0) and left_count < Int32(topk) and right_before < left_next
            )
            if active:
                if too_many_left:
                    high = left_count - Int32(1)
                elif too_many_right:
                    low = left_count + Int32(1)
                else:
                    low = left_count
                    high = left_count - Int32(1)
                    active = cutlass.Boolean(False)

        left_count = low
        right_count = first_rank - left_count
        for item in cutlass.range_constexpr(output_items):
            rank = first_rank + Int32(item)
            left_next_index = left_count if left_count < Int32(topk) else Int32(topk - 1)
            right_next_index = right_count if right_count < Int32(topk) else Int32(topk - 1)
            left_next = Int64(local_keys[left_base + left_next_index])
            right_next = Int64(local_keys[right_base + right_next_index])
            left_next = left_next if left_count < Int32(topk) else Int64(_INVALID_KEY)
            right_next = right_next if right_count < Int32(topk) else Int64(_INVALID_KEY)
            take_left = left_next > right_next
            merged = left_next if take_left else right_next
            if rank < Int32(topk):
                local_keys[write_base + rank] = merged
            if take_left:
                left_count = left_count + Int32(1)
            else:
                right_count = right_count + Int32(1)


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
    causal: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx & Int32(_WARP_SIZE - 1)
    query, batch, _ = cute.arch.block_idx()

    num_candidates = cute.size(mK_sdb.shape[0])
    num_heads = cute.size(mQ_hdtb.shape[0])
    mK_sd = mK_sdb[None, None, batch]
    mQ_hd = mQ_hdtb[None, None, query, batch]

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
    local_stride = _WARPS * topk
    local_keys = smem.allocate_tensor(
        Int64,
        cute.make_layout((2 * local_stride,)),
        byte_alignment=128,
    )

    values_per_lane = cute.ceil_div(topk, _WARP_SIZE)
    local_warp_base = warp_idx * Int32(topk)
    for item in cutlass.range_constexpr(values_per_lane):
        rank = lane + Int32(item * _WARP_SIZE)
        if rank < Int32(topk):
            local_keys[local_warp_base + rank] = Int64(_INVALID_KEY)
    cute.arch.sync_warp()

    if warp_idx == Int32(0):
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)

    bytes_per_stage = cute.size_in_bytes(
        io_dtype,
        cute.select(k_smem_layout, mode=[0, 1, 2]),
    ) + cute.size_in_bytes(
        io_dtype,
        cute.select(q_smem_layout, mode=[0, 1, 2]),
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=_AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=bytes_per_stage,
        barrier_storage=storage.ab_barriers.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=_ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, _THREADS),
        barrier_storage=storage.acc_barriers.data_ptr(),
    ).make_participants()

    fragment_k = tiled_mma.make_fragment_A(sK)
    fragment_q = tiled_mma.make_fragment_B(sQ)
    accumulator_shape = tiled_mma.partition_shape_C(_MMA_TILE[:2])
    accumulator_template = tiled_mma.make_fragment_C(accumulator_shape)

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=_THREADS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr,
        barrier_for_retrieve=tmem_barrier,
    )
    tmem.allocate(utils.get_num_tmem_alloc_cols(accumulator_template))
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tmem.relinquish_alloc_permit()
    accumulator = cute.make_tensor(tmem_ptr, accumulator_template.layout)

    # The MMA maps one candidate to each thread.  That thread receives all 64
    # head logits from TMEM and keeps the cross-head reduction in registers.
    mma_thread = tiled_mma.get_slice(tidx)
    identity_c = cute.make_identity_tensor(_MMA_TILE[:2])
    tmem_load_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x16),
        Float32,
    )
    tmem_copy = tcgen05.make_tmem_copy(tmem_load_atom, accumulator)
    tmem_thread = tmem_copy.get_slice(tidx)
    tmem_source = tmem_thread.partition_S(accumulator)
    coord_fragment = tmem_thread.partition_D(mma_thread.partition_C(identity_c))
    register_shape = tmem_thread.partition_D(cute.make_identity_tensor(accumulator.shape)).shape
    head_logits = cute.make_rmem_tensor(register_shape, Float32)
    candidate_offset = Int32(coord_fragment[0][0])

    candidate_tiles = (
        cute.ceil_div(query + Int32(1), _TILE_CANDIDATES)
        if causal
        else cute.ceil_div(num_candidates, _TILE_CANDIDATES)
    )
    head_tiles = cute.ceil_div(num_heads, _TILE_HEADS)
    for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
        reduced_score = Float32(0.0)

        for head_tile in cutlass.range(head_tiles, unroll=0):
            mma_coord = (candidate_tile, head_tile, None)
            gK = cute.local_tile(mK_sd, _MMA_TILE, mma_coord, proj=(1, None, 1))
            gQ = cute.local_tile(mQ_hd, _MMA_TILE, mma_coord, proj=(None, 1, 1))
            mma_zero = tiled_mma.get_slice(0)
            mma_k = mma_zero.partition_A(gK)
            mma_q = mma_zero.partition_B(gQ)
            part_s_k, part_g_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(mma_k, 0, 3),
            )
            part_s_q, part_g_q = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_q, 0, 3),
            )
            d_tiles = cute.size(gK, mode=[2])
            if warp_idx == Int32(0):
                acc_empty = acc_producer.acquire_and_advance()
                for d_tile in cutlass.range(d_tiles, unroll=0):
                    ab_empty = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        part_g_k[(None, d_tile)],
                        part_s_k[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
                    cute.copy(
                        tma_atom_q,
                        part_g_q[(None, d_tile)],
                        part_s_q[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
                    ab_full = ab_consumer.wait_and_advance()
                    for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                        tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(d_tile != Int32(0) or d_block != 0),
                        )
                        coord = (None, None, d_block, ab_full.index)
                        cute.gemm(
                            tiled_mma,
                            accumulator,
                            fragment_k[coord],
                            fragment_q[coord],
                            accumulator,
                        )
                    ab_full.release()
                acc_empty.commit()

            acc_full = acc_consumer.wait_and_advance()
            cute.copy(tmem_copy, tmem_source, head_logits)
            cute.arch.fence_view_async_tmem_load()
            acc_full.release()

            for local_head in cutlass.range_constexpr(_TILE_HEADS):
                global_head = head_tile * Int32(_TILE_HEADS) + Int32(local_head)
                if global_head < num_heads:
                    logit = Float32(head_logits[local_head])
                    logit = logit if logit > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    weight = Float32(mW_bth[batch, query, global_head])
                    reduced_score = reduced_score + weight * logit

        candidate_index = candidate_tile * Int32(_TILE_CANDIDATES) + candidate_offset
        candidate_is_valid = (
            candidate_index <= query if causal else candidate_index < num_candidates
        )
        candidate_key = Int64(_INVALID_KEY)
        if candidate_is_valid:
            candidate_key = _make_key(reduced_score * score_scale, candidate_index)

        incoming = _warp_sort_32_desc(candidate_key, lane)
        read_bank = candidate_tile & Int32(1)
        write_bank = read_bank ^ Int32(1)
        read_base = read_bank * Int32(local_stride) + local_warp_base
        write_base = write_bank * Int32(local_stride) + local_warp_base
        _merge_local_topk(local_keys, read_base, write_base, incoming, lane, topk)
        cute.arch.sync_warp()

    # The first CTA barrier is required: warp 0/1 may otherwise read the final
    # lists for warp 2/3 before those warps have finished their last merge.
    cute.arch.barrier()
    final_bank = candidate_tiles & Int32(1)
    pair_bank = final_bank ^ Int32(1)
    final_bank_base = final_bank * Int32(local_stride)
    pair_bank_base = pair_bank * Int32(local_stride)
    if warp_idx < Int32(2):
        left_list = warp_idx * Int32(2)
        right_list = left_list + Int32(1)
        _merge_shared_topk(
            local_keys,
            final_bank_base + left_list * Int32(topk),
            final_bank_base + right_list * Int32(topk),
            pair_bank_base + warp_idx * Int32(topk),
            lane,
            topk,
        )
    cute.arch.barrier()
    if warp_idx == Int32(0):
        _merge_shared_topk(
            local_keys,
            pair_bank_base,
            pair_bank_base + Int32(topk),
            final_bank_base,
            lane,
            topk,
        )
    cute.arch.barrier()

    for write_round in cutlass.range_constexpr(cute.ceil_div(topk, _THREADS)):
        slot = tidx + Int32(write_round * _THREADS)
        if slot < Int32(topk):
            key = Int64(local_keys[final_bank_base + slot])
            mOut_btk[batch, query, slot] = ~Int32(key & Int64(0xFFFFFFFF))

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    weights: cute.Tensor,
    output: cute.Tensor,
    score_scale: Float32,
    topk: cutlass.Constexpr,
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
        _AB_STAGES,
    )
    q_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        _MMA_TILE,
        q.element_type,
        _AB_STAGES,
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
        causal,
    ).launch(
        grid=(q.shape[1], q.shape[0], 1),
        block=(_THREADS, 1, 1),
        min_blocks_per_mp=1,
    )


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
        causal,
    )
    return output


__all__ = ["index"]
