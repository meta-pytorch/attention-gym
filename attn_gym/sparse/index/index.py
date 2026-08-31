"""Tensor-core DeepSeek/CSA indexer Top-K for Blackwell GPUs.

The input layout is deliberately BTHD: the 64 indexer heads belonging to one
query token are contiguous and form the N dimension of the tensor-core tile.
For every candidate tile the kernel computes

    sum_h w[b, t, h] * relu(dot(q[b, t, h], k[b, s])) / sqrt(D * H)

completely before offering the resulting scalar to the single Top-K state for
``(b, t)``. No per-head Top-K and no dense ``[B, T, S]`` score tensor exist.

This is intentionally a direct CuTeDSL program. There is no ``torch.library``
registration, fake implementation, manual compilation cache, or fallback.
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
_CANDIDATES_PER_CTA = 1024
_MMA_INSTRUCTION = (128, 64, 16)
_MMA_TILE = (_TILE_CANDIDATES, _TILE_HEADS, _TILE_D)
_THREADS = 128
_AB_STAGES = 2
_ACC_STAGES = 1
_ALIGNMENT = 16
_MAX_SEQUENCE = 1 << 20
_INVALID_KEY = -(1 << 63)


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
    """Pack ``(score desc, index asc)`` into one signed Int64 sort key."""
    bits = _bitcast_f32_to_i32(score)
    magnitude = bits & Int32(0x7FFFFFFF)
    # Float comparisons treat -0 and +0 as equal; normalize both so the
    # candidate-index tie break matches torch.topk exactly.
    bits = Int32(0) if magnitude == Int32(0) else bits
    ordinal = bits ^ ((bits >> Int32(31)) & Int32(0x7FFFFFFF))
    # Match largest-first Top-K: every NaN, including negative NaN, is above
    # finite values. The candidate index supplies a deterministic tie break.
    ordinal = Int32(0x7FFFFFFF) if magnitude > Int32(0x7F800000) else ordinal
    ordinal64 = Int64(ordinal) & Int64(0xFFFFFFFF)
    inverse_index = Int64(~index) & Int64(0xFFFFFFFF)
    return (ordinal64 << Int64(32)) | inverse_index


@cute.jit
def _sort_keys_desc(
    keys: cute.Tensor,
    tidx: Int32,
):
    """Cooperatively bitonic-sort one fixed candidate split in descending order."""
    num_stages = int(math.log2(_CANDIDATES_PER_CTA))
    values_per_thread = _CANDIDATES_PER_CTA // _THREADS
    for stage in cutlass.range_constexpr(num_stages):
        for substage_reverse in cutlass.range_constexpr(stage + 1):
            step = 1 << (stage - substage_reverse)
            for item in cutlass.range_constexpr(values_per_thread):
                left = tidx + Int32(item * _THREADS)
                right = left ^ Int32(step)
                if right > left:
                    left_key = Int64(keys[left])
                    right_key = Int64(keys[right])
                    descending = ((left >> Int32(stage + 1)) & Int32(1)) == Int32(0)
                    swap = left_key < right_key if descending else left_key > right_key
                    if swap:
                        keys[left] = right_key
                        keys[right] = left_key
            cute.arch.barrier()


@cute.kernel
def _index_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_k: cute.CopyAtom,
    mK_sdb: cute.Tensor,
    tma_atom_q: cute.CopyAtom,
    mQ_hdtb: cute.Tensor,
    mW_bth: cute.Tensor,
    mPartialKeys_btck: cute.Tensor,
    k_smem_layout: cute.ComposedLayout,
    q_smem_layout: cute.ComposedLayout,
    io_dtype: cutlass.Constexpr,
    score_scale: Float32,
    topk: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    query, batch, candidate_split = cute.arch.block_idx()

    num_candidates = cute.size(mK_sdb.shape[0])
    num_heads = cute.size(mQ_hdtb.shape[0])

    # Remove the batch/query modes before making the two logical GEMM operands:
    # A = shared indexer K [candidate, D], B = Q [head, D].
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
    candidate_keys = smem.allocate_tensor(
        Int64,
        cute.make_layout((_CANDIDATES_PER_CTA,)),
        byte_alignment=128,
    )

    if warp_idx == Int32(0):
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)

    for init_round in cutlass.range_constexpr(_CANDIDATES_PER_CTA // _THREADS):
        slot = tidx + Int32(init_round * _THREADS)
        candidate_keys[slot] = Int64(_INVALID_KEY)
    cute.arch.barrier()

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

    # Each of the 128 threads receives all 64 head logits for one candidate
    # directly from TMEM. This is the requested head-parallel mapping.
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

    candidate_tiles = cute.ceil_div(num_candidates, _TILE_CANDIDATES)
    head_tiles = cute.ceil_div(num_heads, _TILE_HEADS)
    first_candidate_tile = candidate_split * Int32(_CANDIDATES_PER_CTA // _TILE_CANDIDATES)
    remaining_candidate_tiles = candidate_tiles - first_candidate_tile
    local_candidate_tiles = remaining_candidate_tiles
    if local_candidate_tiles > Int32(  # noqa: PLR1730
        _CANDIDATES_PER_CTA // _TILE_CANDIDATES
    ):
        local_candidate_tiles = Int32(_CANDIDATES_PER_CTA // _TILE_CANDIDATES)

    for local_candidate_tile in cutlass.range(local_candidate_tiles, unroll=0):
        candidate_tile = first_candidate_tile + local_candidate_tile
        # One scalar per candidate. It remains private until every head tile
        # has contributed, so Top-K can never observe a partial head sum.
        reduced_score = Float32(0.0)

        for head_tile in cutlass.range(head_tiles, unroll=0):
            mma_coord = (candidate_tile, head_tile, None)
            gK = cute.local_tile(
                mK_sd,
                _MMA_TILE,
                mma_coord,
                proj=(1, None, 1),
            )
            gQ = cute.local_tile(
                mQ_hd,
                _MMA_TILE,
                mma_coord,
                proj=(None, 1, 1),
            )
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

            # ReLU is per head, and weights are neither normalized nor clamped.
            # The final partial head tile is explicitly masked.
            for local_head in cutlass.range_constexpr(_TILE_HEADS):
                global_head = head_tile * Int32(_TILE_HEADS) + Int32(local_head)
                if global_head < num_heads:
                    logit = Float32(head_logits[local_head])
                    # Keep this explicit select: Python's max() does not lower
                    # to the desired CuTeDSL scalar select for NaNs.
                    logit = logit if logit > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    weight = Float32(mW_bth[batch, query, global_head])
                    reduced_score = reduced_score + weight * logit

        candidate_base = candidate_tile * Int32(_TILE_CANDIDATES)
        candidate_index = candidate_base + candidate_offset
        local_candidate = local_candidate_tile * Int32(_TILE_CANDIDATES) + candidate_offset
        if candidate_index < num_candidates:
            candidate_keys[local_candidate] = _make_key(
                reduced_score * score_scale,
                candidate_index,
            )

    cute.arch.barrier()
    _sort_keys_desc(candidate_keys, tidx)

    # Lists are sorted, not merely heaps. This makes every subsequent pairwise
    # merge parallel and removes the serialized lane-0 sift-down bottleneck.
    for write_round in cutlass.range_constexpr(cute.ceil_div(topk, _THREADS)):
        slot = tidx + Int32(write_round * _THREADS)
        if slot < Int32(topk):
            key = Int64(_INVALID_KEY)
            if slot < Int32(_CANDIDATES_PER_CTA):
                key = Int64(candidate_keys[slot])
            mPartialKeys_btck[batch, query, candidate_split, slot] = key

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.kernel
def _merge_sorted_keys_kernel(
    mSrcKeys_btck: cute.Tensor,
    mDstKeys_btck: cute.Tensor,
    input_lists: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    search_levels: cutlass.Constexpr,
):
    """Merge adjacent sorted Top-K lists with one co-rank search per thread."""
    tidx, _, _ = cute.arch.thread_idx()
    output_list, query, batch = cute.arch.block_idx()
    left_list = output_list * Int32(2)
    right_list = left_list + Int32(1)
    items_per_thread = cute.ceil_div(topk, _THREADS)
    first_rank = tidx * Int32(items_per_thread)

    if first_rank < Int32(topk):
        if right_list < Int32(input_lists):
            # Co-rank this thread's first output diagonal. It then performs a
            # short serial merge over its contiguous stripe, amortizing the
            # search over ceil(K / 128) outputs.
            low = Int32(0)
            high = first_rank
            active = True
            for _ in cutlass.range_constexpr(search_levels + 1):
                if active:
                    left_position = (low + high) // Int32(2)
                    right_position = first_rank - left_position

                    move_high = False
                    if left_position > Int32(0):  # noqa: SIM102
                        if right_position < Int32(topk):
                            right_key = Int64(
                                mSrcKeys_btck[batch, query, right_list, right_position]
                            )
                            left_previous = Int64(
                                mSrcKeys_btck[
                                    batch,
                                    query,
                                    left_list,
                                    left_position - Int32(1),
                                ]
                            )
                            move_high = right_key > left_previous

                    if move_high:
                        high = left_position - Int32(1)
                    else:
                        move_low = False
                        if right_position > Int32(0):  # noqa: SIM102
                            if left_position < Int32(topk):
                                left_key = Int64(
                                    mSrcKeys_btck[
                                        batch,
                                        query,
                                        left_list,
                                        left_position,
                                    ]
                                )
                                right_previous = Int64(
                                    mSrcKeys_btck[
                                        batch,
                                        query,
                                        right_list,
                                        right_position - Int32(1),
                                    ]
                                )
                                move_low = left_key > right_previous

                        if move_low:
                            low = left_position + Int32(1)
                        else:
                            low = left_position
                            high = left_position - Int32(1)
                            active = False

            left_position = low
            right_position = first_rank - left_position
            for item in cutlass.range_constexpr(items_per_thread):
                rank = first_rank + Int32(item)
                if rank < Int32(topk):
                    left_key = Int64(_INVALID_KEY)
                    right_key = Int64(_INVALID_KEY)
                    if left_position < Int32(topk):
                        left_key = Int64(mSrcKeys_btck[batch, query, left_list, left_position])
                    if right_position < Int32(topk):
                        right_key = Int64(mSrcKeys_btck[batch, query, right_list, right_position])
                    take_left = left_key > right_key
                    result = left_key if take_left else right_key
                    if take_left:
                        left_position = left_position + Int32(1)
                    else:
                        right_position = right_position + Int32(1)
                    mDstKeys_btck[batch, query, output_list, rank] = result
        else:
            for item in cutlass.range_constexpr(items_per_thread):
                rank = first_rank + Int32(item)
                if rank < Int32(topk):
                    mDstKeys_btck[batch, query, output_list, rank] = mSrcKeys_btck[
                        batch,
                        query,
                        left_list,
                        rank,
                    ]


@cute.kernel
def _copy_key_indices_kernel(
    mSrcKeys_btck: cute.Tensor,
    mOut_btk: cute.Tensor,
    topk: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    query, batch, _ = cute.arch.block_idx()
    for write_round in cutlass.range_constexpr(cute.ceil_div(topk, _THREADS)):
        slot = tidx + Int32(write_round * _THREADS)
        if slot < Int32(topk):
            key = Int64(mSrcKeys_btck[batch, query, 0, slot])
            inverse_index = Int32(key & Int64(0xFFFFFFFF))
            mOut_btk[batch, query, slot] = ~inverse_index


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    weights: cute.Tensor,
    partial_keys_a: cute.Tensor,
    partial_keys_b: cute.Tensor,
    output: cute.Tensor,
    score_scale: Float32,
    topk: cutlass.Constexpr,
    search_levels: cutlass.Constexpr,
    num_splits: cutlass.Constexpr,
    merge_levels: cutlass.Constexpr,
):
    """Build TMA/MMA objects and launch the score/select and merge kernels."""
    # BTHD -> HDTB and BSD -> SDB. D stays unit-stride for both operands.
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

    _index_kernel(
        tiled_mma,
        tma_atom_k,
        k_tma_tensor,
        tma_atom_q,
        q_tma_tensor,
        weights,
        partial_keys_a,
        k_smem_layout,
        q_smem_layout,
        q.element_type,
        score_scale,
        topk,
    ).launch(
        grid=(q.shape[1], q.shape[0], num_splits),
        block=(_THREADS, 1, 1),
        min_blocks_per_mp=1,
    )

    source_keys = partial_keys_a
    destination_keys = partial_keys_b
    input_lists = num_splits
    for _ in cutlass.range_constexpr(merge_levels):
        output_lists = cute.ceil_div(input_lists, 2)
        _merge_sorted_keys_kernel(
            source_keys,
            destination_keys,
            input_lists,
            topk,
            search_levels,
        ).launch(
            grid=(output_lists, q.shape[1], q.shape[0]),
            block=(_THREADS, 1, 1),
        )
        source_keys, destination_keys = destination_keys, source_keys
        input_lists = output_lists

    _copy_key_indices_kernel(source_keys, output, topk).launch(
        grid=(q.shape[1], q.shape[0], 1),
        block=(_THREADS, 1, 1),
    )


def _validate(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, topk: int) -> None:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,D], got {tuple(q.shape)}")
    if k.ndim != 3:
        raise ValueError(f"k must have shape [B,S,D], got {tuple(k.shape)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B,T,H], got {tuple(weights.shape)}")

    batch, queries, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2] != head_dim:
        raise ValueError(f"q [B,T,H,D]={tuple(q.shape)} and k [B,S,D]={tuple(k.shape)} disagree")
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {(batch, queries, heads)}, got {tuple(weights.shape)}"
        )
    if batch <= 0 or batch % 2:
        raise ValueError(f"batch must be a positive multiple of 2, got {batch}")
    if queries <= 0:
        raise ValueError(f"query length must be positive, got {queries}")
    if heads <= 0 or heads % 2:
        raise ValueError(f"number of heads must be a positive multiple of 2, got {heads}")
    if head_dim <= 0 or head_dim % _HEAD_DIM_GRANULARITY:
        raise ValueError(
            "head dimension must be positive and divisible by "
            f"{_HEAD_DIM_GRANULARITY}, got {head_dim}"
        )
    if k.shape[1] <= 0 or k.shape[1] > _MAX_SEQUENCE:
        raise ValueError(f"candidate length must be in [1, {_MAX_SEQUENCE}], got {k.shape[1]}")
    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"topk must be an int, got {type(topk).__name__}")
    if topk < 0 or topk > k.shape[1]:
        raise ValueError(f"topk must be in [0, {k.shape[1]}], got {topk}")

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
) -> torch.Tensor:
    """Return one Top-K candidate set for every ``(batch, query)`` row.

    ``q`` is contiguous ``[B,T,H,D]``, ``k`` is shared ``[B,S,D]``, and
    ``weights`` is contiguous ``[B,T,H]``. Inputs are FP16 or BF16. The result
    is INT32 ``[B,T,topk]``; slot order is unspecified.

    There is no fallback. Invalid contracts raise, and allocation, compilation,
    or launch failures propagate directly to the caller.
    """
    _validate(q, k, weights, topk)
    if topk == 0:
        return torch.empty((*q.shape[:2], 0), dtype=torch.int32, device=q.device)

    num_splits = math.ceil(k.shape[1] / _CANDIDATES_PER_CTA)
    merge_levels = (num_splits - 1).bit_length()
    partial_shape = (*q.shape[:2], num_splits, topk)
    partial_keys_a = torch.empty(partial_shape, dtype=torch.int64, device=q.device)
    partial_keys_b = torch.empty(partial_shape, dtype=torch.int64, device=q.device)
    output = torch.empty((*q.shape[:2], topk), dtype=torch.int32, device=q.device)
    _launch(
        from_dlpack(q, assumed_align=_ALIGNMENT),
        from_dlpack(k, assumed_align=_ALIGNMENT),
        from_dlpack(weights, assumed_align=_ALIGNMENT),
        from_dlpack(partial_keys_a, assumed_align=_ALIGNMENT),
        from_dlpack(partial_keys_b, assumed_align=_ALIGNMENT),
        from_dlpack(output, assumed_align=_ALIGNMENT),
        1.0 / math.sqrt(q.shape[2] * q.shape[3]),
        topk,
        (topk - 1).bit_length(),
        num_splits,
        merge_levels,
    )
    return output


def _test() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the executable test requires an SM100 CUDA device")

    torch.manual_seed(2026)
    batch, queries, heads, head_dim = 2, 3, 66, 128
    candidates, topk, hidden = 257, 37, 192
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(
        batch,
        queries,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(batch, candidates, head_dim, device=device, dtype=dtype)
    x = torch.randn(batch, queries, hidden, device=device, dtype=torch.float32)
    weight_projection = torch.randn(hidden, heads, device=device, dtype=torch.float32)
    weights = (x @ weight_projection).to(dtype)
    assert (weights < 0).any() and (weights > 0).any()

    actual = index(q, k, weights, topk)

    # Literal DSA/CSA aggregation: ReLU independently per head, unconstrained
    # dynamic weight, variance scaling, then one Top-K over candidates.
    dots = torch.einsum("bthd,bsd->bths", q.float(), k.float())
    scores = (torch.relu(dots / math.sqrt(head_dim)) * weights.float().unsqueeze(-1)).sum(
        dim=2
    ) / math.sqrt(heads)
    expected = scores.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)

    assert actual.dtype == torch.int32
    assert actual.shape == (batch, queries, topk)
    assert bool(((actual >= 0) & (actual < candidates)).all())
    assert bool((actual.sort(dim=-1).values.diff(dim=-1) != 0).all())
    torch.testing.assert_close(
        actual.sort(dim=-1).values,
        expected.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )
    print(
        "PASS: multi-head DSA/CSA indexer ",
        f"B={batch} T={queries} H={heads} D={head_dim} S={candidates} K={topk}",
    )


if __name__ == "__main__":
    _test()


__all__ = ["index"]
