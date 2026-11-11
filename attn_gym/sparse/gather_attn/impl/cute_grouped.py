"""Common loader, dynamic ABI and launcher for grouped warpdrive gather attention.

Each CTA owns a query and up to 16 heads of one KV group. Loading warps gather
32 selected K/V rows per stage, preserving duplicates and masking invalid indices.
A named FULL/EMPTY pair protects each of two shared-memory stages; the initially
empty stages seed progress. See cute_grouped_mma.py for the tensor-core roles.
"""

from dataclasses import dataclass

import cutlass
import torch
from cutlass import Boolean, Float32, Int32, Int64, cute

from attn_gym._backends.cute.utils import requires_int64_abi, tensor_supports_contiguous_dim
from attn_gym.warpdrive import (
    Local,
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@dataclass(frozen=True)
class GatherParams:
    q: cute.Tensor
    sk: cute.Tensor
    lk: cute.Tensor
    sv: cute.Tensor
    lv: cute.Tensor
    indices: cute.Tensor
    sink: cute.Tensor
    cu_q: cute.Tensor
    cu_k: cute.Tensor
    out: cute.Tensor
    lse: cute.Tensor
    go: cute.Tensor
    dq: cute.Tensor
    dsk: cute.Tensor
    dlk: cute.Tensor
    dsv: cute.Tensor
    dlv: cute.Tensor
    dsink: cute.Tensor
    scale: Float32
    window: Int32


class GatherGrouped(WarpSpecialized):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        has_window: bool,
        *,
        backward: bool = False,
        separate_values: bool = True,
        has_sink: bool = True,
        packed: bool = False,
        index64: bool = False,
        sink_fp32: bool = False,
        wide: bool = False,
        vectorized: bool = False,
        block_k: int = 32,
        stages: int = 2,
    ):
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.group = heads // kv_heads
        self.head_tiles = cute.ceil_div(self.group, 16)
        self.has_window = has_window
        self.backward = backward
        self.separate_values = separate_values
        self.has_sink = has_sink
        self.packed = packed
        self.index64 = index64
        self.sink_fp32 = sink_fp32
        self.wide = wide
        self.vectorized = vectorized
        self.block_k = block_k
        self.stages = stages
        self.smem_plan = SmemPlan()
        k = self.smem_plan.buffer("K", cutlass.BFloat16, align=128)
        v = self.smem_plan.buffer("V", cutlass.BFloat16, align=128)
        index = self.smem_plan.buffer("index", Int32)
        # This is a purely thread-to-thread exchange. Named barriers synchronize
        # all writers/readers directly, with one FULL/EMPTY pair per stage.
        self.kv = Ring(
            stages, Threads, Threads, sync="named", buffer={"K": k, "V": v, "index": index}
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"))

    def get_name(self):
        direction = "bwd" if self.backward else "fwd"
        return f"gather_grouped_{direction}_d{self.dim}_h{self.heads}_kv{self.kv_heads}"

    @cute.jit
    def address(self, value):
        return Int64(value) if cutlass.const_expr(self.wide) else Int32(value)

    @cute.jit
    def bounds(self, p: GatherParams, token):
        q_start, k_start, k_end = Int32(0), Int32(0), Int32(p.sk.shape[2])
        active = Boolean(True)
        if cutlass.const_expr(self.packed):
            # upper_bound handles repeated offsets (empty documents).
            lo, hi = Int32(0), Int32(p.cu_q.shape[0] - 1)
            active = token < p.cu_q[hi]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if p.cu_q[mid] <= token:
                    lo = mid
                else:
                    hi = mid - 1
            if active:
                q_start = p.cu_q[lo]
                k_start, k_end = p.cu_k[lo], p.cu_k[lo + 1]
        local_n = Int32(0)
        if cutlass.const_expr(self.has_window):
            local_n = cutlass.min(p.window, token - q_start + 1)
        local_n = cutlass.select_(active, local_n, Int32(0))
        total = cutlass.select_(active, local_n + Int32(p.indices.shape[2]), Int32(0))
        return local_n, total, k_start, k_end, active

    @role(warps=range(1, 4))
    @cute.jit
    def load(self, p: GatherParams, *, kv: Producer, work: WorkCursor, me: Local):
        while work.valid:
            token, head_tile, batch = work.item
            kv_head = head_tile // self.head_tiles
            local_n, total, k_start, k_end, _ = self.bounds(p, token)
            for base in cutlass.range(0, total, self.block_k, unroll=1):
                with kv.produce() as tile:
                    data = tile.data
                    # Each loader warp gathers complete rows, keeping global accesses
                    # contiguous across feature lanes even when selected rows are scattered.
                    for row in cutlass.range(me.warp, self.block_k, me.warps, unroll=1):
                        slot = base + row
                        index = Int32(-1)
                        local = slot < local_n
                        if slot < total:
                            if local:
                                index = token - local_n + 1 + slot
                            else:
                                selected = p.indices[
                                    self.address(batch),
                                    self.address(token),
                                    self.address(slot - local_n),
                                ]
                                if (selected >= 0) & (selected < k_end - k_start):
                                    index = Int32(selected) + k_start
                        if me.lane == 0:
                            data["index"][row] = index
                        if cutlass.const_expr(self.vectorized):  # noqa: SIM102
                            if me.lane * 8 < self.dim:
                                key_vec = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                                value_vec = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                                key_vec.fill(0)
                                value_vec.fill(0)
                                if index >= 0:
                                    coord = (
                                        self.address(batch),
                                        self.address(kv_head),
                                        self.address(index),
                                        None,
                                    )
                                    if cutlass.const_expr(self.has_window):
                                        if local:
                                            cute.autovec_copy(
                                                cute.local_tile(p.lk[coord], (8,), (me.lane,)),
                                                key_vec,
                                            )
                                            cute.autovec_copy(
                                                cute.local_tile(p.lv[coord], (8,), (me.lane,)),
                                                value_vec,
                                            )
                                        else:
                                            cute.autovec_copy(
                                                cute.local_tile(p.sk[coord], (8,), (me.lane,)),
                                                key_vec,
                                            )
                                            cute.autovec_copy(
                                                cute.local_tile(p.sv[coord], (8,), (me.lane,)),
                                                value_vec,
                                            )
                                    else:
                                        cute.autovec_copy(
                                            cute.local_tile(p.sk[coord], (8,), (me.lane,)),
                                            key_vec,
                                        )
                                        cute.autovec_copy(
                                            cute.local_tile(p.sv[coord], (8,), (me.lane,)),
                                            value_vec,
                                        )
                                cute.autovec_copy(
                                    key_vec,
                                    cute.local_tile(data["K"][None, row], (8,), (me.lane,)),
                                )
                                cute.autovec_copy(
                                    value_vec,
                                    cute.local_tile(data["V"][None, row], (8,), (me.lane,)),
                                )
                        for part in cutlass.range_constexpr(
                            0 if self.vectorized else cute.ceil_div(self.dim, 32)
                        ):
                            d = me.lane + part * 32
                            if d < self.dim:
                                key, value = Float32(0), Float32(0)
                                if index >= 0:
                                    coord = (
                                        self.address(batch),
                                        self.address(kv_head),
                                        self.address(index),
                                        self.address(d),
                                    )
                                    if cutlass.const_expr(self.has_window):
                                        if local:
                                            key = p.lk[coord].to(Float32)
                                            value = p.lv[coord].to(Float32)
                                        else:
                                            key = p.sk[coord].to(Float32)
                                            value = p.sv[coord].to(Float32)
                                    else:
                                        key = p.sk[coord].to(Float32)
                                        value = p.sv[coord].to(Float32)
                                data["K"][d, row] = key.to(cutlass.BFloat16)
                                data["V"][d, row] = value.to(cutlass.BFloat16)
            work.advance()
        return kv, work

    @cute.jit
    def __call__(
        self,
        q,
        sk,
        lk,
        sv,
        lv,
        indices,
        sink,
        cu_q,
        cu_k,
        out,
        lse,
        go,
        dq,
        dsk,
        dlk,
        dsv,
        dlv,
        dsink,
        scale: Float32,
        window: Int32,
        stream,
    ):
        p = GatherParams(
            q,
            sk,
            lk,
            sv,
            lv,
            indices,
            sink,
            cu_q,
            cu_k,
            out,
            lse,
            go,
            dq,
            dsk,
            dlk,
            dsv,
            dlv,
            dsink,
            scale,
            window,
        )
        bound = self.work.bind((q.shape[2], self.head_tiles * self.kv_heads, q.shape[0]))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(p, bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_grouped(op: GatherGrouped):
    size = cute.sym_int64 if op.wide else cute.sym_int
    batch, tokens, sparse_tokens, topk, docs = (size() for _ in range(5))

    def tensor(dtype, shape, contiguous=-1, vector=False):
        return make_fake_strided_tensor(
            dtype,
            shape,
            contiguous_dim=contiguous,
            use_int64_strides=op.wide,
            stride_divisibility=8 if vector else 1,
        )

    q_shape = (batch, op.heads, tokens, op.dim)
    k_shape = (batch, op.kv_heads, sparse_tokens, op.dim)
    l_shape = (batch, op.kv_heads, tokens, op.dim)
    indices = tensor(Int64 if op.index64 else Int32, (batch, tokens, topk), None)

    def optional(dtype, shape, enabled=True, contiguous=-1, vector=False):
        return tensor(dtype, shape, contiguous, vector) if enabled else None

    return compile_tvm_ffi(
        op,
        tensor(cutlass.BFloat16, q_shape),
        tensor(cutlass.BFloat16, k_shape, vector=op.vectorized),
        optional(cutlass.BFloat16, l_shape, op.has_window, vector=op.vectorized),
        tensor(cutlass.BFloat16, k_shape, vector=op.vectorized),
        optional(cutlass.BFloat16, l_shape, op.has_window, vector=op.vectorized),
        indices,
        optional(Float32 if op.sink_fp32 else cutlass.BFloat16, (op.heads,), op.has_sink, None),
        optional(Int32, (docs,), op.packed),
        optional(Int32, (docs,), op.packed),
        tensor(cutlass.BFloat16, q_shape),
        tensor(Float32, (batch, op.heads, tokens)),
        optional(cutlass.BFloat16, q_shape, op.backward),
        optional(cutlass.BFloat16, q_shape, op.backward),
        optional(Float32, k_shape, op.backward),
        optional(Float32, l_shape, op.backward and op.has_window),
        optional(Float32, k_shape, op.backward and op.separate_values),
        optional(Float32, l_shape, op.backward and op.separate_values and op.has_window),
        optional(Float32, (op.heads,), op.backward and op.has_sink),
        Float32(1),
        Int32(1),
        name=op.get_name(),
    )


def _launch(
    query,
    local_kv,
    sparse_kv,
    kv_indices,
    attention_sink,
    cu_seqlens,
    cu_seqlens_k,
    sliding_window_size,
    scale,
    *,
    sparse_v=None,
    local_v=None,
    output=None,
    lse=None,
    grad_output=None,
):
    from .routing import require_grouped_schedule

    require_grouped_schedule(
        query,
        [x for x in (sparse_kv, local_kv, sparse_v, local_v) if x is not None],
        sliding_window_size=sliding_window_size,
        backend="cute",
    )
    backward = grad_output is not None
    if backward:
        # Honor deterministic mode even if enabled only after the forward.
        from .cute import _check_backward_mode

        _check_backward_mode()
    query = query if query.stride(-1) == 1 else query.contiguous()
    local = local_kv if sliding_window_size else None
    values = sparse_kv if sparse_v is None else sparse_v
    local_values = (local if local_v is None else local_v) if sliding_window_size else None
    sink = attention_sink
    batch, heads, tokens, dim = query.shape
    if not backward:
        output = query.new_empty((batch, tokens, heads, dim)).permute(0, 2, 1, 3)
        lse = query.new_empty((batch, heads, tokens), dtype=torch.float32)
        grad_output, dq = None, None
        dsk = dlk = dsv = dlv = dsink = None
    else:
        grad_output = grad_output if grad_output.stride(-1) == 1 else grad_output.contiguous()
        dq = torch.zeros(query.shape, dtype=query.dtype, device=query.device)
        dsk = torch.zeros(sparse_kv.shape, dtype=torch.float32, device=query.device)
        dlk = (
            None
            if local_kv is None
            else torch.zeros(local_kv.shape, dtype=torch.float32, device=query.device)
        )
        dsv = (
            None
            if sparse_v is None
            else torch.zeros(sparse_v.shape, dtype=torch.float32, device=query.device)
        )
        dlv = (
            None
            if local_v is None
            else torch.zeros(local_v.shape, dtype=torch.float32, device=query.device)
        )
        dsink = (
            None
            if attention_sink is None
            else torch.zeros((heads,), dtype=torch.float32, device=query.device)
        )
    args = (
        query,
        sparse_kv,
        local,
        values,
        local_values,
        kv_indices,
        sink,
        cu_seqlens,
        cu_seqlens_k,
        output,
        lse,
        grad_output,
        dq,
        dsk,
        dlk if sliding_window_size else None,
        dsv,
        dlv if sliding_window_size else None,
        dsink,
    )
    geometry = (dim, heads, sparse_kv.shape[1], sliding_window_size > 0)
    options = {
        "backward": backward,
        "separate_values": backward and sparse_v is not None,
        "has_sink": attention_sink is not None,
        "packed": cu_seqlens is not None,
        "index64": kv_indices.dtype == torch.int64,
        "sink_fp32": sink is not None and sink.dtype == torch.float32,
        "wide": requires_int64_abi(*args),
        "vectorized": all(
            tensor_supports_contiguous_dim(x, alignment_bytes=16)
            for x in (sparse_kv, local, values, local_values)
            if x is not None
        ),
    }
    from .cute_grouped_mma import GatherGroupedMma, GatherGroupedMmaBackward

    kernel_type = GatherGroupedMmaBackward if backward else GatherGroupedMma
    compiled = compile_grouped.by_static_key(
        (*geometry, *options.values()),
        lambda: compile_grouped(kernel_type(*geometry, **options)),
    )
    compiled(*args, scale, sliding_window_size)
    if not backward:
        return output, lse, ()
    return (
        dq,
        dsk.to(query.dtype),
        None if dlk is None else dlk.to(query.dtype),
        None,
        None,
        None,
        None if attention_sink is None else dsink.to(attention_sink.dtype),
        None if dsv is None else dsv.to(query.dtype),
        None if dlv is None else dlv.to(query.dtype),
    )
