"""Warpdrive gather attention for separate values and grouped KV heads.

One CTA owns a query and up to four heads of one KV group. Four loading warps gather
K/V into a two-stage SMEM ring; four compute warps each own one query head. Every
stage exchanges ``block_k`` occurrences, including duplicates and invalid padding.
The first two empty stages seed progress. All readers release before a stage is
overwritten, including warps outside a partial head group.

The forward uses FP32 online softmax. Backward recomputes probabilities and accumulates
K/V and sink gradients with FP32 atomics; it is not deterministic. Only selected pool
rows are read. There are no raw barriers, asynchronous copies, or storage aliases.
"""

from dataclasses import dataclass

import cutlass
import torch
from cutlass import Boolean, Float32, Int32, Int64, cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive import (
    Consumer,
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
        block_k: int = 16,
        stages: int = 2,
    ):
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.group = heads // kv_heads
        self.head_tiles = cute.ceil_div(self.group, 4)
        self.has_window = has_window
        self.backward = backward
        self.separate_values = separate_values
        self.has_sink = has_sink
        self.packed = packed
        self.index64 = index64
        self.sink_fp32 = sink_fp32
        self.wide = wide
        self.block_k = block_k
        self.stages = stages
        self.smem_plan = SmemPlan()
        k = self.smem_plan.buffer("K", cutlass.BFloat16)
        v = self.smem_plan.buffer("V", cutlass.BFloat16)
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

    @role(warps=range(4, 8))
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
                        for part in cutlass.range_constexpr(cute.ceil_div(self.dim, 32)):
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

    @role(warps=range(4))
    @cute.jit
    def compute(self, p: GatherParams, *, kv: Consumer, work: WorkCursor, me: Local):
        width = cute.ceil_div(self.dim, 32)
        q = cute.make_rmem_tensor((width,), Float32)
        acc = cute.make_rmem_tensor((width,), Float32)
        go = cute.make_rmem_tensor((width,), Float32)
        kr = cute.make_rmem_tensor((width,), Float32)
        vr = cute.make_rmem_tensor((width,), Float32)
        while work.valid:
            token, head_tile, batch = work.item
            kv_head = head_tile // self.head_tiles
            group_head = head_tile % self.head_tiles * 4 + me.warp
            head = kv_head * self.group + group_head
            local_n, total, _, _, active = self.bounds(p, token)
            valid_head = (group_head < self.group) & active
            delta, row_lse = Float32(0), Float32(0)
            maximum, denominator = Float32(-float("inf")), Float32(0)
            if cutlass.const_expr(self.has_sink):  # noqa: SIM102
                if valid_head:
                    maximum = p.sink[self.address(head)].to(Float32)
                    denominator = cutlass.select_(maximum > -float("inf"), Float32(1), Float32(0))
            for part in cutlass.range_constexpr(width):
                d = me.lane + part * 32
                q[part], acc[part], go[part] = Float32(0), Float32(0), Float32(0)
                if valid_head & (d < self.dim):
                    coord = (
                        self.address(batch),
                        self.address(head),
                        self.address(token),
                        self.address(d),
                    )
                    q[part] = p.q[coord].to(Float32)
                    if cutlass.const_expr(self.backward):
                        go[part] = p.go[coord].to(Float32)
                        delta += go[part] * p.out[coord].to(Float32)
            if cutlass.const_expr(self.backward):
                delta = cute.arch.warp_reduction_sum(delta)
                if valid_head:
                    row_lse = p.lse[self.address(batch), self.address(head), self.address(token)]
            for base in cutlass.range(0, total, self.block_k, unroll=1):
                # Every declared consumer participates, even in partial head groups.
                with kv.consume() as tile:
                    data = tile.data
                    for row in cutlass.range(self.block_k, unroll=1):
                        index = data["index"][row]
                        if valid_head & (index >= 0):
                            score, dp = Float32(0), Float32(0)
                            for part in cutlass.range_constexpr(width):
                                d = me.lane + part * 32
                                kr[part], vr[part] = Float32(0), Float32(0)
                                if d < self.dim:
                                    kr[part] = data["K"][d, row].to(Float32)
                                    vr[part] = data["V"][d, row].to(Float32)
                                score += q[part] * kr[part]
                                if cutlass.const_expr(self.backward):
                                    dp += go[part] * vr[part]
                            score = cute.arch.warp_reduction_sum(score) * p.scale
                            if cutlass.const_expr(self.backward):
                                prob = cute.math.exp2((score - row_lse) * LOG2E)
                                dp = cute.arch.warp_reduction_sum(dp)
                                ds = prob * (dp - delta) * p.scale
                                for part in cutlass.range_constexpr(width):
                                    d = me.lane + part * 32
                                    acc[part] += ds * kr[part]
                                    if d < self.dim:
                                        coord = (
                                            self.address(batch),
                                            self.address(kv_head),
                                            self.address(index),
                                            self.address(d),
                                        )
                                        dk = ds * q[part]
                                        dv = prob * go[part]
                                        if cutlass.const_expr(not self.separate_values):
                                            dk += dv
                                        if cutlass.const_expr(self.has_window):
                                            if base + row < local_n:
                                                cute.arch.atomic_add(
                                                    p.dlk.iterator + p.dlk.layout(coord),
                                                    dk,
                                                    sem="relaxed",
                                                    scope="gpu",
                                                )
                                                if cutlass.const_expr(self.separate_values):
                                                    cute.arch.atomic_add(
                                                        p.dlv.iterator + p.dlv.layout(coord),
                                                        dv,
                                                        sem="relaxed",
                                                        scope="gpu",
                                                    )
                                            else:
                                                cute.arch.atomic_add(
                                                    p.dsk.iterator + p.dsk.layout(coord),
                                                    dk,
                                                    sem="relaxed",
                                                    scope="gpu",
                                                )
                                                if cutlass.const_expr(self.separate_values):
                                                    cute.arch.atomic_add(
                                                        p.dsv.iterator + p.dsv.layout(coord),
                                                        dv,
                                                        sem="relaxed",
                                                        scope="gpu",
                                                    )
                                        else:
                                            cute.arch.atomic_add(
                                                p.dsk.iterator + p.dsk.layout(coord),
                                                dk,
                                                sem="relaxed",
                                                scope="gpu",
                                            )
                                            if cutlass.const_expr(self.separate_values):
                                                cute.arch.atomic_add(
                                                    p.dsv.iterator + p.dsv.layout(coord),
                                                    dv,
                                                    sem="relaxed",
                                                    scope="gpu",
                                                )
                            else:
                                next_max = cutlass.max(maximum, score)
                                alpha = cute.math.exp2((maximum - next_max) * LOG2E)
                                prob = cute.math.exp2((score - next_max) * LOG2E)
                                denominator = denominator * alpha + prob
                                maximum = next_max
                                for part in cutlass.range_constexpr(width):
                                    acc[part] = acc[part] * alpha + prob * vr[part]
            if valid_head:
                row_coord = (self.address(batch), self.address(head), self.address(token))
                for part in cutlass.range_constexpr(width):
                    d = me.lane + part * 32
                    if d < self.dim:
                        coord = (*row_coord, self.address(d))
                        if cutlass.const_expr(self.backward):
                            p.dq[coord] = acc[part].to(cutlass.BFloat16)
                        else:
                            value = Float32(0)
                            if denominator > 0:
                                value = acc[part] / denominator
                            p.out[coord] = value.to(cutlass.BFloat16)
                if me.lane == 0:
                    if cutlass.const_expr(self.backward):
                        if cutlass.const_expr(self.has_sink):  # noqa: SIM102
                            if row_lse > -float("inf"):
                                dsink = -cute.math.exp2((maximum - row_lse) * LOG2E) * delta
                                cute.arch.atomic_add(
                                    p.dsink.iterator + p.dsink.layout((self.address(head),)),
                                    dsink,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                    else:
                        result_lse = Float32(-float("inf"))
                        if denominator > 0:
                            result_lse = maximum + cute.math.log2(denominator) * LN2
                        p.lse[row_coord] = result_lse
            work.advance()
        return kv, work

    @cute.kernel
    def kernel(self, p: GatherParams, bound):
        self.begin_roles(
            work=bound,
            layouts={
                "K": cute.make_layout((self.dim, self.block_k, self.stages)),
                "V": cute.make_layout((self.dim, self.block_k, self.stages)),
                "index": cute.make_layout((self.block_k, self.stages)),
            },
        )
        self.run_roles(p)

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

    def tensor(dtype, shape, contiguous=-1):
        return make_fake_strided_tensor(
            dtype,
            shape,
            contiguous_dim=contiguous,
            use_int64_strides=op.wide,
        )

    q_shape = (batch, op.heads, tokens, op.dim)
    k_shape = (batch, op.kv_heads, sparse_tokens, op.dim)
    l_shape = (batch, op.kv_heads, tokens, op.dim)
    indices = tensor(Int64 if op.index64 else Int32, (batch, tokens, topk), None)

    def optional(dtype, shape, enabled=True, contiguous=-1):
        return tensor(dtype, shape, contiguous) if enabled else None

    return compile_tvm_ffi(
        op,
        tensor(cutlass.BFloat16, q_shape),
        tensor(cutlass.BFloat16, k_shape),
        optional(cutlass.BFloat16, l_shape, op.has_window),
        tensor(cutlass.BFloat16, k_shape),
        optional(cutlass.BFloat16, l_shape, op.has_window),
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
    }
    compiled = compile_grouped.by_static_key(
        (*geometry, *options.values()),
        lambda: compile_grouped(GatherGrouped(*geometry, **options)),
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
