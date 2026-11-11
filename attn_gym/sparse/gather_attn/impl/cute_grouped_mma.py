"""Tensor-core tiles for warpdrive grouped gather attention.

One CTA owns one query and up to 16 heads of its KV group. Three loader warps
publishes 32 selected K/V rows per channel epoch; a compute warp uses synchronous
mma.sync for QK and PV. The compute warp alone owns Q and P scratch, with a
32-thread named rendezvous before reading values written by other lanes.
The K/V handle remains live through all matrix products. Backward uses the same
handoff, recomputing probabilities and reducing heads before scattering K/V gradients.
"""

import cutlass
from cutlass import Float32, cute
from cutlass.cute.nvgpu import warp

from attn_gym.warpdrive import Consumer, Local, NamedBarrier, RoleConfig, WorkCursor, role

from .cute_grouped import LN2, LOG2E, GatherGrouped, GatherParams


class GatherGroupedMma(GatherGrouped):
    def __init__(self, *args, block_k=32, loader_warps=3, **kwargs):
        super().__init__(*args, block_k=block_k, **kwargs)
        self.roles = {"load": RoleConfig(warps=range(1, 1 + loader_warps))}
        self.smem_plan.buffer("Q", cutlass.BFloat16, align=128)
        self.smem_plan.buffer("P", cutlass.BFloat16, align=128)
        if self.backward:
            self.smem_plan.buffer("GO", cutlass.BFloat16, align=128)
            self.smem_plan.buffer("DS", cutlass.BFloat16, align=128)
        self.compute_sync = NamedBarrier(threads=32)

    def get_name(self):
        return super().get_name() + "_mma"

    @cute.jit
    def gemm(self, mma, a, b, acc, lane, *, transpose_a=False, transpose_b=False):
        thr = mma.get_slice(lane)
        ca = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=transpose_a, num_matrices=4),
                cutlass.BFloat16,
            ),
            mma,
        )
        cb = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=transpose_b, num_matrices=4),
                cutlass.BFloat16,
            ),
            mma,
        )
        ta, tb = ca.get_slice(lane), cb.get_slice(lane)
        sa, sb = ta.partition_S(a), tb.partition_S(b)
        ra = thr.make_fragment_A(thr.partition_A(a))
        rb = thr.make_fragment_B(thr.partition_B(b))
        for k in cutlass.range_constexpr(cute.size(sa, mode=[2])):
            cute.copy(ca, sa[None, None, k], ta.retile(ra)[None, None, k])
            cute.copy(cb, sb[None, None, k], tb.retile(rb)[None, None, k])
            cute.gemm(mma, acc, ra[None, None, k], rb[None, None, k], acc)

    @role(warps=0)
    @cute.jit
    def compute(self, p: GatherParams, *, kv: Consumer, work: WorkCursor, me: Local):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        thr = mma.get_slice(me.lane)
        sq, sp = self.buffer("Q"), self.buffer("P")
        cs = thr.partition_C(cute.make_identity_tensor((16, self.block_k)))
        co = thr.partition_C(cute.make_identity_tensor((16, self.dim)))
        acc = cute.make_rmem_tensor(co.shape, Float32)
        scores = cute.make_rmem_tensor(cs.shape, Float32)
        # mma.sync's C lanes form groups of four, each owning two head rows.
        maximum = cute.make_rmem_tensor((2,), Float32)
        denom = cute.make_rmem_tensor((2,), Float32)
        alpha = cute.make_rmem_tensor((2,), Float32)
        while work.valid:
            token, head_tile, batch = work.item
            kv_head = head_tile // self.head_tiles
            first_head = head_tile % self.head_tiles * 16
            _, total, _, _, active = self.bounds(p, token)
            for i in cutlass.range_constexpr(16 * self.dim // 32):
                linear = me.lane + i * 32
                h, d = linear // self.dim, linear % self.dim
                q_value = cutlass.BFloat16(0)
                if active & (first_head + h < self.group):
                    q_value = p.q[
                        self.address(batch),
                        self.address(kv_head * self.group + first_head + h),
                        self.address(token),
                        self.address(d),
                    ]
                sq[h, d] = q_value
            self.compute_sync.sync()
            acc.fill(0)
            maximum.fill(-float("inf"))
            denom.fill(0)
            for r in cutlass.range_constexpr(2):
                h = first_head + me.lane // 4 + r * 8
                if cutlass.const_expr(self.has_sink):  # noqa: SIM102
                    if active & (h < self.group):
                        maximum[r] = p.sink[self.address(kv_head * self.group + h)].to(Float32)
                        denom[r] = cutlass.select_(
                            maximum[r] > -float("inf"), Float32(1), Float32(0)
                        )
            for base in cutlass.range(0, total, self.block_k, unroll=1):
                with kv.consume() as tile:
                    data = tile.data
                    scores.fill(0)
                    sk = cute.composition(
                        data["K"],
                        cute.make_layout((self.block_k, self.dim), stride=(self.dim, 1)),
                    )
                    self.gemm(mma, sq, sk, scores, me.lane)
                    for i in cutlass.range_constexpr(cute.size(scores)):
                        h, k = cs[i]
                        score = scores[i] * p.scale
                        if (first_head + h >= self.group) | (data["index"][k] < 0):
                            score = Float32(-float("inf"))
                        scores[i] = score
                    for r in cutlass.range_constexpr(2):
                        mx = maximum[r]
                        for i in cutlass.range_constexpr(cute.size(scores)):
                            if cutlass.const_expr(i // 2 % 2 == r):
                                mx = cutlass.max(mx, scores[i])
                        mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, 1))
                        mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, 2))
                        safe_mx = cutlass.select_(mx > -float("inf"), mx, Float32(0))
                        alpha[r] = cute.math.exp2((maximum[r] - safe_mx) * LOG2E)
                        maximum[r] = mx
                        sm = Float32(0)
                        for i in cutlass.range_constexpr(cute.size(scores)):
                            if cutlass.const_expr(i // 2 % 2 == r):
                                prob = cute.math.exp2((scores[i] - safe_mx) * LOG2E)
                                sp[cs[i]] = prob.to(cutlass.BFloat16)
                                sm += prob
                        sm += cute.arch.shuffle_sync_bfly(sm, 1)
                        sm += cute.arch.shuffle_sync_bfly(sm, 2)
                        denom[r] = denom[r] * alpha[r] + sm
                    for i in cutlass.range_constexpr(cute.size(acc)):
                        acc[i] *= alpha[i // 2 % 2]
                    self.compute_sync.sync()
                    self.gemm(mma, sp, data["V"], acc, me.lane, transpose_b=True)
            for i in cutlass.range_constexpr(cute.size(acc)):
                h, d = co[i]
                if active & (first_head + h < self.group):
                    value = Float32(0)
                    if denom[i // 2 % 2] > 0:
                        value = acc[i] / denom[i // 2 % 2]
                    p.out[
                        self.address(batch),
                        self.address(kv_head * self.group + first_head + h),
                        self.address(token),
                        self.address(d),
                    ] = value.to(cutlass.BFloat16)
            if me.lane % 4 == 0:
                for r in cutlass.range_constexpr(2):
                    h = first_head + me.lane // 4 + r * 8
                    if active & (h < self.group):
                        lse = Float32(-float("inf"))
                        if denom[r] > 0:
                            lse = maximum[r] + cute.math.log2(denom[r]) * LN2
                        p.lse[
                            self.address(batch),
                            self.address(kv_head * self.group + h),
                            self.address(token),
                        ] = lse
            work.advance()
        return kv, work

    @cute.kernel
    def kernel(self, p: GatherParams, bound):
        # SmemPlan places the swizzle on the pointer, where it operates in bytes.
        # Preserve the low four bits for ldmatrix's 16-byte row alignment.
        swizzle_bits = min(3, (self.dim & -self.dim).bit_length() - 4)
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3),
            0,
            cute.make_layout((8, 8 << swizzle_bits), stride=(8 << swizzle_bits, 1)),
        )
        q_layout = cute.tile_to_shape(atom, (16, self.dim), (0, 1))
        kv_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3),
            0,
            cute.make_layout((8 << swizzle_bits, 8)),
        )
        kv_layout = cute.tile_to_shape(kv_atom, (self.dim, self.block_k, self.stages), (0, 1, 2))
        layouts = {
            "K": kv_layout,
            "V": kv_layout,
            "index": cute.make_layout((self.block_k, self.stages)),
            "Q": q_layout,
            "P": cute.make_layout((16, self.block_k), stride=(self.block_k, 1)),
        }
        if cutlass.const_expr(self.backward):
            layouts["GO"] = q_layout
            layouts["DS"] = layouts["P"]
        self.begin_roles(
            work=bound,
            layouts=layouts,
        )
        self.run_roles(p)


class GatherGroupedMmaBackward(GatherGroupedMma):
    @cute.jit
    def scatter(self, p, values, coords, indices, batch, kv_head, base, local_n, feature, is_v):
        for i in cutlass.range_constexpr(cute.size(values)):
            row, d = coords[i]
            index = indices[row]
            if index >= 0:
                coord = (
                    self.address(batch),
                    self.address(kv_head),
                    self.address(index),
                    self.address(feature + d),
                )
                if cutlass.const_expr(self.has_window):
                    if base + row < local_n:
                        dst = p.dlv if cutlass.const_expr(is_v) else p.dlk
                        cute.arch.atomic_add(
                            dst.iterator + dst.layout(coord), values[i], sem="relaxed", scope="gpu"
                        )
                    else:
                        dst = p.dsv if cutlass.const_expr(is_v) else p.dsk
                        cute.arch.atomic_add(
                            dst.iterator + dst.layout(coord), values[i], sem="relaxed", scope="gpu"
                        )
                else:
                    dst = p.dsv if cutlass.const_expr(is_v) else p.dsk
                    cute.arch.atomic_add(
                        dst.iterator + dst.layout(coord), values[i], sem="relaxed", scope="gpu"
                    )

    @role(warps=0)
    @cute.jit
    def compute(self, p: GatherParams, *, kv: Consumer, work: WorkCursor, me: Local):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        thr = mma.get_slice(me.lane)
        sq, sp = self.buffer("Q"), self.buffer("P")
        sgo, sds = self.buffer("GO"), self.buffer("DS")
        cs = thr.partition_C(cute.make_identity_tensor((16, self.block_k)))
        co = thr.partition_C(cute.make_identity_tensor((16, self.dim)))
        feature_tile = 32 if self.dim % 32 == 0 else 16
        cg = thr.partition_C(cute.make_identity_tensor((self.block_k, feature_tile)))
        dq = cute.make_rmem_tensor(co.shape, Float32)
        scores = cute.make_rmem_tensor(cs.shape, Float32)
        dp = cute.make_rmem_tensor(cs.shape, Float32)
        dg = cute.make_rmem_tensor(cg.shape, Float32)
        delta = cute.make_rmem_tensor((2,), Float32)
        lse = cute.make_rmem_tensor((2,), Float32)
        while work.valid:
            token, head_tile, batch = work.item
            kv_head = head_tile // self.head_tiles
            first_head = head_tile % self.head_tiles * 16
            local_n, total, _, _, active = self.bounds(p, token)
            delta.fill(0)
            lse.fill(0)
            dq.fill(0)
            for i in cutlass.range_constexpr(cute.size(co)):
                h, d = co[i]
                q_value, go_value = cutlass.BFloat16(0), cutlass.BFloat16(0)
                if active & (first_head + h < self.group):
                    coord = (
                        self.address(batch),
                        self.address(kv_head * self.group + first_head + h),
                        self.address(token),
                        self.address(d),
                    )
                    q_value, go_value = p.q[coord], p.go[coord]
                    delta[i // 2 % 2] += go_value.to(Float32) * p.out[coord].to(Float32)
                sq[h, d], sgo[h, d] = q_value, go_value
            self.compute_sync.sync()
            for r in cutlass.range_constexpr(2):
                delta[r] += cute.arch.shuffle_sync_bfly(delta[r], 1)
                delta[r] += cute.arch.shuffle_sync_bfly(delta[r], 2)
                h = first_head + me.lane // 4 + r * 8
                if active & (h < self.group):
                    lse[r] = p.lse[
                        self.address(batch),
                        self.address(kv_head * self.group + h),
                        self.address(token),
                    ]
            for base in cutlass.range(0, total, self.block_k, unroll=1):
                with kv.consume() as tile:
                    data = tile.data
                    kt = cute.composition(
                        data["K"],
                        cute.make_layout((self.block_k, self.dim), stride=(self.dim, 1)),
                    )
                    vt = cute.composition(
                        data["V"],
                        cute.make_layout((self.block_k, self.dim), stride=(self.dim, 1)),
                    )
                    scores.fill(0)
                    dp.fill(0)
                    self.gemm(mma, sq, kt, scores, me.lane)
                    self.gemm(mma, sgo, vt, dp, me.lane)
                    for i in cutlass.range_constexpr(cute.size(scores)):
                        h, k = cs[i]
                        prob, ds = Float32(0), Float32(0)
                        if (first_head + h < self.group) & (data["index"][k] >= 0):
                            prob = cute.math.exp2((scores[i] * p.scale - lse[i // 2 % 2]) * LOG2E)
                            ds = prob * (dp[i] - delta[i // 2 % 2]) * p.scale
                        sp[h, k] = prob.to(cutlass.BFloat16)
                        sds[h, k] = ds.to(cutlass.BFloat16)
                    self.compute_sync.sync()
                    self.gemm(mma, sds, data["K"], dq, me.lane, transpose_b=True)
                    pt = cute.composition(sp, cute.make_layout((self.block_k, 16), stride=(16, 1)))
                    dst = cute.composition(
                        sds, cute.make_layout((self.block_k, 16), stride=(16, 1))
                    )
                    qt = cute.composition(sq, cute.make_layout((self.dim, 16), stride=(16, 1)))
                    got = cute.composition(sgo, cute.make_layout((self.dim, 16), stride=(16, 1)))
                    for feature in cutlass.range_constexpr(self.dim // feature_tile):
                        q_part = cute.local_tile(qt, (feature_tile, 16), (feature, 0))
                        go_part = cute.local_tile(got, (feature_tile, 16), (feature, 0))
                        dg.fill(0)
                        self.gemm(
                            mma, dst, q_part, dg, me.lane, transpose_a=True, transpose_b=True
                        )
                        if cutlass.const_expr(not self.separate_values):
                            self.gemm(
                                mma, pt, go_part, dg, me.lane, transpose_a=True, transpose_b=True
                            )
                        self.scatter(
                            p,
                            dg,
                            cg,
                            data["index"],
                            batch,
                            kv_head,
                            base,
                            local_n,
                            feature * feature_tile,
                            False,
                        )
                        if cutlass.const_expr(self.separate_values):
                            dg.fill(0)
                            self.gemm(
                                mma, pt, go_part, dg, me.lane, transpose_a=True, transpose_b=True
                            )
                            self.scatter(
                                p,
                                dg,
                                cg,
                                data["index"],
                                batch,
                                kv_head,
                                base,
                                local_n,
                                feature * feature_tile,
                                True,
                            )
            for i in cutlass.range_constexpr(cute.size(dq)):
                h, d = co[i]
                if active & (first_head + h < self.group):
                    p.dq[
                        self.address(batch),
                        self.address(kv_head * self.group + first_head + h),
                        self.address(token),
                        self.address(d),
                    ] = dq[i].to(cutlass.BFloat16)
            if cutlass.const_expr(self.has_sink):  # noqa: SIM102
                if me.lane % 4 == 0:
                    for r in cutlass.range_constexpr(2):
                        h = first_head + me.lane // 4 + r * 8
                        if active & (h < self.group) & (lse[r] > -float("inf")):
                            head = self.address(kv_head * self.group + h)
                            dsink = (
                                -cute.math.exp2((p.sink[head].to(Float32) - lse[r]) * LOG2E)
                                * delta[r]
                            )
                            cute.arch.atomic_add(
                                p.dsink.iterator + p.dsink.layout((head,)),
                                dsink,
                                sem="relaxed",
                                scope="gpu",
                            )
            work.advance()
        return kv, work
