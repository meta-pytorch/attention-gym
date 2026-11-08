"""Device helpers shared by generated fuzz kernels (``gen.py`` emits calls to these).

Payload words carry their epoch: word ``idx`` of epoch ``e`` in slot ``slot`` of a channel with
``salt`` holds ``e * 4096 + tag(idx, slot, salt)``. A consumer recomputes the value it expects
from its own epoch counter, so a stale stage, a skipped publication, a slot swap, or a word no
producer wrote shows up as a mismatch; the host model checks the logged epochs independently.
"""

import cutlass
from cutlass import Int32, Int64, cute

EPOCH_SHIFT = 12
TAG_MASK = (1 << EPOCH_SHIFT) - 1


def word_tag(idx, slot, salt):
    """Host and device: the low bits of payload word ``idx`` (Python ints or Int32)."""
    return (idx * 7 + salt + slot * 1031) & TAG_MASK


def tma_value(tile, r, c, cols, salt):
    """Host and device: element ``(r, c)`` of TMA source tile ``tile``."""
    return tile * (1 << EPOCH_SHIFT) + (((r * cols + c) * 7 + salt) & TAG_MASK)


def mma_a(tile, r, k, salt):
    """Host and device: A operand ``(r, k)`` of MMA tile ``tile``, an integer in ``[-4, 3]``."""
    return ((tile * 5 + r * 3 + k * 7 + salt) & 7) - 4


def mma_b(tile, c, k, salt):
    """Host and device: B operand ``(c, k)`` (``D = A @ B^T``), an integer in ``[-4, 3]``."""
    return ((tile * 3 + c * 5 + k + salt * 3) & 7) - 4


@cute.jit
def cta_id():
    """Linear CTA id over a 1-, 2- or 3-D launch."""
    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    return Int32(bx) + Int32(gx) * (Int32(by) + Int32(gy) * Int32(bz))


@cute.jit
def jitter(e: Int32, tid: Int32, salt: cutlass.Constexpr[int], scale: cutlass.Constexpr[int]):
    """Spin for a pseudo-random, thread-dependent number of cycles on some epochs."""
    if cutlass.const_expr(scale > 0):
        h = (e * 1103515245 + tid * 12345 + salt) & 0x7FFFFFFF
        if (h >> 9) % 3 == 0:
            cycles = Int64((h & 255) * scale)
            t0 = cute.arch.clock64()
            while cute.arch.clock64() - t0 < cycles:
                pass


@cute.jit
def write_words(
    data,
    e: Int32,
    tid: Int32,
    slot: cutlass.Constexpr[int],
    off: cutlass.Constexpr[int],
    salt: cutlass.Constexpr[int],
):
    idx = tid + off
    data[idx] = e * (1 << EPOCH_SHIFT) + word_tag(idx, slot, salt)


@cute.jit
def check_words(
    data,
    e: Int32,
    tid: Int32,
    slot: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    salt: cutlass.Constexpr[int],
):
    """Mismatching words among ``tid, tid + nthreads, ...`` below ``width``."""
    bad = Int32(0)
    for k in cutlass.range_constexpr((width + nthreads - 1) // nthreads):
        idx = tid + k * nthreads
        if idx < width:
            want = e * (1 << EPOCH_SHIFT) + word_tag(idx, slot, salt)
            if data[idx] != want:
                bad = bad + 1
    return bad


@cute.jit
def check_tile(
    s,
    tile: Int32,
    tid: Int32,
    rows: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    nthreads: cutlass.Constexpr[int],
    salt: cutlass.Constexpr[int],
):
    """Mismatching elements of a TMA stage ``s`` (``(rows, cols)``) against source tile
    ``tile``."""
    bad = Int32(0)
    for k in cutlass.range_constexpr((rows * cols + nthreads - 1) // nthreads):
        flat = tid + k * nthreads
        if flat < rows * cols:
            r = flat // cols
            c = flat - r * cols
            if s[r, c] != tma_value(tile, r, c, cols, salt):
                bad = bad + 1
    return bad


@cute.jit
def put(clog, idx: Int32, value: Int32, pred):
    if pred:
        clog[idx] = value


@cute.jit
def put_capped(clog, base: Int32, k: Int32, cap: cutlass.Constexpr[int], value: Int32, pred):
    if pred & (k < cap):
        clog[base + k] = value


@cute.jit
def add(clog, idx: Int32, value: Int32):
    if value != 0:
        cute.arch.atomic_add(clog.iterator + idx, value)


@cute.jit
def mark_warp(clog, base: Int32, code: cutlass.Constexpr[int]):
    """Lane 0 of every executing warp adds ``code`` to its warp's presence word."""
    if cute.arch.lane_idx() == 0:
        warp = Int32(cute.arch.warp_idx())
        cute.arch.atomic_add(clog.iterator + (base + warp), Int32(code))


@cute.jit
def item_hash(m: Int32, n: Int32, li: Int32):
    return (m * 1009 + n * 917 + li * 101) & 0x3FFFFFFF


@cute.jit
def log_item(
    clog,
    hits,
    base: Int32,
    k: Int32,
    cap: cutlass.Constexpr[int],
    index,
    words,
    hit_idx: Int32,
    pred,
):
    m, n, li = words
    if pred:
        if k < cap:
            clog[base + k] = Int32(index)
            clog[base + cap + k] = item_hash(Int32(m), Int32(n), Int32(li))
        cute.arch.atomic_add(hits.iterator + hit_idx, Int32(1))


@cute.jit
def turn_log(clog, base: Int32, seat: cutlass.Constexpr[int], cap: cutlass.Constexpr[int], pred):
    """Append ``seat`` to a turnstile log (``clog[base]`` counts entries); run under the turn."""
    if pred:
        pos = clog[base]
        if pos < cap:
            clog[base + 1 + pos] = Int32(seat)
        clog[base] = pos + 1
        cute.arch.fence_acq_rel_cta()


@cute.jit
def warp_arrive(bar):
    """One arrival per warp, after the warp's earlier memory operations (``__syncwarp``)."""
    cute.arch.sync_warp()
    if cute.arch.lane_idx() == 0:
        cute.arch.mbarrier_arrive(bar)


@cute.jit
def log_raw_item(clog, base: Int32, cap: cutlass.Constexpr[int], valid, words, pred):
    """A raw role's ``single_item()``: count (0 or 1) and the item's hash."""
    m, n, li = words
    if pred:
        clog[base] = Int32(0)
        if valid:
            clog[base] = Int32(1)
            clog[base + 1 + cap] = item_hash(Int32(m), Int32(n), Int32(li))


@cute.jit
def table_row(table, bad, index, total):
    """Row ``index`` of an item table; a read outside ``[0, total)`` counts in ``bad[0]``."""
    i = Int32(index)
    if (i < 0) | (i >= total):
        cute.arch.atomic_add(bad.iterator, Int32(1))
    i = cutlass.max(cutlass.min(i, total - 1), 0)
    return table[i, 0], table[i, 1], table[i, 2]


PRODUCER_DELAY_NS = 20000


@cute.jit
def produce_rows(table, length, stage, plen: Int32):
    """PDL primary: let the dependent launch, wait, then copy ``stage`` into ``table`` and set
    ``length[0] = plen``. The dependent reads them after ``griddepcontrol.wait`` (PdlWait)."""
    cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    start = cute.arch.globaltimer_lo()
    while cute.arch.globaltimer_lo() - start < PRODUCER_DELAY_NS:
        pass
    for i in cutlass.range(tidx, table.shape[0], cute.arch.block_dim()[0], unroll=1):
        for w in cutlass.range_constexpr(3):
            table[i, w] = stage[i, w]
    if tidx == 0:
        length[0] = plen


@cute.jit
def mma_check(
    acc,
    crd,
    e: Int32,
    kblocks: cutlass.Constexpr[int],
    tiles: cutlass.Constexpr[int],
    k: cutlass.Constexpr[int],
    salt: cutlass.Constexpr[int],
):
    """Accumulator elements of epoch ``e`` (``crd``: their ``(row, col)``) that differ from the
    exact integer ``sum_j A(t_j) @ B(t_j)^T`` over the epoch's ``kblocks`` A/B stages."""
    bad = Int32(0)
    for j in cutlass.range_constexpr(cute.size(acc)):
        r, c = crd[j]
        want = Int32(0)
        for jb in cutlass.range_constexpr(kblocks):
            t = (e * kblocks + jb) % tiles
            for kk in cutlass.range(k, unroll=4):
                want = want + mma_a(t, Int32(r), kk, salt) * mma_b(t, Int32(c), kk, salt)
        if acc[j] != cutlass.Float32(want):
            bad = bad + 1
    return bad
