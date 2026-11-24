"""Tagged words, the megakernels' cross-CTA protocol.

Each activation crossing CTAs (or launches) is one 64-bit word ``(fp32 value, epoch)``
published with a relaxed gpu-scope store; a consumer spins on relaxed loads until every word it
needs carries the current epoch (NCCL's LL protocol). The word is single-copy atomic, so a
matching tag means the value is this epoch's: one write and one read propagation per
dependency, no fences and no separate flag. Buffers are int64 words in a flat scratch tensor
(``scratch_offsets``, ``tagged_views``).

Producers publish and consumers poll with the epoch of their dependency chain: the launches
whose tagged words feed each other (Qwen: every layer launch of a step, which overlap; Inkling:
one launch). The chain's last launch advances the epoch once all its CTAs have read it
(``advance_epoch``), so the next chain's words never match a stale tag and buffers need no
reset.
"""

import cutlass
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass._mlir.extras import types as T
from cutlass._mlir_helpers.arith import bitcast
from cutlass.cutlass_dsl import dsl_user_op


def scratch_offsets(words: dict[str, int]) -> tuple[dict[str, int], int]:
    """``(offsets, size)`` in words of tagged buffers of ``words`` words each, packed in order
    and 128-byte aligned."""
    offsets, offset = {}, 0
    for name, size in words.items():
        offsets[name] = offset
        offset += (size + 15) // 16 * 16
    return offsets, offset


# Plain Python, not cute.jit: a loop staged in a jit body would leak the views out of its region.
def tagged_views(scratch, words: dict[str, int]) -> dict:
    """Flat int64 views of the tagged buffers of ``words`` words each in ``scratch``, laid out
    by ``scratch_offsets``."""
    offsets = scratch_offsets(words)[0]
    return {
        name: cute.make_tensor(scratch.iterator + offsets[name], cute.make_layout((size,)))
        for name, size in words.items()
    }


def pack_tagged(value, tag):
    """The 64-bit word (value bits low, tag high)."""
    low = arith.extui(T.i64(), bitcast(Float32(value).ir_value(), T.i32()))
    high = arith.shli(arith.extui(T.i64(), Int32(tag).ir_value()), arith.constant(T.i64(), 32))
    return Int64(arith.ori(low, high))


def unpack_tagged(word):
    bits = Int64(word).ir_value()
    low = arith.trunci(T.i32(), bits)
    high = arith.trunci(T.i32(), arith.shrui(bits, arith.constant(T.i64(), 32)))
    return Float32(bitcast(low, T.f32())), Int32(high)


@cute.jit
def publish(buf, index, value, epoch):
    cute.arch.store(buf.iterator + index, pack_tagged(value, epoch), sem="relaxed", scope="gpu")


def load_vector(ptr, n: int) -> list:
    """``n`` (2 or 4) int64 words from ``ptr`` (``8 * n``-byte aligned) in one relaxed load."""
    vec = cute.arch.load(ptr, ir.VectorType.get([n], Int64.mlir_type), sem="relaxed", scope="gpu")
    return [Int64(vector.extract(vec, dynamic_position=[], static_position=[k])) for k in range(n)]


@cute.jit
def load_words(buf, index, count: cutlass.Constexpr):
    """``count`` tagged words from ``buf[index]`` as relaxed gpu-scope loads: 32 B vectors, then
    a 16 B vector, then a scalar. Each strong load is one L2 request per lane, so wider is
    cheaper. ``index`` must be a multiple of 4 when ``count >= 4`` and even when ``count >= 2``."""
    words = []
    quads = count // 4 * 4
    for e in cutlass.range_constexpr(0, quads, 4):
        words += load_vector(buf.iterator + (index + e), 4)
    for e in cutlass.range_constexpr(quads, count - 1, 2):
        words += load_vector(buf.iterator + (index + e), 2)
    if cutlass.const_expr(count % 2):
        last = buf.iterator + (index + count - 1)
        words.append(cute.arch.load(last, Int64, sem="relaxed", scope="gpu"))
    return words


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None):
    """Suspend the thread for about ``ns`` nanoseconds (PTX ``nanosleep``; the hardware picks a
    duration in [0, 2 * ns])."""
    llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def fetch(segments, out, epoch, uniform: cutlass.Constexpr = False,
          backoff: cutlass.Constexpr = 0):  # fmt: skip
    """Spin until every word of ``segments`` carries ``epoch``; the values fill ``out`` in order.
    A segment is a ``(buffer, start, count)`` run aligned as ``load_words`` requires.

    ``uniform``: poll until every lane of the warp has all its words, so the lanes leave the
    loop together (every lane of the warp must call it): lanes leaving a divergent poll loop one
    by one and then waiting at a shuffle for the rest stalled warps a fixed ~17k cycles on
    random SMs (GB300, Inkling MLP launch traces; 2026-11-21). ``backoff``: a failed uniform
    poll sleeps about that many nanoseconds, so long waits issue few L2 requests while other
    CTAs' work runs."""
    ready = Int32(0)
    while ready == 0:
        mine = Int32(1)
        start = 0
        for s in cutlass.range_constexpr(len(segments)):
            buf, index, count = segments[s]
            words = load_words(buf, index, count)
            for e in cutlass.range_constexpr(count):
                value, tag = unpack_tagged(words[e])
                out[start + e] = value
                mine = mine & Int32(tag == epoch)
            start += count
        if cutlass.const_expr(uniform):
            ready = Int32(cute.arch.vote_all_sync(mine != 0))
            if cutlass.const_expr(backoff > 0):
                if ready == 0:
                    nanosleep(backoff)
        else:
            ready = mine


@cute.jit
def advance_epoch(state, epoch, tid, launch=0, launches: cutlass.Constexpr = 1):
    """Count this CTA finished in the chain's int32 ``state`` (the epoch, then a finished-CTA
    counter for each of the chain's ``launches`` launches): the last CTA of launch ``launch``
    resets its counter and, in the chain's final launch, stores ``epoch`` to ``state[0]`` for
    the chain's next run. Call it once per CTA after its last read of the epoch. Overlapping
    launches of one chain count in their own counters, each reset only once every CTA of its
    launch has counted; the launch's next run starts a whole chain later, and gpu-scope atomics
    keep the reset ordered with that run's counting."""
    if tid == 0:
        ctas = cute.arch.grid_dim()[0]
        counter = state.iterator + 1 + launch
        finished = cute.arch.atomic_add(counter, Int32(1), sem="acq_rel", scope="gpu")
        if finished == ctas - 1:
            cute.arch.atomic_exch(counter, Int32(0), scope="gpu")
            if launch == launches - 1:
                state[0] = epoch
