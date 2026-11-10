"""Device side of the profiler: ``%globaltimer`` reads and profile-buffer records (CuTeDSL).

Both record formats give each profiling unit (usually a CTA) a slice of int64 words:

* compact (``compact_*``): one int64 per record (``slice = 1 + records``),
  ``tag(8) | dur_ns(24) | ts_lo32(32)``. Word 0 is a 64-bit timer anchor the decoder rebases
  ``ts_lo32`` on. Recorded to global memory directly, or to a shared-memory slice flushed at
  kernel exit.
* full (:func:`event_stop`): four int64 per record (``slice = 1 + 4 * records``, word 0 unused)
  holding ``start_ns``, ``dur_ns``, the tag word (:func:`tag_word`) and the recording warp.
  Warpdrive's ``"globaltimer"`` trace backend records these, since they carry a warp, a
  payload and durations past 2^24 ns.

This module imports no other package module: a kernel's compile key hashes the modules it
imports, and the host decoder and analysis must stay out of it.
"""

from contextlib import contextmanager

import cutlass
from cutlass import Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "PAYLOAD_FLAG",
    "compact_anchor_init",
    "compact_event_stop",
    "compact_event_stop_smem",
    "compact_flush_smem_to_gmem",
    "compact_prepare_smem",
    "compact_profile_region",
    "event_stop",
    "read_globaltimer",
    "read_globaltimer_lo32",
]

# A full record's tag word holds the tag in its low 31 bits. A region recorded with a payload
# (an int32, such as its work item) sets bit 31 and stores the payload in the high 32 bits, so
# records without one decode unchanged.
PAYLOAD_FLAG = 1 << 31


def tag_word(tag, payload=None) -> Int64:
    """The record's tag word: ``tag``, or ``tag | PAYLOAD_FLAG | payload << 32``."""
    if payload is None:
        return Int64(tag)
    return Int64(tag) | Int64(PAYLOAD_FLAG) | (Int64(Int32(payload)) << 32)


@cute.jit
def event_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ns: Int64,
    tag: Int32,
    warp: Int32,
    max_events_per_unit: Int32,
    payload: Int32 | None = None,
) -> None:
    """Read the end timer and store one full record in slot ``event_idx`` of ``unit_id``.

    Pair with :func:`read_globaltimer` for ``start_ns``. Unguarded: the caller elects the
    recording thread and keeps ``0 <= event_idx < max_events_per_unit``. ``warp`` is stored as
    the record's warp; ``payload`` is an optional runtime int32 decoded as ``Range.payload``.
    """
    end_ns = read_globaltimer()
    base_ptr = buf.iterator.toint()
    slice_size = Int64(1 + 4 * max_events_per_unit)
    base_offset = Int64(unit_id) * slice_size + Int64(1 + 4 * event_idx)
    byte_offset = base_offset * Int64(8)
    dur_ns = end_ns - start_ns
    _store_i64(base_ptr + byte_offset, start_ns)
    _store_i64(base_ptr + byte_offset + Int64(8), dur_ns)
    _store_i64(base_ptr + byte_offset + Int64(16), tag_word(tag, payload))
    _store_i64(base_ptr + byte_offset + Int64(24), Int64(warp))


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the 64-bit ``%globaltimer`` in nanoseconds (PTX ``mov.u64 $0, %globaltimer;``).

    Available on sm_70+, in nanoseconds on Hopper and Blackwell. Unlike ``clock64``, it is
    synchronized across SMs, so timestamps from different CTAs compare.
    """
    result = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@dsl_user_op
def read_globaltimer_lo32(*, loc=None, ip=None) -> Int32:
    """Read the low 32 bits of ``%globaltimer`` (PTX ``mov.u32 ... %globaltimer_lo;``).

    Lowers to one ``CS2R.32 Rn, SR_GLOBALTIMERLO`` on Blackwell, with a third of the register
    pressure of :func:`read_globaltimer` and no 64-bit packing, but it wraps every ~4 s. The
    host decoder rebases it on a per-unit 64-bit anchor (see :func:`compact_anchor_init`).
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %globaltimer_lo;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _store_i64(ptr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Store an int64 to global memory with ``st.global.cs``.

    Evict-first: device code never rereads profile records.
    """
    llvm.inline_asm(
        None,
        [ptr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.global.cs.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def compact_record(start_ts32: Int32, end_ts32: Int32, tag: Int32) -> Int64:
    """Pack one compact record: ``tag(8) | dur_ns(24) | ts_lo32(32)``."""
    ts_u = Int64(start_ts32) & Int64(0xFFFFFFFF)
    dur_u = Int64(end_ts32 - start_ts32) & Int64(0xFFFFFF)
    tag_u = Int64(tag) & Int64(0xFF)
    return ts_u | (dur_u << 32) | (tag_u << 56)


@cute.jit
def compact_anchor_init(
    buf: cute.Tensor,
    unit_id: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Write the 64-bit ``%globaltimer`` anchor to slot 0 of this unit's compact slice.

    Compact records keep only the timer's low 32 bits. The decoder rebuilds each start as the
    anchor plus the signed difference of their low 32 bits, so an event may start up to ~2.1 s
    before or after the anchor and other warps need not wait for it. Call once per CTA at the
    top of the ``@cute.kernel``, before any nested warp guards; lane 0 of warp 0 stores.
    """
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    if warp_idx == 0 and lane_idx == 0:
        ts = read_globaltimer()
        base_ptr = buf.iterator.toint()
        slice_size = Int64(1 + max_events_per_unit)
        byte_offset = Int64(unit_id) * slice_size * Int64(8)
        _store_i64(base_ptr + byte_offset, ts)


@cute.jit
def compact_event_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ts32: Int32,
    tag: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Record one compact event to global memory, one ``STG.E.EF.64`` per event.

    Pair with :func:`read_globaltimer_lo32` for the start timestamp. The record is
    ``tag(8) | dur_ns(24) | ts_lo32(32)`` (bits 63..56, 55..32, 31..0); ``tag`` is masked to
    8 bits and ``dur_ns`` wraps at 2^24 ns (~16.7 ms), so use :func:`event_stop` for longer
    regions. For the same workload, :func:`compact_event_stop_smem` has less observer overhead
    per event, at the cost of an end-of-kernel flush.

    Lane 0 of the calling warp stores, and only for ``0 <= event_idx < max_events_per_unit``:
    an out-of-range slot is dropped rather than written over a neighboring unit. The caller
    owns any outer warp guard. The inner ``if`` yields no value, so this is safe in deeply
    nested control flow.
    """
    lane_idx = cute.arch.lane_idx()
    end_ts32 = read_globaltimer_lo32()
    if (lane_idx == 0) & (event_idx >= 0) & (event_idx < max_events_per_unit):
        base_ptr = buf.iterator.toint()
        slice_size = Int64(1 + max_events_per_unit)
        byte_offset = (Int64(unit_id) * slice_size + Int64(1 + event_idx)) * Int64(8)
        _store_i64(base_ptr + byte_offset, compact_record(start_ts32, end_ts32, tag))


@cute.jit
def compact_event_stop_smem(
    smem_buf: cute.Tensor,
    event_idx: Int32,
    start_ts32: Int32,
    tag: Int32,
    max_events_per_unit: Int32,
) -> None:
    """Record one compact event into the CTA's shared-memory slice, one ``STS.64`` per event.

    The record matches :func:`compact_event_stop`, but shared memory needs no global address
    math, so an inner region's ``dur_ns`` reads smaller for the same work. Pair with
    :func:`read_globaltimer_lo32` for the start timestamp and call
    :func:`compact_flush_smem_to_gmem` once at kernel exit, outside the recorded regions.

    Lane 0 of the calling warp stores, and only for ``0 <= event_idx < max_events_per_unit``
    (the slice holds ``1 + max_events_per_unit`` words): an out-of-range slot is dropped
    rather than written over the kernel's neighboring shared memory. The caller owns any
    outer warp guard.
    """
    lane_idx = cute.arch.lane_idx()
    end_ts32 = read_globaltimer_lo32()
    if (lane_idx == 0) & (event_idx >= 0) & (event_idx < max_events_per_unit):
        smem_buf[Int32(1) + event_idx] = compact_record(start_ts32, end_ts32, tag)


@cute.jit
def compact_prepare_smem(
    smem_buf: cute.Tensor,
    max_events_per_unit: Int32,
    record=True,
) -> None:
    """Clear and anchor a compact shared-memory profile slice (CTA-wide, with barriers).

    ``record``: CTA-uniform runtime predicate; a CTA that does not record skips the slice.
    """
    if record:
        tidx, _, _ = cute.arch.thread_idx()
        bdim, _, _ = cute.arch.block_dim()
        slice_size = Int32(1) + max_events_per_unit
        i = tidx
        while i < slice_size:
            smem_buf[i] = Int64(0)
            i = i + bdim
        cute.arch.barrier()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if warp_idx == 0 and lane_idx == 0:
            smem_buf[0] = read_globaltimer()
        cute.arch.barrier()


@cute.jit
def compact_flush_smem_to_gmem(
    smem_buf: cute.Tensor,
    gmem_buf: cute.Tensor,
    unit_id: Int32,
    max_events_per_unit: Int32,
    record=True,
) -> None:
    """Copy the CTA's shared-memory profile slice to its unit's slice of the global buffer.

    Call once at kernel end, after this CTA's last :func:`compact_event_stop_smem`. A CTA-wide
    barrier makes the recorded slots visible, then the threads copy the
    ``1 + max_events_per_unit`` words with a block-dim stride. ``record`` is the CTA-uniform
    predicate passed to :func:`compact_prepare_smem`; a CTA that does not record flushes
    nothing.
    """
    if record:
        cute.arch.barrier()
        tidx, _, _ = cute.arch.thread_idx()
        bdim, _, _ = cute.arch.block_dim()
        slice_size = Int32(1) + max_events_per_unit
        base_ptr = gmem_buf.iterator.toint()
        base_byte_offset = Int64(unit_id) * Int64(slice_size) * Int64(8)
        i = tidx
        while i < slice_size:
            _store_i64(base_ptr + base_byte_offset + Int64(i) * Int64(8), smem_buf[i])
            i = i + bdim


@cute.jit
def compact_warp_start(target_warp: Int32, record=True) -> Int32:
    """Read a compact start timestamp on lane 0 of ``target_warp``; other threads get 0."""
    start_ts32 = Int32(0)
    if record:
        if cute.arch.warp_idx() == target_warp and cute.arch.lane_idx() == 0:
            start_ts32 = read_globaltimer_lo32()
    return start_ts32


@cute.jit
def compact_warp_stop(
    buf: cute.Tensor,
    unit_id: Int32,
    event_idx: Int32,
    start_ts32: Int32,
    tag: Int32,
    max_events_per_unit: Int32,
    target_warp: Int32,
    record=True,
    smem=False,
) -> None:
    """Close a compact region from lane 0 of ``target_warp``, in shared or global memory."""
    if record:
        if cute.arch.warp_idx() == target_warp and cute.arch.lane_idx() == 0:
            if cutlass.const_expr(smem):
                compact_event_stop_smem(buf, event_idx, start_ts32, tag, max_events_per_unit)
            else:
                compact_event_stop(buf, unit_id, event_idx, start_ts32, tag, max_events_per_unit)


@contextmanager
def compact_profile_region(
    buf,
    max_events_per_unit,
    tag,
    unit_id,
    event_idx,
    target_warp=None,
    enabled=True,
    record=True,
    smem=False,
):
    """Record one compact region in static slot ``event_idx``.

    Lane 0 of ``target_warp`` (default warp 0) reads the start timer on entry and stores the
    record on exit, to the shared-memory slice ``buf`` (``smem=True``; flush it with
    :func:`compact_flush_smem_to_gmem`) or to unit ``unit_id`` of the global buffer ``buf``.

    Args:
        enabled: Compile-time switch. When false, the region emits no profiling code.
        record: Runtime predicate selecting whether this unit records the region.
    """
    if not cutlass.const_expr(enabled):
        yield
        return
    if target_warp is None:
        target_warp = Int32(0)
    start_ts32 = compact_warp_start(target_warp, record)
    try:
        yield
    finally:
        compact_warp_stop(
            buf,
            unit_id,
            event_idx,
            start_ts32,
            tag,
            max_events_per_unit,
            target_warp,
            record,
            smem,
        )
