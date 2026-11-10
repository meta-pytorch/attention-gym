"""Test device recording and host decoding for all three profiler formats.

Formats: full 4xi64 records, compact 1xi64 global-memory records, and compact 1xi64
shared-memory records.
"""

import pytest
import torch

try:
    import cutlass
    from cutlass import Int32, cute
    from cutlass.cute.runtime import from_dlpack

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except ImportError:
    pytest.skip("CUTE not available", allow_module_level=True)

from attn_gym.testing.perfetto_trace import read_slices
from attn_gym.warpdrive.profiler import (
    compact_anchor_init,
    compact_event_stop,
    compact_event_stop_smem,
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    event_stop,
    profile_session,
    read_globaltimer,
    read_globaltimer_lo32,
)

NUM_BLOCKS = 4
NUM_ITERS = 8
THREADS = 64


# Full records (event_stop), with and without payloads


@cute.kernel
def _kernel_full(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    if cute.arch.warp_idx() == 1 and cute.arch.lane_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer()
            # Alternate plain records with payloads, some negative.
            if i % 2 == 0:
                event_stop(prof_buf, bidx, Int32(i), start, Int32(1), Int32(1), max_events)
            else:
                payload = (Int32(i) - 4) * Int32(1000) + bidx
                event_stop(
                    prof_buf, bidx, Int32(i), start, Int32(2), Int32(1), max_events, payload
                )


@cute.jit()
def _launch_full(prof_buf, max_events):
    _kernel_full(prof_buf, max_events).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1))


def test_full_record_roundtrip(tmp_path):
    """Full records keep their slot, warp, and tag.

    Payloads decode as Range.payload and Perfetto args; plain records keep payload None.
    """
    trace_path = tmp_path / "payload.pftrace"
    with profile_session(
        max_events_per_unit=NUM_ITERS,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=["unused", "plain", "item"],
        trace_path=trace_path,
    ) as session:
        _launch_full(from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit))

    ranges = session.ranges
    assert len(ranges) == NUM_BLOCKS * NUM_ITERS
    for r in ranges:
        assert r.start_ns > 0 and r.warp == 1
        if r.slot % 2 == 0:
            assert (r.name, r.payload) == ("plain", None)
        else:
            assert (r.name, r.payload) == ("item", (r.slot - 4) * 1000 + r.cta[0])
    slices = read_slices(trace_path)
    assert {(s.process, s.track) for s in slices} == {
        (f"Block {block}", "warp 1") for block in range(NUM_BLOCKS)
    }
    assert sorted(s.args["payload"] for s in slices if "payload" in s.args) == sorted(
        r.payload for r in ranges if r.payload is not None
    )


# Compact 1xi64 gmem mode


@cute.kernel
def _kernel_compact_gmem(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init(prof_buf, bidx, max_events)
    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer_lo32()
            compact_event_stop(prof_buf, bidx, Int32(i), start, Int32(i & 0xF), max_events)


@cute.jit()
def _launch_compact_gmem(prof_buf, max_events):
    _kernel_compact_gmem(prof_buf, max_events).launch(
        grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1)
    )


def test_compact_gmem_roundtrip(tmp_path):
    """Compact records decode with their static slots; the trace has one track per tag."""
    trace_path = tmp_path / "compact_gmem.pftrace"
    with profile_session(
        max_events_per_unit=NUM_ITERS + 1,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
        compact=True,
        expected_events_per_unit=NUM_ITERS,
        trace_path=trace_path,
    ) as session:
        _launch_compact_gmem(
            from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
        )

    ranges = session.ranges
    assert len(ranges) == NUM_BLOCKS * NUM_ITERS
    for unit in range(NUM_BLOCKS):
        assert {r.slot for r in ranges if r.cta == (unit,)} == set(range(NUM_ITERS))
    for r in ranges:
        # Anchors put all ranges near the same 64-bit timer band.
        assert r.start_ns > (1 << 40), "start_ns should be a reconstructed 64-bit globaltimer"
    slices = read_slices(trace_path)
    assert {(s.process, s.track, s.args["slot"]) for s in slices} == {
        (f"Block {r.cta[0]}", r.name, r.slot) for r in ranges
    }


def test_expected_event_count_validation_rejects_missing_events():
    """profile_session checks every unit's decoded count."""
    with (
        pytest.raises(RuntimeError, match=f"Expected {NUM_ITERS + 1} events"),
        profile_session(
            max_events_per_unit=NUM_ITERS + 1,
            num_units=(NUM_BLOCKS, "Block"),
            tag_names=[f"t{i}" for i in range(NUM_ITERS)],
            compact=True,
            expected_events_per_unit=NUM_ITERS + 1,
        ) as session,
    ):
        _launch_compact_gmem(
            from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
        )


# Compact 1xi64 smem mode (the cheap-hot-path variant)


PROF_SMEM_SLOTS = NUM_ITERS + 1


@cute.kernel
def _kernel_compact_smem(prof_gmem: cute.Tensor, max_events: cutlass.Int32):
    @cute.struct
    class S:
        prof_smem: cute.struct.MemRange[cutlass.Int64, PROF_SMEM_SLOTS]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(S)
    smem_tensor = storage.prof_smem.get_tensor(cute.make_layout((PROF_SMEM_SLOTS,)))

    bidx, _, _ = cute.arch.block_idx()
    compact_prepare_smem(smem_tensor, max_events)

    if cute.arch.warp_idx() == 0:
        for i in cutlass.range(NUM_ITERS):
            start = read_globaltimer_lo32()
            compact_event_stop_smem(smem_tensor, Int32(i), start, Int32(i & 0xF), max_events)

    compact_flush_smem_to_gmem(smem_tensor, prof_gmem, bidx, max_events)


@cute.jit()
def _launch_compact_smem(prof_gmem, max_events):
    _kernel_compact_smem(prof_gmem, max_events).launch(
        grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1)
    )


def test_compact_smem_roundtrip():
    """Shared-memory records flush to global memory and decode like records written there."""
    with profile_session(
        max_events_per_unit=NUM_ITERS,
        num_units=(NUM_BLOCKS, "Block"),
        tag_names=[f"t{i}" for i in range(NUM_ITERS)],
        compact=True,
    ) as session:
        _launch_compact_smem(
            from_dlpack(session.prof.tensor), Int32(session.prof.max_events_per_unit)
        )

    ranges = session.ranges
    assert len(ranges) == NUM_BLOCKS * NUM_ITERS
    assert {r.cta for r in ranges} == {(unit,) for unit in range(NUM_BLOCKS)}
    assert {r.name for r in ranges} == {f"t{i}" for i in range(NUM_ITERS)}
    for unit in range(NUM_BLOCKS):
        assert {r.slot for r in ranges if r.cta == (unit,)} == set(range(NUM_ITERS))
    for r in ranges:
        assert r.start_ns > (1 << 40)


# Out-of-range slots are dropped, not written over neighboring records


@cute.kernel
def _kernel_compact_slots(prof_buf: cute.Tensor, max_events: cutlass.Int32):
    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init(prof_buf, bidx, max_events)
    cute.arch.barrier()
    if cute.arch.warp_idx() == 0:
        # Block 0's slots -1 and max_events would overwrite its anchor and block 1's.
        for slot in (Int32(0) - bidx, max_events * bidx):
            start = read_globaltimer_lo32()
            compact_event_stop(prof_buf, Int32(1) - bidx, slot, start, Int32(0), max_events)


@cute.jit()
def _launch_compact_slots(prof_buf, max_events):
    _kernel_compact_slots(prof_buf, max_events).launch(grid=(2, 1, 1), block=(THREADS, 1, 1))


def test_compact_out_of_range_slots_are_dropped():
    with profile_session(
        max_events_per_unit=2, num_units=2, tag_names=["t"], compact=True
    ) as session:
        _launch_compact_slots(from_dlpack(session.prof.tensor), Int32(2))
    # Block 0 records slot 0 of unit 1 twice; block 1's slots -1 and 2 of unit 0 are dropped.
    assert [(r.cta, r.slot) for r in session.ranges] == [((1,), 0)]
    anchors = session.prof.tensor.view(2, 3)[:, 0]
    assert (anchors > 1 << 50).all(), "an out-of-range record overwrote an anchor"


SENTINEL = 0x5E17


@cute.kernel
def _kernel_compact_smem_slot(prof_gmem: cute.Tensor, out: cute.Tensor):
    @cute.struct
    class S:
        prof_smem: cute.struct.MemRange[cutlass.Int64, 2]
        neighbor: cute.struct.MemRange[cutlass.Int64, 1]

    storage = cutlass.utils.SmemAllocator().allocate(S)
    smem_tensor = storage.prof_smem.get_tensor(cute.make_layout((2,)))
    neighbor = storage.neighbor.get_tensor(cute.make_layout((1,)))
    if cute.arch.thread_idx()[0] == 0:
        neighbor[0] = cutlass.Int64(SENTINEL)
    compact_prepare_smem(smem_tensor, Int32(1))
    if cute.arch.warp_idx() == 0:
        start = read_globaltimer_lo32()
        compact_event_stop_smem(smem_tensor, Int32(1), start, Int32(0), Int32(1))  # past the end
    compact_flush_smem_to_gmem(smem_tensor, prof_gmem, Int32(0), Int32(1))
    if cute.arch.thread_idx()[0] == 0:
        out[0] = neighbor[0]


@cute.jit()
def _launch_compact_smem_slot(prof_gmem, out):
    _kernel_compact_smem_slot(prof_gmem, out).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def test_compact_smem_slot_past_the_slice_is_dropped():
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    with profile_session(
        max_events_per_unit=1, num_units=1, tag_names=["t"], compact=True
    ) as session:
        _launch_compact_smem_slot(from_dlpack(session.prof.tensor), from_dlpack(out))
    assert session.ranges == []
    assert out.item() == SENTINEL
