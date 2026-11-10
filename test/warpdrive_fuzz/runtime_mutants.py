"""Inject layer bugs that the GPU tests must catch.

Each mutant patches one derivation. test_mutants.py runs its listed tests in a child
pytest with ``-p warpdrive_fuzz.runtime_mutants`` and WDFUZZ_MUTANT set. Their kernels
compile against the broken layer. Passing tests mean that failure class is not covered.
"""

from __future__ import annotations

import os

from cutlass import cute

from attn_gym.warpdrive import channels, cluster, kernel, scheduling
from attn_gym.warpdrive.channels import (
    Channel,
    ConsumerEnd,
    Handle,
    Threads,
    Warps,
    _End,
)
from attn_gym.warpdrive.tracing import wait_range


def join_short():
    """Joins expect one warp fewer: the empty barrier completes before the last consumer."""
    orig = Channel.slot_counts

    def slot_counts(self, side):
        counts = orig(self, side)
        if side == "consumer" and self.consumer is Threads and not self.internal:
            return [c - 32 if c > 32 else c for c in counts]
        return counts

    Channel.slot_counts = slot_counts


def warps_short():
    """``Warps`` sides expect one warp fewer: a stage completes before its last warp arrives."""
    orig = Channel.slot_counts

    def slot_counts(self, side):
        counts = orig(self, side)
        if (self.consumer if side == "consumer" else self.producer) is Warps:
            return [c - 1 if c > 1 else c for c in counts]
        return counts

    Channel.slot_counts = slot_counts


def ring_wait_phase():
    """Ring consumers wait on the wrong parity (a fresh full barrier passes immediately)."""

    def wait(self, slot=None):
        index, phase, label = self._next(slot, "wait")
        if not self._chan.fixed and not self._chan.internal:
            phase = phase ^ 1
        self._trace.record(self._chan.name, "wait", label)
        with wait_range(self._trace, f"{self._chan.name}.wait"):
            cute.arch.mbarrier_wait(self._full(index), phase)
        return Handle(self, index, phase, label, False)

    ConsumerEnd.wait = wait
    ConsumerEnd.consume = wait


def slots_shared_phase():
    """Using one slot flips every slot's phase bit (slots lose their independence)."""
    orig = _End._next

    def _next(self, slot, op):
        out = orig(self, slot, op)
        chan = self._chan
        if chan.fixed and not chan.internal and chan.n > 1:
            self._state = self._state ^ (((1 << chan.n) - 1) ^ (1 << slot))
        return out

    _End._next = _next


def static_stride():
    """StaticPersistent strides one unit too far: items are skipped."""
    scheduling.StaticPersistent.next_index = lambda self, index, problem: (
        index + problem.units() + 1
    )


@cute.jit
def _no_report(workspace, units):
    pass


def atomic_no_reset():
    """AtomicPersistent never reports or resets its counter: the next launch starts dirty."""
    scheduling._report_finished = _no_report


def turn_no_wait():
    """Turnstile seats enter without waiting for their turn."""

    def __enter__(self):
        e = self._end
        if e._ts.enabled:
            e._trace.record(e._ts.name, "turn", e._seat)

    channels._TurnScope.__enter__ = __enter__


def pair_join_short():
    """Pair-scoped Threads joins expect one warp fewer on the leader's barrier."""
    orig = cluster.derive_scope

    def derive_scope(chan, clu):
        scope = orig(chan, clu)
        if scope.pair and not chan.internal:
            for counts, mech in (
                (scope.full_counts, chan.producer),
                (scope.empty_counts, chan.consumer),
            ):
                if mech is Threads:
                    for rank in counts:
                        counts[rank] = [c - 32 if c > 32 else c for c in counts[rank]]
        return scope

    cluster.derive_scope = derive_scope
    kernel.derive_scope = derive_scope


def override_ignored():
    """Count overrides are dropped: barriers keep the derived counts, so per-warp manual
    arrivals (``signal=False`` plus an override) never complete a phase."""
    kernel.apply_overrides = lambda chan, scope, clu: scope


def signal_false_signals():
    """``commit(signal=False)``/``release(signal=False)`` still signal: arrivals double."""
    orig_commit, orig_release = Handle.commit, Handle.release

    def commit(self, signal=True):
        orig_commit(self, signal)
        if not signal:
            self._end._signal(self.index)

    def release(self, signal=True):
        orig_release(self, signal)
        if not signal:
            self._end._signal(self.index)

    Handle.commit, Handle.release = commit, release


def decode_unguarded():
    """Every mapping is decoded at any index, as if it declared ``decodes_any_index``."""
    scheduling._decode = lambda stream, index, problem, state: scheduling._keep_words(
        stream.mapping, index, problem.space, state
    )[1:]


def pdl_no_wait():
    """``PdlWait`` issues no ``griddepcontrol.wait``: a dependent reads the table early."""
    scheduling.PdlWait.wait = lambda self: None


def raw_counts_wrong():
    """``Barriers(counts=...)`` initializes each barrier with twice its count."""
    orig = kernel.WarpSpecialized._barrier_plan

    def _barrier_plan(self):
        plan, off = orig(self)
        out = []
        for name, o, counts in plan:
            if name in self._ws_raw_bars and self._ws_raw_bars[name].counts:
                counts = {c: [2 * x for x in v] for c, v in counts.items()}
            out.append((name, o, counts))
        return out, off

    kernel.WarpSpecialized._barrier_plan = _barrier_plan


def umma_commit_early():
    """Replace tcgen05.commit with an elected arrive, publishing before MMA completion.

    The epilogue may read the accumulator before MMA writes finish, and TMA may overwrite
    A/B stages before MMA reads finish.
    """
    orig = _End._arrive

    def _arrive(self, mech, bar, fence, mask):
        if mech is channels.Umma:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar)
            return
        orig(self, mech, bar, fence, mask)

    _End._arrive = _arrive


def view_endpoint_stage():
    """``h.data``/``h.view`` address the endpoint's next ring stage instead of the handle's
    own: wrong whenever another handle of the end opened in between."""
    data, view = Handle.data.fget, Handle.view

    def at_endpoint(h):
        if h._end._chan.fixed:
            return h
        moved = object.__new__(Handle)
        moved.__dict__.update(h.__dict__)
        moved.index = h._end._state.index
        return moved

    Handle.data = property(lambda self: data(at_endpoint(self)))
    Handle.view = lambda self, *args, **kwargs: view(at_endpoint(self), *args, **kwargs)


# mutant -> (patch, test file in test/warpdrive, test that fails or hangs with it applied)
MUTANTS = {
    "join_short": (join_short, "test_warpdrive.py", "test_multi_warp_arrival_counts"),
    "warps_short": (
        warps_short,
        "test_warpdrive_warps.py",
        "test_warps_publish_and_release_whole_warps",
    ),
    "ring_wait_phase": (
        ring_wait_phase,
        "test_warpdrive_barrier_init.py",
        "test_deep_ring_arrival_counts_and_wraparound[1]",
    ),
    "slots_shared_phase": (
        slots_shared_phase,
        "test_warpdrive.py",
        "test_slots_payload_round_trip",
    ),
    "static_stride": (
        static_stride,
        "test_warpdrive_scheduling.py",
        "test_static_assignment_strides_by_the_grid",
    ),
    "atomic_no_reset": (
        atomic_no_reset,
        "test_warpdrive_scheduling.py",
        "test_concurrent_launches_use_one_workspace_per_stream",
    ),
    "turn_no_wait": (turn_no_wait, "test_warpdrive.py", "test_turnstile_alternates_seats"),
    "pair_join_short": (
        pair_join_short,
        "test_warpdrive_cluster.py",
        "test_toy_runs[ClusterToy-pair-ok]",
    ),
    "override_ignored": (
        override_ignored,
        "test_warpdrive_hardening.py",
        "test_raw_interop_consumer_sees_every_epoch[1]",
    ),
    "signal_false_signals": (
        signal_false_signals,
        "test_warpdrive_exclusive.py",
        "test_unequal_owners_with_overridden_counts_and_dynamic_skip",
    ),
    "decode_unguarded": (
        decode_unguarded,
        "test_warpdrive_scheduling.py",
        "test_empty_and_overflowing_spaces[static_checked-single]",
    ),
    "pdl_no_wait": (
        pdl_no_wait,
        "test_warpdrive_scheduler_interface.py",
        "test_device_built_list_under_pdl[atomic-single-packed]",
    ),
    "raw_counts_wrong": (
        raw_counts_wrong,
        "test_warpdrive_hardening.py",
        "test_layer_initialized_barriers_complete_after_their_count",
    ),
    "umma_commit_early": (umma_commit_early, "test_warpdrive.py", "test_gemm[128-256-64]"),
    "view_endpoint_stage": (
        view_endpoint_stage,
        "test_warpdrive_storage.py",
        "test_buffer_views_project_each_handles_stage",
    ),
}


def apply(name: str) -> None:
    MUTANTS[name][0]()


def pytest_configure(config) -> None:
    """Apply ``WDFUZZ_MUTANT`` when loaded with ``-p warpdrive_fuzz.runtime_mutants``."""
    if name := os.environ.get("WDFUZZ_MUTANT"):
        apply(name)
