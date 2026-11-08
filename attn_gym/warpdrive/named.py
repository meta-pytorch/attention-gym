"""Declared CTA named barriers: one hardware barrier ID and one thread count per declaration.

``NamedBarrier(threads=N)`` is a kernel attribute, declared in ``__init__`` like a channel. The
kernel allocates its ID together with the named-barrier channels' IDs (``kernel.py``); role
bodies call ``self.<name>.sync()`` or ``.arrive()``. Each call records a trace event, so the
alias diagnostic sees the barrier, the kernel rejects hand-picked ``alias.named_barrier`` IDs
that collide with it, and ``describe()`` lists the roles that used it. Pipeline plans
(``pipeline_plan.py``) model channels only and leave named barriers out.

The resource supplies the ID and a consistent count, nothing more: which threads arrive in
which generation, and why a fast participant cannot arrive twice before slow ones finish a
generation, remain the author's protocol (docs/warpdrive/channels.md, "Named barriers").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutlass import cute
from cutlass._mlir import ir

from attn_gym.warpdrive.channels import _named_arrive, _named_sync
from attn_gym.warpdrive.protocol import ProtocolError, current_trace
from attn_gym.warpdrive.tracing import wait_range

WARP_SIZE = 32
MAX_THREADS = 1024  # CUDA's block limit


@dataclass(eq=False)
class NamedBarrier:
    """One CTA named barrier whose ID the kernel allocates; ``threads`` arrivals complete it.

    ``sync()`` arrives and waits for the generation to complete (``barrier.cta.sync``),
    ``arrive()`` arrives without waiting (``barrier.cta.arrive``). Both are unaligned, so lanes
    and warps may reach them from different code sites. ``aligned=True`` emits the aligned
    forms (``bar.sync``/``bar.arrive``), which require every thread of each participating warp
    to execute the same instruction convergently; use it only where that holds and it matters.

    ``id`` is the allocated barrier ID for native calls (``cute.arch.barrier(barrier_id=...)``,
    a library that takes an ID). The calls themselves are invisible to the layer, but reading
    ``id`` while a kernel is traced records that native code may join the barrier: the count
    check then reports incomplete observed participation as unverified instead of raising.
    """

    threads: int
    name: str = field(default="", init=False)
    _id: int | None = field(default=None, init=False, repr=False)
    # ``id`` read by kernel code outside the role bodies in the current trace (reset by
    # ``begin_roles``; the kernel's named-barrier check reads it).
    _native_outside_roles: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.threads) is not int
            or not WARP_SIZE <= self.threads <= MAX_THREADS
            or self.threads % WARP_SIZE
        ):
            raise ValueError(
                f"NamedBarrier threads must be an int multiple of {WARP_SIZE} in "
                f"[{WARP_SIZE}, {MAX_THREADS}] (bar.sync counts whole warps), got {self.threads!r}"
            )

    @property
    def id(self) -> int:
        barrier_id = self._allocated_id()
        trace = current_trace()
        if trace is not None:
            trace.event("barrier_native", barrier=self)
        elif _tracing_device_code():  # kernel code outside the roles: every thread may join
            self._native_outside_roles = True
        return barrier_id

    def _allocated_id(self) -> int:
        if self._id is None:
            raise ProtocolError(
                f"NamedBarrier {self.name or '(unnamed)'} has no ID yet: declare it as a kernel "
                "attribute; the kernel allocates IDs when it collects its declarations "
                "(describe(), compile)"
            )
        return self._id

    def sync(self, *, aligned: bool = False):
        """Arrive, then wait until ``threads`` threads have arrived in this generation. With trace
        waits enabled, the wait is a ``<name>.sync`` range."""
        with wait_range(current_trace(), f"{self.name}.sync"):
            if aligned:
                cute.arch.barrier(barrier_id=self._allocated_id(), number_of_threads=self.threads)
            else:
                _named_sync(self._allocated_id(), self.threads)
        self._record("barrier")

    def arrive(self, *, aligned: bool = False):
        """Arrive without waiting; the generation completes when ``threads`` have arrived."""
        if aligned:
            cute.arch.barrier_arrive(
                barrier_id=self._allocated_id(), number_of_threads=self.threads
            )
        else:
            _named_arrive(self._allocated_id(), self.threads)
        self._record("barrier_arrive")

    def _record(self, kind: str):
        # "barrier" is the event alias.named_barrier records; an arrive orders nothing for the
        # arriving thread, so it gets its own kind the alias diagnostic ignores.
        trace = current_trace()
        if trace is not None:
            trace.event(kind, threads=self.threads, id=self._allocated_id(), barrier=self)


def _tracing_device_code() -> bool:
    try:
        ir.InsertionPoint.current  # noqa: B018
    except ValueError:
        return False
    return True
