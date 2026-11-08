"""Declared CTA named barriers with kernel-allocated IDs and fixed thread counts.

Declare ``NamedBarrier(threads=N)`` as a kernel attribute and call ``sync()`` or ``arrive()``
in roles. IDs share an allocator with named-barrier channels. Calls record trace events
for alias diagnostics and ``describe()``; hand-picked ``alias.named_barrier`` IDs cannot
collide with them. Pipeline plans model channels only, not named barriers.

The author owns participation and generation order, including what stops a fast thread
from arriving twice before slow threads finish a generation. See
``docs/warpdrive/channels.md`` (Named barriers).
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
    """A CTA named barrier with a kernel-allocated ID and ``threads`` arrivals per generation.

    ``sync()`` arrives and waits (``barrier.cta.sync``); ``arrive()`` does not wait
    (``barrier.cta.arrive``). Both default to unaligned forms, so lanes and warps may call
    from different sites. ``aligned=True`` uses ``bar.sync``/``bar.arrive`` and requires
    every thread of each participating warp to execute the same instruction convergently.

    ``id`` exposes the ID for native calls, which the layer cannot observe. Reading it
    during a kernel trace marks native participation as possible: incomplete observed
    participation is then unverified rather than an error.
    """

    threads: int
    name: str = field(default="", init=False)
    _id: int | None = field(default=None, init=False, repr=False)
    # Native ID access outside roles; reset by begin_roles for the participation check.
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
        """Arrive and wait for ``threads`` arrivals in this generation.

        With wait tracing enabled, records a ``<name>.sync`` range.
        """
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
        # Only sync orders accesses for the caller. Keep arrive events out of alias ordering;
        # "barrier" matches alias.named_barrier's sync event.
        trace = current_trace()
        if trace is not None:
            trace.event(kind, threads=self.threads, id=self._allocated_id(), barrier=self)


def _tracing_device_code() -> bool:
    try:
        ir.InsertionPoint.current  # noqa: B018
    except ValueError:
        return False
    return True
