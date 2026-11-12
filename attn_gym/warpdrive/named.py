"""Declared CTA named barriers with kernel-allocated IDs and fixed thread counts.

Declare ``NamedBarrier(threads=N)`` as a kernel attribute and call ``sync()`` or ``arrive()``
in roles. IDs share an allocator with named-barrier channels. Calls record trace events
for alias diagnostics and ``describe()``; hand-picked ``alias.named_barrier`` IDs cannot
collide with them. Pipeline plans model channels only, not named barriers.

The author owns participation and generation order, including what stops a fast thread
from arriving twice before slow threads finish a generation. See
``docs/warpdrive/channels.md`` (Named barriers).

``check`` is the kernel's trace-time check of barrier IDs and ``NamedBarrier`` participation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutlass import cute
from cutlass._mlir import ir

from attn_gym.warpdrive.alias import RoleEvents
from attn_gym.warpdrive.channels import _named_arrive, _named_sync
from attn_gym.warpdrive.protocol import ProtocolError, current_trace
from attn_gym.warpdrive.tracing import wait_range

WARP_SIZE = 32
MAX_THREADS = 1024  # CUDA's block limit
# Named barrier ids owned by the layer; user code must not reuse them.
TMEM_PTR_BARRIER_ID = 1
TMEM_TEARDOWN_BARRIER_ID = 2
# Named-barrier channels take IDs from the top down and declared NamedBarriers from the bottom
# up, so adding a NamedBarrier leaves every channel's IDs where they were.
FIRST_FREE_BARRIER_ID = 3
MAX_NAMED_BARRIER_ID = 15


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


def check(kernel, roles: list[RoleEvents]) -> dict:
    """Check named-barrier IDs and ``NamedBarrier`` participation after a trace.

    Rejects hand-picked ``alias.named_barrier`` IDs that are out of range or owned by the
    layer, a named channel, or a ``NamedBarrier``. Returns each ``NamedBarrier``'s users
    (``{"sync"/"arrive"/"native": role names, "unverified": notes}``) and rejects one whose
    users cannot supply its count in some CTA. Native code that read ``.id`` may supply the
    rest, so short observed participation in a CTA where such code runs is only
    unverified; a count above the threads that run in that CTA still raises.
    """
    taken = {
        i: f"channel {name}" for name, ids in kernel._ws_named.items() for i in (*ids[0], *ids[1])
    }
    taken.update({TMEM_PTR_BARRIER_ID: "the layer", TMEM_TEARDOWN_BARRIER_ID: "the layer"})
    taken.update(
        {
            nb._allocated_id(): f"NamedBarrier {name} (call {name}.sync() or {name}.arrive())"
            for name, nb in kernel._ws_named_bars.items()
        }
    )
    users = {
        name: {"sync": [], "arrive": [], "native": [], "unverified": []}
        for name in kernel._ws_named_bars
    }
    channel_bars = {
        nb.name: nb
        for ch in kernel._ws_chans.values()
        for nb in (*ch.full_barriers, *ch.empty_barriers)
    }
    ops = {"barrier": "sync", "barrier_arrive": "arrive", "barrier_native": "native"}
    for role_events in roles:
        for ev in role_events.events:
            if ev.kind not in ops:
                continue
            if "barrier" in ev.data:  # a NamedBarrier operation or ID read
                name = ev.data["barrier"].name
                if channel_bars.get(name) is ev.data["barrier"]:
                    continue  # a named channel's slot: participation is the channel's
                if kernel._ws_named_bars.get(name) is not ev.data["barrier"]:
                    raise ProtocolError(
                        f"{role_events.name}: NamedBarrier {name} at {ev.site} belongs to "
                        "another kernel; declare one on this kernel"
                    )
                op = ops[ev.kind]
                if role_events.name not in users[name][op]:
                    users[name][op].append(role_events.name)
                continue
            barrier_id = ev.data["id"]
            if barrier_id is None:  # a dynamic ID: not checkable
                continue
            if not 0 <= barrier_id <= MAX_NAMED_BARRIER_ID:
                raise ProtocolError(
                    f"{role_events.name}: named_barrier({barrier_id}) at {ev.site}: barrier "
                    f"IDs are 0-{MAX_NAMED_BARRIER_ID}"
                )
            if barrier_id in taken:
                raise ProtocolError(
                    f"{role_events.name}: named_barrier({barrier_id}) at {ev.site} uses a "
                    f"barrier ID owned by {taken[barrier_id]}; pick an unused ID"
                )
    instances = {r.name: r for r in kernel._ws_roles}
    for name, nb in kernel._ws_named_bars.items():
        used = users[name]
        outside = nb._native_outside_roles  # kernel code every role's threads may run
        if outside:
            used["native"].append("kernel code")
        used_by = [instances[r] for r in dict.fromkeys(used["sync"] + used["arrive"])]
        native = [instances[r] for r in used["native"] if r in instances]
        for rank in range(kernel.cluster.size):
            in_rank = [r for r in kernel._ws_roles if rank in kernel.cluster.ctas(r.cta)]
            here = [r for r in used_by if r in in_rank]
            total = sum(r.num_threads for r in here)
            where = f" in CTA rank {rank}" if kernel.cluster.size > 1 else ""
            if not (outside or any(r in in_rank for r in native)):
                if 0 < total < nb.threads:
                    raise ProtocolError(
                        f"NamedBarrier {name} counts {nb.threads} threads, but the roles "
                        f"using it{where} ({', '.join(r.name for r in here)}) have only "
                        f"{total}; it can never complete"
                    )
                continue
            # Native callers can be any thread that runs here (a role, or kernel code).
            runs = kernel.num_threads if outside else sum(r.num_threads for r in in_rank)
            if runs < nb.threads:
                raise ProtocolError(
                    f"NamedBarrier {name} counts {nb.threads} threads, but only {runs} run "
                    f"the roles{where}; it can never complete"
                )
            if total < nb.threads:
                used["unverified"].append(
                    f"roles calling it{where} have {total} of {nb.threads} threads; native "
                    f"code reading .id ({', '.join(used['native'])}) must supply the rest"
                )
    return users
