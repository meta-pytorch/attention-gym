"""Channels (rings and fixed slots), turnstiles, and their device-side ends and handles.

A channel is a full/empty mbarrier pair per stage or slot. Each side names how it signals, which
fixes the arrivals its barrier expects:

* ``Tma(tx_bytes, multicast=...)``: one elected ``arrive.expect_tx``; the TMA engine completes
  the phase.
* ``Umma``: one elected ``tcgen05.commit``, which completes once every prior MMA of the issuing
  thread has finished.
* ``Threads``: every thread of every role instance bound to the slot arrives.
* ``Warps``: every thread of those instances signals, but each warp syncs (``__syncwarp``) and
  one elected lane arrives: one arrival per warp.

The layer derives each slot's arrival count from the role instances bound to it, so a channel
with several consumer roles (fan-out) has a join barrier that completes only after all of them
release. A ``Tma`` or ``Umma`` side signals once, so it binds to one single-warp role, the warp
that issues the copies or MMAs; round-robin and exclusive sides (below) bind several such roles,
one signalling per epoch. ``Channel(override={...})`` replaces any derived value. ``Barriers``
reserves mbarriers that user code drives directly, and ``external="consumer"`` (or
``"producer"``) leaves one side of a channel to raw code.

A joining consumer side may give each role its own mechanism, ``{"mma": Umma, "cuda": Threads}``:
the dict names the consumer roles, and each role's instances signal and fence with their own
mechanism, so the count sums 1 for the MMA warp and every thread of the ``cuda`` role.

Each side shares the channel's epochs among its role instances in one of three ways
(``producer_sharing=``/``consumer_sharing=``, a ``Sharing``):

* ``JOIN`` (the default): every instance takes every epoch it is bound to, and the counts sum.
* ``ROUND_ROBIN`` (``Ring`` only, either side): the instances take the epochs in turn. With
  ``k`` instances, instance ``j`` takes epochs ``j, j + k, j + 2k, ...``, and each epoch counts
  one instance's contribution. On an mbarrier ring ``k`` must divide the stage count: a parity
  wait is only exact when the waiter itself saw the stage's previous epoch, and with
  ``n % k != 0`` another instance did.
* ``EXCLUSIVE`` (consumers of a ``Ring`` only): each epoch belongs to exactly one of the
  declared ``consumer_roles``, which waits and releases it; the other consumers ``skip()`` it,
  advancing their end without waiting or arriving. Each epoch counts one consumer's
  contribution. Which consumer owns which epoch is the author's protocol. The layer checks the
  topology (CTA-local, one producer, equal contributions unless ``override`` sets
  ``empty_counts``) and, when every wait and skip is static, that each epoch has one owner. A
  wait is exact only if the waiter's previous wait was at most one ring lap earlier; otherwise
  the author must order it after the stage's previous epoch (FA4's decode uses two named
  barriers; see ``docs/warpdrive/channels.md``). Round-robin sides get the same ownership
  check: instance ``j`` of ``k`` is an owner that skips ``k - 1`` epochs after each of its own.

``sync="named"`` (``Threads`` on both sides) makes each slot a pair of CTA named barriers
(unaligned ``bar.sync``/``bar.arrive``) instead of mbarriers. A named barrier has no phase: it
completes when all producer and consumer threads have arrived, so its count is their sum. Every
consumer instance arrives once on its slots' empty barriers at role start, the equivalent of an
mbarrier ring starting empty. IDs are allocated from 15 down; see
``WarpSpecialized.named_barrier_ids``.

A named barrier cannot tell two instances apart. If round-robin instances share a stage's
barrier (``n % k != 0``, e.g. a one-stage mailbox two warpgroups write in turn), their threads
could complete it together without the other side. Such a channel names a ``Turnstile`` over
those instances (``turnstile="mufu"``) and acquires/waits only while holding the turn, which is
checked at trace time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cutlass
from cutlass import Int32, cute, pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import if_generate

from attn_gym.warpdrive.cluster import (
    ALL,
    MULTICAST_MODES,
    SCOPE_OVERRIDES,
    rank_in,
    rank_value,
)
from attn_gym.warpdrive.compat import NATIVE_MULTI_ADVANCE, mbarrier_arrive
from attn_gym.warpdrive.fences import FENCE_KINDS, derived_fences
from attn_gym.warpdrive.protocol import (
    FENCE_SKIPS,
    Lifetime,
    Lineage,
    ProtocolError,
    current_block,
    fence_skip,
    user_site,
)
from attn_gym.warpdrive.storage import SmemBuffer, StageView, project, stage_extent
from attn_gym.warpdrive.thread_sync import after_sync, before_sync
from attn_gym.warpdrive.tma import TmaUse, stage_closed
from attn_gym.warpdrive.tracing import wait_range


@dataclass(frozen=True)
class Tma:
    """Producer side filled by TMA, with ``tx_bytes`` per stage: the bytes landing in each
    producing CTA's SMEM.

    ``multicast``: the cluster modes ("m", "n", or both, as a string or tuple) along which the
    copies are multicast (``Cluster.multicast``). A copy multicast along N lands in every CTA
    with this CTA's coordinates except along N, and the consumers there must also release this
    CTA's stage; the layer derives that from the declaration.
    """

    tx_bytes: int
    multicast: tuple[str, ...] | str = ()

    def __post_init__(self):
        if type(self.tx_bytes) is not int or self.tx_bytes <= 0:
            raise ValueError(f"Tma.tx_bytes must be a positive int, got {self.tx_bytes!r}")
        modes = self.multicast
        if not isinstance(modes, (str, tuple, list)) or any(
            m not in MULTICAST_MODES for m in modes
        ):
            raise ValueError(
                f"Tma.multicast must name cluster modes from {MULTICAST_MODES}, got {modes!r}"
            )
        object.__setattr__(self, "multicast", tuple(sorted(set(modes))))


class Umma:
    """Side signalled by one elected ``tcgen05.commit``."""


class Threads:
    """Side signalled by ``mbarrier.arrive`` from every thread of the bound role instances."""


class Warps:
    """Side signalled by one elected ``mbarrier.arrive`` per warp of the bound role instances.

    Every thread still reaches the signal, as with ``Threads``: it runs the side's fence, then
    the warp's ``bar.warp.sync``, which orders the memory accesses of the participating lanes
    before the elected lane's arrive (PTX ISA, ``bar.warp.sync``). The arrival thus publishes
    the whole warp's work. The count is warps, not threads: one shared-memory arrival per
    warp instead of 32; FA4 and CUTLASS pipelines signal this way.
    """


def arrivals(mech, threads: int) -> int:
    if mech is Threads:
        return threads
    return threads // 32 if mech is Warps else 1


def _fence(kinds: tuple[str, ...]):
    """Emit the side's fences, in declaration order, before its managed signal."""
    for kind in kinds:
        match kind:
            case "tmem_store":  # tcgen05.st -> tcgen05.mma reads TMEM
                cute.arch.fence_view_async_tmem_store()
            case "tmem_load":  # finish tcgen05.ld before MMA overwrites its columns
                cute.arch.fence_view_async_tmem_load()
            case "smem_async":  # st.shared -> TMA store / UMMA reads SMEM (async proxy)
                cute.arch.fence_view_async_shared()
            case _:
                raise ValueError(f"unknown fence kind {kind!r}")


def _fences(side: str, value) -> tuple[str, ...]:
    """Normalize ``commit_fence``/``release_fence``: None -> (), a kind -> (kind,), and an
    ordered tuple of distinct kinds (possibly empty) as is."""
    if value is None:
        return ()
    kinds = (value,) if isinstance(value, str) else value
    if (
        not isinstance(kinds, tuple)
        or any(k not in FENCE_KINDS for k in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError(
            f"{side} must be None, one of {FENCE_KINDS}, or an ordered tuple of distinct kinds, "
            f"got {value!r}"
        )
    return kinds


class Sharing(Enum):
    """How the role instances on one side of a ``Ring`` share its epochs.

    ``JOIN``: every instance takes every epoch it is bound to, and an epoch's count sums their
    contributions (a fan-out join). ``ROUND_ROBIN``: instance ``j`` of ``k``, in binding order,
    takes epochs ``j, j + k, ...``; the count is one instance's. ``EXCLUSIVE`` (consumers only):
    one of ``consumer_roles`` owns each epoch and the others ``skip()`` it; the count is one
    consumer's.
    """

    JOIN = "join"
    ROUND_ROBIN = "round_robin"
    EXCLUSIVE = "exclusive"


SYNC_KINDS = ("mbarrier", "named")
WAIT_KINDS = ("suspend", "spin")
SIDES = ("producer", "consumer")


@dataclass
class Channel:
    """Shared declaration for ``Ring`` and ``Slots``.

    ``producer_role``/``consumer_roles``: optional names of the roles allowed on each side,
    checked against the role annotations.

    ``consumer`` may be a ``{role: mechanism}`` dict (``Threads``, ``Warps`` or ``Umma``) for a
    joining consumer side whose roles signal differently, e.g. an MMA warp and CUDA threads
    that both read each stage. The keys are the consumer roles (``consumer_roles`` is implied)
    and each must bind a consumer end. Each instance contributes its own role's arrivals and
    signals with its own mechanism; ``release_fence`` (derived or declared) applies to the
    ``Threads``/``Warps`` roles, and ``Umma`` roles, which read through the async proxy, emit
    none. The side must be an mbarrier ``JOIN`` side of a CTA-local channel; each ``Umma`` role
    is a single-warp, single-instance role.

    ``payload=(dtype, shape)``: a per-slot SMEM buffer the layer owns, exposed as
    ``handle.data``. ``buffer=`` binds buffers of the kernel's ``SmemPlan`` instead (one
    ``SmemBuffer``, or a ``{name: SmemBuffer}`` dict): ``handle.data`` is then the stage's
    canonical view and ``handle.view(name)`` a registered partition (``storage.py``). Unlike a
    payload, a buffer does not change ``remote_scope``, and pair-scoped channels may use one.

    ``commit_fence``/``release_fence`` order the producer's/consumer's accesses before signalling:
    ``"tmem_store"`` completes TMEM stores; ``"tmem_load"`` completes TMEM loads before the other
    side's MMA overwrites their columns; ``"smem_async"`` orders generic SMEM accesses before
    async-proxy reads or writes. An ordered tuple combines kinds. Each managed arrival emits
    exactly those fences in order.

    ``commit_fence`` defaults to ``"smem_async"`` on a ``Threads``/``Warps`` -> ``Umma`` channel
    with SMEM storage (``payload=`` or ``buffer=``). Each writing thread must fence its generic
    stores before arrival: tcgen05.mma reads through the async proxy (PTX ISA, tcgen05 "Shared
    Memory Accesses"). Use ``commit_fence=()`` only if producers make no generic writes to that
    storage or fence them by hand. Without SMEM storage, no commit fence is derived.

    ``release_fence`` defaults to ``"smem_async"`` on a ``Tma``-filled channel released by
    ``Threads`` or ``Warps``. Those threads may read the stage with generic loads, and the next
    fill writes it through the async proxy, a write the read could otherwise observe (PTX ISA,
    Memory Consistency Model: Proxies and the causality axiom). Each reading thread fences
    before it arrives. ``release_fence=()`` declares that the consumers made no generic access
    to what the producer fills (they only issued TMA stores or MMAs from it, or fenced
    themselves); ``describe()`` prints any value that replaces either default.

    ``remote_scope``: memory scope of thread arrivals on a pair-scoped channel's barriers in the
    other CTA. ``"cta"`` only counts arrivals: right for TMEM/TMA/MMA handoffs ordered by their
    own fences and for stage reuse, and what CUTLASS and FA4 pipelines use. ``"cluster"`` is a
    cluster-scope release. The default is ``"cluster"`` when thread arrivals publish a layer
    payload, else ``"cta"``; a ``Tma``-filled payload keeps ``"cta"``.
    **Rule:** a pair-scoped channel whose signal publishes ordinary memory (SMEM read through
    DSMEM, or global memory) that the other CTA reads outside a layer payload must set
    ``remote_scope="cluster"``. The layer cannot detect this. Cluster scope is measurably
    slower on hot channels.

    ``override``: replace derived values of the channel's ``cluster.Scope``. The user value
    wins, and ``describe()`` prints it next to the derived one, flagging differences. Keys:
    ``full_counts``/``empty_counts`` (an int, a per-slot list, or ``{cta rank: list}``),
    ``tx`` (bytes armed per full barrier), ``full_mask``/``empty_mask`` (``tcgen05.commit``
    multicast masks). For example, a consumer that releases each stage with one raw arrival
    needs ``empty_counts=1`` and closes with ``handle.release(signal=False)``; per-warp
    elected releases are the ``Warps`` mechanism (``docs/warpdrive/dropping-down.md``).

    ``producer_sharing``/``consumer_sharing`` (a ``Sharing``) and ``sync`` are described in the
    module docstring. A named channel takes no override and must stay CTA-local; after
    collection, ``full_barriers``/``empty_barriers`` hold its ``NamedBarrier`` per slot.

    ``wait``: how this channel's blocking acquires, waits, span begins and previews poll their
    mbarrier. ``"suspend"`` (the default) is CuTeDSL's ``mbarrier.try_wait.parity`` with a
    suspend-time hint, which ptxas lowers to a ``NANOSLEEP.SYNCS`` fallback; ``"spin"`` retries
    a hint-less ``try_wait`` (the C++ ``ClusterBarrier::wait`` loop). Spinning can shorten a
    serial handoff chain whose waits are usually satisfied quickly, but it competes for issue
    slots with co-resident warps, so measure per channel. Probes, tails and named channels are
    unaffected.

    ``external``: ``"producer"`` or ``"consumer"`` when raw code (a raw role, or kernel code)
    drives that side through ``WarpSpecialized.barrier_ptr``. No role binds that side, so
    ``override`` must give its counts (``full_counts`` for an external producer,
    ``empty_counts`` for an external consumer; ``derived_scope`` has 0 there), and it takes no
    role names or fences. The layer checks nothing the external side does. The channel stays
    CTA-local: no TMA multicast, and one placement for the managed side's roles.
    """

    n: int
    producer: object = Threads
    consumer: object = Threads
    producer_role: str | None = None
    consumer_roles: tuple[str, ...] | None = None
    payload: tuple | None = None
    buffer: SmemBuffer | dict | None = None
    commit_fence: str | tuple[str, ...] | None = None
    release_fence: str | tuple[str, ...] | None = None
    remote_scope: str | None = None
    override: dict | None = None
    producer_sharing: Sharing = Sharing.JOIN
    consumer_sharing: Sharing = Sharing.JOIN
    sync: str = "mbarrier"
    turnstile: str | None = None
    external: str | None = None
    wait: str = "suspend"
    name: str = field(default="", init=False)
    fixed: bool = field(default=False, init=False)
    producers: list = field(default_factory=list, init=False)  # (RoleInstance, slot | None)
    consumers: list = field(default_factory=list, init=False)
    # Derived by the layer from the kernel's cluster and role placement (cluster.derive_scope);
    # ``scope`` has the overrides applied, ``derived_scope`` does not.
    scope: object = field(default=None, init=False, repr=False)
    derived_scope: object = field(default=None, init=False, repr=False)
    cluster: object = field(default=None, init=False, repr=False)
    derived_remote_scope: str = field(default="cta", init=False, repr=False)
    remote_scope_overridden: bool = field(default=False, init=False, repr=False)
    derived_commit_fence: tuple = field(default=(), init=False, repr=False)
    commit_fence_overridden: bool = field(default=False, init=False, repr=False)
    derived_release_fence: tuple = field(default=(), init=False, repr=False)
    release_fence_overridden: bool = field(default=False, init=False, repr=False)
    # Layer-owned channels (the scheduler's rings) publish without TMA copies.
    internal: bool = field(default=False, init=False, repr=False)
    # ``buffer`` normalized to {name: SmemBuffer}; a single buffer is keyed by its plan name.
    buffers: dict = field(default_factory=dict, init=False, repr=False)
    # sync="named": a NamedBarrier per slot for each side, allocated by the kernel.
    full_barriers: tuple = field(default=(), init=False, repr=False)
    empty_barriers: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        if type(self.n) is not int:
            raise ValueError(f"channel stages must be a positive int, got {self.n!r}")
        if self.n <= 0:
            raise ValueError("channels need at least one stage")
        if isinstance(self.producer, dict):
            raise ValueError(  # noqa: TRY004 (an invalid channel setting, like the rest)
                "per-role mechanisms are consumer-only: a producer side takes one mechanism"
            )
        for side in SIDES:
            for mech in self.mechanisms(side):
                if mech not in (Threads, Warps, Umma) and not isinstance(mech, Tma):
                    raise ValueError(
                        f"channel {side} mechanism must be Threads, Warps, Umma, or Tma(bytes), "
                        f"got {mech!r}"
                    )
        if any(isinstance(mech, Tma) for mech in self.mechanisms("consumer")):
            raise ValueError("TMA is a producer mechanism")
        self.commit_fence_overridden = self.commit_fence is not None
        self.commit_fence = _fences("commit_fence", self.commit_fence)
        self.release_fence_overridden = self.release_fence is not None
        self.release_fence = _fences("release_fence", self.release_fence)
        if self.per_role and self.release_fence and set(self.consumer.values()) == {Umma}:
            raise ValueError(
                "release_fence applies to the Threads/Warps roles of a per-role consumer, and "
                "this one has none"
            )
        if isinstance(self.producer, Tma) and self.commit_fence:
            raise ValueError("TMA producers do not arrive, so they have no commit to fence")
        if self.remote_scope not in (None, "cta", "cluster"):
            raise ValueError(f"remote_scope must be 'cta' or 'cluster', got {self.remote_scope!r}")
        if self.buffer is not None:
            if self.payload is not None:
                raise ValueError("a channel takes payload= or buffer=, not both")
            named = (
                {self.buffer.name: self.buffer}
                if isinstance(self.buffer, SmemBuffer)
                else self.buffer
            )
            if (
                not isinstance(named, dict)
                or not named
                or any(
                    not isinstance(k, str) or not isinstance(v, SmemBuffer)
                    for k, v in named.items()
                )
            ):
                raise ValueError(
                    "channel buffer must be a SmemBuffer or a non-empty {name: SmemBuffer} dict, "
                    f"got {self.buffer!r}"
                )
            self.buffers = dict(named)
        for side in SIDES:
            if not isinstance(self.sharing(side), Sharing):
                raise ValueError(  # noqa: TRY004 (an invalid channel setting, like the rest)
                    f"{side}_sharing must be a Sharing (JOIN, ROUND_ROBIN, EXCLUSIVE), got "
                    f"{self.sharing(side)!r}"
                )
        if self.sync not in SYNC_KINDS:
            raise ValueError(f"channel sync must be one of {SYNC_KINDS}, got {self.sync!r}")
        if self.wait not in WAIT_KINDS:
            raise ValueError(f"channel wait must be one of {WAIT_KINDS}, got {self.wait!r}")
        if self.wait != "suspend" and self.sync == "named":
            raise ValueError("wait='spin' selects an mbarrier wait; named channels have none")
        if self.per_role:
            self._check_per_role()
        if self.turnstile is not None and not (self.named and self.rotates):
            raise ValueError(
                "turnstile= serializes round-robin instances of a named channel: it needs "
                "sync='named' and a Sharing.ROUND_ROBIN side"
            )
        if self.producer_sharing is Sharing.EXCLUSIVE:
            raise ValueError(
                "producer_sharing=Sharing.EXCLUSIVE is not supported: only consumer ends skip() "
                "epochs; producer instances taking turns use Sharing.ROUND_ROBIN"
            )
        if self.exclusive:
            if self.named:
                raise ValueError(
                    "consumer_sharing=Sharing.EXCLUSIVE needs an mbarrier ring: exclusive "
                    "consumers skip epochs by phase, and a named barrier has no phase"
                )
            if self.producer_sharing is not Sharing.JOIN:
                raise ValueError(
                    "consumer_sharing=Sharing.EXCLUSIVE needs one producer filling every epoch "
                    "in order; producer_sharing must be Sharing.JOIN"
                )
            if not self.consumer_roles:
                raise ValueError(
                    "consumer_sharing=Sharing.EXCLUSIVE needs consumer_roles=(...) naming the "
                    "roles that take the epochs"
                )
        if self.external is not None:
            if self.external not in SIDES:
                raise ValueError(
                    f"external must be 'producer' or 'consumer', got {self.external!r}"
                )
            if self.named:
                raise ValueError(
                    f"external={self.external!r} needs an mbarrier channel: raw code drives the "
                    "side through barrier pointers, which a named-barrier channel has none of"
                )
            if self.owned_epochs:
                raise ValueError(
                    f"external={self.external!r} takes Sharing.JOIN on both sides: the layer "
                    "cannot share epochs with raw code"
                )
            if isinstance(self.producer, Tma) and self.producer.multicast:
                raise ValueError(
                    f"external={self.external!r} needs a CTA-local channel: raw code drives the "
                    "external side in each CTA, so its TMA copies take no multicast"
                )
            settings = {
                "producer": ("producer_role", "commit_fence"),
                "consumer": ("consumer_roles", "release_fence"),
            }[self.external]
            unused = [key for key in settings if getattr(self, key)]
            if unused:
                raise ValueError(
                    f"external={self.external!r} takes no {' or '.join(unused)}: no role binds "
                    "that side, and raw code fences and signals it itself"
                )
            if self.external_counts not in (self.override or {}):
                raise ValueError(
                    f"external={self.external!r} needs override={{{self.external_counts!r}: "
                    "...}: no role binds that side, so nothing derives its count"
                )
        if self.named:
            if self.producer is not Threads or self.consumer is not Threads:
                raise ValueError(
                    "a named-barrier channel counts threads: both sides must be Threads (not Warps)"
                )
            if self.override:
                raise ValueError(
                    "a named-barrier channel takes no override: its count is the producer and "
                    "consumer threads of each slot"
                )
        unknown = set(self.override or {}) - set(SCOPE_OVERRIDES)
        if unknown:
            raise ValueError(f"unknown override keys {sorted(unknown)}; one of {SCOPE_OVERRIDES}")
        # Thread arrivals publish a payload, so a remote reader needs a cluster-scope release.
        # A Tma producer completes its payload by bytes instead (the scheduler's st.async into
        # the work ring). The only thread arrival left is then the consumers' release, which
        # hands the slot back and publishes nothing, so it counts at CTA scope as in CUTLASS's
        # pipelines; a cluster-scope release costs a GPU-scope membar per thread.
        published = self.payload is not None and not isinstance(self.producer, Tma)
        self.derived_remote_scope = "cluster" if published else "cta"
        self.remote_scope_overridden = self.remote_scope is not None
        if self.remote_scope is None:
            self.remote_scope = self.derived_remote_scope
        # The proxy rules behind both defaults are fences.RULES.
        self.derived_commit_fence, self.derived_release_fence = derived_fences(self)
        if not self.release_fence_overridden:
            self.release_fence = self.derived_release_fence
        if not self.commit_fence_overridden:
            self.commit_fence = self.derived_commit_fence

    def _check_per_role(self):
        """Check a ``{role: mechanism}`` consumer and set ``consumer_roles`` to its keys."""
        roles = self.consumer
        if not roles or any(not isinstance(k, str) for k in roles):
            raise ValueError(
                f"a per-role consumer must be a non-empty {{role name: mechanism}} dict, got "
                f"{roles!r}"
            )
        if self.consumer_roles is not None:
            raise ValueError(
                "a per-role consumer dict names the consumer roles itself; drop consumer_roles="
            )
        if self.consumer_sharing is not Sharing.JOIN:
            raise ValueError(
                f"per-role consumer mechanisms need consumer_sharing=Sharing.JOIN, got "
                f"{self.consumer_sharing}: every role takes every epoch and the counts sum"
            )
        if self.sync != "mbarrier":
            raise ValueError(
                "per-role consumer mechanisms need an mbarrier channel: a named barrier counts "
                "threads on both sides"
            )
        if self.external == "consumer":
            raise ValueError("external='consumer' takes no consumer mechanisms: raw code signals")
        if isinstance(self.producer, Tma) and self.producer.multicast:
            raise ValueError(
                "per-role consumer mechanisms are CTA-local only: the producer's TMA copies take "
                "no multicast"
            )
        self.consumer_roles = tuple(roles)

    @property
    def per_role(self) -> bool:
        """The consumer side gives each role its own mechanism (``{role: mechanism}``)."""
        return isinstance(self.consumer, dict)

    def mechanisms(self, side: str) -> tuple:
        """The distinct mechanisms declared on ``side``."""
        mech = self.producer if side == "producer" else self.consumer
        return tuple(dict.fromkeys(mech.values())) if isinstance(mech, dict) else (mech,)

    def mechanism(self, side: str, role: str):
        """The mechanism instances of ``role`` signal ``side`` with."""
        mech = self.producer if side == "producer" else self.consumer
        return mech[role] if isinstance(mech, dict) else mech

    def fences(self, side: str, role: str) -> tuple[str, ...]:
        """The fences ``role``'s managed signal on ``side`` emits.

        ``Umma`` roles of a per-role consumer read through the async proxy and emit none;
        ``release_fence`` is the thread roles'.
        """
        if side == "producer":
            return self.commit_fence
        if self.per_role and self.consumer[role] is Umma:
            return ()
        return self.release_fence

    @property
    def named(self) -> bool:
        return self.sync == "named"

    @property
    def external_counts(self) -> str | None:
        """The ``override`` key that gives the external side's counts."""
        return {"producer": "full_counts", "consumer": "empty_counts"}.get(self.external)

    @property
    def checks_tma(self) -> bool:
        """A TMA producer the layer runs and whose copies it checks (``tma.py``)."""
        return isinstance(self.producer, Tma) and not self.internal and self.external != "producer"

    def bound(self, side: str) -> list:
        """Bound ``(role instance, slot)`` pairs on ``side`` ('producer' or 'consumer')."""
        return self.producers if side == "producer" else self.consumers

    def sharing(self, side: str) -> Sharing:
        return self.producer_sharing if side == "producer" else self.consumer_sharing

    @property
    def exclusive(self) -> bool:
        return self.consumer_sharing is Sharing.EXCLUSIVE

    @property
    def rotates(self) -> bool:
        """Some side is ``Sharing.ROUND_ROBIN``."""
        return Sharing.ROUND_ROBIN in (self.producer_sharing, self.consumer_sharing)

    @property
    def owned_epochs(self) -> bool:
        """Some side gives each epoch one owner (round-robin or exclusive) instead of joining;
        the kernel checks the traced ownership (``ownership.check``)."""
        return (self.producer_sharing, self.consumer_sharing) != (Sharing.JOIN, Sharing.JOIN)

    @property
    def empty_counts_overridden(self) -> bool:
        """The author set the empty counts: exclusive owners then need not contribute equally,
        and their releases must supply the counts."""
        return "empty_counts" in (self.override or {})

    def turns(self, side: str) -> int:
        """Instances taking turns on ``side``: ``k`` for a round-robin side, else 1."""
        bound = self.bound(side)
        return len(bound) if self.sharing(side) is Sharing.ROUND_ROBIN and bound else 1

    def slot_counts(self, side: str) -> list[int]:
        bound = self.bound(side)

        def contribution(ri):
            return arrivals(self.mechanism(side, ri.role), ri.num_threads)

        if self.sharing(side) is not Sharing.JOIN and bound:
            # One instance signals each epoch (the layer checks that they contribute equally).
            return [contribution(bound[0][0])] * self.n
        return [
            sum(contribution(ri) for ri, slot in bound if slot in (None, j)) for j in range(self.n)
        ]

    def named_counts(self) -> list[int]:
        """Per-slot thread count of a named channel's barriers: producers plus consumers."""
        return [p + c for p, c in zip(self.slot_counts("producer"), self.slot_counts("consumer"))]


@dataclass
class Ring(Channel):
    """Rotating stages: ``acquire()``/``wait()`` take the next stage in order."""


@dataclass
class Slots(Channel):
    """Fixed slots: the caller names the slot; each slot keeps its own phase bit."""

    def __post_init__(self):
        super().__post_init__()
        for side in SIDES:
            if self.sharing(side) is not Sharing.JOIN:
                raise ValueError(
                    f"{side}_sharing={self.sharing(side)} needs a Ring: a Slots caller names the "
                    "slot itself"
                )
        self.fixed = True
        if self.n > 31:
            raise ValueError("Slots supports at most 31 slots (phase bits share one Int32)")


@dataclass
class Turnstile:
    """Round-robin critical section between the instances of one role.

    Seat p waits on barrier p and, on leaving, arrives on barrier (p + 1) % n; seat 0 starts
    with the turn. ``enabled=False`` keeps the declaration but emits no synchronization and
    gives no exclusion.
    """

    enabled: bool = True
    name: str = field(default="", init=False)
    parties: list = field(default_factory=list, init=False)


@dataclass
class Barriers:
    """``n`` mbarriers that user code drives directly: an unmanaged channel, with no protocol
    checks.

    The layer allocates them with its own barriers, so they count toward SMEM, and covers them
    with its init fence and CTA/cluster sync. ``counts`` (an int, or a list of ``n``): arrival
    counts the layer initializes them with. ``counts=None``: the user initializes them in the
    kernel between ``begin_roles()`` and ``run_roles()``, for example with
    ``cutlass.pipeline.PipelineUmmaAsync.create(barrier_storage=self.barrier_ptr(name),
    defer_sync=True)``. Roles receive the base pointer through a ``name: Barriers`` parameter.
    """

    n: int
    counts: int | list[int] | None = None
    name: str = field(default="", init=False)

    def __post_init__(self):
        if type(self.n) is not int:
            raise ValueError(f"Barriers.n must be a positive int, got {self.n!r}")
        if self.n <= 0:
            raise ValueError("Barriers needs at least one mbarrier")
        if type(self.counts) is int:
            self.counts = [self.counts] * self.n
        if self.counts is not None and (
            not isinstance(self.counts, (list, tuple))
            or len(self.counts) != self.n
            or any(type(c) is not int or c <= 0 for c in self.counts)
        ):
            raise ValueError(f"Barriers counts must be {self.n} positive ints, got {self.counts}")


# --------------------------------------------------------------------------- #
# Device side
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class HandleRecord(Lifetime):
    chan: object = None  # the Channel
    slot: int | None = None  # the Slots slot; None for a Ring stage (dynamic)
    producer: bool = False
    role: str = ""  # the role whose end opened it: its mechanism and fences close it
    order: int = 0  # how many acquires/waits/skips of its end (and slot) were traced before it
    resets: int = 0  # its end's raw state accesses before it opened
    # (op, site, epochs apart) of the first later wait on its own end (and slot) whose epoch is
    # ready only after this handle closes (``_End._check_hold``), or None.
    held_across: tuple | None = None
    # A span's loop body when it signals in that loop's last trip: it is held until the loop ends.
    held_in: object = None


@dataclass(frozen=True)
class StageBuffer:
    """A channel buffer as an end sees it: the whole staged tensor and its stage mode."""

    tensor: object
    axis: int
    storage: str | None  # the SmemPlan buffer name; None for a layer payload


AGENTS = ("threads", "tma", "umma")


class _Stage:
    """One stage or slot of a channel as an end sees it: views of its buffers, and access
    annotations for the alias diagnostic. ``_slot`` and ``_owner()`` give the stage and the
    handle record the annotations belong to."""

    def view(
        self,
        name: str | None = None,
        *coord,
        read: bool = False,
        write: bool = False,
        by="threads",
    ):
        """Buffer ``name`` at this stage: the partition registered with ``end.views(name=...)``,
        else the canonical view. ``name`` may be None when the channel has one buffer.
        ``coord``: indices of the other modes, taken in the same slice as the stage
        (``h.view("A", None, None, kb)`` is ``tCrA[None, None, kb, h.index]``). ``read``/
        ``write`` with ``by`` ("threads", "tma", "umma") annotate the access for the alias
        diagnostic."""
        key, b = self._buffer(name)
        part = self._end._views.get(key)
        tensor, axis = (part.tensor, part.axis) if part is not None else (b.tensor, b.axis)
        if read or write:
            self._record_access(b, read, write, by)
        else:
            self._record_raw(b)
        return project(tensor, axis, self.index, coord)

    def access(
        self, name: str | None = None, *, read: bool = False, write: bool = False, by="threads"
    ):
        """Annotate an access to buffer ``name`` at this stage made through a tensor obtained
        elsewhere (for example a TMA copy function that selects the stage itself). No IR."""
        if not (read or write):
            raise ValueError("access() needs read=True and/or write=True")
        _, b = self._buffer(name)
        self._record_access(b, read, write, by)

    def _buffer(self, name):
        buffers, chan = self._end._buffers, self._end._chan
        if name is None:
            if len(buffers) != 1:
                raise ProtocolError(
                    f"channel {chan.name} has buffers {sorted(buffers)}; name the one to view"
                )
            name = next(iter(buffers))
        if name not in buffers:
            raise ProtocolError(
                f"channel {chan.name} has no buffer {name!r} (have {sorted(buffers)})"
            )
        return name, buffers[name]

    def _record_access(self, b: StageBuffer, read: bool, write: bool, by):
        if by not in AGENTS:
            raise ValueError(f"access agent must be one of {AGENTS}, got {by!r}")
        if b.storage is None:
            return None  # a layer payload is never aliased
        return self._end._trace.event(
            "access",
            storage=("smem", b.storage),
            slot=self._slot,
            read=read,
            write=write,
            agent=by,
            handle=self._owner(),
        )

    def _record_raw(self, b: StageBuffer):
        if b.storage is not None:
            self._end._trace.event("raw", storage=("smem", b.storage))


class Handle(_Stage):
    """One acquired stage or slot. Close it exactly once, in the dynamic region that opened it.

    Raw state: ``index`` (stage or slot), ``phase`` (the parity this handle waited on),
    ``full_barrier``/``barrier`` and ``empty_barrier`` (this CTA's pointers; a pair's peer has
    its barriers at the same offsets). ``commit(signal=False)``/``release(signal=False)`` close
    the handle for the checker while the caller signals the barrier itself.

    ``skip_fence=..., reason=...`` on the call that opened the handle (``acquire``, ``wait``,
    ``span``) also covers its closing signal, which a ``with`` block runs implicitly; a
    ``commit``/``release`` that passes its own replaces it.
    """

    def __init__(
        self, end, index, phase, label, is_producer: bool, opened: bool = True, skip=None
    ):
        self._end = end
        self.index = index
        self.phase = phase
        self._label = label
        self._is_producer = is_producer
        self._skip = skip  # the opening call's FenceSkip, which also covers the close
        # Trace-time lifetime record, shared by loop-carried copies of this handle.
        key = index if end._chan.fixed else None
        self.rec = HandleRecord(
            self.describe(),
            user_site(),
            current_block(),
            chan=end._chan,
            slot=key,
            producer=is_producer,
            role=end._role,
            order=end._rec.taken(key) - 1,  # its own acquire/wait already counted
            resets=end._rec.resets,
        )
        if opened:
            self._open()

    def _open(self, once: str | None = None):
        """Register the handle as open. ``once`` ("first"/"last") marks a span's event that
        runs in one iteration of its loop only, here and in ``_close``."""
        self._end._trace.opened(self.rec)
        self._end._trace.event("open", handle=self.rec, once=once)

    # The preprocessor carries every name a loop or branch reassigns, so reusing a handle name
    # across a dynamic region boundary (``with kv.produce() as h`` before and inside a loop)
    # makes the handle a carried value. It carries its index and phase; the trace-time record
    # is shared, so a carried copy still reports double closes and cross-region closes.
    def __extract_mlir_values__(self):
        return [v.ir_value() for v in (self.index, self.phase) if not isinstance(v, int)]

    def __new_from_mlir_values__(self, values):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        values = list(values)
        if not isinstance(self.index, int):
            new.index = Int32(values.pop(0))
        if not isinstance(self.phase, int):
            new.phase = Int32(values.pop(0))
        return new

    def describe(self) -> str:
        return (
            f"{self._end._chan.name}[{self._label}] {'acquire' if self._is_producer else 'wait'}"
        )

    @property
    def full_barrier(self):
        """This stage's full-barrier pointer. On a TMA producer, pass it as ``tma_bar_ptr``: the
        layer checks the copies that complete on it (``tma.py``)."""
        self._end._mbarriers("full_barrier", self if self._is_producer else None)
        return self._end._handed_full(self.index, self.rec if self._is_producer else None)

    barrier = full_barrier

    @property
    def _transaction_ctas(self) -> tuple[int, ...] | None:
        """CTAs whose full barriers complete by transaction bytes, or None for arrivals."""
        chan = self._end._chan
        if not isinstance(chan.producer, Tma):
            return None
        return chan.scope.full_ctas if chan.scope.pair else (0,)

    @property
    def empty_barrier(self):
        """This stage's empty-barrier pointer (escape hatch for raw ``cute.arch`` signalling)."""
        self._end._mbarriers("empty_barrier")
        return self._end._empty(self.index)

    @property
    def data(self):
        """This stage's canonical view of the channel's storage: the ``payload`` tensor, the
        bound buffer, or ``{name: view}`` for named buffers, keeping every mode but the stage
        mode. It records an unannotated use; for aliased storage prefer
        ``view(..., read=/write=)``."""
        buffers = self._end._buffers
        if not buffers:
            raise ProtocolError(f"channel {self._end._chan.name} has no payload or buffer")
        for b in buffers.values():
            self._record_raw(b)
        views = {k: project(b.tensor, b.axis, self.index) for k, b in buffers.items()}
        return next(iter(views.values())) if len(views) == 1 else views

    @property
    def _slot(self):
        return self.rec.slot

    def _owner(self):
        return self.rec

    def _close_skip(self, skip_fence, reason, op: str, allowed=("before",)):
        """The ``FenceSkip`` of a closing call: its own, else the opening call's."""
        skip = fence_skip(skip_fence, reason, f"{self.describe()} {op}", allowed)
        return self._skip if skip is None else skip

    def commit(self, signal: bool = True, wait: bool = False, *, skip_fence=None, reason=None):
        """Producer: publish the stage by signalling full. ``signal=False``: the caller signals
        the full barrier itself, with the derived (or overridden) counts.

        ``wait=True`` (named channels only) makes the commit a rendezvous: the producer threads
        also wait on the full barrier. When it returns every producer and consumer thread has
        arrived, and the producer threads see each other's writes made before the commit.
        ``skip_fence="before"`` (``"after"``/``"both"`` with ``wait=True``) and ``reason``
        drop the tcgen05 thread-sync fences this signal emits (``thread_sync.py``)."""
        if wait and not self._end._chan.named:
            raise ProtocolError(
                f"{self._end._trace.role_name}: commit(wait=True) on {self.describe()}: only a "
                "named-barrier channel's producer can wait on its own full barrier"
            )
        if wait and not signal:
            raise ProtocolError("commit(wait=True) signals the full barrier; drop signal=False")
        skip = self._close_skip(skip_fence, reason, "commit", FENCE_SKIPS if wait else ("before",))
        self._close(True, "commit" if signal else "commit(raw signal)")
        if wait:
            trace = self._end._trace
            trace.record(self._end._chan.name, "rendezvous", self._label, order=self.rec.order)
            _fence(self._end._chan.commit_fence)
            self._end._named_wait(self._end._chan.full_barriers, self.index, "commit", skip)
        elif signal:
            self._end._signal(self.index, skip=skip)

    def retire(self):
        """Producer: close without publishing, for a final acquire whose stage is never filled
        (e.g. the MMA's last S/P/O acquire, which only waits for P and O)."""
        if self._end._chan.scope is not None and self._end._chan.scope.tail:
            raise ProtocolError(
                f"{self._end._trace.role_name}: retire() on {self.describe()}: producers of this "
                "pair-scoped channel drain every stage before exit, and a retired stage is never "
                "released; commit it instead"
            )
        self._close(True, "retire")

    def release(self, signal: bool = True, *, skip_fence=None, reason=None):
        """Consumer: hand the stage back by signalling empty. ``signal=False``: the caller
        signals the empty barrier itself, with the derived (or overridden) counts.
        ``skip_fence="before"`` and ``reason`` as for ``commit``."""
        skip = self._close_skip(skip_fence, reason, "release")
        self._close(False, "release" if signal else "release(raw signal)")
        if signal:
            self._end._signal(self.index, skip=skip)

    def _close(
        self,
        producer: bool,
        op: str,
        once: str | None = None,
        hint: str = "",
        fenced: bool = True,
    ):
        trace = self._end._trace
        if producer != self._is_producer:
            raise ProtocolError(
                f"{trace.role_name}: {op}() on a {self.describe()} handle; producer handles "
                "commit or retire, consumer handles release"
            )
        trace.closed(self.rec, op, last_trip=once == "last", hint=hint)
        if op in ("commit", "release") and self.rec.held_across is not None:
            self._end._hold_error(self.rec, *self.rec.held_across)
        trace.record(self._end._chan.name, op, self._label, order=self.rec.order)
        trace.event(
            "close",
            handle=self.rec,
            signal=op in ("commit", "release"),
            retire=op == "retire",
            once=once,
            fenced=fenced,  # False: the signal runs without the channel's fences
        )
        if producer and isinstance(self._end._chan.producer, Tma):
            stage_closed(trace, self.rec)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit() if self._is_producer else self.release()
        return False


@dataclass(eq=False)
class _SpanProgress:
    """Trace-time progress of a span, shared by its loop-carried copies."""

    begin_site: str | None = None
    loop: object = None  # the loop body ``begin()`` was traced in


class Span(Handle):
    """One stage held across the iterations of a counted loop (``end.span()``).

    ``span()`` takes the stage and advances the end without waiting. ``begin(i)`` waits for
    the stage in the loop's first iteration, and ``commit(i)``/``release(i)`` signal in its
    last; ``commit()``/``release()`` without an index signal after the loop, in the region
    that opened the span. ``i`` is the index of ``for i in cutlass.range(n)`` with a Python
    int ``n``. The stage is this role's only between that wait and the signal.
    """

    def __init__(self, end, index, phase, label, is_producer: bool, skip=None):
        super().__init__(end, index, phase, label, is_producer, opened=False, skip=skip)
        end._rec.spans.append(self.rec)  # held from here, not only from begin()
        self._progress = _SpanProgress()
        end._trace.checks.append(self._check_begun)

    def describe(self) -> str:
        return f"{self._end._chan.name}[{self._label}] span"

    def _where(self, op: str) -> str:
        return (
            f"{self._end._trace.role_name}: {self.describe()} opened at {self.rec.site}: "
            f"{op} at {user_site()}"
        )

    def _check_begun(self):
        if self._progress.begin_site is None:
            raise ProtocolError(
                f"{self._end._trace.role_name}: {self.describe()} opened at {self.rec.site} took "
                "an epoch but never waited for it; call begin(i) in its loop"
            )

    def _trips(self, i, op: str) -> int:
        """Check that ``op(i)`` runs directly in a counted loop of the span's opening region
        and that ``i`` is the loop's index; returns the trip count."""
        block = current_block()
        loop = _counted_loop(block)
        if loop is None or _parent_block(block) != self.rec.block:
            raise ProtocolError(
                f"{self._where(f'{op}(i)')} must run directly in the body of a "
                "`for i in cutlass.range(n)` loop with a Python int n >= 1 (start 0, step 1) "
                "placed in the region that opened the span: not in a branch, a nested loop, a "
                "constexpr loop or a loop with dynamic bounds"
            )
        index, trips = loop
        if not isinstance(i, Int32) or not ir.Value.__eq__(i.ir_value(), index):
            raise ProtocolError(f"{self._where(f'{op}(i)')}: `i` must be that loop's index")
        return trips

    def begin(self, i):
        """Wait for the stage (producer: empty, consumer: full) when ``i == 0``."""
        progress = self._progress
        if progress.begin_site is not None:
            raise ProtocolError(
                f"{self._where('begin()')} after begin() at {progress.begin_site}; a span "
                "waits once"
            )
        self._trips(i, "begin")
        progress.begin_site, progress.loop = user_site(), current_block()
        self._end._check_hold(self.rec.slot, "span begin", self.rec.order)
        self._open(once="first")
        end, op = self._end, "acquire" if self._is_producer else "wait"

        def wait():
            bar = end._empty(self.index) if self._is_producer else end._full(self.index)
            with wait_range(end._trace, f"{end._chan.name}.{op}"):
                _parity_wait(bar, self.phase, end._chan.wait)
            after_sync(end._trace, self._skip)

        if_generate(i == 0, wait)

    def commit(self, i=None, *, signal: bool = True, skip_fence=None, reason=None):
        """Producer: publish the stage, in the last iteration of ``begin``'s loop when given
        its index ``i``, else here, after that loop. ``signal=False`` and ``skip_fence`` as for
        ``Handle``."""
        self._finish(True, "commit", i, signal, self._close_skip(skip_fence, reason, "commit"))

    def release(self, i=None, *, signal: bool = True, skip_fence=None, reason=None):
        """Consumer: hand the stage back, in the last iteration of ``begin``'s loop when given
        its index ``i``, else here, after that loop. ``signal=False`` and ``skip_fence`` as for
        ``Handle``."""
        self._finish(False, "release", i, signal, self._close_skip(skip_fence, reason, "release"))

    def retire(self):
        raise ProtocolError(f"{self._where('retire()')}: a span cannot be retired; commit it")

    def _finish(self, producer: bool, op: str, i, signal: bool, skip):
        if isinstance(i, bool):  # Handle's positional `signal`, which Span takes by keyword
            raise ProtocolError(
                f"{self._where(f'{op}({i})')}: a span's {op}() takes the loop index (or "
                f"nothing after the loop); pass signal={i} by keyword"
            )
        progress = self._progress
        if progress.begin_site is None:
            raise ProtocolError(f"{self._where(f'{op}()')} before begin(i) waited for the stage")
        if i is not None:
            trips = self._trips(i, op)
            if current_block() != progress.loop:
                raise ProtocolError(
                    f"{self._where(f'{op}(i)')} is in another loop than begin() at "
                    f"{progress.begin_site}"
                )
            if trips > 1:
                self.rec.held_in = progress.loop  # signals only in the loop's last trip
        self._close(
            producer,
            op if signal else f"{op}(raw signal)",
            once=None if i is None else "last",
            hint=f"; inside the loop of begin() pass the loop index, {op}(i)",
        )
        if not signal:
            return
        if i is None:
            self._end._signal(self.index, skip=skip)
        else:
            if_generate(i == trips - 1, lambda: self._end._signal(self.index, skip=skip))


class Preview(_Stage):
    """A read-only borrowed view of the epoch a consumer end's next ``wait()`` takes, returned
    by ``end.peek()`` once that epoch is full.

    It owns nothing: it has no release, adds no arrival and does not advance the end. The end's
    next ``wait()`` on the same stage or slot, in trace order, owns the epoch, and its release
    must cover the preview's accesses; the alias diagnostic records them under that handle. Use
    a preview only in the dynamic region of its ``peek()`` and before that handle closes.

    Raw state: ``index`` (stage or slot) and ``phase`` (the parity it waited on).
    """

    def __init__(self, end, index, phase, slot):
        self._end = end
        self.index = index
        self.phase = phase
        self._slot = slot if end._chan.fixed else None
        self._key = slot  # the end's advance key: the slot, None for a ring
        self._resets = end._rec.resets  # a later raw state access leaves no known owner
        self._site = user_site()
        self._block = current_block()
        self._owner_rec = None  # the adopting wait's HandleRecord
        self._events = []  # access events recorded before adoption
        self._life = Lifetime(self.describe(), self._site, self._block, region_bound=False)

    def describe(self) -> str:
        label = "i" if self._key is None else self._key
        return f"{self._end._chan.name}[{label}] peek"

    def view(
        self,
        name: str | None = None,
        *coord,
        read: bool = True,
        write: bool = False,
        by="threads",
    ):
        """Buffer ``name`` at the previewed stage (as ``Handle.view``), always a read."""
        if write or not read:
            self._read_only("view(..., write=True)" if write else "view(..., read=False)")
        return super().view(name, *coord, read=True, by=by)

    def access(
        self, name: str | None = None, *, read: bool = True, write: bool = False, by="threads"
    ):
        """Annotate a read of buffer ``name`` at the previewed stage (as ``Handle.access``)."""
        if write or not read:
            self._read_only("access(..., write=True)" if write else "access(..., read=False)")
        super().access(name, read=True, by=by)

    def commit(self, *args, **kwargs):
        self._read_only("commit()")

    def release(self, *args, **kwargs):
        self._read_only("release()")

    def retire(self, *args, **kwargs):
        self._read_only("retire()")

    def _read_only(self, op: str):
        raise ProtocolError(
            f"{self._end._trace.role_name}: {op} on {self.describe()} (taken at {self._site}) "
            f"at {user_site()}: a preview is a read-only borrow; the end's next wait() owns, "
            "and releases, this epoch"
        )

    def _owner(self):
        return self._owner_rec

    def _record_access(self, b: StageBuffer, read: bool, write: bool, by):
        who, site = self._end._trace.role_name, user_site()
        if current_block() != self._block:
            raise ProtocolError(
                f"{who}: {self.describe()} taken at {self._site} is used at {site} in a different "
                "dynamic region (loop or branch); a region running more often would read an "
                "epoch the preview never waited for. Use a preview where it was taken"
            )
        owner = self._owner_rec
        if owner is not None and owner.closed_at is not None:
            raise ProtocolError(
                f"{who}: {self.describe()} taken at {self._site} is used at {site}, after its "
                f"epoch was handed back ({owner.closed_op} at {owner.closed_at}): a stale preview"
            )
        ev = super()._record_access(b, read, write, by)
        if ev is not None and owner is None:
            self._events.append(ev)

    def _adopt(self, rec: HandleRecord):
        """The wait that owns this epoch: earlier access annotations move to its record."""
        self._owner_rec = rec
        for ev in self._events:
            ev.data["handle"] = rec
        self._events.clear()


class Token:
    """Result of one readiness probe (``end.try_wait()``/``try_acquire()``) of one epoch of one
    end; ``ready`` is the probe's dynamic Boolean.

    Pass it as ``token=`` to that end's next ``wait()``/``acquire()``, which then skips its
    blocking wait when ``ready`` holds. A token is readiness, not ownership: it cannot release,
    commit or index a stage. Only the end that probed it can use it, for that epoch, before any
    other acquire/wait on the end (or slot) and before ``end.state`` is read or assigned.

    A token is a DSL value, so a loop may carry it: probe before the loop and again at the end
    of each iteration, and consume it in the next iteration (CUTLASS's mainloop idiom). A
    carried token is checked at both ends of the carry: the value entering the loop and the one
    each iteration passes on must still name the end's next epoch, and the body must consume it
    at its top level before any other acquire/wait on the end. A token merged by a branch is
    unusable after it; probe in the region that consumes it.
    """

    def __init__(self, end, key, ready):
        self.ready = ready
        self._rec = end._rec
        self._what = (
            f"{end._chan.name} {'producer' if isinstance(end, ProducerEnd) else 'consumer'} "
            f"end of {end._trace.role_name}"
        )
        self._key = key  # the slot, None for a ring
        self._epoch = end._epoch(key)
        self._block = current_block()
        self._site = user_site()
        self._minted = False  # set by the end's probe; a token built elsewhere stays False
        self._stale = None  # a carried copy: why a value flowing into it is stale, if one is
        self._bodies = []  # (loop body, why the entering value is stale) of each loop carry
        self._carried_in = None  # a body copy: the loop body it was carried into
        self._crossed = []  # blocks this value left a region from (entering a loop or branch)

    def _why_stale(self) -> str | None:
        if self._stale is not None:
            return self._stale
        current = (self._rec.resets, self._rec.taken(self._key))
        if not self._minted or self._epoch != current:
            return (
                "an acquire/wait on this end (or raw end.state access) follows the probe, or it "
                "was not returned by a probe"
            )
        return None

    def __extract_mlir_values__(self):
        # A region boundary: record whether the value crossing it is still fresh.
        block = current_block()
        self._crossed.append(block)
        self._rec.token_exports.append((block, self._key, self._why_stale(), self._site))
        return [self.ready.ir_value()]

    def __new_from_mlir_values__(self, values):
        """A copy of this token carried into a loop body or branch, or out of either."""
        block = current_block()
        copy = object.__new__(Token)
        copy.__dict__.update(self.__dict__)
        copy.ready = cutlass.Boolean(values[0])
        copy._block = block
        copy._epoch = (self._rec.resets, self._rec.taken(self._key))
        copy._bodies, copy._carried_in, copy._crossed = [], None, []
        loops = [(b, entry) for b, entry in self._bodies if _parent_block(b) == block]
        if loops:  # out of a loop: the last iteration's value, or (no trips) the entering one
            for body, entry in loops:
                copy._stale = entry or _stale_back_edge(self._rec, self._key, body)
            return copy
        if block in self._crossed:  # out of a branch, which merges values of unknown freshness
            copy._stale = "a branch merged it"
            return copy
        if self._why_stale() is not None:
            copy._stale = self._stale or (
                f"the value entering at {user_site()} (probed at {self._site}) is stale"
            )
        if _region_op(block) in ("scf.for", "scf.while"):  # into a loop body: the back-edge
            self._bodies.append((block, copy._stale))  # is checked when the body uses it
            copy._carried_in = block
        return copy


@dataclass
class _EndRecord:
    """Trace-time record shared by loop-carried copies of one end.

    ``taken(slot)`` counts traced acquires, waits and skips per slot (``None`` for a ring).
    Together with ``resets``, the count of raw ``end.state`` accesses, it identifies a token's
    epoch. A fixed-slot tail drains only slots recorded in ``steps``. ``previews``: previews
    no wait has adopted yet. ``lineage``: which copies hold out-of-date phases.
    """

    resets: int = 0
    # Per slot (None for a ring), each traced acquire/wait/skip in order: (epochs it advances
    # the end, 0 if dynamic; the block it ran in). ``HandleRecord.order`` indexes it.
    steps: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)  # HandleRecords of the spans this end took
    previews: list = field(default_factory=list)
    # (block, slot, why stale or None, probe site) of each token crossing a region boundary.
    token_exports: list = field(default_factory=list)
    lineage: Lineage = field(default_factory=Lineage)

    def taken(self, key) -> int:
        return len(self.steps.get(key, ()))


class _End:
    """One role instance's end of a channel. A DSL value whose only runtime state is phases:
    a ``PipelineState`` for rings, an Int32 phase-bit mask for slots.

    ``cta``: the owning role's placement. With the channel's derived scope it selects local or
    remote arrivals and which CTA arms transaction barriers. ``role``: the owning role's name,
    which selects its mechanism and fences on a per-role consumer side.
    """

    def __init__(self, chan, bars, buffers, trace, bound_slot, state, cta=ALL, stride=1, role=""):
        self._chan = chan
        self._role = role
        self._stride = stride  # round-robin: epochs between this end's turns
        self._bars = bars  # full barriers at [0, n), empty barriers at [n, 2n)
        self._buffers = buffers  # {name: StageBuffer}: the payload or the bound plan buffers
        self._trace = trace
        self._bound = bound_slot
        self._state = state
        self._cta = cta
        self._rec = _EndRecord()
        self._views = {}  # {buffer name: StageView} registered with views()
        self._copy = self._rec.lineage.add()

    def __extract_mlir_values__(self):
        if self._chan.fixed:
            return [self._state.ir_value()]
        return self._state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        if self._chan.fixed:
            state = Int32(values[0])
        else:
            state = self._state.__new_from_mlir_values__(values)
        return self._rebuilt(state, self._views)

    def _rebuilt(self, state, views):
        """A new copy of this end with ``state`` and ``views``, registered in the shared
        record's lineage."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._state = state
        new._views = views or {}
        new._copy = self._rec.lineage.add()
        return new

    def _check_current(self, returned_by: str | None = None):
        """Reject this copy if a newer copy advanced past it (``Lineage``)."""
        self._rec.lineage.check(self._copy, self._trace, self._chan.name, "end", returned_by)

    def views(self, **named: StageView):
        """This end with partitions of its channel's buffers: ``ab = ab.views(A=StageView(tAsA,
        axis=1))`` makes ``h.view("A")`` return ``tAsA`` at ``h.index``. Each name must be one of
        the channel's buffers, and each view's stage mode must have the channel's stage count.
        The returned end shares this end's state; use and return it in place of this one."""
        chan = self._chan
        for name, part in named.items():
            if name not in self._buffers or self._buffers[name].storage is None:
                raise ProtocolError(
                    f"{self._trace.role_name}: {chan.name}.views({name}=...): the channel has no "
                    f"buffer {name!r} (have {sorted(k for k, b in self._buffers.items() if b.storage)})"
                )
            if not isinstance(part, StageView):
                raise ProtocolError(
                    f"{chan.name}.views({name}=...) takes a StageView, got {type(part).__name__}"
                )
            extent = stage_extent(part.tensor, part.axis, f"{chan.name}.views({name}=...)")
            if extent != chan.n:
                raise ProtocolError(
                    f"{self._trace.role_name}: {chan.name}.views({name}=...): the view's stage "
                    f"mode {part.axis} has {extent} stages, but {chan.name} has {chan.n}"
                )
        return self._rebuilt(self._state, {**self._views, **named})

    def _mbarriers(self, what: str, handle=None):
        if self._chan.named:
            raise ProtocolError(
                f"{self._trace.role_name}: {self._chan.name} is a named-barrier channel and has no "
                f"mbarriers ({what}); its barriers are channel.full_barriers/empty_barriers"
            )
        if handle is not None and isinstance(self._chan.producer, Tma):
            # The pointer a TMA copy completes on: where liveness places the stage's fill.
            self._trace.record(
                self._chan.name, "tma pointer", handle._label, order=handle.rec.order
            )
        else:
            self._trace.record(self._chan.name, "barrier pointer", None)

    @staticmethod
    def _named(barriers, index):
        """(barrier ID, thread count) of a named channel's slot ``index`` on one side."""
        if isinstance(index, int):
            nb = barriers[index]
            return nb._allocated_id(), nb.threads
        # Ring stages have consecutive IDs and one count.
        return Int32(barriers[0]._allocated_id()) + index, barriers[0].threads

    def _named_wait(self, barriers, index, op, skip=None):
        """``bar.sync`` on a named channel's slot: an arrival and a wait, so in a tcgen05 role
        it takes both thread-sync fences."""
        before_sync(self._trace, skip)
        with wait_range(self._trace, f"{self._chan.name}.{op}"):
            _named_sync(*self._named(barriers, index))
        after_sync(self._trace, skip)

    def _named_arrive(self, barriers, index):
        _named_arrive(*self._named(barriers, index))

    def _check_turn(self, side: str, op: str):
        chan = self._chan
        if chan.turnstile is None or chan.turns(side) == 1:
            return
        if chan.turnstile not in self._trace.open_turns:
            raise ProtocolError(
                f"{self._trace.role_name}: {chan.name}.{op}() at {user_site()} outside "
                f"`with {chan.turnstile}.turn():`; round-robin {side}s of this named channel share "
                "a barrier and must take it in turn"
            )

    def seed(self, slots, when=None):
        """Named channel, consumer: arrive once on the empty barriers of ``slots``. The layer
        emits this at role start, the equivalent of mbarriers starting empty. ``when(j)``: a
        condition on each arrival, a Python bool or a runtime one (a shared body's turns)."""
        for j in slots:
            cond = True if when is None else when(j)
            if isinstance(cond, bool):
                if cond:
                    self._named_arrive(self._chan.empty_barriers, j)
            else:
                _named_arrive_if(cond, *self._named(self._chan.empty_barriers, j))

    def _full(self, index):
        return self._bars + index

    def _empty(self, index):
        return self._bars + (self._chan.n + index)

    def _handed_full(self, index, handle_rec=None):
        """A full-barrier pointer handed to user code. On a TMA producer the layer records it
        and, at role exit, checks the TMA copies that complete on it (``tma.py``)."""
        ptr = self._full(index)
        chan, trace = self._chan, self._trace
        if isinstance(self, ProducerEnd) and isinstance(chan.producer, Tma) and not chan.internal:
            site = user_site()
            opted_out = trace.opt_out("tma", f"{chan.name} barrier at {site}")
            trace.tma_bars.append(TmaUse(chan, ptr.value, site, handle_rec, opted_out))
        return ptr

    # Escape hatches: everything the layer derived, for a role that drops to raw
    # cute.arch / cutlass.pipeline calls on one edge. A raw signal must keep the derived (or
    # overridden) counts in ``scope``.
    def full_barrier(self, index):
        """This CTA's full barrier of a stage/slot (a pair's peer: same offset)."""
        self._mbarriers("full_barrier")
        return self._handed_full(index)

    def empty_barrier(self, index):
        """This CTA's empty barrier of a stage/slot (a pair's peer: same offset)."""
        self._mbarriers("empty_barrier")
        return self._empty(index)

    @property
    def channel(self):
        """The channel declaration (``n``, mechanisms, ``remote_scope``, ``override``)."""
        return self._chan

    @property
    def scope(self):
        """The ``cluster.Scope`` in effect (overrides applied): barrier CTAs, per-CTA arrival
        counts, multicast masks, armed tx bytes, tail. ``channel.derived_scope`` has the
        layer's own values."""
        return self._chan.scope

    @property
    def state(self):
        """Ring: the ``PipelineState`` (index, phase, count); Slots: the Int32 phase-bit mask.

        Assign a new value to resynchronize after raw waits or signals on this edge. Reading it
        hands out the live state, which raw code may advance in place, so reading invalidates
        the end's outstanding try tokens and unadopted previews, as assigning does.
        """
        self._check_current()
        self._raw_state_access()
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._raw_state_access()

    def _raw_state_access(self):
        self._rec.resets += 1  # outstanding tokens no longer name the end's epoch
        self._rec.lineage.advanced(self._copy)
        # Raw code may advance it: the end's epochs are unknown from here on.
        self._trace.record(self._chan.name, "state", None)

    @property
    def index(self):
        """Ring: the stage the next acquire/wait takes."""
        if self._chan.fixed:
            raise ProtocolError(f"{self._chan.name} is Slots: the caller names the slot")
        self._check_current()
        return self._state.index

    def phase(self, slot=None):
        """The parity the next acquire/wait on the stage (Ring) or ``slot`` (Slots) waits for."""
        self._check_current()
        if not self._chan.fixed:
            return self._state.phase
        if not isinstance(slot, int):
            raise ProtocolError(f"{self._chan.name}.phase(slot) needs a Python int slot")
        return (self._state >> slot) & 1

    @property
    def ctas(self) -> tuple[int, ...]:
        """CTA ranks this end's role runs in."""
        cluster = self._chan.cluster
        return (0,) if cluster is None else cluster.ctas(self._cta)

    @property
    def targets(self) -> dict:
        """For each CTA rank this end runs in, the ranks whose barrier its signal reaches."""
        sc = self._chan.scope
        table = sc.full_targets if isinstance(self, ProducerEnd) else sc.empty_targets
        return {r: table.get(r, ()) for r in self.ctas}

    def _signal(self, index, fenced: bool = True, skip=None):
        """Signal stage ``index``; ``fenced=False`` skips the channel's fences, ``skip`` (a
        ``FenceSkip``) the thread-sync fence."""
        chan = self._chan
        producer = isinstance(self, ProducerEnd)
        side = "producer" if producer else "consumer"
        mech = chan.mechanism(side, self._role)
        if isinstance(mech, Tma):
            return  # completed by the TMA transaction count
        fence = chan.fences(side, self._role) if fenced else ()
        if chan.named:
            _fence(fence)
            before_sync(self._trace, skip)
            self._named_arrive(chan.full_barriers if producer else chan.empty_barriers, index)
            return
        self._arrive(
            mech,
            self._full(index) if producer else self._empty(index),
            fence,
            chan.scope.full_mask if producer else chan.scope.empty_mask,
            skip,
        )

    def _arrive(self, mech, bar, fence, mask, skip=None):
        """Signal ``bar`` in each CTA of ``self.targets``; ``mask``: the ``tcgen05.commit``
        multicast mask (an int, or ``{rank: mask}``)."""
        _fence(fence)
        cluster, targets = self._chan.cluster, self.targets
        if mech is Umma:
            # A tcgen05.commit performs tcgen05.fence::before_thread_sync for its MMAs itself.
            # Stage reuse assumes tcgen05.commit orders MMA SMEM reads before later generic
            # writes. See docs/warpdrive/guarantees.md, Hardware assumptions.
            #
            # Without pairs a commit to this CTA's own barrier needs no mask. Every tcgen05 op of
            # a pair kernel is cta_group::2; the multicast mask names the CTAs whose barrier
            # (same offset) receives the arrival.
            key = "full_mask" if isinstance(self, ProducerEnd) else "empty_mask"
            multicast = cluster is not None and (
                cluster.pair
                or any(cs != (r,) for r, cs in targets.items())
                or key in (self._chan.override or {})
            )
            if multicast and isinstance(mask, dict):
                mask = rank_value({r: mask[r] for r in self.ctas}, cluster_rank)
            with cute.arch.elect_one():
                if multicast:
                    cute.nvgpu.tcgen05.commit(bar, mask, cluster.cta_group)
                else:
                    cute.nvgpu.tcgen05.commit(bar)
        elif mech is Threads:
            before_sync(self._trace, skip)
            self._thread_arrive(bar)
        elif mech is Warps:
            before_sync(self._trace, skip)
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                self._thread_arrive(bar)
        else:
            raise AssertionError(mech)

    def _thread_arrive(self, bar):
        """One ``mbarrier.arrive`` from this thread on ``bar`` in each target CTA."""
        targets = self.targets
        if not self._chan.scope.pair:
            cute.arch.mbarrier_arrive(bar)
            return
        # See Channel.remote_scope: a cluster-scope release orders generic-memory data for
        # the other CTA's waiter; "cta" (CUTLASS's default) only counts the arrival.
        # Ordering this thread's reads before remote refill is an assumption at CTA scope
        # (docs/warpdrive/guarantees.md, Hardware assumptions).
        scope = _nvvm.MemScopeKind.CLUSTER if self._chan.remote_scope == "cluster" else None
        shared = set(targets.values())
        if len(shared) == 1:  # every CTA of the role signals the same ranks
            for c in shared.pop():
                if self.ctas == (c,):
                    mbarrier_arrive(bar, scope=scope)
                else:
                    mbarrier_arrive(bar, c, scope=scope)
            return
        # Targets relative to the signalling CTA (its MMA pair): pick them by rank.
        widths = {len(cs) for cs in targets.values()}
        assert len(widths) == 1, targets
        rank = cluster_rank()
        for k in range(widths.pop()):
            c = rank_value({r: cs[k] for r, cs in targets.items()}, rank)
            mbarrier_arrive(bar, c, scope=scope)

    def _next(self, slot, op):
        """(index, phase, label) of the next use, advancing this end's state."""
        index, phase, label = self._current(slot, op)
        if self._chan.fixed:
            self._state = self._state ^ (1 << slot)
        else:
            advance_state(self._state, self._stride)
        self._advanced(slot, 1 if self._chan.fixed else self._stride)
        return index, phase, label

    def _advanced(self, key, epochs: int):
        """Count this copy's move ``epochs`` (0 if dynamic) past the next epoch of ``key`` (a
        slot, None for a ring)."""
        self._rec.lineage.check_carried(self._copy, self._trace, self._chan.name, "end")
        self._rec.steps.setdefault(key, []).append((epochs, current_block()))
        self._rec.lineage.advanced(self._copy)

    def _check_hold(self, key, op: str, step: int | None = None):
        """Note each open handle of this end that the wait for step ``step`` of ``key`` (default:
        the next) needs back. An epoch is ready only after every epoch a ring lap (on a slot,
        one epoch) or more before it was handed back, so a wait that far past a handle its own
        role holds never returns. The handle's close raises if it signals; a raw signal or a
        TMA fill (complete without the commit) may hand the stage back earlier, so those pass.

        Distances count only the steps that run before every run of the wait: those in its own
        block or an enclosing one, at their static epoch counts."""
        chan, trace = self._chan, self._trace
        producer = isinstance(self, ProducerEnd)
        # Raw code drives the other side of an external channel, so it may not wait for the
        # hand-back at all.
        if (producer and isinstance(chan.producer, Tma)) or chan.external is not None:
            return
        steps = self._rec.steps.get(key, [])
        step = len(steps) if step is None else step
        enclosing = trace.blocks()
        lap = 1 if chan.fixed else chan.n
        for held in dict.fromkeys([*trace.open, *self._rec.spans]):
            same_end = held.chan is chan and held.producer == producer and held.slot == key
            if not same_end or held.order >= step or held.resets != self._rec.resets:
                continue
            # A handle opened in a branch arm or loop that does not enclose the wait is not held
            # when the wait runs. A span closed in its loop's last trip is held until then.
            if held.block not in enclosing:
                continue
            # Only a wait that runs in every trip of the span's loop provably runs before its
            # last-trip signal.
            if held.closed_at is not None and (
                held.held_in is None or current_block() != held.held_in
            ):
                continue
            apart = sum(n for n, block in steps[held.order : step] if block in enclosing)
            if apart < lap:
                continue
            where = (op, user_site(), apart)
            if held.closed_at is not None:  # it signals in a later trip of this loop
                if not held.closed_op.endswith("(raw signal)"):
                    self._hold_error(held, *where, close=f"{held.closed_op} in the last trip")
            elif held.held_across is None:
                held.held_across = where

    def _hold_error(self, held, op: str, site: str, apart: int, close: str | None = None):
        chan = self._chan
        lap = "slot" if chan.fixed else f"{chan.n}-stage ring"
        stages = -(-(apart + 1) // self._stride) * self._stride  # round-robin: a multiple of k
        fix = "" if chan.fixed or close else f", or give {chan.name} at least {stages} stages"
        close = close or f"{held.closed_op} at {held.closed_at}"
        raise ProtocolError(
            f"{self._trace.role_name}: {chan.name}.{op}() at {site} takes an epoch at least "
            f"{apart} after {held.what} opened at {held.site}, which this role holds until "
            f"{close}. On a {lap} that epoch is ready only after the held one is handed back, "
            f"so the wait never returns (deadlock). Hand it back first{fix}"
        )

    def _current(self, slot, op):
        """(index, phase, label) of the next use, without advancing."""
        self._check_slot(slot, op)
        self._check_current()
        if not self._chan.fixed:
            return self._state.index, self._state.phase, "i"
        return slot, (self._state >> slot) & 1, slot

    def _check_slot(self, slot, op):
        chan, who = self._chan, self._trace.role_name
        if not chan.fixed:
            if slot is not None:
                raise ProtocolError(f"{who}: ring {chan.name}.{op}() takes no slot")
            return
        if not isinstance(slot, int):
            raise ProtocolError(
                f"{who}: {chan.name}.{op}(slot) needs a Python int slot (use "
                f"cutlass.range_constexpr), got {type(slot).__name__}"
            )
        if not 0 <= slot < chan.n:
            raise ProtocolError(f"{who}: {chan.name} has {chan.n} slots, got {slot}")
        if self._bound is not None and slot != self._bound:
            raise ProtocolError(
                f"{who}: this {chan.name} end is bound to slot {self._bound}, but {op}({slot}) "
                f"was called at {user_site()}"
            )

    def _epoch(self, key):
        """Trace-time identity of the epoch the end's next use of ``key`` (a slot, None for a
        ring) takes, within one dynamic region."""
        return self._rec.resets, self._rec.taken(key)

    def _probeable(self, op: str):
        if self._chan.named:
            raise ProtocolError(
                f"{self._trace.role_name}: {self._chan.name}.{op}() at {user_site()}: a "
                "named-barrier channel cannot be probed or previewed; waiting on a named "
                "barrier (bar.sync) is an arrival that joins the handoff"
            )

    def _probe(self, slot, op: str, bar) -> Token:
        """One ``mbarrier.try_wait.parity`` on ``bar(index)`` for the end's next use."""
        self._probeable(op)
        index, phase, label = self._current(slot, op)
        self._trace.record(self._chan.name, op, label)
        token = Token(self, slot, cute.arch.mbarrier_try_wait(bar(index), phase))
        token._minted = True
        return token

    def _take_token(self, token, slot, op: str):
        """Check that ``token`` is this end's probe of the epoch ``op`` is about to take, and
        return its readiness."""
        chan, who, site = self._chan, self._trace.role_name, user_site()
        probe = "try_acquire" if isinstance(self, ProducerEnd) else "try_wait"
        where = f"{who}: {chan.name}.{op}(token=...) at {site}"
        if chan.named:
            raise ProtocolError(f"{where}: a named-barrier channel has no tokens")
        if not isinstance(token, Token):
            raise ProtocolError(
                f"{where} takes the Token returned by {chan.name}.{probe}(), got "
                f"{type(token).__name__}. Carry the Token through the loop, or wait on "
                "the end's barrier with raw code."
            )
        if token._rec is not self._rec:
            raise ProtocolError(
                f"{where} got a token of the {token._what} (probed at {token._site}); a token "
                "serves only the end that probed it"
            )
        if not token._minted:
            raise ProtocolError(f"{where}: this token was not returned by {chan.name}.{probe}()")
        self._check_slot(slot, op)
        if token._key != slot:
            raise ProtocolError(
                f"{where}: the token probed slot {token._key} (at {token._site}), not {slot}"
            )
        if token._stale is not None:
            raise ProtocolError(
                f"{where}: the token probed at {token._site} was carried through a loop or "
                f"branch, and {token._stale}; carry a token only from a probe after the "
                "end's last acquire/wait, or probe in the region that consumes it"
            )
        if token._epoch != self._epoch(slot):
            raise ProtocolError(
                f"{where}: stale token probed at {token._site}; an acquire/wait on this end (or "
                "raw end.state access) since the probe may have moved it to another epoch"
            )
        if token._block != current_block():
            raise ProtocolError(
                f"{where}: the token probed at {token._site} is used in a different dynamic "
                "region (loop or branch), where it may not name the epoch this wait takes; "
                "probe in the region that consumes it"
            )
        body = token._carried_in
        if body is not None:  # the next iteration's value is traced after this use
            rec, key = self._rec, slot

            def check_back_edge():
                stale = _stale_back_edge(rec, key, body)
                if stale is not None:
                    raise ProtocolError(f"{where}: the token carried by the loop: {stale}")

            self._trace.checks.append(check_back_edge)
        return token.ready

    def _span(self, slot, op: str, skip) -> Span:
        """Take the next stage for ``span``, without waiting for it."""
        chan, producer = self._chan, isinstance(self, ProducerEnd)
        if chan.named or (producer and isinstance(chan.producer, Tma)):
            kind = "a named-barrier channel" if chan.named else "a Tma producer"
            raise ProtocolError(
                f"{self._trace.role_name}: {chan.name}.span() at {user_site()}: {kind} cannot "
                "hold a span (a named wait is an arrival, and a Tma producer arms its stage "
                "when it acquires); use an ordinary handle"
            )
        index, phase, label = self._next(slot, op)
        self._trace.record(chan.name, op, label, order=self._rec.taken(slot) - 1, span=True)
        return Span(self, index, phase, label, producer, skip)

    def _take(self, slot, token, op: str, skip):
        """Take the next stage for ``acquire``/``wait``: check ``token`` before advancing, then
        wait for empty (producer) or full (consumer). Only an mbarrier wait may be skipped when
        the token's probe found it ready. Returns (index, phase, label)."""
        ready = None if token is None else self._take_token(token, slot, op)
        self._check_slot(slot, op)
        self._check_hold(slot, op)
        index, phase, label = self._next(slot, op)
        chan, producer = self._chan, isinstance(self, ProducerEnd)
        self._trace.record(chan.name, op, label, order=self._rec.taken(slot) - 1)
        if chan.named:
            self._check_turn("producer" if producer else "consumer", op)
            barriers = chan.empty_barriers if producer else chan.full_barriers
            self._named_wait(barriers, index, op, skip)
        else:
            with wait_range(self._trace, f"{chan.name}.{op}"):
                bar = self._empty(index) if producer else self._full(index)
                _wait(bar, phase, ready, chan.wait)
            after_sync(self._trace, skip)
        return index, phase, label


class ProducerEnd(_End):
    def try_acquire(self, slot=None) -> Token:
        """Probe once whether the stage the next ``acquire`` takes is empty, without advancing
        or arming. The probe (``mbarrier.try_wait``) may suspend the thread for a
        hardware-dependent time: it is one probe, not a latency bound."""
        return self._probe(slot, "try_acquire", self._empty)

    def acquire(self, slot=None, *, token=None, skip_fence=None, reason=None) -> Handle:
        """Wait until the stage is empty; returns a handle to commit or retire. ``token``: this
        end's ``try_acquire()`` of the same stage; the wait runs only if that probe failed. A
        ``Tma`` producer arms the stage's bytes here either way, once.

        ``skip_fence="after"`` drops the tcgen05 thread-sync fence after the wait,
        ``"before"`` the ones before the handle's arrivals (TMA arming, a named wait, the
        closing commit), ``"both"`` all of them; ``reason`` says why the role needs none there
        (``docs/warpdrive/guarantees.md``, thread-sync)."""
        skip = fence_skip(skip_fence, reason, f"{self._chan.name}.acquire")
        index, phase, label = self._take(slot, token, "acquire", skip)
        if isinstance(self._chan.producer, Tma):  # never a named channel (Threads only)
            self._arm(self._full(index), skip)
        return Handle(self, index, phase, label, True, skip=skip)

    produce = acquire  # ``with end.produce(...) as h:`` reads better for nested scopes

    def span(self, slot=None, *, skip_fence=None, reason=None) -> Span:
        """Take the next stage without waiting, to fill it across the iterations of a counted
        loop: ``begin(i)`` waits until it is empty in the first iteration, ``commit(i)``
        publishes it in the last (``commit()`` after the loop). ``skip_fence`` as for
        ``acquire``."""
        return self._span(
            slot, "acquire", fence_skip(skip_fence, reason, f"{self._chan.name}.span")
        )

    def _arm(self, full, skip=None):
        """Expect the stage's transaction bytes on every waited full barrier. In a cross-CTA
        channel one CTA arms each barrier, once, with the bytes of every CTA's copies that
        complete there, as CUTLASS's ``PipelineTmaUmma`` does."""
        scope = self._chan.scope
        before_sync(self._trace, skip)  # arming is an arrival
        if not scope.pair:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full, scope.tx)
            return
        arms = {r: cs for r, cs in self.targets.items() if cs}
        if all(cs == (r,) for r, cs in arms.items()):
            targets = (None,)  # each arming CTA arms its own barrier
        else:  # one CTA arms the barriers of its targets (rank 0's work-ring publication)
            ((rank, targets),) = arms.items()
            targets = tuple(None if c == rank else c for c in targets)
        if len(arms) == len(self.ctas):
            _arm_targets(full, scope.tx, targets)
        else:
            _arm_if(full, scope.tx, targets, tuple(arms), self._chan.cluster.size)

    def tail(self, *, skip_fence=None, reason=None):
        """Wait until every stage this end filled has been released. The layer emits this at
        role exit on pair-scoped channels whose consumer signals from another CTA.
        ``skip_fence="after"`` and ``reason`` as for ``acquire``."""
        skip = fence_skip(skip_fence, reason, f"{self._chan.name}.tail", ("after",))
        self._check_current()
        self._trace.record(self._chan.name, "tail", None)
        n = self._chan.n
        with wait_range(self._trace, f"{self._chan.name}.tail"):
            if self._chan.fixed:
                # Only slots this end acquired: another instance's slot keeps its own parity.
                for j in sorted(self._rec.steps):
                    cute.arch.mbarrier_wait(self._empty(j), (self._state >> j) & 1)
            else:
                st = self._state.clone()
                for _ in range(n):
                    cute.arch.mbarrier_wait(self._empty(st.index), st.phase)
                    st.advance()
        after_sync(self._trace, skip)


class ConsumerEnd(_End):
    def try_wait(self, slot=None) -> Token:
        """Probe once whether the stage the next ``wait`` takes is full, without advancing. The
        probe (``mbarrier.try_wait``) may suspend the thread for a hardware-dependent time: it
        is one probe, not a latency bound."""
        return self._probe(slot, "try_wait", self._full)

    def wait(self, slot=None, *, token=None, skip_fence=None, reason=None) -> Handle:
        """Wait until the stage is full; returns a handle to release. ``token``: this end's
        ``try_wait()`` of the same stage; the wait runs only if that probe failed.
        ``skip_fence`` as for ``acquire`` (``"before"``: the release's fence)."""
        skip = fence_skip(skip_fence, reason, f"{self._chan.name}.wait")
        index, phase, label = self._take(slot, token, "wait", skip)
        handle = Handle(self, index, phase, label, False, skip=skip)
        if self._rec.previews:  # never on a named channel, which cannot peek
            self._adopt(slot, handle.rec)
        return handle

    consume = wait

    def consume_unread(self, slot=None, *, skip_fence=None, reason=None):
        """Wait for the next epoch and release it at once, without accessing it: a joining
        consumer passing over an epoch another role or instance processes. The release runs
        no fence, because the end made no access to order: between the wait and the release
        nothing runs, so this end's reads of the stage all precede an earlier, fenced release.
        A previewed epoch was read through its preview, so its release keeps the fences.
        ``skip_fence`` as for ``wait``."""
        previewed = any(p._key == slot for p in self._rec.previews)
        handle = self.wait(slot, skip_fence=skip_fence, reason=reason)
        handle._close(False, "release", fenced=previewed)
        self._signal(handle.index, fenced=previewed, skip=handle._skip)

    def span(self, slot=None, *, skip_fence=None, reason=None) -> Span:
        """Take the next stage without waiting, to read it across the iterations of a counted
        loop: ``begin(i)`` waits until it is full in the first iteration, ``release(i)`` hands
        it back in the last (``release()`` after the loop). ``skip_fence`` as for ``wait``."""
        span = self._span(slot, "wait", fence_skip(skip_fence, reason, f"{self._chan.name}.span"))
        if self._rec.previews:
            self._adopt(slot, span.rec)
        return span

    def skip(self, n: int | Int32 = 1):
        """Exclusive ring: advance past ``n`` epochs other consumers own, without waiting or
        arriving (``advance_state(state, n)``). Outstanding try tokens of this end go stale,
        and an unadopted preview raises. ``n`` is a positive Python int or a dynamic ``Int32``
        (at least 0); a dynamic count leaves the ring's ownership check unverified."""
        chan, who, site = self._chan, self._trace.role_name, user_site()
        if not chan.exclusive:
            mode = "round-robin" if chan.turns("consumer") > 1 else "joining"
            raise ProtocolError(
                f"{who}: {chan.name}.skip() at {site}: only consumers of an exclusive ring "
                f"(consumer_sharing=Sharing.EXCLUSIVE) skip epochs; {mode} consumers take every "
                "epoch they are bound to"
            )
        dynamic = isinstance(n, Int32)
        if not dynamic and (type(n) is not int or n < 1):
            raise ProtocolError(
                f"{who}: {chan.name}.skip({n!r}) at {site}: the skip count must be a positive "
                "Python int or a dynamic Int32"
            )
        if self._rec.previews:
            raise ProtocolError(
                f"{who}: {chan.name}.skip() at {site} with an unconsumed preview "
                f"({self._rec.previews[0].describe()} at {self._rec.previews[0]._site}): the "
                "next wait would take another epoch than the one previewed"
            )
        self._check_current()
        self._trace.record(chan.name, "skip", None, count=None if dynamic else n)
        advance_state(self._state, n)
        self._advanced(None, 0 if dynamic else n)

    def peek(self, slot=None, *, skip_fence=None, reason=None) -> Preview:
        """Wait until the stage the next ``wait`` takes is full and return a read-only
        ``Preview`` of it, without advancing, arriving or opening a handle.
        ``skip_fence="after"`` and ``reason`` as for ``wait``."""
        skip = fence_skip(skip_fence, reason, f"{self._chan.name}.peek", ("after",))
        self._probeable("peek")
        index, phase, label = self._current(slot, "peek")
        self._check_hold(slot, "peek")
        self._trace.record(self._chan.name, "peek", label)
        with wait_range(self._trace, f"{self._chan.name}.peek"):
            _parity_wait(self._full(index), phase, self._chan.wait)
        after_sync(self._trace, skip)
        preview = Preview(self, index, phase, slot)
        self._rec.previews.append(preview)
        self._trace.opened(preview._life)
        return preview

    def _adopt(self, slot, rec: HandleRecord):
        """The wait taking ``slot``'s next epoch owns the previews of that epoch."""
        for preview in [p for p in self._rec.previews if p._key == slot]:
            if preview._resets != self._rec.resets:
                raise ProtocolError(
                    f"{self._trace.role_name}: {preview.describe()} taken at {preview._site}: "
                    f"end.state was read or assigned before the wait at {rec.site}, so that "
                    "wait may take another epoch than the one previewed"
                )
            preview._adopt(rec)
            self._rec.previews.remove(preview)
            self._trace.closed(preview._life, "adopted")


def _region_op(block) -> str | None:
    try:
        return block.owner.operation.name
    except (AttributeError, ValueError, RuntimeError):
        return None


def _counted_loop(block) -> tuple[ir.Value, int] | None:
    """``(index, trips)`` if ``block`` is the body of an ``scf.for`` over the constants
    ``0, 1, ..., trips - 1`` with at least one trip; None for any other region or bounds."""
    try:
        loop = block.owner.operation
        if loop.name != "scf.for":
            return None
        bounds = []
        for bound in list(loop.operands)[:3]:  # lower, upper, step; then carried values
            constant = bound.owner  # the defining op, or the block of a block argument
            if isinstance(constant, ir.Block) or constant.name != "arith.constant":
                return None
            bounds.append(constant.attributes["value"].value)
    except (AttributeError, ValueError, RuntimeError):
        return None
    lower, upper, step = bounds
    return (block.arguments[0], upper) if lower == 0 and step == 1 and upper >= 1 else None


def _parent_block(block):
    try:
        return block.owner.operation.block
    except (AttributeError, ValueError, RuntimeError):
        return None


def _stale_back_edge(rec: _EndRecord, key, body) -> str | None:
    """Why a token a loop ``body`` passes to its next iteration (or out of the loop) may not
    name the end's next epoch: an acquire/wait on the end after its probe. None if fresh."""
    for block, k, why, site in rec.token_exports:
        if block == body and k == key and why is not None:
            return f"the value one iteration passes to the next (probed at {site}) is stale: {why}"
    return None


def advance_state(state: pipeline.PipelineState, count: int | Int32) -> pipeline.PipelineState:
    """Advance ``state`` in place by ``count`` epochs and return it. ``count`` is a positive
    Python int or a dynamic ``Int32`` (at least 0). On CuTeDSL 4.8 this is
    ``state.advance(count)``; 4.6 and 4.7 advance one epoch per call, so there it applies 4.8's
    index, phase and count arithmetic.

    A dynamic loop carries a state only if its body reassigns it or calls a method on it, so
    reassign the result there: ``st = advance_state(st, 2)``."""
    if not isinstance(count, Int32) and count < 1:
        raise ValueError(f"advance_state: a Python int count must be at least 1, got {count!r}")
    if not isinstance(count, Int32) and count == 1:
        state.advance()
        return state
    if NATIVE_MULTI_ADVANCE:
        state.advance(count)
        return state
    state._count += count
    state._index += count
    phase_flips = state._index // state.stages
    state._index = state._index % state.stages
    state._phase ^= phase_flips % 2
    return state


def _wait(bar, phase, ready, mode: str):
    """Blocking parity wait on ``bar``; with a probe's ``ready``, only if the probe failed."""
    if ready is None:
        _parity_wait(bar, phase, mode)
    else:
        if_generate(ready == 0, lambda: _parity_wait(bar, phase, mode))


def _parity_wait(bar, phase, mode: str):
    """Wait for ``bar`` to complete phase ``phase``: ``"suspend"`` with the suspend-time hint,
    ``"spin"`` by retrying a hint-less ``try_wait`` (``Channel.wait``)."""
    if mode == "spin":
        _spin_wait(bar, phase)
    else:
        cute.arch.mbarrier_wait(bar, phase)


@cute.jit
def _spin_wait(bar, phase):
    # The peeled first probe is the fast path of a satisfied wait: a single-site loop is smaller
    # but made the GDN backward summary 4% slower (229.0 vs 220.9 us at 1x131072 H2).
    ready = cute.arch.mbarrier_try_wait(bar, phase)
    while not ready:
        ready = cute.arch.mbarrier_try_wait(bar, phase)


def _named_sync(barrier_id, threads: int):
    """Unaligned ``barrier.cta.sync``: roles reach a named barrier (a channel's, a
    ``NamedBarrier``, TMEM teardown) from different code sites, and lanes of one warp may
    arrive from divergent paths."""
    _nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(), thread_count=Int32(threads).ir_value(), aligned=False
    )


def _named_arrive(barrier_id, threads: int):
    cute.arch.barrier_arrive(barrier_id=barrier_id, number_of_threads=threads, aligned=False)


@cute.jit
def _named_arrive_if(cond, barrier_id: cutlass.Constexpr, threads: cutlass.Constexpr):
    if cond:
        _named_arrive(barrier_id, threads)


def cluster_rank():
    """This CTA's cluster rank without the warp-uniform hint (safe in divergent code)."""
    return cute.arch.block_idx_in_cluster()


def _arm_targets(full, tx: int, targets):
    """Arm each target: None (this CTA) or a rank."""
    with cute.arch.elect_one():
        for cta in targets:
            if cta is None:
                cute.arch.mbarrier_arrive_and_expect_tx(full, tx)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(full, tx, cta)


@cute.jit
def _arm_if(
    full,
    tx: cutlass.Constexpr,
    targets: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    size: cutlass.Constexpr,
):
    """Arm ``targets`` only from the CTAs in ``ctas`` (a pair's leader)."""
    if rank_in(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()), ctas, size):
        _arm_targets(full, tx, targets)


class TurnEnd:
    """One role instance's seat at a turnstile. The phase bit is loop-carried."""

    def __init__(self, ts: Turnstile, bars, seat: int, trace, phase):
        self._ts, self._bars, self._seat, self._trace, self._phase = ts, bars, seat, trace, phase

    def __extract_mlir_values__(self):
        return [self._phase.ir_value()]

    def __new_from_mlir_values__(self, values):
        return TurnEnd(self._ts, self._bars, self._seat, self._trace, Int32(values[0]))

    def turn(self, *, skip_fence=None, reason=None):
        """``with seat.turn():`` runs the body when it is this seat's turn, then passes it on.
        ``skip_fence`` drops the thread-sync fence after the turn's wait (``"after"``), before
        its pass (``"before"``) or both, as for ``acquire``."""
        return _TurnScope(self, fence_skip(skip_fence, reason, f"{self._ts.name}.turn"))


class _TurnScope:
    def __init__(self, end: TurnEnd, skip=None):
        self._end, self._skip = end, skip

    def __enter__(self):
        e = self._end
        e._trace.open_turns.add(e._ts.name)
        if e._ts.enabled:
            e._trace.record(e._ts.name, "turn", e._seat)
            with wait_range(e._trace, f"{e._ts.name}.turn"):
                cute.arch.mbarrier_wait(e._bars + e._seat, e._phase)
            after_sync(e._trace, self._skip)

    def __exit__(self, exc_type, exc, tb):
        e = self._end
        e._trace.open_turns.discard(e._ts.name)
        if exc_type is None and e._ts.enabled:
            nxt = (e._seat + 1) % len(e._ts.parties)
            before_sync(e._trace, self._skip)
            cute.arch.mbarrier_arrive(e._bars + nxt)
            e._phase = e._phase ^ 1
            e._trace.record(e._ts.name, "pass", nxt)
        return False
