"""warpdrive: authoring code for warp-specialized CuTeDSL kernels.

A kernel declares channels (``Ring`` for rotating stages, ``Slots`` for caller-chosen slots),
turnstiles, and a ``TmemPlan`` as attributes, and writes roles as methods decorated
``@role(warps, regs)`` directly above ``@cute.jit``; ``self.roles``, a ``{name: RoleConfig}``
set in ``__init__``, replaces those defaults per instance. The resolved configuration is fixed
on first collection; ``jit_cache`` keys it through ``__cute_cache_key__``. Role parameters
receive channel ends by name and annotation (``Producer``, ``Consumer``,
``Consumer[Instance]``), the TMEM view (``Tmem``), the instance index (``Instance``), the
instance's own thread/warp/lane indices and size (``Local``), or a turnstile seat (``Turn``).
The context passed to ``run_roles`` (a frozen dataclass of prologue values in the examples)
goes to the parameter annotated with its type, or to the role's single unannotated parameter;
any other annotation is a ``ProtocolError``.
``Ring(..., consumer_sharing=Sharing.ROUND_ROBIN)`` (or ``producer_sharing=``) lets the
instances of one role take a side's epochs in turn;
``Ring(..., consumer_sharing=Sharing.EXCLUSIVE)`` gives each epoch one owner among the consumer
roles, which ``skip()`` the epochs they do not own;
``Channel(external="consumer", override=...)`` leaves one side to raw code;
``sync="named"`` builds a Threads-to-Threads channel from CTA named barriers (``channels.py``);
``end.try_wait()``/``try_acquire()`` probe a stage once and ``wait``/``acquire(token=...)`` skip
the blocking wait when it was ready; ``end.peek()`` returns a read-only ``Preview`` of the epoch
the next ``wait()`` takes;
``NamedBarrier(threads=N)`` declares one named barrier whose ID the kernel allocates, for
syncs inside a role or one-shot flags between roles (``named.py``).

Ends are DSL values: dynamic loops carry their phases when the end is a local name that the
loop body uses through a method call or reassignment. Helpers that take ends must return them
and the caller must reassign (``s, stats = step(s, stats)``), as with any CuTeDSL value. Managed
roles return every stateful argument: producer/consumer ends, work cursors and turnstile seats,
e.g. ``return ab, acc, work``. The kernel-specific input dataclass is called ``params`` in the
examples, with signatures written as ``self, params, *, ...state``. The dispatcher respects
keyword-only parameters; positional signatures also work. Self, params, instance indices, TMEM
views and raw barrier pointers are not returned.
The layer uses the returned state directly; no execution state is recovered from Python frames.
Work scheduling (``scheduling.py``): declare ``self.work = WorkStream(mapping=...,
schedule=...)`` and give roles ``work: WorkCursor``; the index schedule (``OneShot``,
``StaticPersistent``, ``AtomicPersistent``, ``ClusterLaunchControl``, or an ``IndexSchedule``
subclass) and the mapping (``Raster``, a ``WorkMap`` subclass, or an ``adapters.py`` wrapper
around a CUTLASS/FA4 scheduler's tile order) are one-line switches. Clusters
(``cluster.py``): declare ``self.cluster = CtaPair()`` (or ``Cluster(m, n, pair=...)``, with
multicast ``Tma`` loads) and pin the MMA role with ``@role(..., cta="leader")``. See
``examples/`` for a GEMM, a persistent GEMM, and an FA4-style attention forward.

Storage (``storage.py``): an optional ``SmemPlan`` of named SMEM buffers (dtype, alignment,
stage axis; layouts bind at ``begin_roles(layouts=...)``) that channels bind with ``buffer=``;
``h.data`` is the handle's stage view and ``end.views(A=StageView(tAsA, axis=1))`` registers
partitions that ``h.view("A")`` projects at ``h.index``. The alias diagnostic (``alias.py``)
reports potential hazards between annotated accesses (``tmem.access``/``h.view(..., read=...)``)
to declared overlaps; it is a bug finder, never a proof (``alias_check``).

Design principles
-----------------
The layer exists so that agents and people can write and iterate on warp-specialized kernels
quickly. Explicit code is preferred over clever inference. The layer hides only what has one
correct, checkable answer, removes the footguns around it, and always lets the author drill down
to the underlying CuTeDSL/cutlass objects. State moves through explicit arguments and returns.

The layer derives: arrival counts per slot and per CTA; phase and stage bookkeeping; armed TMA
bytes; barrier storage, initialization, init fences, and teardown; TMEM allocation and free;
``setmaxnreg`` budgets and warpgroup padding; scheduler plumbing; cross-CTA targets and
multicast masks; and pairing/protocol checks (handle lifetimes, TMA atoms vs channel scope and
bytes). ``describe()`` prints every derived value.

The author decides: tiling, layouts, stage counts, the role split, schedules and work order,
which synchronizations to skip, and the math. Each is a constructor argument or a declaration,
so trying another one is a one-line change that the checks re-verify.

Dropping down, per edge:

* Raw state: ``end.state``/``index``/``phase(slot)``/``full_barrier(i)``/``empty_barrier(i)``/
  ``targets``/``scope`` (per-CTA counts, masks, ``tx``) and ``handle.index``/``phase``/
  ``full_barrier``/``empty_barrier``. ``handle.commit(signal=False)``/``release(signal=False)``
  close a handle while the role signals with raw ``cute.arch`` calls.
* ``Channel(override={...})`` replaces derived counts, ``tx``, or masks; ``describe()`` shows
  derived vs overridden values and flags differences.
* ``Barriers(n, counts=None)`` reserves mbarriers that user code (or a ``cutlass.pipeline``
  built on ``self.barrier_ptr(name)``) drives; ``@role(..., raw=True)`` reserves warps and a
  register budget for a function the layer only dispatches. Both still count in SMEM, padding,
  init fences, and teardown.
* ``with unchecked(check, reason=...)`` turns off one trace-time check at one site;
  ``describe()`` lists every opt-out with its reason, line, and what it let through.
* Raw ``cute.copy(..., tma_bar_ptr=p)`` with a pointer the layer did not hand out stays legal;
  ``describe()`` marks that channel's TMA check unverified.

Tracing: ``kernel.enable_trace(backend)`` before compiling makes the layer emit ranges named
after the declarations: ``<role>`` per work item (or per role body), ``<channel>.acquire``/
``.wait``/``.peek``/``.tail`` around blocking waits, ``<turnstile>.turn``, and
``with self.trace_region(name, payload=...)`` user regions (``waits=False``: role ranges and
regions only), recorded by the profiler's ``"iket"`` backend (``enable_iket()``) or its
``"globaltimer"`` backend (decoded by ``trace_session.trace_session``). Disabled, it emits
nothing. ``plan_from_warpdrive(kernel, cta=0)`` exports the matching ``PipelinePlan``, so a
capture of either backend feeds ``join``, ``analyze_pipeline``, and ``write_pipeline_perfetto``
from ``attn_gym.profiler.pipeline`` with no hand annotations (``tracing.py``).

Cluster memory scope: thread arrivals that cross CTAs default to ``remote_scope="cta"`` (what
CUTLASS and FA4 use), which counts the arrival but does not order ordinary memory. A pair-scoped
channel whose signal publishes ordinary memory (DSMEM or global) that the other CTA reads
outside a layer payload must declare ``remote_scope="cluster"``. The layer cannot detect this
case. The cluster scope costs about 30% on FA4's hot S channel.
"""

import importlib

from attn_gym.warpdrive.adapters import CutlassRaster, Fa4Mapping
from attn_gym.warpdrive.alias import (
    bulk_commit_group,
    bulk_wait_group,
    fence_async_shared,
    named_barrier,
)
from attn_gym.warpdrive.channels import (
    Barriers,
    Channel,
    Handle,
    Preview,
    Ring,
    Sharing,
    Slots,
    Threads,
    Tma,
    Token,
    Turnstile,
    Umma,
    Warps,
    advance_state,
)
from attn_gym.warpdrive.cluster import Cluster, CtaPair, SingleCta, TmaMulticast
from attn_gym.warpdrive.kernel import (
    Consumer,
    Instance,
    Local,
    Producer,
    RoleConfig,
    Tmem,
    Turn,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.named import NamedBarrier
from attn_gym.warpdrive.protocol import ProtocolError, ProtocolLog, unchecked
from attn_gym.warpdrive.scheduling import (
    AtomicPersistent,
    BoundWork,
    ClusterLaunchControl,
    IndexSchedule,
    OneShot,
    PdlWait,
    Raster,
    Readiness,
    StaticPersistent,
    WorkCursor,
    WorkMap,
    WorkStream,
)
from attn_gym.warpdrive.storage import SmemBuffer, SmemPlan, StageView
from attn_gym.warpdrive.tmem import TmemPlan, TmemView

__all__ = [
    "AtomicPersistent",
    "Barriers",
    "BoundWork",
    "Channel",
    "Cluster",
    "ClusterLaunchControl",
    "Consumer",
    "CtaPair",
    "CutlassRaster",
    "Fa4Mapping",
    "Handle",
    "IndexSchedule",
    "Instance",
    "Local",
    "NamedBarrier",
    "OneShot",
    "PdlWait",
    "Preview",
    "Producer",
    "ProtocolError",
    "ProtocolLog",
    "Raster",
    "Readiness",
    "Ring",
    "RoleConfig",
    "Sharing",
    "SingleCta",
    "Slots",
    "SmemBuffer",
    "SmemPlan",
    "StageView",
    "StaticPersistent",
    "Threads",
    "Tma",
    "TmaMulticast",
    "Tmem",
    "TmemPlan",
    "TmemView",
    "Token",
    "Turn",
    "Turnstile",
    "Umma",
    "WarpSpecialized",
    "Warps",
    "WorkCursor",
    "WorkMap",
    "WorkStream",
    "advance_state",
    "bulk_commit_group",
    "bulk_wait_group",
    "fence_async_shared",
    "named_barrier",
    "plan_from_warpdrive",
    "role",
    "unchecked",
]


def __getattr__(name: str):
    # Host-only: every kernel's compile key hashes the modules this package imports, and
    # ``plan_from_warpdrive`` (with the profiler's plan model) never runs while tracing.
    if name == "plan_from_warpdrive":
        return importlib.import_module(".pipeline_plan", __package__).plan_from_warpdrive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
