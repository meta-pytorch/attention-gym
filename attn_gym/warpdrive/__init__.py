"""Author warp-specialized CuTeDSL kernels with explicit roles, channels and state.

Declare ``Ring`` (rotating stages), ``Slots`` (caller-chosen slots), turnstiles and
``TmemPlan`` as kernel attributes. Put ``@role(warps, regs)`` directly above ``@cute.jit``
on role methods. ``self.roles = {name: RoleConfig(...)}`` in ``__init__`` overrides defaults.
Configuration freezes at first collection; ``jit_cache`` keys it via ``__cute_cache_key__``.

Role parameters bind by name and annotation: ``Producer``, ``Consumer`` and
``Consumer[Instance]`` select channel ends; ``Tmem`` selects the TMEM view; ``Instance`` the
instance index; ``Local`` the instance's thread/warp/lane indices and size; ``Turn`` a
turnstile seat. The context passed to ``run_roles`` binds by its type annotation or the
single unannotated parameter. Other annotations raise ``ProtocolError``. Examples use a
frozen dataclass named ``params`` and ``self, params, *, ...state`` signatures; positional
signatures also work.

Endpoints are DSL values. A dynamic loop carries their phases only when its body uses the
local name through a method call or reassignment. Helpers must return updated ends and
callers must reassign them, e.g. ``s, stats = step(s, stats)``. Managed roles return every
endpoint, work cursor and turnstile seat, e.g. ``return ab, acc, work``. Do not return self,
params, instance indices, TMEM views or raw barrier pointers. The layer uses returned state;
it does not recover execution state from Python frames.

Channels support these handoffs (``channels.py``):

* ``consumer_sharing=Sharing.ROUND_ROBIN`` or ``producer_sharing=Sharing.ROUND_ROBIN`` lets
  instances of one role take a side's epochs in turn. ``Sharing.EXCLUSIVE`` gives one
  consumer role ownership per epoch; the others ``skip()`` it.
* ``Channel(external="consumer", override=...)`` leaves one side to raw code.
* ``sync="named"`` uses CTA named barriers for Threads-to-Threads handoffs.
* ``try_wait()``/``try_acquire()`` probe readiness; ``wait``/``acquire(token=...)`` skip the
  blocking wait when ready. ``peek()`` returns a read-only ``Preview`` of the next wait's epoch.
* ``NamedBarrier(threads=N)`` allocates a named barrier for intra-role syncs or one-shot
  flags between roles (``named.py``).

Declare ``self.work = WorkStream(mapping=..., schedule=...)`` and bind ``work: WorkCursor``.
Schedules are ``OneShot``, ``StaticPersistent``, ``AtomicPersistent``, ``ClusterLaunchControl``
or an ``IndexSchedule`` subclass. Mappings are ``Raster``, a ``WorkMap`` subclass or an
``adapters.py`` CUTLASS/FA4 tile-order adapter. ``self.cluster = CtaPair()`` or
``Cluster(m, n, pair=...)`` supplies cluster scope and multicast TMA loads; pin MMA to
``@role(..., cta="leader")``. See ``examples/`` for GEMM, persistent GEMM and FA4 forward.

``SmemPlan`` declares buffer dtype, alignment and stage axis; layouts bind at
``begin_roles(layouts=...)`` and channels bind buffers with ``buffer=``. ``h.data`` selects
the handle's stage. ``end.views(A=StageView(tAsA, axis=1))`` registers partitions for
``h.view("A")`` at ``h.index``. The alias diagnostic checks annotated accesses
(``tmem.access``/``h.view(..., read=...)``) to declared overlaps. ``alias_check`` can find
potential hazards, not prove safety.

Hide only decisions with one correct, checkable answer; leave algorithmic choices explicit.
The layer owns counts per slot/CTA, phases, stages, armed TMA bytes, barrier storage/init/
fences/teardown, TMEM allocation/free, register budgets and warpgroup padding, scheduler
plumbing, cross-CTA targets, multicast masks and protocol checks. ``describe()`` reports
derived values. Authors own tiling, layouts, stage counts, roles, schedules, work order,
skipped synchronization and math. Keep state explicit; raw CuTeDSL remains available:

* ``end.state``, ``index``, ``phase(slot)``, ``full_barrier(i)``, ``empty_barrier(i)``,
  ``targets`` and ``scope`` expose state, counts, masks and ``tx``. Handles expose ``index``,
  ``phase``, ``full_barrier`` and ``empty_barrier``. ``commit(signal=False)`` or
  ``release(signal=False)`` closes a handle while raw code supplies the signal.
* ``Channel(override={...})`` replaces counts, ``tx`` or masks. ``describe()`` compares
  effective and derived values.
* ``Barriers(n, counts=None)`` reserves user-driven mbarriers, including for a raw
  ``cutlass.pipeline`` via ``self.barrier_ptr(name)``. ``@role(..., raw=True)`` reserves
  warps and registers but only dispatches the function. Both still count in SMEM, padding,
  init fences and teardown.
* ``unchecked(check, reason=...)`` disables one trace-time check at one site.
  ``describe()`` records the reason, line and suppressed checks.
* Copies using barrier pointers the layer did not supply stay legal and mark TMA checks
  unverified.

Call ``enable_trace(backend)`` before compiling to emit ``<role>`` ranges per work item (or
role body), ``<channel>.acquire/.wait/.peek/.tail`` around blocking waits and
``<turnstile>.turn``. ``trace_region(name, payload=...)`` adds user ranges.
``waits=False`` keeps only role and user ranges. Backends are ``"iket"`` (``enable_iket()``)
and ``"globaltimer"`` (``trace_session.trace_session`` decodes it); disabled tracing emits
nothing. ``plan_from_warpdrive(kernel, cta=0)`` exports matching ``PipelinePlan`` names for
``join``, ``analyze_pipeline`` and ``write_pipeline_perfetto`` in
``attn_gym.warpdrive.profiler.pipeline``. See ``tracing.py``.

Cross-CTA thread arrivals default to ``remote_scope="cta"``: they count arrivals but do not
order ordinary memory. A pair-scoped channel publishing DSMEM or global memory read by
another CTA outside a layer payload must use ``remote_scope="cluster"``. The layer cannot
detect this requirement.
"""

import importlib

# Lazy imports let torch-only code use warpdrive.cache without loading the layer or cutlass.
# Absolute module names keep these imports visible to compile keys.
_EXPORTS = {
    "attn_gym.warpdrive.adapters": ("CutlassRaster", "Fa4Mapping"),
    "attn_gym.warpdrive.alias": (
        "bulk_commit_group",
        "bulk_wait_group",
        "fence_async_shared",
        "named_barrier",
    ),
    "attn_gym.warpdrive.channels": (
        "Barriers",
        "Channel",
        "Handle",
        "Preview",
        "Ring",
        "Sharing",
        "Slots",
        "Threads",
        "Tma",
        "Token",
        "Turnstile",
        "Umma",
        "Warps",
        "advance_state",
    ),
    "attn_gym.warpdrive.cluster": ("Cluster", "CtaPair", "SingleCta", "TmaMulticast"),
    "attn_gym.warpdrive.kernel": (
        "Consumer",
        "Instance",
        "Local",
        "Producer",
        "RoleConfig",
        "Tmem",
        "Turn",
        "WarpSpecialized",
        "role",
    ),
    "attn_gym.warpdrive.named": ("NamedBarrier",),
    "attn_gym.warpdrive.protocol": ("ProtocolError", "ProtocolLog", "unchecked"),
    "attn_gym.warpdrive.scheduling": (
        "AtomicPersistent",
        "BoundWork",
        "ClusterLaunchControl",
        "IndexSchedule",
        "OneShot",
        "PdlWait",
        "Raster",
        "Readiness",
        "StaticPersistent",
        "WorkCursor",
        "WorkMap",
        "WorkStream",
    ),
    "attn_gym.warpdrive.storage": ("SmemBuffer", "SmemPlan", "StageView"),
    "attn_gym.warpdrive.tmem": ("TmemPlan", "TmemView"),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

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
    if name in _MODULE_OF:
        value = getattr(importlib.import_module(_MODULE_OF[name]), name)
        globals()[name] = value
        return value
    # Relative dynamic import keeps the host-only profiler plan out of kernel compile keys.
    if name == "plan_from_warpdrive":
        return importlib.import_module(".pipeline_plan", __package__).plan_from_warpdrive
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
