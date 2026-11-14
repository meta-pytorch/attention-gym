"""Role declarations and the ``WarpSpecialized`` kernel base."""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
import itertools
import pickle
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from types import SimpleNamespace

import cutlass
from cutlass import Int32, Uint32, cute, pipeline
from cutlass._mlir.dialects import llvm

from attn_gym.warpdrive import alias, fences, liveness, named, ownership, tracing
from attn_gym.warpdrive.alias import RoleEvents
from attn_gym.warpdrive.cache._key import canonical_key, module_source_digest
from attn_gym.warpdrive.channels import (
    SIDES,
    Barriers,
    Channel,
    ConsumerEnd,
    ProducerEnd,
    Sharing,
    StageBuffer,
    Threads,
    TurnEnd,
    Turnstile,
    Umma,
    Warps,
    _named_sync,
    advance_state,
    arrivals,
)
from attn_gym.warpdrive.cluster import (
    ALL,
    LEADER,
    RANK0,
    ROLE_PLACEMENTS,
    Cluster,
    SingleCta,
    apply_overrides,
    check_atoms,
    derive_scope,
    rank_in,
    rank_value,
)
from attn_gym.warpdrive.compat import SmemAllocator, TmemAllocator
from attn_gym.warpdrive.describe import KernelChecks, describe_kernel, kernel_checks
from attn_gym.warpdrive.named import (
    FIRST_FREE_BARRIER_ID,
    MAX_NAMED_BARRIER_ID,
    TMEM_PTR_BARRIER_ID,
    TMEM_TEARDOWN_BARRIER_ID,
    ByInstance,
    NamedBarrier,
    NamedBarriers,
)
from attn_gym.warpdrive.protocol import (
    ProtocolError,
    ProtocolLog,
    RoleTrace,
    current_trace,
    take_opt_outs,
    tracing_role,
    user_site,
)
from attn_gym.warpdrive.scheduling import (
    WorkCursor,
    WorkStream,
    _Fetch,
    scheduler_role,
)
from attn_gym.warpdrive.storage import (
    SmemPlan,
    allocate,
    stage_extent,
)
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync
from attn_gym.warpdrive.tma import TmaReport, clear_multicast_masks
from attn_gym.warpdrive.tma import check_role as check_tma
from attn_gym.warpdrive.tmem import TmemPlan, TmemView
from attn_gym.warpdrive.tracing import IketRanges, IketTrace

WARP_SIZE = 32
WARPGROUP = 4
REGS_PER_SM = 65536
MIN_REGS = 24  # setmaxnreg floor: the budget of a warpgroup no role owns
MAX_WARPS = 32  # CUDA's 1024-thread block limit


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


class _Bound:
    """``Consumer[Instance]``: an end restricted to the slot equal to the instance index."""

    def __init__(self, side):
        self.side = side


class _SideMeta(type):
    def __getitem__(cls, key) -> _Bound:
        if key is not Instance:
            raise TypeError(f"{cls.__name__}[...] takes Instance, got {key!r}")
        return _Bound(cls)


class Producer(metaclass=_SideMeta):
    """Annotation: producer end of the channel with the parameter's name."""


class Consumer(metaclass=_SideMeta):
    """Annotation: consumer end of the channel with the parameter's name."""


class Tmem:
    """Annotation: the role receives a ``TmemView``."""


class Instance:
    """Annotation: the role receives its instance index, a Python int (constexpr), or a
    warp-uniform ``Int32`` in a ``shared_body`` role."""


class Local:
    """Annotation and value: the role instance's own thread, warp and lane indices.

    Hardware ids (``cute.arch.thread_idx()``, ``warp_idx()``) depend on where the role is
    placed, and ``roles=`` overrides move and resize roles, so role code that indexes by its own
    threads should use these. ``tid`` and ``warp`` count from the instance's first thread and
    warp (free at offset 0); ``warp`` is warp-uniform. ``warps`` and ``threads`` are the
    instance's size as Python ints. TMEM lanes follow the hardware warp instead; ``tmem_row``
    is the TMEM row this thread may access.
    """

    def __init__(self, role_name: str, first_warp: int | Int32, warps: int):
        self._role_name = role_name
        self._first_warp = first_warp  # a runtime value in a shared body
        self.warps = warps
        self.threads = warps * WARP_SIZE

    def _from_first(self, index, per_warp: int):
        """``index`` (a thread or warp index) relative to the instance's first one."""
        if isinstance(self._first_warp, int) and self._first_warp == 0:
            return index
        return index - self._first_warp * per_warp

    # Holds no DSL values: a loop or branch that captures it carries nothing.
    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self

    @property
    def tid(self) -> Int32:
        """Thread index within the instance, ``0 .. threads - 1``."""
        return _assume_below(self._from_first(cute.arch.thread_idx()[0], WARP_SIZE), self.threads)

    @property
    def warp(self) -> Int32:
        """Warp index within the instance, ``0 .. warps - 1`` (warp-uniform)."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        return _assume_below(self._from_first(warp, 1), self.warps)

    @property
    def lane(self) -> Int32:
        """Lane within the warp (instances are whole warps, so the hardware lane)."""
        return cute.arch.lane_idx()

    @property
    def tmem_row(self) -> Int32:
        """This thread's TMEM row, ``0 .. threads - 1``; it may access lane ``row % 128``.

        ``tcgen05.ld``/``st`` from hardware warp ``w`` reach only lanes ``32 * (w % 4)`` to
        ``32 * (w % 4) + 31``, wherever the role is placed. The row is therefore the thread
        index modulo the instance's size, which is right for any contiguous placement of ``4k``
        warps. Raises ``ProtocolError`` at trace time when the instance's warp count is not a
        multiple of 4.
        """
        if self.warps % WARPGROUP:
            raise ProtocolError(
                f"role {self._role_name}: tmem_row needs whole warpgroups (each covers the 128 "
                f"TMEM lanes once), but the instance has {self.warps} warps"
            )
        return cute.arch.thread_idx()[0] % self.threads


def _assume_below(value: Int32, bound: int) -> Int32:
    """``value``, with ``0 <= value < bound`` stated to LLVM.

    Role dispatch guarantees the range, but LLVM does not derive it from the dispatch branch, so
    predicates and index math on role-local indices would stay runtime work. The unsigned compare
    states both bounds; the assumption emits no instruction.
    """
    llvm.intr_assume((Uint32(value) < Uint32(bound)).ir_value(), [], [])
    return value


class Turn:
    """Annotation: the role receives its seat at the turnstile with the parameter's name."""


class _SchedulerCursor:
    """Internal annotation: the scheduler warp's own cursor (it decodes and publishes)."""


class _SchedulerFetch:
    """Internal annotation: the scheduler warp's private schedule channels (both ends)."""


class _Context:
    """Internal annotation: the parameter receives the context passed to ``run_roles``."""


_ANNOTATIONS = {
    "Producer": Producer,
    "Consumer": Consumer,
    "Tmem": Tmem,
    "Instance": Instance,
    "Local": Local,
    "Turn": Turn,
    "WorkCursor": WorkCursor,
    "Barriers": Barriers,
}
# Parameters a raw role may take: no channel ends, cursors, or seats (they carry protocol state).
_RAW_ANNOTATIONS = (Instance, Local, Tmem, Barriers, _Context)
# Layer classes a role might write in place of an annotation, and the annotation meant.
_LOOKALIKES = (
    (ProducerEnd, "Producer"),
    (ConsumerEnd, "Consumer"),
    (Channel, "Producer or Consumer"),
    (WorkStream, "WorkCursor"),
    (TmemPlan, "Tmem"),
    (TmemView, "Tmem"),
    (Turnstile, "Turn"),
    (TurnEnd, "Turn"),
)
_CTX_RULE = (
    "a parameter receives the context when it is annotated with the kernel's context type (the "
    "type of the object passed to run_roles, or a base of it) or is the role's single "
    "unannotated parameter"
)


def _resolve_param(role_name: str, pname: str, ann, fn) -> tuple[object, type | None]:
    """``(annotation, context type)`` of one role parameter. A context parameter gets
    ``_Context`` and its annotated type, or ``None`` when unannotated."""
    if ann is inspect.Parameter.empty:
        return _Context, None
    if isinstance(ann, str):
        # String annotations (``from __future__ import annotations``) resolve in the module
        # that defines ``fn``, not the module of its ``@cute.jit`` wrapper.
        namespace = {**inspect.unwrap(fn).__globals__, **_ANNOTATIONS}
        try:
            ann = eval(ann, namespace)
        except Exception as exc:
            raise ProtocolError(
                f"role {role_name}: parameter {pname} is annotated {ann!r}, which does not "
                f"resolve ({type(exc).__name__}: {exc}); string annotations resolve in the "
                "role's module globals, so a context type must be module-level"
            ) from exc
    if isinstance(ann, _Bound) or ann in _ANNOTATIONS.values():
        return ann, None
    if isinstance(ann, type):
        for cls, meant in _LOOKALIKES:
            if issubclass(ann, cls):
                raise ProtocolError(
                    f"role {role_name}: parameter {pname} is annotated {ann.__name__}, a layer "
                    f"class rather than a role annotation; did you mean `{pname}: {meant}`?"
                )
        return _Context, ann
    raise ProtocolError(
        f"role {role_name}: parameter {pname} has unrecognized annotation {ann!r}; {_CTX_RULE}"
    )


def _context_type(ctx_types: dict) -> tuple[type | None, str | None]:
    """Return ``(context type, first "role.param" annotated with it)`` from
    ``{annotated type: first "role.param" using it}``, or ``(None, None)`` when no role
    annotates its context."""
    for t, site in ctx_types.items():
        if not (is_dataclass(t) and t.__dataclass_params__.frozen):
            raise ProtocolError(
                f"{site} is annotated {t.__name__}, which is not a frozen dataclass; an "
                "annotated context reaches the roles as itself, which the layer supports "
                "for frozen dataclasses (leave a SimpleNamespace context unannotated)"
            )
    # The context type is the most derived annotation; the others must be its bases.
    derived = [t for t in ctx_types if all(issubclass(t, u) for u in ctx_types)]
    if ctx_types and not derived:
        sites = ", ".join(f"{site}: {t.__name__}" for t, site in ctx_types.items())
        raise ProtocolError(
            f"roles annotate the context with unrelated types ({sites}); {_CTX_RULE}"
        )
    if not derived:
        return None, None
    return derived[0], ctx_types[derived[0]]


@dataclass
class RoleInstance:
    name: str  # "softmax[0]", or the method name for single-instance roles
    role: str
    index: int
    first_warp: int
    num_warps: int
    regs: int | None
    fn: object
    params: list
    # "all": every CTA of the cluster; "leader": the leader of each MMA pair; "rank0": rank 0
    # (the broadcast scheduler only)
    cta: str = ALL
    raw: bool = False  # dispatched without ends, protocol log, or trace checks
    keyword_only: frozenset[str] = frozenset()
    shared: bool = False  # traced once for all instances of its role (RoleConfig.shared_body)

    @property
    def warps(self) -> range:
        return range(self.first_warp, self.first_warp + self.num_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE

    @property
    def uses_tmem(self) -> bool:
        return any(ann is Tmem for _, ann in self.params)


@dataclass(frozen=True)
class RoleConfig:
    """Host-side role placement, resolved before channels and launch resources are built.

    ``warps``: an int names one warp, a range a contiguous group, and a sequence of ranges one
    role instance per range; it is stored as a tuple of ranges. ``regs=None`` leaves the
    register budget alone; it does not tune it. All roles in a warpgroup must agree on ``regs``.
    ``shared_body=True`` compiles one body for all instances, which receive their index at run
    time (``docs/warpdrive/roles.md``, "Shared bodies"); with one instance it changes nothing.
    """

    warps: int | range | list[range] | tuple[range, ...]
    regs: int | None = None
    cta: str = ALL
    raw: bool = False
    shared_body: bool = False

    def __post_init__(self):
        if self.cta not in ROLE_PLACEMENTS:
            raise ValueError(f"role cta must be one of {ROLE_PLACEMENTS}, got {self.cta!r}")
        if self.regs is not None and (
            type(self.regs) is not int or not MIN_REGS <= self.regs <= 256 or self.regs % 8 != 0
        ):
            raise ValueError(
                f"role regs must be None or an int multiple of 8 in [24, 256], got {self.regs!r}"
            )
        if type(self.raw) is not bool:
            raise ValueError(f"role raw must be a bool, got {self.raw!r}")
        if type(self.shared_body) is not bool:
            raise ValueError(f"role shared_body must be a bool, got {self.shared_body!r}")
        if type(self.warps) is int:
            ranges = (range(self.warps, self.warps + 1),)
        elif isinstance(self.warps, range):
            ranges = (self.warps,)
        else:
            try:
                ranges = tuple(self.warps)
            except TypeError as exc:
                raise ValueError(
                    f"role warps must be an int, range, or list of ranges, got {self.warps!r}"
                ) from exc
        if not ranges:
            raise ValueError(f"role warps must contain at least one range, got {self.warps!r}")
        for r in ranges:
            if not isinstance(r, range) or r.step != 1 or len(r) == 0:
                raise ValueError(
                    f"role warps must be non-empty unit-step ranges, got {self.warps!r}"
                )
            if r.start < 0 or r.stop > MAX_WARPS:
                raise ValueError(f"role warps must lie in [0, {MAX_WARPS}), got {self.warps!r}")
        object.__setattr__(self, "warps", ranges)


def role(
    warps: int | range | list[range] | tuple[range, ...],
    regs: int | None = None,
    cta: str = ALL,
    raw: bool = False,
    shared_body: bool = False,
):
    """Mark a method as a warp role with its default placement.

    ``@role(warps=4, regs=40)`` is the role's default ``RoleConfig``. A kernel instance can
    replace it per role through ``self.roles``, a ``{method name: RoleConfig}`` dict set in
    ``__init__``, usually from a constructor argument merged over the kernel's defaults. The
    placement is fixed on first collection (``describe()``, compile-cache lookup, or tracing);
    build a new instance to change it.

    A list of warp ranges instantiates the method once per range; an ``Instance`` parameter
    receives the index. Each instance compiles its own body, with the index as a Python int;
    ``shared_body=True`` compiles one body for all of them, with the index as a runtime
    ``Int32`` (``docs/warpdrive/roles.md``, "Shared bodies"). ``cta="leader"`` runs the role
    only in each MMA pair's leader (every CTA without pairs); ``"all"`` is the default.
    ``raw=True`` roles take ``Instance``, ``Local``, ``Tmem``, ``Barriers`` and a context, but no
    managed channel ends or work cursors; they still take part in register allocation and TMEM
    teardown.

    Put this decorator directly above ``@cute.jit``: CuTeDSL re-applies it when it preprocesses
    the source.
    """
    declaration = RoleConfig(warps, regs, cta, raw, shared_body)

    def deco(fn):
        fn._warpdrive = declaration
        return fn

    return deco


# --------------------------------------------------------------------------- #
# Kernel base
# --------------------------------------------------------------------------- #


# Keyed settings that may change after construction. Collection freezes the cluster (it fixes
# the channel scopes), the work streams (the scheduler roles) and trace_protocol (the protocol
# log); the rest are read at trace time.
_HOST_SETTINGS = (
    "alias_check",
    "liveness_check",
    "trace_protocol",
    "trace_backend",
    "trace_waits",
    "trace_events_per_role",
)
_COLLECTED_SETTINGS = ("cluster", "trace_protocol")
# Trace results for describe(), pipeline plans and trace-time warnings.
# __cute_trace_record__ and __cute_restore_trace__ save and restore exactly these attributes;
# _reset_trace gives each its empty value.
_TRACE_RESULTS = (
    "_ws_tma",
    "_ws_smem",
    "_ws_alias",
    "_ws_fences",
    "_ws_opt_outs",
    "_ws_uses",
    "_ws_named_users",
    "_ws_ownership",
    "_ws_liveness",
    "_ws_trace_tags",
)


@functools.cache
def _init_signature(cls) -> inspect.Signature:
    return inspect.signature(cls.__init__)


def _rebuild(cls, args, kwargs, settings):
    kernel = cls(*args, **kwargs)
    for name, value in settings.items():
        setattr(kernel, name, value)
    return kernel


class _KernelType(type):
    """Metaclass that records the constructor call.

    Everything any ``__init__`` of the call assigns belongs to the constructor. The arguments
    are deep-copied before the call runs; that copy becomes the kernel's cache key and is
    replayed on unpickling, so a later in-place change to an argument affects neither.
    """

    def __call__(cls, *args, **kwargs):
        try:
            init = copy.deepcopy((args, kwargs))
        except Exception as error:  # noqa: BLE001 (raised by __cute_cache_key__ and pickling)
            init = TypeError(
                f"{cls.__name__}: the constructor arguments cannot be copied, so the kernel has "
                f"no cache key and cannot be pickled ({error})"
            )
        kernel = super().__call__(*args, **kwargs)
        if isinstance(init, TypeError):
            frozen = init
        else:
            call = _init_signature(cls).bind(kernel, *init[0], **init[1])
            call.apply_defaults()
            try:
                frozen = canonical_key(tuple(call.arguments.items())[1:])  # without self
            except TypeError as error:
                frozen = error
        object.__setattr__(kernel, "_ws_init_key", frozen)
        object.__setattr__(kernel, "_ws_init", init)
        return kernel


class WarpSpecialized(metaclass=_KernelType):
    """Base for warp-specialized kernels.

    Subclasses declare ``Ring``/``Slots``/``Turnstile`` attributes and an optional
    ``tmem_plan`` in ``__init__``, and mark role methods with ``@role``. Inside the subclass's
    ``@cute.kernel``::

        smem = self.begin_roles()   # allocate and init barriers and payloads, TMEM alloc
        ...user SMEM tensors and context...
        self.run_roles(ctx)         # init fence + CTA sync, dispatch, TMEM teardown

    The layer owns barrier storage and initialization, arrival counts, TMEM allocation and
    teardown, ``setmaxnreg`` per role, warpgroup padding, and the trace-time protocol checks.
    """

    tmem_plan: TmemPlan | None = None
    smem_plan: SmemPlan | None = None  # SMEM buffers channels bind with buffer= (storage.py)
    # ``{method name: RoleConfig}`` replacing those roles' ``@role`` declarations. Set it in
    # ``__init__`` from a constructor argument so the compile key sees it, and use a dict: the
    # key canonicalizes dicts, not other mappings. Unknown names raise at collection.
    roles: dict[str, RoleConfig] | None = None
    # Alias diagnostic (alias.py) and fence check (fences.py): "warn" (describe() plus one
    # warning on findings), "strict" (raise on a potential hazard or missing fence), or "off".
    # Host-only: it never changes generated code.
    alias_check: str = "warn"
    liveness_check: str = "strict"
    trace_protocol: bool = False
    # Ranges named after roles and channels (tracing.py), set with enable_trace().
    trace_backend: str | None = None  # "iket", "globaltimer", or None: no ranges
    trace_waits: bool = True  # also a range around every blocking channel wait
    trace_events_per_role: int = 1024  # globaltimer: records per role instance and CTA
    cluster: Cluster = SingleCta()  # or CtaPair() for 2-CTA MMAs
    _ws_roles: list | None = None  # the role instances, once collected (_collect)
    _ws_traced: bool = False  # trace results are present, from tracing or a cached build

    def __reduce__(self):
        """Pickle as the constructor call plus what the cache key lets change after it: the
        host settings, the cluster and the work streams. A fresh compiler process rebuilds the
        kernel from that. Any other attribute assigned after construction makes the key raise,
        so none are kept."""
        if isinstance(self._ws_init, TypeError):
            raise self._ws_init
        state = vars(self)
        settings = {k: state[k] for k in (*_HOST_SETTINGS, "cluster") if k in state}
        # A stream refers back to its owner; ship its declaration and let setattr re-own it.
        settings |= {k: replace(v) for k, v in state.items() if isinstance(v, WorkStream)}
        return _rebuild, (type(self), *self._ws_init, settings)

    def __setattr__(self, name, value):
        state = vars(self)
        collected_setting = isinstance(value, WorkStream) or name in _COLLECTED_SETTINGS
        if collected_setting and self._ws_roles is not None:
            raise ProtocolError(
                f"{type(self).__name__}.{name} cannot change after collection (describe, "
                "compile-cache lookup, or tracing derived the channels from it); construct a "
                "new kernel"
            )
        if isinstance(value, WorkStream):
            value.owner = self  # the stream's work unit follows this kernel's cluster
        super().__setattr__(name, value)  # a read-only property raises before anything is kept
        if "_ws_init" in state and not (  # assigned after construction
            collected_setting or name.startswith("_") or name in _HOST_SETTINGS
        ):
            state.setdefault("_ws_late", set()).add(name)  # not keyed: see __cute_cache_key__

    # ---- host-side analysis ------------------------------------------------ #

    def _collect(self):
        if self._ws_roles is not None:
            return
        chans = {k: v for k, v in vars(self).items() if isinstance(v, Channel)}
        turns = {k: v for k, v in vars(self).items() if isinstance(v, Turnstile)}
        raw_bars = {k: v for k, v in vars(self).items() if isinstance(v, Barriers)}
        named_bars = {k: v for k, v in vars(self).items() if isinstance(v, NamedBarrier)}
        groups = {k: v for k, v in vars(self).items() if isinstance(v, NamedBarriers)}
        streams = {k: v for k, v in vars(self).items() if isinstance(v, WorkStream)}
        if len(streams) > 1:
            raise ProtocolError(
                "one WorkStream per kernel (every role starts at this CTA's first item), got "
                f"{list(streams)}"
            )
        seen = {}
        declarations = (
            *chans.items(),
            *turns.items(),
            *raw_bars.items(),
            *named_bars.items(),
            *groups.items(),
        )
        for name, obj in (*declarations, *streams.items()):
            if id(obj) in seen:
                raise ProtocolError(
                    f"attributes {seen[id(obj)]!r} and {name!r} use the same declaration; "
                    "give each channel, turnstile, barrier set, named barrier, and stream its "
                    "own object"
                )
            seen[id(obj)] = name
        cluster = self.cluster
        fetch_keys = {}
        for name, st in streams.items():
            st.name = name
            if st.owner is not None and st.owner is not self:
                # The stream sizes its work unit from its owner's cluster and looks its problem
                # up there, so a second kernel would read this one's.
                raise ProtocolError(
                    f"{type(self).__name__}.{name}: this WorkStream now belongs to a "
                    f"{type(st.owner).__name__}; a stream belongs to one kernel (create it in "
                    "__init__)"
                )
            st.owner = self
            if st.schedule.broadcast:
                generated = {st._channel_name(name, "ring"): st.channel()}
                fetch = st.schedule.fetch_channels()
                fetch_keys[name] = tuple(fetch)
                generated.update({st._channel_name(name, k): ch for k, ch in fetch.items()})
                for k, ch in generated.items():
                    if k in vars(self):
                        raise ProtocolError(
                            f"WorkStream {name}: attribute {k!r} is reserved for its scheduler"
                        )
                    chans[k] = ch
        for name, group in groups.items():  # consecutive IDs: declared as consecutive barriers
            for i, nb in enumerate(group.barriers):
                named_bars[f"{name}[{i}]"] = nb
        for name, obj in (
            *chans.items(),
            *turns.items(),
            *raw_bars.items(),
            *named_bars.items(),
            *groups.items(),
        ):
            obj.name = name
        for ch in chans.values():
            ch.producers.clear()
            ch.consumers.clear()
        for ts in turns.values():
            ts.parties.clear()
        plan = self.smem_plan
        for name, ch in chans.items():
            for key, buf in ch.buffers.items():
                if plan is None or plan.buffers.get(buf.name) is not buf:
                    raise ProtocolError(
                        f"channel {name}: buffer {key!r} ({buf.name}) is not a buffer of this "
                        "kernel's smem_plan; declare it with self.smem_plan.buffer(...)"
                    )
        roles = []
        ctx_types = {}  # annotated context type -> first "role.param" using it
        declared = {
            name: config
            for name in dir(type(self))
            if (config := getattr(getattr(type(self), name), "_warpdrive", None)) is not None
        }
        for name, config in _role_configs(self.roles, declared).items():
            fn = getattr(type(self), name)
            shared = config.shared_body and len(config.warps) > 1
            if not hasattr(fn, "__wrapped__"):
                raise TypeError(f"role {name}: put @role(...) directly above @cute.jit")
            signature = inspect.signature(fn)
            for p in signature.parameters.values():
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    raise ProtocolError(
                        f"role {name}: variadic parameter {p.name} cannot be injected; "
                        "declare each parameter explicitly"
                    )
            keyword_only = frozenset(
                p.name
                for p in signature.parameters.values()
                if p.kind is inspect.Parameter.KEYWORD_ONLY
            )
            params = []
            for pname, p in list(signature.parameters.items())[1:]:
                ann, ctx_type = _resolve_param(name, pname, p.annotation, fn)
                if config.raw and ann not in _RAW_ANNOTATIONS:
                    raise ProtocolError(
                        f"raw role {name}: parameter {pname} is {getattr(ann, '__name__', ann)}; "
                        "raw roles take Instance, Local, Tmem, Barriers, and the context only "
                        "(drive edges through Barriers, or make the role managed)"
                    )
                if ann is Barriers and pname not in raw_bars:
                    raise ProtocolError(f"role {name}: no Barriers named {pname!r}")
                if ann is WorkCursor and pname not in streams:
                    raise ProtocolError(f"role {name}: no WorkStream named {pname!r}")
                if ctx_type is not None:
                    ctx_types.setdefault(ctx_type, f"{name}.{pname}")
                params.append((pname, ann))
            ctx_params = [p for p, a in params if a is _Context]
            if len(ctx_params) > 1:
                raise ProtocolError(
                    f"role {name}: parameters {ctx_params} would all receive the context; "
                    f"{_CTX_RULE}"
                )
            if shared:
                _check_shared_body(name, config.warps, params)
            for idx, wr in enumerate(config.warps):
                inst = f"{name}[{idx}]" if len(config.warps) > 1 else name
                ri = RoleInstance(
                    inst,
                    name,
                    idx,
                    wr.start,
                    len(wr),
                    config.regs,
                    fn,
                    params,
                    cta=config.cta,
                    raw=config.raw,
                    keyword_only=keyword_only,
                    shared=shared,
                )
                roles.append(ri)
                self._bind(ri, chans, turns)
        self._ws_ctx_type, self._ws_ctx_site = _context_type(ctx_types)
        for name, st in streams.items():
            self._bind_stream(name, st, roles, chans, cluster, fetch_keys.get(name, ()))
        roles.sort(key=lambda r: r.first_warp)
        self._validate(chans, turns, roles)
        self._plan_registers(roles)
        self._validate_placement(chans, turns, roles)
        self._ws_named = _allocate_named(chans, named_bars)
        for name, nb in named_bars.items():
            if nb.threads > self._ws_num_warps * WARP_SIZE:
                raise ProtocolError(
                    f"NamedBarrier {name} counts {nb.threads} threads, but the kernel launches "
                    f"{self._ws_num_warps * WARP_SIZE}; it can never complete"
                )
        self._ws_chans, self._ws_turns, self._ws_roles = chans, turns, roles
        self._ws_raw_bars = raw_bars
        self._ws_named_bars = named_bars
        self._ws_streams = streams
        self._ws_fetch_keys = fetch_keys
        self._reset_trace()

    def _reset_trace(self):
        """Clear trace state so a failed retrace cannot expose an earlier trace's reports.

        This is the only reset, called at collection and the start of ``begin_roles``.
        ``_ws_traced`` stays false until ``_finish_trace`` or ``__cute_restore_trace__`` marks
        the reports ready.
        """
        self._ws_traced = False
        # Results, saved and restored by cached builds (_TRACE_RESULTS).
        self._ws_tma = {}  # channel name -> tma.TmaReport, filled while tracing
        self._ws_smem = {}  # SmemPlan buffer name -> storage.ResolvedBuffer, after begin_roles
        self._ws_alias = None  # alias.AliasReport, after a trace
        self._ws_fences = None  # fences.FenceReport, after a trace
        self._ws_opt_outs = []
        self._ws_uses = {}  # (role, channel) -> acquires/waits per work item (0: inner loop)
        self._ws_named_users = {}  # NamedBarrier name -> {"sync"/"arrive": role names}
        self._ws_ownership = {}  # owned-epoch channel -> epoch-ownership verdict
        self._ws_liveness = None  # liveness.Liveness, after a trace
        self._ws_trace_tags = []  # globaltimer range names
        # The protocol log is not saved: a cached build has none (trace_key).
        self._ws_log = ProtocolLog() if self.trace_protocol else None
        # What a trace records for _finish_trace's checks.
        self._ws_role_events = []  # RoleEvents of every dispatched role
        self._ws_epoch_seqs = {}  # (owned-epoch channel, role) -> RoleTrace.epoch_sequence
        self._ws_raw_barrier_access = ""

    @staticmethod
    def _bind_stream(
        name: str, st: WorkStream, roles: list, chans: dict, cluster, fetch_keys: tuple[str, ...]
    ):
        """Check the stream's users. A broadcast schedule also gets the scheduler role, the work
        ring's producer; the scheduler and every stream user consume the ring."""
        users = [r for r in roles if any(a is WorkCursor for _, a in r.params)]
        st.roles = [r.name for r in sorted(users, key=lambda r: r.first_warp)]
        if st.schedule.persistent:
            idle = sorted({r.role for r in roles if not r.raw} - {r.role for r in users})
            if idle:
                raise ProtocolError(
                    f"{st.schedule.describe()} runs several work items per CTA, but roles {idle} "
                    f"do not take `{name}: WorkCursor`; every role must loop over the stream "
                    "(a role with no per-item work drains it: "
                    f"`while {name}.valid: {name}.advance()`)"
                )
        if not st.schedule.broadcast:
            return
        owned = {w for r in roles for w in r.warps}
        warp = st.scheduler_warp
        if warp is None:
            warp = next(w for w in range(max(owned, default=-1) + 2) if w not in owned)
        elif warp in owned:
            raise ValueError(f"WorkStream {name}: scheduler_warp {warp} is claimed by a role")
        group = range(warp // WARPGROUP * WARPGROUP, (warp // WARPGROUP + 1) * WARPGROUP)
        # The scheduler shares its warpgroup's budget (setmaxnreg is per warpgroup), including a
        # deliberate None; only an otherwise idle warpgroup takes the smallest declared budget.
        budgets = {r.regs for r in roles if set(r.warps) & set(group)}
        any_regs = [r.regs for r in roles if r.regs is not None]
        if budgets:
            regs = budgets.pop() if len(budgets) == 1 else min(b for b in budgets if b is not None)
        else:
            regs = min(any_regs) if any_regs else None
        ring = st._channel_name(name, "ring")
        fetch = [st._channel_name(name, k) for k in fetch_keys]
        params = [(ring, Producer), (name, _SchedulerCursor), (name, _SchedulerFetch)]
        # In a cluster only rank 0 claims, and it publishes each item into every CTA. When rank
        # 0 is the only leader (a CTA pair), the role keeps the leader placement.
        placement = ALL
        if cluster.size > 1:
            placement = LEADER if cluster.ctas(LEADER) == (0,) else RANK0
        sched = RoleInstance(
            f"{name}.scheduler",
            f"{name}.scheduler",
            0,
            warp,
            1,
            regs,
            scheduler_role,
            params,
            cta=placement,
        )
        roles.append(sched)
        st.roles.append(sched.name)
        chans[ring].producers.append((sched, None))
        for r in [*users, sched]:
            chans[ring].consumers.append((r, None))
        for f in fetch:
            chans[f].producers.append((sched, None))
            chans[f].consumers.append((sched, None))

    @staticmethod
    def _bind(ri: RoleInstance, chans, turns):
        for pname, ann in ri.params:
            side = ann.side if isinstance(ann, _Bound) else ann
            if side is Producer or side is Consumer:
                if pname not in chans:
                    raise ProtocolError(f"role {ri.name}: no channel named {pname!r}")
                chan = chans[pname]
                slot = None
                if isinstance(ann, _Bound):
                    if not chan.fixed:
                        raise ProtocolError(f"role {ri.name}: ring {pname} cannot be slot-bound")
                    slot = ri.index
                    if slot >= chan.n:
                        raise ProtocolError(f"role {ri.name}: slot {slot} >= {pname}.n={chan.n}")
                if side is Producer:
                    declared = chan.producer_role
                    if declared is not None and declared != ri.role:
                        raise ProtocolError(
                            f"channel {pname}: producer is declared as role {declared!r}, but "
                            f"role {ri.name} takes its Producer end"
                        )
                    chan.producers.append((ri, slot))
                else:
                    declared = chan.consumer_roles
                    if declared is not None and ri.role not in declared:
                        raise ProtocolError(
                            f"channel {pname}: consumers are declared as {declared}, but role "
                            f"{ri.name} takes its Consumer end"
                        )
                    chan.consumers.append((ri, slot))
            elif ann is Turn:
                if pname not in turns:
                    raise ProtocolError(f"role {ri.name}: no turnstile named {pname!r}")
                turns[pname].parties.append(ri)

    def _validate_placement(self, chans, turns, roles):
        """Cluster checks on role placement, then derive every channel's scope."""
        cluster = self.cluster
        leaders = set(cluster.ctas(LEADER))
        if cluster.size > 1:
            for name, ch in chans.items():
                for side in SIDES:
                    for ri, _ in ch.bound(side):
                        commits = ch.mechanism(side, ri.role) is Umma
                        if commits and cluster.pair and not set(cluster.ctas(ri.cta)) <= leaders:
                            raise ProtocolError(
                                f"role {ri.name} signals channel {name} ({side} side) with "
                                f"tcgen05.commit, so it issues the MMAs; in a "
                                f"{type(cluster).__name__} kernel CtaGroup.TWO MMAs are issued "
                                "by the leader CTA only: pin it with @role(..., cta='leader')"
                            )
            for name, ts in turns.items():
                if len({cluster.ctas(r.cta) for r in ts.parties}) != 1:
                    raise ProtocolError(f"turnstile {name}: seats run in different CTAs")
            tmem_roles = [r for r in roles if r.uses_tmem]
            everywhere = cluster.ctas(ALL)
            if tmem_roles and not any(cluster.ctas(r.cta) == everywhere for r in tmem_roles):
                raise ProtocolError(
                    "TMEM is allocated by one warp in each CTA of the cluster (2-CTA TMEM: of "
                    f"each pair), but every Tmem role {[r.name for r in tmem_roles]} is "
                    "leader-only"
                )
        for name, ch in chans.items():
            ch.cluster = cluster
            ch.derived_scope = derive_scope(ch, cluster)
            ch.scope = apply_overrides(ch, ch.derived_scope, cluster)
            if ch.scope.pair and ch.external:
                managed = "consumer" if ch.external == "producer" else "producer"
                raise ProtocolError(
                    f"channel {name}: its {managed} roles run in different CTAs, but its "
                    f"{ch.external} side is external, driven by raw code in each CTA; give the "
                    f"{managed} roles one placement"
                )
            if ch.scope.pair and (ch.named or ch.rotates or ch.exclusive):
                what = (
                    "named barriers are per CTA"
                    if ch.named
                    else "round-robin sides are CTA-local"
                    if ch.rotates
                    else "exclusive consumers are CTA-local"
                )
                raise ProtocolError(
                    f"channel {name}: it crosses CTAs (its roles run in different CTAs, or its "
                    f"copies are multicast), but {what}; give both sides the same placement "
                    "without multicast"
                )

    def _validate(self, chans, turns, roles):
        if not roles:
            raise ValueError("warp-specialized kernels need at least one role")
        owner = {}
        for r in roles:
            for w in r.warps:
                if w in owner:
                    raise ValueError(f"warp {w} claimed by {owner[w]} and {r.name}")
                owner[w] = r.name
        for name, chan in chans.items():
            if chan.per_role:
                unbound = [k for k in chan.consumer if all(r.role != k for r, _ in chan.consumers)]
                if unbound:
                    raise ProtocolError(
                        f"channel {name}: per-role consumer(s) {unbound} take no Consumer end; "
                        "every key of the consumer dict must bind one"
                    )
            for side in SIDES:
                bound = chan.bound(side)
                if side == chan.external:
                    if bound:
                        raise ProtocolError(
                            f"channel {name}: its {side} side is external (driven by raw code), "
                            f"but roles {[r.name for r, _ in bound]} take a {side.title()} end"
                        )
                    continue
                counts = chan.slot_counts(side)
                if 0 in counts:
                    raise ProtocolError(
                        f"channel {name}: slot {counts.index(0)} has no {side} role"
                    )
                if chan.sharing(side) is Sharing.ROUND_ROBIN:
                    if len(bound) < 2:
                        raise ProtocolError(
                            f"channel {name}: {side}_sharing=Sharing.ROUND_ROBIN needs at least "
                            f"two instances of one role on the {side} side, got "
                            f"{[r.name for r, _ in bound]}"
                        )
                    _check_turns(name, side, bound)
                # A per-role side's roles signal independently, each with its own mechanism.
                if side == "consumer" and chan.per_role:
                    groups = [[b for b in bound if b[0].role == k] for k in chan.consumer]
                else:
                    groups = [bound]
                for group in groups:
                    _check_signal_once(name, side, chan, group)
            if chan.exclusive:
                _check_exclusive(name, chan)
            if chan.named:
                _check_named_turns(name, chan, turns)
            else:
                for side in SIDES:
                    k = chan.turns(side)
                    if k > 1 and chan.n % k:
                        raise ProtocolError(
                            f"channel {name}: {k} round-robin {side}s need a multiple of {k} "
                            f"stages, got {chan.n}: an instance's parity wait is only exact when "
                            "it saw the stage's previous epoch itself"
                        )
        for name, ts in turns.items():
            if len(ts.parties) < 2 or len({r.role for r in ts.parties}) != 1:
                raise ProtocolError(f"turnstile {name} needs >= 2 instances of one role")
            ts.parties.sort(key=lambda r: r.index)
        if self.tmem_plan is not None:
            _ = self.tmem_plan.alloc_cols  # raises when over capacity
        elif any(r.uses_tmem for r in roles):
            raise ProtocolError("roles take Tmem but the kernel has no tmem_plan")

    def _plan_registers(self, roles):
        """Size the launch and give every warpgroup its ``setmaxnreg`` budget.

        Sets ``_ws_num_warps``, ``_ws_default_regs``, ``_ws_padding`` and
        ``_ws_budget_groups``; rejects budgets the launch cannot supply.
        """
        owner = {w: r for r in roles for w in r.warps}
        num_warps = max(owner) + 1
        if min(owner) < 0 or num_warps > MAX_WARPS:
            raise ValueError(f"role warps must lie in [0, {MAX_WARPS})")
        uses_regs = any(r.regs is not None for r in roles)
        # setmaxnreg is .aligned per warpgroup: pad to whole warpgroups and require every warp
        # of a warpgroup to request the same budget.
        self._ws_num_warps = -(-num_warps // WARPGROUP) * WARPGROUP if uses_regs else num_warps
        self._ws_default_regs = min(REGS_PER_SM // (self._ws_num_warps * WARP_SIZE) // 8 * 8, 256)
        # (warps, regs) for each run of warps no role owns, holes included: they take their
        # warpgroup's budget (MIN_REGS for an all-idle warpgroup).
        self._ws_padding = []
        # One setmaxnreg budget per warpgroup, set before dispatch; None when no role sets regs.
        wg_regs = [None] * -(-self._ws_num_warps // WARPGROUP)
        if uses_regs:
            for g in range(len(wg_regs)):
                group = range(g * WARPGROUP, (g + 1) * WARPGROUP)
                budgets = {owner[w].name: owner[w].regs for w in group if w in owner}
                vals = set(budgets.values())
                if len(vals) > 1:
                    raise ValueError(f"warpgroup {g} has mixed register budgets {budgets}")
                regs = vals.pop() if vals else MIN_REGS
                wg_regs[g] = regs
                idle = [w for w in group if w not in owner]
                self._ws_padding += [(run, regs) for run in _runs(idle)]
            total = sum(
                (regs if regs is not None else self._ws_default_regs) * len(warps) * WARP_SIZE
                for warps, regs in [*((r.warps, r.regs) for r in roles), *self._ws_padding]
            )
            # Every warp starts with default_regs, and setmaxnreg.inc only takes registers other
            # warps released. The budgets therefore share default_regs per launched thread,
            # which is below REGS_PER_SM when 65536 / threads is not a multiple of 8 (12, 20,
            # 24, 28 warps). A larger total compiles, but its setmaxnreg.inc waits forever.
            pool = self._ws_default_regs * self._ws_num_warps * WARP_SIZE
            if total > pool:
                raise ValueError(
                    f"register budgets need {total} registers, but {self._ws_num_warps} warps start "
                    f"with {self._ws_default_regs} each ({pool} in all; setmaxnreg can only move them)"
                )
        # One dispatch branch per run of adjacent warpgroups with one budget. Non-adjacent
        # warpgroups never share a branch even when their budgets agree: ptxas allocates a merged
        # branch jointly and can spill one role under the other's pressure
        # (docs/warpdrive/configuration.md, "Placement and registers").
        # (regs, warpgroups, dispatch units): a unit is the indices of the role instances one
        # traced body runs, one instance or every instance of a shared-body role in declared
        # (Instance) order.
        self._ws_budget_groups = []
        for regs, run in itertools.groupby(range(len(wg_regs)), key=lambda g: wg_regs[g]):
            wgs = tuple(run)
            units = {}
            for i, r in enumerate(roles):
                if r.first_warp // WARPGROUP in wgs:
                    units.setdefault(r.role if r.shared else i, []).append(i)
            units = tuple(tuple(sorted(u, key=lambda i: roles[i].index)) for u in units.values())
            self._ws_budget_groups.append((regs, wgs, units))
        for role_name in {r.role for r in roles if r.shared}:
            split = [
                wgs
                for _, wgs, units in self._ws_budget_groups
                if any(roles[u[0]].role == role_name for u in units)
            ]
            if len(split) > 1:
                raise ValueError(
                    f"role {role_name}: shared_body instances fall in different register-budget "
                    f"dispatch branches (warpgroups {split}), which cannot share one body; place "
                    "them in adjacent warpgroups with one budget"
                )

    def role_config_key(self) -> tuple:
        """Resolved role placement and budgets, part of ``__cute_cache_key__``. The first call
        collects and freezes the role configuration."""
        self._collect()
        return tuple(
            (r.role, r.index, r.first_warp, r.num_warps, r.regs, r.cta, r.raw, r.shared)
            for r in self._ws_roles
        )

    def __cute_cache_key__(self) -> tuple:
        """This kernel's key as a ``jit_cache`` compile argument (the examples' ``compile_*``).

        ``jit_cache`` adds the class's module and name. The key holds:

        * the constructor call: bound arguments with defaults filled in, copied before
          ``__init__`` runs, so a later in-place change to a list argument does not change it;
        * what may change after construction: the cluster, each work stream, and the host
          settings (trace backend and options, ``trace_protocol``, ``alias_check``; a "warn"
          build must not satisfy a "strict" one);
        * the resolved role placement and budgets;
        * the source of every module defining the kernel's, cluster's and streams' classes, so
          editing a kernel the compile function does not import (a test's subclass)
          recompiles it.

        Raises ``TypeError`` for any other public attribute assigned after construction: set it
        in ``__init__``. In-place changes to objects the constructor built (``op.ab.n = 4``) are
        not keyed; don't make them.
        """
        late = vars(self).get("_ws_late")
        if late:
            raise TypeError(
                f"{type(self).__name__}: {sorted(late)} assigned after construction, but a "
                "cached compile keys the constructor call. Pass the value to the constructor or "
                "set it in __init__ (a subclass works); only cluster, work streams, alias_check, "
                "liveness_check, trace_protocol and the trace settings may change after "
                "construction"
            )
        if isinstance(self._ws_init_key, TypeError):
            raise self._ws_init_key
        streams = [v for v in vars(self).values() if isinstance(v, WorkStream)]
        parts = [self, self.cluster]
        parts += [part for s in streams for part in (s.schedule, s.mapping, s.ready)]
        modules = sorted({c.__module__ for part in parts for c in type(part).__mro__})
        return (
            self._ws_init_key,
            self.cluster,
            tuple((s.schedule, s.mapping, s.ready, s.scheduler_warp) for s in streams),
            (self.alias_check, self.liveness_check, self.trace_protocol, self._trace_settings()),
            self.role_config_key(),
            tuple((m, module_source_digest(m)) for m in modules),
        )

    def __cute_trace_record__(self) -> dict:
        """What a trace left for ``describe()``, pipeline plans and the trace-time warnings, as
        plain data; None when this instance was not traced. ``jit_cache`` stores it beside the
        compiled kernel and passes it to ``__cute_restore_trace__`` on every call it serves from
        memory or disk."""
        if not self._ws_traced:
            return None
        record = {name: getattr(self, name) for name in _TRACE_RESULTS}
        record["_ws_smem"] = {
            name: replace(res, buffer=None, tensor=None) for name, res in self._ws_smem.items()
        }
        return record

    def __cute_restore_trace__(self, record: dict) -> None:
        """Load a cached build's trace record, so ``describe()`` and the warnings match a cold
        compile. An instance that traced, or already took a record, keeps its own. Restored
        SMEM entries have no tensors: ``buffer()`` works only while tracing."""
        self._collect()
        if record is None or self._ws_traced:
            return
        record = pickle.loads(pickle.dumps(record))  # this instance's own copies
        buffers = self.smem_plan.buffers if self.smem_plan is not None else {}
        record["_ws_smem"] = {
            name: replace(res, buffer=buffers[name]) for name, res in record["_ws_smem"].items()
        }
        for name in _TRACE_RESULTS:
            setattr(self, name, record[name])
        self._ws_traced = True
        self._report_trace()

    def enable_trace(
        self, backend: str | None, *, waits: bool = True, events_per_role: int = 1024
    ):
        """Emit ranges named after roles and channels in the next compile.

        ``backend``: ``"iket"`` records IKET ranges for ``run-iket``; ``"globaltimer"`` writes
        profiler records into the buffer passed to ``run_roles(..., trace_buffer=...)``,
        decoded by ``trace_session.trace_session``; ``None`` emits no ranges and no IR.
        ``waits=False`` keeps only the role, work-item and ``trace_region`` ranges: wait ranges
        are costly, role ranges nearly free.
        ``events_per_role``: globaltimer records per role instance and CTA; later ranges are
        dropped.

        ``__cute_cache_key__`` keys the settings that change code; ``trace_key()`` names the
        backend for kernel names.
        """
        if backend is not None and backend not in tracing.BACKENDS:
            raise ValueError(
                f"trace backend must be one of {tracing.BACKENDS} or None, got {backend!r}"
            )
        if type(events_per_role) is not int or events_per_role < 1:
            raise ValueError(f"events_per_role must be a positive int, got {events_per_role!r}")
        self.trace_backend, self.trace_waits = backend, waits
        self.trace_events_per_role = events_per_role
        return self

    def _trace_settings(self) -> tuple:
        """The trace settings that change the generated code (the compile key's part)."""
        if self.trace_backend is None:
            return (None,)
        if self.trace_backend == "iket":
            return ("iket", self.trace_waits)
        return (self.trace_backend, self.trace_waits, self.trace_events_per_role)

    def enable_iket(self, enabled: bool = True, *, waits: bool = True):
        """``enable_trace("iket", waits=waits)``, or ``enable_trace(None)`` when not enabled."""
        return self.enable_trace("iket" if enabled else None, waits=waits)

    def trace_key(self) -> str:
        """Kernel-name suffix: ``_iket``, ``_iketroles``, ``_gtimer`` or ``_gtimerroles`` (empty
        when tracing is off), then ``_trace`` with ``trace_protocol``. A cache hit does not
        retrace the body, so examples use ``jit_cache(uncached=...)`` to get protocol logs; see
        ``docs/warpdrive/tracing.md`` ("Trace the protocol")."""
        backend = ""
        if self.trace_backend is not None:
            backend = {"iket": "_iket", "globaltimer": "_gtimer"}[self.trace_backend]
            backend += "" if self.trace_waits else "roles"
        return backend + ("_trace" if self.trace_protocol else "")

    @contextmanager
    def trace_region(self, name: str, payload=None):
        """``with self.trace_region("exp", payload=row):`` inside a role: a named range when a
        trace backend is on, nothing otherwise.

        ``payload``: an optional runtime ``Int32`` recorded with the range, such as its work
        item (globaltimer: ``Range.payload``; IKET: ``payloadVal`` in the capture). Globaltimer
        records per role, so its ranges must be inside a role; IKET ranges may be anywhere.
        """
        if self.trace_backend is None:
            yield
            return
        trace = current_trace()
        if trace is not None:
            ranges = trace.ranges
        elif self.trace_backend == "iket":
            ranges = IketRanges()
        else:
            raise ProtocolError(
                f"trace_region({name!r}) outside a role: globaltimer ranges are recorded per role"
            )
        ranges.push(name, payload)
        yield
        ranges.pop(name, payload)

    @property
    def num_warps(self) -> int:
        """Warps launched: the roles' warps, padded to whole warpgroups when a role sets regs."""
        self._collect()
        return self._ws_num_warps

    @property
    def num_threads(self) -> int:
        return self.num_warps * WARP_SIZE

    def describe(self, format: str = "text") -> str:
        """Plain-text summary of what the layer derived (``describe.py``): roles and register
        budgets, channels (per-CTA arrival counts, TMA bytes, masks, overrides, TMA check
        status), streams and TMEM; after a compile, also check opt-outs and the traced protocol.

        ``format="mermaid"`` or ``"dot"`` draws the declarations as a graph
        (``describe_graph.py``): roles as nodes, channels as producer -> consumer edges,
        barriers and storage aliases. ``format="html"`` renders that graph as a self-contained
        page with details on hover (``describe_html.py``)."""
        self._collect()
        if format == "text":
            return describe_kernel(self)
        # Host-only renderers. Importing them at call time by relative name keeps them out of
        # every kernel's compile key, which hashes the modules a kernel imports (AGENTS.md,
        # "Adding CuTeDSL Kernels").
        return importlib.import_module(".describe_graph", __package__).describe_graph(self, format)

    def checks(self) -> KernelChecks:
        """Check verdicts from ``describe()``, as ``describe.KernelChecks``.

        Applicable channel checks are ``"not checked yet"`` before compilation;
        ``jit_cache`` restores saved results.
        """
        self._collect()
        return kernel_checks(self)

    def _barrier_plan(self):
        """``(owner, offset, {cta rank: arrival counts})`` of every mbarrier in allocation
        order, and the total barrier count.

        Every CTA of a cluster allocates the same barriers at the same offsets, because remote
        arrivals address the peer's barrier by offset; counts may differ per rank.
        """
        plan, off = [], 0
        ranks = range(self.cluster.size)
        for name, ch in self._ws_chans.items():
            if ch.named:
                continue  # hardware named barriers: no SMEM, no initialization
            sc = ch.scope
            plan.append((name, off, {c: sc.full_counts[c] + sc.empty_counts[c] for c in ranks}))
            off += 2 * ch.n
        for name, ts in self._ws_turns.items():
            if ts.enabled:
                n = len(ts.parties)
                counts = [ts.parties[(p - 1) % n].num_threads for p in range(n)]
                plan.append((name, off, {c: counts for c in ranks}))
                off += n
        for name, b in self._ws_raw_bars.items():
            counts = b.counts or [None] * b.n  # None: initialized by user code
            plan.append((name, off, {c: counts for c in ranks}))
            off += b.n
        return plan, off

    @property
    def named_barrier_ids(self) -> dict:
        """``{channel: ([full id per slot], [empty id per slot])}`` of the named-barrier
        channels, the IDs behind their ``full_barriers``/``empty_barriers``.

        Channels take IDs from 15 down, the layer owns 1 and 2, and declared ``NamedBarrier``s
        take IDs from 3 up (each has its own ``.id``). Hand-picked IDs in user code must avoid
        all of them; ``alias.named_barrier`` calls are checked at trace time.
        """
        self._collect()
        return dict(self._ws_named)

    def barrier_ptr(self, name: str):
        """Pointer to the first mbarrier of a channel (full barriers at ``[0, n)``, empty at
        ``[n, 2n)``), an enabled turnstile, or a ``Barriers`` declaration. Valid after
        ``begin_roles()``; every CTA of a cluster has them at the same offsets."""
        if name not in self._ws_bars:
            raise ProtocolError(f"no barriers named {name!r} (have {sorted(self._ws_bars)})")
        self._ws_raw_barrier_access = f"{name} at {user_site()}"
        return self._ws_bars[name]

    def launch_config(self, work=None, grid=None, min_blocks_per_mp: int | None = None) -> dict:
        """``grid``/``block``/``cluster`` for ``kernel(...).launch(**cfg, stream=stream)``.
        A kernel with a ``WorkStream`` passes its ``BoundWork``, whose schedule sets the grid (a
        ``OneShot`` over plain ``Raster()`` launches the tile space).

        When a role declares ``regs``, the config also sets ``min_blocks_per_mp=1``: ptxas keeps
        ``setmaxnreg`` only with a ``.minnctapersm`` (or ``.maxnreg``) entry directive, and
        without it silently compiles every warp at one count, spilling a role that needs its
        larger budget. ``min_blocks_per_mp`` passes another value through; with budgets only 1
        fits, since they split the whole register file among one CTA's warps.
        """
        if work is not None:
            grid = work.grid()
        cfg = {
            "grid": grid,
            "block": (self.num_threads, 1, 1),
            "cluster": self.cluster.launch_cluster(),
        }
        budgets = any(r.regs is not None for r in self._ws_roles)
        if min_blocks_per_mp is not None:
            if type(min_blocks_per_mp) is not int or min_blocks_per_mp < 1:
                raise ValueError(
                    f"min_blocks_per_mp must be a positive int, got {min_blocks_per_mp!r}"
                )
            if budgets and min_blocks_per_mp != 1:
                raise ValueError(
                    f"{type(self).__name__}: role register budgets split the whole register "
                    f"file among one CTA, so min_blocks_per_mp must be 1, got {min_blocks_per_mp}"
                )
            cfg["min_blocks_per_mp"] = min_blocks_per_mp
        elif budgets:
            cfg["min_blocks_per_mp"] = 1
        return cfg

    # ---- device-side ------------------------------------------------------- #

    def begin_roles(self, layouts: dict | None = None, **problems):
        """Allocate and initialize barriers and payload buffers, allocate TMEM, then allocate
        the ``smem_plan`` buffers.

        ``problems``: the ``BoundWork`` of each declared ``WorkStream``, by name
        (``self.begin_roles(work=bound)``). ``layouts``: the layout, plain or swizzled, of every
        ``smem_plan`` buffer, by name. Plan buffers are allocated last, in declaration order, so
        later ``smem`` allocations follow them; ``self.buffer(name)`` returns each.

        Plain Python at trace time; only barrier initialization is a device-side branch.
        """
        self._collect()
        self._reset_trace()
        # Trace state outside this kernel: process-wide recorders and flags on its declarations.
        # Collection may run inside another kernel's trace, so _reset_trace leaves these alone.
        take_opt_outs()  # drop opt-outs left over from an earlier, failed trace
        for nb in self._ws_named_bars.values():  # count only this trace's native ID reads
            nb._native_outside_roles = False
        clear_multicast_masks()
        if set(problems) != set(self._ws_streams):
            raise ProtocolError(
                f"begin_roles needs a problem for each WorkStream {sorted(self._ws_streams)}, "
                f"got {sorted(problems)}"
            )
        self._ws_problems = problems
        for name, problem in problems.items():
            self._ws_streams[name]._bound = problem
            problem.start()  # per-CTA start values, computed once ahead of role dispatch
        smem = SmemAllocator()
        plan, total = self._barrier_plan()
        bars = smem.allocate_array(cutlass.Int64, max(total, 1))
        self._ws_bars = {name: bars + off for name, off, _ in plan}
        self._init_barriers(bars, self._barrier_init_runs(plan))
        self._ws_trace_starts = None
        if self.trace_backend == "globaltimer":  # start times of open ranges, per role
            depth = tracing.MAX_DEPTH
            starts = smem.allocate_array(cutlass.Int64, len(self._ws_roles) * depth)
            layout = cute.make_layout((len(self._ws_roles), depth), stride=(depth, 1))
            self._ws_trace_starts = cute.make_tensor(starts, layout)
        self._ws_buffers = {}  # channel name -> {buffer name: StageBuffer}
        for name, ch in self._ws_chans.items():
            if ch.payload is not None:
                dtype, shape = ch.payload
                payload = smem.allocate_tensor(dtype, cute.make_layout((*shape, ch.n)), 16)
                self._ws_buffers[name] = {"payload": StageBuffer(payload, -1, None)}
        self._ws_tmem = None
        clustered, two_cta = self.cluster.size > 1, self.cluster.pair
        tmem_roles = [r for r in self._ws_roles if r.uses_tmem]
        if tmem_roles:
            alloc_role, alloc_warp = self._tmem_allocator()
            holder = smem.allocate_array(cutlass.Int32, 1)
            dealloc_bar = smem.allocate_array(cutlass.Int64, 1) if two_cta else None
            tmem = TmemAllocator(
                holder,
                barrier_for_retrieve=pipeline.NamedBarrier(
                    barrier_id=TMEM_PTR_BARRIER_ID, num_threads=self.num_threads
                ),
                allocator_warp_id=alloc_warp,
                is_two_cta=two_cta,
                two_cta_tmem_dealloc_mbar_ptr=dealloc_bar,
            )
            self._ws_tmem = tmem
            self._ws_tmem_alloc_role = alloc_role
            self._ws_tmem_alloc_warp = alloc_warp
            self._ws_tmem_dealloc_bar = dealloc_bar
            if not clustered:  # a cluster allocates after the cluster sync in run_roles
                # Only the allocator warp executes tcgen05.alloc (checked inside).
                tmem.allocate(self.tmem_plan.alloc_cols)
                tmem.relinquish_alloc_permit()
        self._allocate_plan(smem, layouts or {})
        return smem

    def _tmem_allocator(self) -> tuple[RoleInstance, int]:
        """The TMEM role, and the warp in it that allocates, relinquishes and frees TMEM.

        Every CTA of a cluster allocates (a pair's CTAs together, with
        ``tcgen05.alloc.cta_group::2``), so the role must run in every CTA. A single CTA
        allocates before the CTA sync, while warp 0 initializes the barriers, so it picks the
        role's lowest warp other than 0 (FA4 allocates from warp 1) and falls back to warp 0. A
        cluster allocates after the cluster sync, where any warp works, so it takes the first.
        """
        everywhere = self.cluster.ctas(ALL)
        roles = [
            r for r in self._ws_roles if r.uses_tmem and self.cluster.ctas(r.cta) == everywhere
        ]
        if self.cluster.size == 1:
            others = [(r, w) for r in roles for w in r.warps if w != 0]
            if others:
                return min(others, key=lambda rw: rw[1])
        return roles[0], roles[0].first_warp

    def _allocate_plan(self, smem, layouts: dict):
        """Allocate every ``smem_plan`` buffer and check each channel binding's stage count."""
        buffers = {} if self.smem_plan is None else self.smem_plan.buffers
        if set(layouts) != set(buffers):
            raise ProtocolError(
                f"begin_roles(layouts=...) needs a layout for each smem_plan buffer "
                f"{sorted(buffers)}, got {sorted(layouts)}"
            )
        for buf in buffers.values():
            self._ws_smem[buf.name] = allocate(smem, buf, layouts[buf.name], self._ws_smem)
        for name, ch in self._ws_chans.items():
            if not ch.buffers:
                continue
            bound = {}
            for key, buf in ch.buffers.items():
                res = self._ws_smem[buf.name]
                res.channels.append(name)
                what = f"channel {name} buffer {key!r}"
                extent = stage_extent(res.tensor, buf.stage_axis, what)
                if extent != ch.n:
                    raise ProtocolError(
                        f"{what}: stage mode {buf.stage_axis} of {buf.name}'s layout has "
                        f"{extent} stages, but {name} has {ch.n}"
                    )
                bound[key] = StageBuffer(res.tensor, buf.stage_axis, buf.name)
            self._ws_buffers[name] = bound

    def buffer(self, name: str):
        """The whole tensor of ``smem_plan`` buffer ``name`` (valid after ``begin_roles``)."""
        if name not in self._ws_smem:
            raise ProtocolError(
                f"no allocated SMEM buffer named {name!r} (have {sorted(self._ws_smem)}); "
                "declare it in smem_plan and call begin_roles(layouts=...) first"
            )
        return self._ws_smem[name].tensor

    def _barrier_init_runs(self, plan: list) -> list[tuple[int, int, dict[int, int]]]:
        """Runs of contiguous barriers from ``_barrier_plan`` with equal per-rank counts, at most
        ``WARP_SIZE`` long: one barrier per warp-0 lane."""
        per_rank = [
            [c for _, _, counts in plan for c in counts[rank]] for rank in range(self.cluster.size)
        ]
        runs = []
        offset = 0
        while offset < len(per_rank[0]):
            counts = tuple(counts[offset] for counts in per_rank)
            end = offset + 1
            while (
                end < len(per_rank[0])
                and end - offset < WARP_SIZE
                and tuple(counts[end] for counts in per_rank) == counts
            ):
                end += 1
            if counts[0] is not None:  # user-initialized Barriers stay untouched
                runs.append((offset, end - offset, dict(enumerate(counts))))
            offset = end
        return runs

    @cute.jit
    def _init_barriers(self, bars, runs):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            rank = self.cluster.rank()
            lane = cute.arch.lane_idx()
            for i in cutlass.range_constexpr(len(runs)):
                offset, count, arrivals = runs[i]
                # One lane initializes each barrier. The converged init fence and CTA/cluster
                # sync in run_roles publish every lane's initialization.
                if lane < count:
                    cute.arch.mbarrier_init(bars + offset + lane, rank_value(arrivals, rank))

    @cute.jit
    def run_roles(self, params, trace_buffer=None):
        """Dispatch the roles with the kernel's params, then tear down. ``trace_buffer``: the
        int64 profile buffer of a ``"globaltimer"`` trace (any shape; allocated by
        ``trace_session.trace_session``); ignored by other backends."""
        params = self._role_context(params)
        self._ws_trace = self._start_trace(trace_buffer)
        clustered = self.cluster.size > 1
        if cutlass.const_expr(not clustered):
            # Fences mbarrier.init and syncs the CTA; the same bar.sync publishes the TMEM base
            # address written by tcgen05.alloc in begin_roles().
            pipeline.pipeline_init_arrive()
            pipeline.pipeline_init_wait()
        else:
            # Every barrier of the cluster is initialized before any CTA arrives remotely.
            pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster.shape, is_relaxed=True)
            pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster.shape)
            if cutlass.const_expr(self._ws_tmem is not None):
                self._ws_tmem.allocate(self.tmem_plan.alloc_cols)
                self._ws_tmem.relinquish_alloc_permit()
                cute.arch.sync_threads()  # publishes the TMEM base address
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = self.cluster.rank()
        # setmaxnreg is .sync.aligned over a warpgroup: every warp of a group must execute the
        # same instruction. Branch once per run of adjacent warpgroups with one budget (a
        # warpgroup-uniform condition), adjust there, then split into roles inside that branch
        # (as CUTLASS does), so ptxas still allocates each role's code under its budget.
        groups = self._ws_budget_groups
        for gi in cutlass.range_constexpr(len(groups)):
            if _in_warpgroups(warp_idx, groups[gi][1]):
                self._dispatch_group(groups[gi], params, warp_idx, rank)
        if cutlass.const_expr(clustered):
            # Remote arrivals into a CTA must land before it exits. Each thread-issued remote
            # arrive precedes that thread's cluster arrive (a release), and producers whose
            # stages another CTA's tcgen05.commit releases drained them above (tails).
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        self._finish_trace()

    def _start_trace(self, buffer):
        """Kernel-wide trace state that makes each role's ranges: ``tracing.GlobaltimerTrace``
        (device code computes this CTA's unit), ``IketTrace``, or None when tracing is off."""
        if self.trace_backend is None:
            return None
        if self.trace_backend == "iket":
            return IketTrace()
        if buffer is None:
            raise ProtocolError(
                "enable_trace('globaltimer') records into a profile buffer: pass "
                "run_roles(params, trace_buffer=...) (see trace_session.py)"
            )
        trace = tracing.GlobaltimerTrace(
            buffer, self._ws_trace_starts, len(self._ws_roles), self.trace_events_per_role
        )
        self._ws_trace_tags = trace.tags
        return trace

    def _role_context(self, ctx):
        """The context the roles receive: a frozen dataclass as is; any other object, such as a
        ``SimpleNamespace``, as a ``RoleContext`` copy of its attributes. Checked against the
        roles' annotated context type and the cluster's MMA/TMA atoms."""
        want = self._ws_ctx_type
        if want is not None and not isinstance(ctx, want):
            raise ProtocolError(
                f"run_roles got a {type(ctx).__name__} context, but {self._ws_ctx_site} is "
                f"annotated {want.__name__}; {_CTX_RULE}"
            )
        if not hasattr(ctx, "__dict__") and not is_dataclass(ctx):
            raise ProtocolError(
                f"context {type(ctx).__name__} has no attributes to share; pass a frozen "
                "dataclass (or a SimpleNamespace)"
            )
        if is_dataclass(ctx):
            if not type(ctx).__dataclass_params__.frozen:
                raise ProtocolError(
                    f"context {type(ctx).__name__} is a mutable dataclass; declare it "
                    "@dataclass(frozen=True): every role shares it, and a field rebound inside "
                    "one role's dynamic branch would leak that region's values into other roles"
                )
            check_atoms({f.name: getattr(ctx, f.name) for f in fields(ctx)}, self.cluster)
            return ctx
        ctx = ctx if isinstance(ctx, RoleContext) else RoleContext(**vars(ctx))
        check_atoms(vars(ctx), self.cluster)
        return ctx

    def _finish_trace(self):
        """Collect opt-outs and check reports, then report findings."""
        self._ws_opt_outs = take_opt_outs()
        self._ws_named_users = named.check(self, self._ws_role_events)
        self._ws_ownership = ownership.check(self, self._ws_epoch_seqs)
        if self.liveness_check not in liveness.MODES:
            raise ValueError(
                f"liveness_check must be one of {liveness.MODES}, got {self.liveness_check!r}"
            )
        if self.liveness_check != "off":
            self._ws_liveness = liveness.check(self, self._ws_role_events)
        if self.alias_check not in alias.MODES:
            raise ValueError(f"alias_check must be one of {alias.MODES}, got {self.alias_check!r}")
        if self.alias_check != "off":
            self._ws_alias = alias.check(self, self._ws_role_events)
            self._ws_fences = fences.check(self, self._ws_role_events)
        for name, ch in self._ws_chans.items():
            if ch.checks_tma:
                report = self._ws_tma.setdefault(name, TmaReport())
                if not (report.copies or report.opted_out or report.unverified):
                    report.unverified.append(
                        "no TMA copy completes on a barrier taken from the layer (raw pointer?)"
                    )
        self._ws_traced = True
        self._report_trace()

    def _report_trace(self):
        """Report findings under each check's mode, after tracing or cache restoration."""
        liveness.report(self._ws_liveness, self.liveness_check, type(self).__name__)
        if self._ws_alias is not None:
            alias.report(self._ws_alias, self.alias_check, type(self).__name__)
        if self._ws_fences is not None:
            fences.report(self._ws_fences, self.alias_check, type(self).__name__)
        unverified = [n for n, rep in self._ws_tma.items() if rep.unverified]
        if unverified:
            details = "; ".join(f"{n}: {self._ws_tma[n].status()}" for n in unverified)
            warnings.warn(
                f"warpdrive: TMA check could not verify channel(s) {unverified} (no error "
                f"raised; see describe()): {details}",
                stacklevel=2,
            )

    @cute.jit
    def _dispatch_group(self, group, ctx, warp_idx, rank):
        regs, _, units = group
        self._set_regs(regs)
        size = self.cluster.size
        for i in cutlass.range_constexpr(len(units)):
            unit = tuple(self._ws_roles[j] for j in units[i])
            r = unit[0]
            ctas = self.cluster.ctas(r.cta)
            here = _in_runs(warp_idx, _runs(sorted(w for ri in unit for w in ri.warps)))
            if cutlass.const_expr(len(ctas) < size):
                here = here & rank_in(rank, ctas, size)
            # Only a call on `self` inside the dynamic `if`: an assignment through `self`
            # would make the preprocessor carry `self` as an if-result.
            if here:
                self._dispatch(unit, ctx)

    def _dispatch(self, unit: tuple[RoleInstance, ...], ctx):
        """Trace one role body: one instance, or every instance of a shared-body role, which
        then sees its index (``instance``) at run time."""
        # Plain Python (no dynamic control flow): runs at trace time inside the role's `if`.
        r = unit[0]
        instance = _instance_index(unit) if len(unit) > 1 else None
        name = f"{r.role}[*]" if instance is not None else r.name
        ranges = None
        if self._ws_trace is not None:
            ranges = self._ws_trace.role(
                _by_instance(instance, [self._ws_roles.index(ri) for ri in unit]),
                _by_instance(instance, [ri.first_warp for ri in unit]),
            )
            if instance is not None:
                ranges.share(instance, {name: [ri.name for ri in unit]})
        log = None if r.raw else self._ws_log
        trace = RoleTrace(
            name, log, ranges, self.trace_waits, instance=instance, instances=len(unit)
        )
        trace.tcgen05 = self._issues_tcgen05(r)
        if trace.tcgen05:
            # Orders the role's first tcgen05 operation after the CTA sync that published the
            # TMEM base address and barrier initialization (PTX ISA 9.7.18.6.3).
            tcgen05_after_thread_sync()
        args = self._role_args(unit, instance, ctx, trace)
        # A role looping over a WorkStream gets one range per work item (WorkCursor); any
        # other role gets one over its whole body.
        whole_range = ranges is not None and not any(isinstance(arg, WorkCursor) for arg in args)
        if whole_range:
            ranges.push(name)
        positional, keywords = [], {}
        for (pname, _), arg in zip(r.params, args):
            if pname in r.keyword_only:
                keywords[pname] = arg
            else:
                positional.append(arg)
        with tracing_role(trace):
            result = r.fn(self, *positional, **keywords)
        if not r.raw:
            state_args = [
                (pname, arg)
                for (pname, _), arg in zip(r.params, args)
                if isinstance(arg, _ROLE_STATE_TYPES)
            ]
            final = _returned_state(r, result, state_args)
            for cur in final:
                if isinstance(cur, WorkCursor):
                    cur._exit_item()
            for end in final:
                if isinstance(end, ProducerEnd) and end.scope.tail:
                    end.tail()
        if whole_range:
            ranges.pop(name)
        if ranges is not None and ranges.names:
            raise ProtocolError(f"{name}: trace ranges {ranges.names} left open at role exit")
        self._record_role(unit, instance, trace)
        if r.uses_tmem:
            self._tmem_teardown(unit)

    def _record_role(self, unit: tuple[RoleInstance, ...], instance, trace: RoleTrace):
        """Check a traced role body and keep its results, under each instance it ran as, for
        ``_finish_trace``'s kernel-wide checks. Host only: it emits no device code."""
        r = unit[0]
        if not r.raw:
            check_tma(trace, self.cluster, self._ws_tma)
            trace.finish()
            if self._ws_log is not None:
                self._ws_log.add(trace)
        if instance is not None and trace.log is not None:  # each instance ran these ops
            for book in (trace.log.ops, trace.log.tmem):
                if trace.role_name in book:
                    shared = book.pop(trace.role_name)
                    book.update({ri.name: copy.copy(shared) for ri in unit})
        # One trace stands for every instance: they differ only in the instance index, which
        # the layer applies itself (round-robin offsets, warp indices); a body that branches on
        # it has dynamic regions, which the checks treat as they do any other.
        uses = trace.uses_per_item()
        for j, ri in enumerate(unit):
            for chan, n in uses.items():
                self._ws_uses[(ri.name, chan)] = n
            for chan_name, ch in self._ws_chans.items():
                if ch.owned_epochs and any(b is ri for b, _ in ch.producers + ch.consumers):
                    self._ws_epoch_seqs[(chan_name, ri.name)] = (
                        "raw role: its waits are not visible"
                        if r.raw
                        else trace.epoch_sequence(chan_name)
                    )
            # The role's regions are live now; the diagnostic keeps plain data only.
            events = RoleEvents.from_trace(trace, ri)
            if instance is not None:
                events.events = [_for_instance(ev, j) for ev in events.events]
            self._ws_role_events.append(events)

    def _issues_tcgen05(self, r: RoleInstance) -> bool:
        """Whether ``r`` issues tcgen05 operations: it takes ``tmem: Tmem`` (tcgen05.ld/st, or
        MMAs into TMEM) or signals a channel side as ``Umma``. Its managed synchronization then
        takes the tcgen05 thread-sync fences (``thread_sync.py``)."""
        if r.uses_tmem:
            return True
        for ch in self._ws_chans.values():
            for side, bound in (("producer", ch.producers), ("consumer", ch.consumers)):
                if any(ri is r for ri, _ in bound) and ch.mechanism(side, r.role) is Umma:
                    return True
        return False

    def _tmem_teardown(self, unit: tuple[RoleInstance, ...]):
        """At role exit, wait for the CTA's other TMEM users, then free from the allocator warp.

        Each user thread orders its tcgen05 operations before the rendezvous
        (``tcgen05.fence::before_thread_sync``), and the allocator warp orders the dealloc after
        it (``::after_thread_sync``), as PTX requires across threads (ISA 9.7.18.6.3)."""
        r = unit[0]
        # Every TMEM user of this CTA finishes before the allocator warp frees it (a pair's
        # peer has no leader-only TMEM roles, so its count is smaller).
        users = [u for u in self._ws_roles if u.uses_tmem]
        threads = {
            c: sum(u.num_threads for u in users if c in self.cluster.ctas(u.cta))
            for c in self.cluster.ctas(r.cta)
        }
        count = rank_value(threads, self.cluster.rank)
        # Roles reach this from different code sites after per-lane mbarrier spins, so use
        # the unaligned form (compute-sanitizer synccheck flags the aligned bar.sync).
        tcgen05_before_thread_sync()
        _named_sync(TMEM_TEARDOWN_BARRIER_ID, count)
        if self._ws_tmem_alloc_role in unit:
            self._free_tmem(self._ws_tmem.retrieve_ptr(cutlass.Float32))

    @cute.jit
    def _free_tmem(self, ptr):
        """``tcgen05.dealloc`` from the allocator warp (a shared body may reach this from other
        warps), fenced after the last synchronization: the teardown rendezvous, or in a pair the
        allocator warps' dealloc mbarrier, which the peer reaches after its own rendezvous."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == self._ws_tmem_alloc_warp:
            if cutlass.const_expr(self.cluster.pair):
                bar = self._ws_tmem_dealloc_bar
                peer = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) ^ 1
                cute.arch.mbarrier_arrive(bar, peer)
                cute.arch.mbarrier_wait(bar, 0)
            tcgen05_after_thread_sync()
            cute.arch.dealloc_tmem(ptr, self.tmem_plan.alloc_cols, is_two_cta=self.cluster.pair)

    def _set_regs(self, regs):
        if regs is None:
            return
        if regs > self._ws_default_regs:
            cute.arch.setmaxregister_increase(regs)
        elif regs < self._ws_default_regs:
            cute.arch.setmaxregister_decrease(regs)

    def _role_args(self, unit: tuple[RoleInstance, ...], instance, ctx, trace: RoleTrace):
        r = unit[0]
        args = []
        for pname, ann in r.params:
            side = ann.side if isinstance(ann, _Bound) else ann
            if side is Producer or side is Consumer:
                bound = r.index if isinstance(ann, _Bound) else None
                args.append(self._make_end(pname, side is Producer, bound, trace, unit, instance))
            elif ann is Tmem:
                base = self._ws_tmem.retrieve_ptr(cutlass.Float32)
                args.append(TmemView(base, self.tmem_plan, trace.role_name, trace.log, trace))
            elif ann is Barriers:
                args.append(self._ws_bars[pname])
            elif ann is WorkCursor or ann is _SchedulerCursor:
                st = self._ws_streams[pname]
                end = None
                if st.schedule.broadcast:
                    end = self._make_end(
                        st._channel_name(pname, "ring"), False, None, trace, unit, instance
                    )
                args.append(
                    WorkCursor.create(
                        st,
                        self._ws_problems[pname],
                        end,
                        trace,
                        scheduler=ann is _SchedulerCursor,
                    )
                )
            elif ann is _SchedulerFetch:
                st = self._ws_streams[pname]
                ends = {
                    k: (
                        self._make_end(
                            st._channel_name(pname, k), True, None, trace, unit, instance
                        ),
                        self._make_end(
                            st._channel_name(pname, k), False, None, trace, unit, instance
                        ),
                    )
                    for k in self._ws_fetch_keys[pname]
                }
                args.append(_Fetch(st.schedule, ends, self._ws_problems[pname]))
            elif ann is Instance:
                args.append(r.index if instance is None else instance)
            elif ann is Local:
                first = _by_instance(instance, [ri.first_warp for ri in unit])
                args.append(Local(trace.role_name, first, r.num_warps))
            elif ann is Turn:
                ts = self._ws_turns[pname]
                seat = ts.parties.index(r)
                args.append(
                    TurnEnd(ts, self._ws_bars.get(pname), seat, trace, Int32(int(seat == 0)))
                )
            else:  # _Context
                args.append(ctx)
        return args

    def _make_end(self, pname: str, is_prod: bool, bound, trace, unit, instance):
        """The end of channel ``pname`` for the instances ``unit`` (several in a shared body,
        where ``instance`` is the runtime index)."""
        ri = unit[0]
        chan = self._ws_chans[pname]
        side = "producer" if is_prod else "consumer"
        # Round-robin: instance j of k starts at epoch j and steps k epochs per operation.
        turns = chan.turns(side)
        turn = 0
        if turns > 1:
            bound_roles = [r for r, _ in chan.bound(side)]
            turn = _by_instance(instance, [bound_roles.index(u) for u in unit])
        if chan.fixed:
            # Producers start each slot at phase 1 (a fresh empty barrier passes).
            state = Int32((1 << chan.n) - 1 if is_prod else 0)
        else:
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer
                if is_prod
                else pipeline.PipelineUserType.Consumer,
                chan.n,
            )
            if not isinstance(turn, int) or turn:
                advance_state(state, turn)
        cls = ProducerEnd if is_prod else ConsumerEnd
        end = cls(
            chan,
            self._ws_bars.get(pname),
            self._ws_buffers.get(pname, {}),
            trace,
            bound,
            state,
            ri.cta,
            stride=turns,
            role=ri.role,
        )
        if chan.named and not is_prod:
            if bound is not None:
                end.seed([bound])
            else:  # seed only the stages this instance takes first (its first turns)
                end.seed(range(chan.n), when=lambda j: turn == j % turns)
        return end


class RoleContext(SimpleNamespace):
    """A namespace context as the roles see it, opaque to CuTeDSL control flow.

    The preprocessor treats ``ctx.x[i] = v`` inside a dynamic ``if``/``for`` as a write of
    ``ctx``, and would carry a plain namespace through the region, rebinding its attributes to
    region results that do not dominate other roles. Context values are defined before role
    dispatch and never rebound, so this class carries nothing and rebuilds as itself. A frozen
    dataclass needs no wrapper: its fields cannot be rebound, and the preprocessor does not
    carry it.
    """

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


_ROLE_STATE_TYPES = (ProducerEnd, ConsumerEnd, WorkCursor, TurnEnd)


def _state_identity(state):
    if isinstance(state, TurnEnd):
        return id(state._trace), id(state._ts), state._seat
    return id(state._rec)


def _returned_state(r: RoleInstance, result, state_args: list[tuple[str, object]]) -> list:
    """Return each state argument's final value in argument order, matching the role's returned
    state independently of names and tuple order.

    Rebuilt ends and cursors share a record; turnstile copies share a role, turnstile and seat.
    A returned end or cursor must be current: not a snapshot saved before a loop or branch that
    advanced it, nor an argument a helper advanced and returned in a copy the role dropped
    (``Lineage``). Turnstile seats are checked for ownership only.
    """
    role_name = r.name
    values = () if result is None else result if isinstance(result, tuple) else (result,)
    owned = {_state_identity(arg) for _, arg in state_args}
    final = {}
    for value in values:
        if not isinstance(value, _ROLE_STATE_TYPES):
            raise ProtocolError(
                f"{role_name}: return channel endpoints, work cursors and turnstile seats "
                f"(a single value or tuple), not {type(value).__name__}"
            )
        key = _state_identity(value)
        if key not in owned:
            raise ProtocolError(f"{role_name}: returned state does not belong to this role")
        if key in final:
            raise ProtocolError(f"{role_name}: returned the same state argument more than once")
        if not isinstance(value, TurnEnd):
            value._check_current(returned_by=r.fn.__name__)
        final[key] = value
    missing = [name for name, arg in state_args if _state_identity(arg) not in final]
    if missing:
        names = ", ".join(name for name, _ in state_args)
        raise ProtocolError(
            f"{role_name}: return all warpdrive state arguments; missing {', '.join(missing)} "
            f"(for example, return {names})"
        )
    return [final[_state_identity(arg)] for _, arg in state_args]


def _check_signal_once(name: str, side: str, chan, bound):
    """A ``Tma`` or ``Umma`` signal is one elected arrival: one single-warp instance signals
    each epoch."""
    mech = chan.mechanism(side, bound[0][0].role)
    if mech is Threads or mech is Warps:
        return
    how = getattr(mech, "__name__", repr(mech))  # Umma, or Tma(tx_bytes=...)
    if len(bound) > 1 and chan.sharing(side) is Sharing.JOIN:
        raise ProtocolError(
            f"channel {name}: {side} side signals once ({how}) but has several "
            f"roles {[r.name for r, _ in bound]}"
        )
    # One elected thread signals; every warp of a wider role would issue its own arrive (and
    # its own TMA copies or MMAs).
    for r, _ in bound:
        if r.num_warps > 1:
            raise ProtocolError(
                f"channel {name}: {side} side signals once ({how}) from one elected thread, "
                f"but role {r.name} spans {r.num_warps} warps and each would signal; issue it "
                "from a single-warp role"
            )


def _check_turns(name: str, side: str, bound):
    roles = {r.role for r, _ in bound}
    widths = {r.num_threads for r, _ in bound}
    if len(roles) != 1 or len(widths) != 1:
        raise ProtocolError(
            f"channel {name}: round-robin {side}s {[r.name for r, _ in bound]} must be "
            "instances of one role with the same width (each epoch counts one instance)"
        )


def _check_exclusive(name: str, chan):
    """Check an exclusive ring: one producer instance and at least two consumer instances of
    the declared roles, each contributing the same arrivals, since each epoch counts one of
    them. When ``override`` sets the empty counts, contributions may differ and the owners'
    releases must supply those counts."""
    producers = [r.name for r, _ in chan.producers]
    consumers = [r for r, _ in chan.consumers]
    if len(producers) != 1:
        raise ProtocolError(
            f"channel {name}: exclusive consumers need one producer instance (it fills every "
            f"epoch in order), got {producers}"
        )
    if len(consumers) < 2:
        raise ProtocolError(
            f"channel {name}: consumer_sharing=Sharing.EXCLUSIVE needs at least two consumer "
            f"instances, got {[r.name for r in consumers]}"
        )
    missing = [role for role in chan.consumer_roles if all(r.role != role for r in consumers)]
    if missing:
        raise ProtocolError(
            f"channel {name}: exclusive consumer role(s) {missing} take no Consumer end"
        )
    contributions = {
        r.name: arrivals(chan.mechanism("consumer", r.role), r.num_threads) for r in consumers
    }
    if not chan.empty_counts_overridden and len(set(contributions.values())) > 1:
        raise ProtocolError(
            f"channel {name}: exclusive consumers must contribute equal arrivals (each epoch "
            f"counts one of them), got {contributions}"
        )


def _check_named_turns(name: str, chan, turns: dict):
    """Round-robin instances of a named channel that share a stage barrier need a turnstile."""
    for side in SIDES:
        k = chan.turns(side)
        if k == 1 or chan.n % k == 0:
            continue
        bound = {r.name for r, _ in chan.bound(side)}
        ts = turns.get(chan.turnstile)
        if ts is None or not ts.enabled or {r.name for r in ts.parties} != bound:
            raise ProtocolError(
                f"channel {name}: {k} round-robin {side}s share the barriers of {chan.n} named "
                f"stage(s) ({chan.n} % {k} != 0), so two of them could complete a barrier "
                f"without the other side; declare an enabled Turnstile over {sorted(bound)} and "
                "pass its name as turnstile=, or use a multiple of the instance count as stages"
            )


def _role_configs(roles, declared: dict) -> dict:
    """Each role's config: its ``@role`` declaration, replaced by the kernel's ``roles`` entry
    when it has one. Checks ``roles`` against the declared role names."""
    if roles is None:
        return declared
    if not isinstance(roles, Mapping):
        raise ProtocolError(
            f"roles must be a mapping from role name to RoleConfig, got {type(roles).__name__}"
        )
    unknown = sorted(set(roles) - set(declared))
    if unknown:
        raise ProtocolError(f"roles names {unknown}, which are not roles; have {sorted(declared)}")
    for name, config in roles.items():
        if not isinstance(config, RoleConfig):
            raise ProtocolError(
                f"roles[{name!r}] must be a RoleConfig, got {type(config).__name__}"
            )
    return {**declared, **roles}


def _allocate_named(chans: dict, named_bars: dict) -> dict:
    """Assign named-barrier IDs and return the channels' IDs.

    Named channels take IDs from ``MAX_NAMED_BARRIER_ID`` down: per channel, ``n`` consecutive
    full IDs then ``n`` empty IDs, so a ring stage can index them. Each ``NamedBarrier`` then
    takes one ID from ``FIRST_FREE_BARRIER_ID`` up, in declaration order. Each slot of a named
    channel becomes a ``NamedBarrier`` (``full_barriers``/``empty_barriers``) counting the
    slot's producer and consumer threads.
    """
    named_chans = {name: ch for name, ch in chans.items() if ch.named}
    need = sum(2 * ch.n for ch in named_chans.values()) + len(named_bars)
    free = MAX_NAMED_BARRIER_ID - FIRST_FREE_BARRIER_ID + 1
    if need > free:
        raise ProtocolError(
            f"named-barrier channels (2 IDs per stage) and NamedBarriers (1 ID each) need {need} "
            f"barrier IDs; {free} are free (0 is __syncthreads, 1 and 2 belong to the layer); "
            "use mbarrier channels for some"
        )
    out, top = {}, MAX_NAMED_BARRIER_ID
    for name, ch in named_chans.items():
        base = top - 2 * ch.n + 1
        ch.full_barriers, ch.empty_barriers = (
            tuple(
                _channel_barrier(f"{name}.{kind}[{s}]", first + s, count)
                for s, count in enumerate(ch.named_counts())
            )
            for kind, first in (("full", base), ("empty", base + ch.n))
        )
        out[name] = tuple(
            [nb._id for nb in bars] for bars in (ch.full_barriers, ch.empty_barriers)
        )
        top = base - 1
    for barrier_id, nb in enumerate(named_bars.values(), FIRST_FREE_BARRIER_ID):
        nb._id = barrier_id
    return out


def _channel_barrier(name: str, barrier_id: int, threads: int) -> NamedBarrier:
    nb = NamedBarrier(threads)
    nb.name, nb._id = name, barrier_id
    return nb


def _runs(warps: list[int]) -> list[range]:
    """Each run of consecutive warps in a sorted list."""
    runs = []
    for w in warps:
        if runs and runs[-1].stop == w:
            runs[-1] = range(runs[-1].start, w + 1)
        else:
            runs.append(range(w, w + 1))
    return runs


def _in_warpgroups(warp_idx, groups: tuple[int, ...]):
    """Warpgroup-uniform predicate: does ``warp_idx`` fall in the adjacent ``groups``?"""
    return _in_warps(warp_idx, groups[0] * WARPGROUP, len(groups) * WARPGROUP)


def _in_warps(warp_idx, first: int, count: int):
    if count == 1:
        return warp_idx == first
    return (warp_idx >= first) & (warp_idx < first + count)


def _in_runs(warp_idx, runs: list[range]):
    here = _in_warps(warp_idx, runs[0].start, len(runs[0]))
    for run in runs[1:]:
        here = here | _in_warps(warp_idx, run.start, len(run))
    return here


def _check_shared_body(name: str, ranges: tuple[range, ...], params: list):
    """Reject a ``shared_body`` role whose instances need their index at trace time."""
    if len({len(r) for r in ranges}) != 1:
        raise ValueError(
            f"role {name}: shared_body instances must have equal warp counts, got "
            f"{[len(r) for r in ranges]}"
        )
    for pname, ann in params:
        if isinstance(ann, _Bound):
            raise ProtocolError(
                f"role {name}: shared_body cannot take slot-bound {pname}: "
                f"{ann.side.__name__}[...] (each instance binds its own slot, a Python int); "
                "drop shared_body or take the unbound end"
            )
        if ann is Turn:
            raise ProtocolError(
                f"role {name}: shared_body cannot take the turnstile seat {pname} (each "
                "instance holds its own seat); drop shared_body"
            )


def _instance_index(unit: tuple[RoleInstance, ...]) -> Int32:
    """The warp-uniform ``Instance`` index of the executing instance among ``unit`` (in
    declared order): a division when the instances sit at ascending, evenly spaced warps, a
    select per instance otherwise."""
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    firsts = [r.first_warp for r in unit]
    spacing = firsts[1] - firsts[0]
    if spacing > 0 and firsts == [firsts[0] + j * spacing for j in range(len(firsts))]:
        rel = _assume_below(warp - firsts[0] if firsts[0] else warp, spacing * len(unit))
        index = Int32(Uint32(rel) // spacing)
    else:
        index = Int32(0)
        for j, r in enumerate(unit[1:], 1):
            index = Int32(
                cutlass.select_(_in_warps(warp, r.first_warp, r.num_warps), Int32(j), index)
            )
    return _assume_below(index, len(unit))


def _by_instance(instance, values: list[int]):
    """``values[instance]``: the value itself when there is no runtime instance or all values
    agree, an affine expression when they are evenly spaced, a select chain otherwise."""
    if instance is None or len(set(values)) == 1:
        return values[0]
    step = values[1] - values[0]
    if values == [values[0] + j * step for j in range(len(values))]:
        out = Int32(instance * step)
        return Int32(out + values[0]) if values[0] else out
    out = Int32(values[0])
    for j in range(1, len(values)):
        out = Int32(cutlass.select_(instance == j, Int32(values[j]), out))
    return out


def _for_instance(ev, j: int):
    """A shared body's event as instance ``j`` ran it: an instance-indexed ``NamedBarriers``
    operation becomes the operation on barrier ``j``."""
    barriers = ev.data.get("barrier")
    if not isinstance(barriers, ByInstance):
        return ev
    member = barriers.barriers[j]
    return replace(ev, data={**ev.data, "barrier": member, "id": member._allocated_id()})
