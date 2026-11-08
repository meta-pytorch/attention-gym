"""Layer-owned tensor-memory plan: named per-slot column regions and aliases."""

from __future__ import annotations

from dataclasses import dataclass

from attn_gym.warpdrive.protocol import ProtocolError, ProtocolLog

TMEM_COLS = 512


@dataclass
class _Region:
    name: str
    cols: int
    slots: int
    offset: int = 0
    alias_of: str | None = None
    alias_offset: int = 0


class TmemPlan:
    """Named TMEM column regions; ``alias`` places a region inside each slot of another.

    Offsets are assigned in declaration order and aliases take no new columns. The layer
    allocates the smallest power-of-two column count (>= 32) covering the plan.
    """

    def __init__(self):
        self.regions: dict[str, _Region] = {}
        self._next = 0

    def region(self, name: str, cols: int, slots: int = 1) -> TmemPlan:
        if name in self.regions:
            raise ValueError(f"TMEM region {name!r} is already declared")
        if type(cols) is not int or type(slots) is not int or cols <= 0 or slots <= 0:
            raise ValueError(f"TMEM region {name}: cols and slots must be positive ints")
        self.regions[name] = _Region(name, cols, slots, offset=self._next)
        self._next += cols * slots
        return self

    def alias(self, name: str, of: str, offset: int, cols: int) -> TmemPlan:
        if name in self.regions:
            raise ValueError(f"TMEM region {name!r} is already declared")
        if of not in self.regions:
            raise ProtocolError(f"no TMEM region named {of!r}")
        if type(cols) is not int or cols <= 0:
            raise ValueError(f"TMEM alias {name}: cols must be a positive int")
        if type(offset) is not int:
            raise ValueError(f"TMEM alias {name}: offset must be a nonnegative int")
        parent = self.regions[of]
        if offset < 0 or offset + cols > parent.cols:
            raise ValueError(
                f"TMEM alias {name} [{offset}, {offset + cols}) exceeds {of} ({parent.cols} cols)"
            )
        self.regions[name] = _Region(name, cols, parent.slots, alias_of=of, alias_offset=offset)
        return self

    def col(self, name: str, slot: int = 0) -> int:
        if name not in self.regions:
            raise ProtocolError(f"no TMEM region named {name!r}")
        r = self.regions[name]
        if type(slot) is not int or not 0 <= slot < r.slots:
            raise ProtocolError(f"TMEM region {name} has {r.slots} slots; got slot {slot!r}")
        if r.alias_of is None:
            return r.offset + slot * r.cols
        return self.col(r.alias_of, slot) + r.alias_offset

    @property
    def used_cols(self) -> int:
        return self._next

    @property
    def alloc_cols(self) -> int:
        cols = 1 << (max(32, self._next) - 1).bit_length()
        if cols > TMEM_COLS:
            raise ValueError(f"TMEM plan needs {self._next} > {TMEM_COLS} columns")
        return cols

    def describe(self) -> str:
        lines = []
        for r in self.regions.values():
            spans = ", ".join(
                f"[{self.col(r.name, j)}, {self.col(r.name, j) + r.cols})" for j in range(r.slots)
            )
            alias = f" (inside {r.alias_of})" if r.alias_of else ""
            lines.append(f"  tmem {r.name:<3} {spans}{alias}")
        return "\n".join(lines)


TMEM_AGENTS = ("threads", "umma")


class TmemView:
    """Device-side TMEM base pointer plus the plan; records which regions each role touches.

    ``ptr``/``cols``/``base`` are unannotated acquisitions. ``view``/``access`` annotate an access
    for the alias diagnostic (``alias.py``) at the point of the call: ``by="threads"``
    (``tcgen05.ld``/``st``) or ``by="umma"`` (a ``tcgen05.mma`` A-operand read or D write).
    Footprint: ``warps`` (default every warp of the role), ``lanes`` and ``cols`` (ranges, or
    callables of the warp index; ``cols`` relative to the region). Defaults: a thread access
    covers its warp's 32-lane subpartition (``32 * (warp % 4)``), an MMA all 128 lanes, and both
    the whole region."""

    def __init__(self, base, plan: TmemPlan, role_name: str, log: ProtocolLog | None, trace=None):
        self._base, self._plan, self._role, self._log = base, plan, role_name, log
        self._trace = trace

    # The preprocessor carries a local name that a loop body calls a method on. The view is
    # defined once per role and never rebound, so it carries nothing and rebuilds as itself.
    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self

    @property
    def base(self):
        """The allocation's FP32 base pointer (an unannotated acquisition of every region)."""
        if self._trace is not None:
            for name in self._plan.regions:
                self._trace.event("raw", storage=("tmem", name))
        return self._base

    def _col(self, name: str, slot: int) -> int:
        col = self._plan.col(name, slot)
        if self._log is not None:
            self._log.tmem.setdefault(self._role, set()).add(f"{name}[{slot}]")
        return col

    def cols(self, name: str, slot: int = 0) -> int:
        """Column offset of ``name[slot]`` relative to the allocation base."""
        col = self._col(name, slot)
        if self._trace is not None:
            self._trace.event("raw", storage=("tmem", name))
        return col

    def ptr(self, name: str, slot: int = 0):
        """FP32 TMEM pointer to ``name[slot]``."""
        return self._base + self.cols(name, slot)

    def view(self, name: str, slot: int = 0, **access):
        """FP32 TMEM pointer to ``name[slot]``, annotating an access here (see ``access``)."""
        self.access(name, slot, **access)
        return self._base + self._col(name, slot)

    def access(
        self,
        name: str,
        slot: int = 0,
        *,
        read: bool = False,
        write: bool = False,
        by: str = "threads",
        warps=None,
        lanes=None,
        cols=None,
    ):
        """Annotate an access to ``name[slot]`` here, through a tensor built earlier. No IR."""
        if not (read or write):
            raise ValueError("a TMEM access needs read=True and/or write=True")
        if by not in TMEM_AGENTS:
            raise ValueError(f"TMEM access agent must be one of {TMEM_AGENTS}, got {by!r}")
        for what, value in (("lanes", lanes), ("cols", cols)):
            if value is not None and not (isinstance(value, range) or callable(value)):
                raise ValueError(f"TMEM access {what} must be a range or a callable of the warp")
        if warps is not None:
            warps = tuple(warps)
            if not warps or any(type(w) is not int for w in warps):
                raise ValueError(f"TMEM access warps must be ints, got {warps!r}")
        self._col(name, slot)  # validates the region and slot
        width = self._plan.regions[name].cols
        for what, value, limit in (("lanes", lanes, 128), ("cols", cols, width)):
            if isinstance(value, range) and (value.start < 0 or value.stop > limit):
                raise ValueError(f"TMEM access {what} {value} leaves {name} ([0, {limit}))")
        if self._trace is not None:
            self._trace.event(
                "access",
                storage=("tmem", name),
                slot=slot,
                read=read,
                write=write,
                agent=by,
                warps=warps,
                lanes=lanes,
                cols=cols,
            )
