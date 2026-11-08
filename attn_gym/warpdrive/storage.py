"""Optional channel storage: a per-CTA plan of named SMEM buffers and stage projections.

A kernel declares its SMEM buffers once, then binds them to channels::

    self.smem_plan = SmemPlan()
    A = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
    B = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
    self.ab = Ring(4, Tma(stage_bytes), Umma, buffer={"A": A, "B": B})

Layouts are trace-time values (they usually come from a tiled MMA built in the entry point), so
they bind in the kernel prologue: ``self.begin_roles(work=bound, layouts={"A": a_layout, ...})``
allocates every buffer, in declaration order, after the layer's own barriers and payloads.
``self.buffer("A")`` returns the whole tensor. Inside a role, a handle projects its own stage
snapshot: ``h.data`` is the stage's canonical view, and ``end.views(A=StageView(tAsA, axis=1))``
registers a partition that ``h.view("A")`` projects. There is no second cursor: the stage is
always ``h.index``.

The plan owns SMEM accounting: a buffer declared with ``overlaps=`` reuses another buffer's
bytes (FA4's V over K) and adds none; several channels may bind one buffer. The plan describes one
CTA; in a CTA pair each CTA holds its own copy at the same offset (2-CTA MMAs read the peer's
operands there). Binding a buffer never implies DSMEM publication and never adds a fence: the
channel's ``remote_scope`` and fences are what they would be without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cutlass import cute

from attn_gym.warpdrive.protocol import ProtocolError


@dataclass(eq=False)
class SmemBuffer:
    """One named SMEM buffer of a ``SmemPlan``: element type, alignment and stage axis now, the
    layout (including its swizzle) at ``begin_roles``. ``overlaps``: share another buffer's bytes,
    starting at its first byte, instead of allocating."""

    name: str
    dtype: type
    align: int = 16
    stage_axis: int = -1
    overlaps: SmemBuffer | None = None

    @property
    def root(self) -> SmemBuffer:
        """The buffer that owns the bytes (itself unless it overlaps another)."""
        return self if self.overlaps is None else self.overlaps.root


class SmemPlan:
    """Named per-CTA SMEM buffers, allocated by ``begin_roles`` in declaration order."""

    def __init__(self):
        self.buffers: dict[str, SmemBuffer] = {}

    def buffer(
        self,
        name: str,
        dtype: type,
        *,
        align: int = 16,
        stage_axis: int = -1,
        overlaps: SmemBuffer | str | None = None,
    ) -> SmemBuffer:
        """Declare a buffer. ``stage_axis`` is the layout mode a channel's stage selects (the
        last mode by default, as in CUTLASS's staged layouts). ``overlaps`` names a buffer whose
        bytes this one reuses; it must fit inside and not need a stricter alignment."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"SMEM buffer name must be a non-empty str, got {name!r}")
        if name in self.buffers:
            raise ValueError(f"SMEM buffer {name!r} is already declared")
        if type(align) is not int or align <= 0 or align & (align - 1):
            raise ValueError(
                f"SMEM buffer {name}: align must be a power-of-two int, got {align!r}"
            )
        if type(stage_axis) is not int:
            raise ValueError(f"SMEM buffer {name}: stage_axis must be an int, got {stage_axis!r}")
        if isinstance(overlaps, str):
            if overlaps not in self.buffers:
                raise ProtocolError(f"SMEM buffer {name} overlaps unknown buffer {overlaps!r}")
            overlaps = self.buffers[overlaps]
        if overlaps is not None:
            if self.buffers.get(overlaps.name) is not overlaps:
                raise ProtocolError(
                    f"SMEM buffer {name} overlaps {overlaps.name!r}, which is not in this plan"
                )
            if overlaps.root.align % align:
                raise ValueError(
                    f"SMEM buffer {name} needs {align}-byte alignment, but it overlaps "
                    f"{overlaps.root.name} ({overlaps.root.align}-byte aligned)"
                )
        buf = SmemBuffer(name, dtype, align, stage_axis, overlaps)
        self.buffers[name] = buf
        return buf

    def __getitem__(self, name: str) -> SmemBuffer:
        if name not in self.buffers:
            raise ProtocolError(f"no SMEM buffer named {name!r} (have {sorted(self.buffers)})")
        return self.buffers[name]


@dataclass(frozen=True)
class StageView:
    """A tensor derived from a staged buffer (for example its TMA or MMA partition) whose mode
    ``axis`` is the buffer's stage mode. ``end.views(A=StageView(tAsA, axis=1))`` binds it to
    buffer A of the end's channel; ``h.view("A")`` then returns ``tAsA`` at ``h.index``."""

    tensor: object
    axis: int


@dataclass
class ResolvedBuffer:
    """A plan buffer after ``begin_roles``: its tensor and the host facts about its bytes.

    ``nbytes``/``stage_span``: from the static layout, ``None`` when it is not static.
    ``stage_span(s)`` is ``[start, stop)`` in bytes from the first byte of the root buffer."""

    buffer: SmemBuffer
    tensor: object
    nbytes: int | None
    stages: int | None
    stage_stride: int | None = None  # bytes between consecutive stages
    stage_bytes: int | None = None
    channels: list = field(default_factory=list)  # channel names binding this buffer

    def stage_span(self, stage: int) -> tuple[int, int] | None:
        if self.stage_stride is None or self.stage_bytes is None:
            return None
        start = stage * self.stage_stride
        return start, start + self.stage_bytes


def split_layout(layout):
    """``(outer, swizzle)`` of a plain or composed (swizzled) layout."""
    if isinstance(layout, cute.ComposedLayout):
        return layout.outer, layout.inner
    return layout, None


def allocate(smem, buf: SmemBuffer, layout, resolved: dict) -> ResolvedBuffer:
    """Allocate (or, for an overlap, alias) one buffer and record its byte facts."""
    outer, swizzle = split_layout(layout)
    if buf.overlaps is None:
        tensor = smem.allocate_tensor(buf.dtype, outer, buf.align, swizzle)
    else:
        parent = resolved[buf.overlaps.name]
        dtype = None if buf.dtype is parent.buffer.dtype else buf.dtype
        tensor = cute.make_tensor(cute.recast_ptr(parent.tensor.iterator, swizzle, dtype), outer)
    width = buf.dtype.width
    cosize = cute.cosize(outer)
    nbytes = cosize * width // 8 if isinstance(cosize, int) else None
    rank = cute.rank(outer)
    axis = buf.stage_axis + rank if buf.stage_axis < 0 else buf.stage_axis
    if not 0 <= axis < rank:
        raise ProtocolError(
            f"SMEM buffer {buf.name}: stage_axis {buf.stage_axis} is outside its rank-{rank} layout"
        )
    stages = cute.size(outer, mode=[axis])
    stride = _stage_stride(outer.shape[axis], outer.stride[axis])
    out = ResolvedBuffer(buf, tensor, nbytes, stages if isinstance(stages, int) else None)
    if nbytes is not None and isinstance(stages, int) and stride is not None:
        stage_stride = stride * width // 8
        stage_bytes = nbytes - (stages - 1) * stage_stride
        if 0 < stage_bytes <= stage_stride:  # the stage mode is outermost: stages are disjoint
            out.stage_stride, out.stage_bytes = stage_stride, stage_bytes
    if buf.overlaps is not None:
        root = resolved[buf.root.name]
        if nbytes is not None and root.nbytes is not None and nbytes > root.nbytes:
            raise ProtocolError(
                f"SMEM buffer {buf.name} ({nbytes} B) overlaps {root.buffer.name}, which has only "
                f"{root.nbytes} B"
            )
    return out


def _stage_stride(shape, stride) -> int | None:
    """The element stride between stages of a (possibly nested) stage mode: the stride of its
    only non-unit sub-mode (CUTLASS's epilogue layouts write a stage mode as ``(1, n)``)."""
    live = [d for n, d in zip(cute.flatten((shape,)), cute.flatten((stride,))) if n != 1]
    if len(live) != 1 or not isinstance(live[0], int):
        return None
    return live[0]


def stage_extent(tensor, axis: int, what: str) -> int:
    """The static extent of ``tensor``'s mode ``axis`` (the stage mode)."""
    rank = cute.rank(tensor)
    norm = axis + rank if axis < 0 else axis
    if not 0 <= norm < rank:
        raise ProtocolError(f"{what}: stage axis {axis} is outside a rank-{rank} tensor")
    extent = cute.size(tensor, mode=[norm])
    if not isinstance(extent, int):
        raise ProtocolError(f"{what}: the stage mode must be static, got extent {extent}")
    return extent


def project(tensor, axis: int, index, coord: tuple = ()):
    """``tensor`` with its mode ``axis`` fixed to ``index``. ``coord`` (one entry per other
    mode, ``None`` keeps a mode) indexes the other modes in the same slice; the default keeps
    them all."""
    rank = cute.rank(tensor)
    norm = axis + rank if axis < 0 else axis
    rest = list(coord) if coord else [None] * (rank - 1)
    if len(rest) != rank - 1:
        raise ProtocolError(
            f"a stage view of a rank-{rank} tensor takes {rank - 1} coordinates, got {len(rest)}"
        )
    return tensor[(*rest[:norm], index, *rest[norm:])]
