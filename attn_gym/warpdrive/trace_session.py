"""Host side of a ``"globaltimer"`` trace (``tracing.py``): ``profile_session`` over a warpdrive
kernel's profile buffer, and its Perfetto layout.

Not imported by the kernel modules or the package ``__init__``: a compile key hashes every
module a kernel imports, and decoding has nothing to do with the compiled code.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from attn_gym.profiler.host import ProfileSession, profile_session
from attn_gym.profiler.perfetto import write_ranges
from attn_gym.profiler.ranges import Range
from attn_gym.warpdrive.protocol import ProtocolError


def trace_session(
    kernel, num_ctas: int, *, trace_path: str | Path | None = None, device=None
) -> AbstractContextManager[ProfileSession]:
    """``profile_session`` for one launch of a compiled ``enable_trace("globaltimer")`` kernel
    with ``num_ctas`` CTAs. Pass ``session.prof.tensor`` as the kernel's trace buffer
    (``run_roles(..., trace_buffer=...)``). After the block, ``session.ranges`` holds the
    decoded ranges (``name`` = range name, ``cta`` = ``(CTA,)``, ``warp`` = the role's first
    warp, ``payload`` = a work item's index or a ``trace_region``'s payload), and
    ``trace_path`` (optional) the Perfetto trace of ``write_trace``.

    Warns when a role filled its ``events_per_role`` records in some CTA (later ranges were
    dropped)."""
    if kernel.trace_backend != "globaltimer":
        raise ProtocolError(
            f"trace_session needs enable_trace('globaltimer'), not {kernel.trace_backend!r}"
        )
    if not getattr(kernel, "_ws_traced", False):
        raise ProtocolError(
            "trace_session needs a compiled kernel: range names come from its trace"
        )
    return capture(kernel, num_ctas, trace_path, device)


@contextmanager
def capture(kernel, num_ctas: int, trace_path, device) -> Iterator[ProfileSession]:
    roles = kernel._ws_roles
    capacity = kernel.trace_events_per_role
    with profile_session(
        max_events_per_unit=len(roles) * capacity,
        num_units=(num_ctas, "CTA"),
        tag_names=kernel._ws_trace_tags,
        device=device,
    ) as session:
        yield session
    lanes = {r.first_warp: r.name for r in roles}
    counts = Counter((item.cta, item.warp) for item in session.ranges)
    full = sorted({lanes[warp] for (_, warp), n in counts.items() if n >= capacity})
    if full:
        warnings.warn(
            f"warpdrive trace: role(s) {full} filled their {capacity} records in some CTA; "
            "later ranges were dropped (raise enable_trace(events_per_role=...))",
            stacklevel=3,
        )
    if trace_path is not None:
        write_trace(trace_path, kernel, session.ranges)


def write_trace(path: str | Path, kernel, ranges: list[Range], **options) -> Path:
    """Write a globaltimer trace of ``kernel`` with one process per CTA and one track per role
    (named after it); ``options`` (``name``, ``args``, ``flows``) go to ``write_ranges``."""
    lanes = {r.first_warp: r.name for r in kernel._ws_roles}
    return write_ranges(
        path,
        ranges,
        process=lambda item: f"CTA {item.cta[0]}",
        thread=lambda item: lanes[item.warp],
        **options,
    )
