"""Inline annotation markers for CuTeDSL role kernels.

``PipelineAnnotations`` emits no device IR unless IKET is enabled. ``extract.extract_plan``
reads the same declarations from the Python source and lowers them to a ``PipelinePlan``. A
kernel imports only this module, so its compile key does not hash the extractor or the plan
model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps

from cutlass.cute.experimental import iket


class PipelineAnnotations:
    """Source markers with optional IKET instrumentation during CuTeDSL tracing."""

    def __init__(self, name: str, *, iteration_name: str) -> None:
        self.name = name
        self.iteration_name = iteration_name
        self.iket_enabled = False
        self.active_role: str | None = None
        self.region_open = False

    def enable_iket(self) -> None:
        """Emit IKET ranges in kernels compiled from now on in this process."""
        self.iket_enabled = True

    def role(
        self,
        name: str,
        *,
        label: str,
        warp_start: int,
        warp_end: int,
    ) -> Callable[[Callable], Callable]:
        """Mark a role function. Its arguments are read from source by the extractor.

        While tracing, this rejects nested roles and, with IKET enabled, a role that returns
        with a region still open.
        """

        def decorate(function: Callable) -> Callable:
            @wraps(function)
            def wrapped(*args, **kwargs):
                if self.active_role is not None:
                    raise RuntimeError(
                        f"Pipeline role {name!r} entered while {self.active_role!r} is active"
                    )
                self.active_role = name
                try:
                    result = function(*args, **kwargs)
                    if self.iket_enabled and self.region_open:
                        raise RuntimeError(
                            f"Pipeline role {name!r} exited with an open region; "
                            "add PIPELINE.iteration_end() after the final phase"
                        )
                    return result
                finally:
                    self.active_role = None
                    self.region_open = False

            return wrapped

        return decorate

    def resource(
        self,
        name: str,
        *,
        label: str,
        depth: int,
        kind: str = "buffer",
    ) -> None:
        """Declare a logical resource. Only the extractor reads it; at runtime it does nothing."""

    def region(
        self,
        name: str,
        *,
        label: str,
        weight: int,
        consumes: Iterable[str] = (),
        produces: Iterable[str] = (),
        releases: Iterable[str] = (),
        count: int = 1,
        index: int | None = None,
    ) -> None:
        """Start a named IKET range, closing the previous one, when IKET is enabled.

        A region inside a ``cutlass.range_constexpr`` loop passes the compile-time loop
        variable as ``index`` and the trip count as a literal ``count``. The traced range is
        named ``name[index]`` and the extractor expands the call into ``count`` regions.
        Tokens containing ``{index}`` are substituted per copy; plain ``consumes`` attach to
        the first copy and plain ``produces``/``releases`` to the last, so from outside the
        group acts as one region.
        """
        if not self.iket_enabled:
            return
        if self.active_role is None:
            raise RuntimeError(f"Pipeline region {name!r} executed outside an annotated role")
        if index is not None:
            if type(index) is not int:
                raise TypeError(
                    f"Pipeline region {name!r} index must be a compile-time int "
                    f"(use cutlass.range_constexpr), got {type(index).__name__}"
                )
            if not 0 <= index < count:
                raise ValueError(f"Pipeline region {name!r} index {index} outside count {count}")
            name = indexed_region_name(name, index)
        if self.region_open:
            iket.range_pop()
        iket.range_push(name)
        self.region_open = True

    def iteration_end(self) -> None:
        """Close the last region of a traced loop iteration."""
        if not self.iket_enabled:
            return
        if not self.region_open:
            raise RuntimeError("PIPELINE.iteration_end() has no active region")
        iket.range_pop()
        self.region_open = False


def indexed_region_name(name: str, index: int) -> str:
    """Name one copy of a ``count``-expanded region."""
    return f"{name}[{index}]"
