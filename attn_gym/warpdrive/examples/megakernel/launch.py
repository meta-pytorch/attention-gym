"""The host side of the megakernels' launches: instruction tables dealt to CTAs, a one-launch
dependency chain's state (``Launch``), and reading tagged scratch."""

import torch

from attn_gym.warpdrive.examples.megakernel.table import layout_static
from attn_gym.warpdrive.examples.megakernel.tagged import scratch_offsets


def deal(rows: list[list[int]], ctas: int) -> list[list[list[int]]]:
    """``rows`` (in dependency order) dealt round-robin over ``ctas`` lists."""
    return [rows[c::ctas] for c in range(ctas)]


def table_of(lists: list[list[list[int]]], words: int) -> torch.Tensor:
    """Per-CTA row lists (each row its first ``words - 1`` words) as an int32 ``(slots * ctas,
    words)`` table (``layout_static``), each row ending in its index."""
    return index_rows(layout_static(lists, words))


def index_rows(rows: list[list[int]], first: int = 0) -> torch.Tensor:
    """``rows`` as an int32 table, each ending in its index ``first + i`` (the trace's row ID)."""
    return torch.tensor([row + [first + i] for i, row in enumerate(rows)], dtype=torch.int32)


def tagged_values(words: torch.Tensor) -> torch.Tensor:
    """The fp32 values of int64 tagged words (``tagged.pack_tagged``: value bits low), ignoring
    their tags."""
    return words.view(torch.int32)[..., 0::2].view(torch.float32)


class Launch:
    """One kernel's launch state as a one-launch dependency chain (``tagged.advance_epoch``):
    its instruction table (``build(op, ctas)``), tagged scratch (``op.scratch_shapes``: words
    per buffer), ``[epoch, finished CTAs]`` state and a placeholder trace buffer argument, which
    a traced runner replaces with its session's. ``trace``: compile ``op`` with that trace
    backend, recording up to ``trace_events`` events per role (default: 32 per table row of a
    CTA, which covers a work item, its regions and a few ring waits). Launches sharing one must
    run one at a time in stream order."""

    def __init__(
        self, op, build, device, trace: str | None = None, trace_events: int | None = None
    ):
        device = torch.device(device)
        self.device = device
        self.workspace, self.max_ctas = op.work.launch_args(device)
        self.table = build(op, self.max_ctas).to(device)
        if trace is not None:
            if trace_events is None:
                trace_events = 32 * -(-self.table.shape[0] // self.max_ctas) + 64
            op = op.with_settings(trace_backend=trace, trace_events_per_role=trace_events)
            self.workspace = op.work.launch_args(device)[0]  # the traced kernel's stream
        self.op = op
        self.offsets, size = scratch_offsets(op.scratch_shapes)
        self.scratch = torch.zeros(size, device=device, dtype=torch.int64)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)

    def values(self, name: str) -> torch.Tensor:
        """The fp32 values of tagged buffer ``name``."""
        offset = self.offsets[name]
        return tagged_values(self.scratch[offset : offset + self.op.scratch_shapes[name]])
