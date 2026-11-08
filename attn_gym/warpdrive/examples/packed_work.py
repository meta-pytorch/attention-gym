"""Map a device-built work list to rows of an int32 table.

A preceding kernel writes one row per item and a device-side length. The host needs only
capacity, not the length. For packed attention::

    self.work = WorkStream(mapping=PackedWork(), schedule=AtomicPersistent(), ready=PdlWait())
    bound = self.work.bind((table, length), workspace, max_ctas)  # in the jit entry point

    while work.valid:  # in a role taking `work: WorkCursor`
        q_block, head, seq = work.item
        ...
        work.advance()

``table`` is row-major int32 ``(capacity, words)``; ``words`` defaults to 3.
``count64`` returns capacity, so the launch covers ``[0, capacity)``. ``decode`` keeps
indices below ``length[0]``. Skipped indices still cost scheduling work; keep capacity
close to the expected length.

The readiness hook runs before decoding. ``PdlWait()`` makes the producer's writes visible
under programmatic dependent launch. The table and length must stay unchanged after readiness,
and ``length[0] >= 0``. A length above capacity keeps every row; decode never exceeds capacity.

``test/warpdrive/test_warpdrive_scheduler_interface.py`` covers a producer kernel followed
by PDL consumers under every schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

from cutlass import Int64

from attn_gym.warpdrive import WorkMap


@dataclass(frozen=True)
class PackedWork(WorkMap):
    """Items are the first ``length[0]`` rows of ``table``; space ``(table, length)``."""

    words: int = 3
    skips = True  # rows past the device-side length are dropped

    def __post_init__(self):
        if type(self.words) is not int or self.words < 1:
            raise ValueError(f"PackedWork.words must be a positive int, got {self.words!r}")

    def count64(self, space):
        table, _ = space
        capacity = table.shape[0]
        return capacity if isinstance(capacity, int) else Int64(capacity)

    def decode(self, index, space, state):
        table, length = space
        # The scheduler bounds index by capacity, so even a skipped row is safe to read.
        row = tuple(table[index, w] for w in range(self.words))
        return index < length[0], row, state

    def get_key(self) -> str:
        return f"packed{self.words}"

    def describe(self) -> str:
        return f"PackedWork(words={self.words})"
