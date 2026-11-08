"""A work list built on the device: ``PackedWork`` maps indices to rows of an int32 table.

Varlen and sparse schedules often build their work list in a preceding kernel: one row per item
that has work, e.g. ``(query_block, head, sequence)`` rows for packed attention, and a
device-side length. The host never reads that length; it only knows the table's capacity::

    self.work = WorkStream(mapping=PackedWork(), schedule=AtomicPersistent(), ready=PdlWait())
    bound = self.work.bind((table, length), workspace, max_ctas)  # in the jit entry point

    while work.valid:  # in a role taking `work: WorkCursor`
        q_block, head, seq = work.item
        ...
        work.advance()

How the contract carries it:

* ``count64`` returns the capacity (``table.shape[0]``), a host-visible extent: the launch is
  sized for a full table, and indices run over ``[0, capacity)``.
* ``decode`` reads ``length[0]`` and keeps an index only below it. The layer calls ``decode``
  only for indices below the capacity, after the readiness hook, so with ``ready=PdlWait()``
  the producer's writes are visible under programmatic dependent launch. Indices past the
  length are dropped like any skipped item (the schedule still visits them, so size the
  capacity close to the usual length).
* The table and length must not change after readiness, and ``0 <= length[0]`` (a length past
  the capacity keeps every row).

The table is ``(capacity, words)`` int32, row-major; ``words`` is a constructor argument (3 by
default). Grouped GEMM or split-KV attention add columns (group id, split id and count) and
filter empty rows the same way. ``test/warpdrive/test_warpdrive_scheduler_interface.py`` builds the
list with a producer kernel and consumes it under every schedule with a PDL launch.
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
        # index < capacity, so the row read is in bounds even past the length (its words are
        # then ignored), and no branch is needed.
        row = tuple(table[index, w] for w in range(self.words))
        return index < length[0], row, state

    def get_key(self) -> str:
        return f"packed{self.words}"

    def describe(self) -> str:
        return f"PackedWork(words={self.words})"
