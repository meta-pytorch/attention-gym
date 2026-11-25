"""The megakernels' instruction tables: an int32 ``(rows, words)`` table of work items, the op
first and the row's index in the step's tables last, that the host deals to CTAs."""

from dataclasses import dataclass

from cutlass import Int64

from attn_gym.warpdrive import WorkMap


@dataclass(frozen=True)
class InstructionTable(WorkMap):
    """Row ``index`` of an int32 ``(rows, words)`` table. A negative op marks a padding row,
    dropped for every role."""

    words: int
    skips = True

    def count64(self, space):
        rows = space.shape[0]
        return rows if isinstance(rows, int) else Int64(rows)

    def decode(self, index, space, state):
        words = tuple(space[index, w] for w in range(self.words))
        return words[0] >= 0, words, state

    def get_key(self) -> str:
        return f"instructions{self.words}"


def layout_static(lists: list[list[list[int]]], words: int) -> list[list[int]]:
    """Per-CTA row lists (each row its first ``words - 1`` words) laid out ``[slot][cta]`` for
    ``StaticPersistent`` (CTA b runs rows b, b + ctas, ...); short lists are padded with dropped
    rows."""
    slots = max(1, max(len(rows) for rows in lists))
    padding = [-1] * (words - 1)
    return [
        lists[cta][slot] if slot < len(lists[cta]) else padding
        for slot in range(slots)
        for cta in range(len(lists))
    ]
