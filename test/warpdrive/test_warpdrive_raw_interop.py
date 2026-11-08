"""The raw-interop example: a managed producer, a raw consumer that drives the ring's external
consumer side by pointer under an overridden empty count, and a raw role that waits natively on
a ``NamedBarrier``'s ID."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples.raw_interop import (
    WORDS,
    RawInterop,
    compile_raw_interop,
    raw_interop,
)


def test_describe_marks_what_the_raw_code_leaves_unverified():
    op = RawInterop()
    compile_raw_interop(op)
    text = op.describe()
    assert "ring  Ring(2)  producer -> raw code\n" in text, text
    assert "empty     1 x2\n" in text, text
    assert "external  consumer side driven by raw code; empty counts are the override's" in text
    row = next(line for line in text.splitlines() if line.strip().startswith("done "))
    assert "arrive by consumer; native id by watcher" in row, row
    assert "unverified: roles calling it have 64 of 96 threads" in row, row


@pytest.mark.parametrize("stages", [1, 2, 3])
@pytest.mark.parametrize("n", [0, 1, 7, 64])
def test_raw_consumer_sees_every_epoch(requires_blackwell, stages, n):
    for _ in range(2):  # repeated launches start from fresh barriers
        bad, total = raw_interop(n, stages).tolist()
        words = torch.arange(n * WORDS, dtype=torch.int64)
        assert bad == 0 and total == int(words.sum())
