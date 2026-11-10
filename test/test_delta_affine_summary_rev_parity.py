"""Temporary bitwise oracle for the reverse-summary warpdrive port."""

from itertools import product

import pytest
import torch

pytest.importorskip("cutlass")

from test_delta_affine_summary import int64_stride_copy, make_summary_inputs

from attn_gym.linear._delta_rule.cute import affine_summary_rev as port
from attn_gym.linear._delta_rule.cute import affine_summary_rev_legacy as legacy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="native reverse summaries require Blackwell",
)


@pytest.mark.parametrize(
    "dtype,heads,wide,whole,fastmath",
    list(
        product(
            (torch.bfloat16, torch.float16), (1, 16), (False, True), (False, True), (False, True)
        )
    ),
)
def test_reverse_summary_bitwise_legacy_parity(dtype, heads, wide, whole, fastmath):
    qg, kg, w, _u, dout, aqk, gate = make_summary_inputs(71, dtype, 2048, heads)
    tensors = (qg, kg, w, dout, aqk, gate)
    if wide:
        tensors = tuple(int64_stride_copy(t) for t in tensors)
    rows = (
        [[0, 0], [0, 1], [64, 127], [128, 192], [192, 257], [257, 768], [800, 1441], [1441, 2048]]
        if whole
        else [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [64, 0, 1, 63],
            [128, 0, 1, 64],
            [192, 0, 2, 65],
            [257, 0, 8, 511],
            [800, 2, 11, 641],
            [1441, 0, 10, 607],
        ]
    )
    work = torch.tensor(rows, dtype=torch.int32, device="cuda")
    expected = torch.empty((len(rows), heads, 256, 128), dtype=torch.float32, device="cuda")
    actual = torch.empty_like(expected)
    args = (port._IO_TYPE_NAMES[dtype], heads, wide, whole, fastmath)
    old = legacy._compile_affine_summary_rev(*args)
    new = port._compile_affine_summary_rev(*args)
    workspace, _max_ctas = port._LAUNCH_WORK.launch_args(actual.device)
    old(*tensors, work, expected, 128**-0.5)
    # Two CTAs force every role through dozens of persistent items, changing between
    # source and transition tiles, empty items, and odd numbers of chunk epochs.
    new(*tensors, work, actual, 128**-0.5, workspace, 2)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        old(*tensors, work, expected, 128**-0.5)
        new(*tensors, work, actual, 128**-0.5, workspace, 2)
    for _ in range(3):
        tensors[3].mul_(0.5)
        graph.replay()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
