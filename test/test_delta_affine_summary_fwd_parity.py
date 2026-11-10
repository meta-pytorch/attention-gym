"""Temporary bitwise oracle for the forward-summary warpdrive port."""

from itertools import product

import pytest
import torch

pytest.importorskip("cutlass")

from test_delta_affine_summary import int64_stride_copy, make_summary_inputs

from attn_gym.linear._delta_rule.cute import affine_summary_fwd as port
from attn_gym.linear._delta_rule.cute import affine_summary_fwd_legacy as legacy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="native forward summaries require Blackwell",
)


@pytest.mark.parametrize(
    "dtype,heads,bn,wide,whole,fastmath",
    list(
        product(
            (torch.bfloat16, torch.float16),
            (1, 16),
            (32, 64),
            (False, True),
            (False, True),
            (False, True),
        )
    ),
)
def test_forward_summary_bitwise_legacy_parity(dtype, heads, bn, wide, whole, fastmath):
    _qg, kg, w, u, _dout, _aqk, gate = make_summary_inputs(79, dtype, 2048, heads)
    tensors = (kg, w, u, gate)
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
    args = (port._IO_TYPE_NAMES[dtype], heads, bn, wide, whole, fastmath)
    old = legacy._compile_affine_summary(*args)
    new = port._compile_affine_summary(*args)
    old(*tensors, work, expected)
    new(*tensors, work, actual)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        old(*tensors, work, expected)
        new(*tensors, work, actual)
    for _ in range(3):
        tensors[2].mul_(0.5)
        graph.replay()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
