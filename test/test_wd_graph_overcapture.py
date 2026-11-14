"""warpdrive GDN/KDA CUDA graphs captured at capacity and replayed at low occupancy.

Fixed-capacity training and serving capture one graph per token capacity, then replay it with new
packed offsets whose terminal offset is far below that capacity. A replay must equal an eager call
on the same buffers: outputs and token gradients up to the terminal offset, and every final state
and state gradient. KDA passes ``schedule="persistent"``, which warpdrive accepts and ignores.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN/KDA kernels require an SM100-family GPU",
)

CAPACITY = 1024
HEADS = 2
DIM = 128
CAPTURE_LENGTHS = (300, 0, 500, 224)  # fills the capacity
REPLAY_LENGTHS = (40, 0, 0, 24)  # terminal offset 64, 1/16 of the capacity


def _inputs(family: str, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, device="cuda", generator=generator)

    shape = (1, CAPACITY, HEADS, DIM)
    q = F.normalize(randn(*shape), dim=-1).bfloat16()
    k = F.normalize(randn(*shape), dim=-1).bfloat16()
    value = randn(*shape).bfloat16()
    gate_shape = shape if family == "kda" else shape[:3]
    gate = -torch.rand(gate_shape, device="cuda", generator=generator) * 0.5
    beta = torch.rand(shape[:3], device="cuda", generator=generator).sigmoid()
    state = randn(len(CAPTURE_LENGTHS), HEADS, DIM, DIM) / 100
    return q, k, value, gate, beta, state


def _step(family: str, stateful: bool, inputs, cu_seqlens, cotangents):
    """One forward and backward; returns outputs then gradients."""
    chunk = chunk_kda if family == "kda" else chunk_gdn
    options = {"backend": "warpdrive"}
    if family == "kda":
        options["schedule"] = "persistent"
    operands = inputs if stateful else inputs[:5]
    output, final_state = chunk(
        *operands, cu_seqlens=cu_seqlens, output_final_state=stateful, kernel_options=options
    )
    results = (output,) if final_state is None else (output, final_state)
    grads = torch.autograd.grad(results, operands, cotangents[: len(results)])
    return (*results, *grads)


def _assign(destinations, generator: torch.Generator, family: str) -> None:
    with torch.no_grad():
        for destination, source in zip(destinations, _inputs(family, generator), strict=True):
            destination.copy_(source)


@pytest.mark.parametrize("stateful", [False, True], ids=["no-state", "state"])
@pytest.mark.parametrize("family", ["gdn", "kda"])
def test_replay_at_low_occupancy_matches_eager(family: str, stateful: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    inputs = tuple(t.requires_grad_() for t in _inputs(family, generator))
    cotangents = (
        torch.randn(1, CAPACITY, HEADS, DIM, device="cuda", generator=generator).bfloat16(),
        torch.randn(len(CAPTURE_LENGTHS), HEADS, DIM, DIM, device="cuda", generator=generator),
    )
    cu_seqlens = cumulative_sequence_offsets(CAPTURE_LENGTHS)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _step(family, stateful, inputs, cu_seqlens, cotangents)  # compile before capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _step(family, stateful, inputs, cu_seqlens, cotangents)

    _assign(inputs, generator, family)
    cu_seqlens.copy_(cumulative_sequence_offsets(REPLAY_LENGTHS))
    graph.replay()
    expected = _step(family, stateful, inputs, cu_seqlens, cotangents)

    terminal = sum(REPLAY_LENGTHS)
    for index, (actual, reference) in enumerate(zip(captured, expected, strict=True)):
        if actual.shape[:2] == (1, CAPACITY):  # token tensors are defined up to the terminal
            actual, reference = actual[:, :terminal], reference[:, :terminal]
        assert torch.isfinite(actual).all(), index
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=f"result {index}")
