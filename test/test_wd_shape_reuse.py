"""warpdrive GDN/KDA specialization: changing sizes must not recompile.

AGENTS.md (Kernel Specialization for Dynamic Shapes) keeps token, sequence, and head counts out of
compile keys. With the plan pinned, one compiled host per direction must serve every shape class
(one token, multiples of the 16- and 64-token chunks, other values, chunk edges, empty
sequences), and a dynamic ``torch.compile`` graph must serve changing sizes without recompiling.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN/KDA kernels require an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
DIM = 128
# (packed lengths, heads): one token, chunk multiples, chunk edges, empty sequences.
SHAPE_CLASSES = (
    ((1,), 1),
    ((16,), 2),
    ((64, 64), 2),
    ((15, 0, 17), 3),
    ((63, 65, 1), 1),
    ((100, 100, 100), 2),
)


def _inputs(family: str, tokens: int, heads: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Operands whose keys correlate with their queries.

    Independent unit Q/K give single-token outputs proportional to a near-zero ``q . k``,
    whose BF16-staged cancellation error dominates a one-token comparison.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, heads, DIM)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(size, device="cuda", generator=generator)

    q = F.normalize(randn(*shape), dim=-1)
    k = F.normalize(q + randn(*shape) / DIM**0.5, dim=-1)
    value = randn(*shape)
    gate_shape = shape if family == "kda" else shape[:3]
    gate = -torch.rand(gate_shape, device="cuda", generator=generator) * 0.69
    beta = torch.rand(shape[:3], device="cuda", generator=generator)
    return q.bfloat16(), k.bfloat16(), value.bfloat16(), gate, beta


@pytest.mark.parametrize("family", ["gdn", "kda"])
def test_one_compile_per_direction_serves_every_shape_class(monkeypatch, family: str) -> None:
    from attn_gym.linear._delta_rule.warpdrive.bundles import (
        gdn_warmup_backward,
        gdn_warmup_forward,
        kda_warmup_backward,
        kda_warmup_forward,
    )
    from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda

    # Pin the uncut plan: plans legitimately bucket by shape and select different hosts.
    driver = gdn if family == "gdn" else kda
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    monkeypatch.setattr(driver.ForwardPlan, "build", lambda *_: driver.ForwardPlan(0, 1, num_sm))
    monkeypatch.setattr(driver.BackwardPlan, "build", lambda *_: driver.BackwardPlan(0, num_sm))
    if family == "kda":
        monkeypatch.setattr(driver, "PREP_TILE_FRACTION", 0.0)
    forward, backward = (
        (gdn_warmup_forward, gdn_warmup_backward)
        if family == "gdn"
        else (kda_warmup_forward, kda_warmup_backward)
    )
    compile_fns = (forward._compile_warmup_forward, backward._compile_warmup_backward)
    if family == "gdn":
        # GDN launches its gradient kernel outside the warmup bundle, through two cache layers.
        from attn_gym.linear._delta_rule.warpdrive import gdn_bprop

        compile_fns += (gdn_bprop._compiled, gdn_bprop.compile_gdn_bprop)
    for compile_fn in compile_fns:
        compile_fn.cache_clear()  # count keys from zero; entries reload from disk

    chunk = chunk_gdn if family == "gdn" else chunk_kda
    for index, (lengths, heads) in enumerate(SHAPE_CLASSES):
        inputs = tuple(
            t.requires_grad_() for t in _inputs(family, sum(lengths), heads, seed=100 + index)
        )
        offsets = cumulative_sequence_offsets(lengths)
        output, _ = chunk(*inputs, cu_seqlens=offsets, kernel_options=WARPDRIVE)
        expected, _ = chunk(
            *(t.detach().float() for t in inputs), cu_seqlens=offsets, impl="reference"
        )
        name = f"{family} output {lengths=} {heads=}"
        assert_matches_low_precision_reference(output, expected, expected, name)
        assert_relative_rms_within(
            output, expected, name, max_eps=1e-2 / torch.finfo(torch.bfloat16).eps
        )
        grads = torch.autograd.grad(output, inputs, torch.ones_like(output))
        assert all(torch.isfinite(grad).all() for grad in grads), (lengths, heads)

    infos = [compile_fn.cache_info() for compile_fn in compile_fns]
    assert [info.hits + info.misses for info in infos[:2]] == [len(SHAPE_CLASSES)] * 2
    assert [info.currsize for info in infos] == [1] * len(compile_fns)


def _kda_step(*inputs: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    return chunk_kda(*inputs, cu_seqlens=cu_seqlens, kernel_options=WARPDRIVE)[0]


def _assert_compiled_matches_eager(compiled, inputs, **kwargs) -> None:
    leaves = clone_kda_inputs(inputs, requires_grad=True)
    expected = _kda_step(*leaves, **kwargs)
    expected_grads = torch.autograd.grad(expected, leaves, torch.ones_like(expected))
    leaves = clone_kda_inputs(inputs, requires_grad=True)
    actual = compiled(*leaves, **kwargs)
    actual_grads = torch.autograd.grad(actual, leaves, torch.ones_like(actual))
    for got, want in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


# Four heads never equal a batch or sequence count below, so duck sizing cannot tie them.
DYNAMIC_HEADS = 4


def test_kda_dynamic_packed_tokens_and_sequence_count_compile_once(fresh_compile_cache) -> None:
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_kda_step, backend=counter, fullgraph=True, dynamic=True)
    frames = []
    for index, lengths in enumerate(((64, 32), (15, 0, 81), (100, 37, 1, 64, 30))):
        inputs = make_kda_test_inputs(
            sum(lengths), heads=DYNAMIC_HEADS, seed=200 + index, gate_scale=0.69
        )
        _assert_compiled_matches_eager(
            compiled, inputs, cu_seqlens=cumulative_sequence_offsets(lengths)
        )
        frames.append(counter.frame_count)
    assert frames == frames[:1] * len(frames), frames


def test_kda_dynamic_dense_batch_and_tokens_compile_once(fresh_compile_cache) -> None:
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_kda_step, backend=counter, fullgraph=True, dynamic=True)
    frames = []
    for index, (batch, tokens) in enumerate(((2, 64), (3, 17), (2, 100))):
        inputs = make_kda_test_inputs(
            tokens, batch=batch, heads=DYNAMIC_HEADS, seed=300 + index, gate_scale=0.69
        )
        _assert_compiled_matches_eager(compiled, inputs)
        frames.append(counter.frame_count)
    assert frames == frames[:1] * len(frames), frames
