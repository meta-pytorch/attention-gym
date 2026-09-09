"""Native stateful Mega backward: registration, fragment handoff parity, and reference accuracy."""

from __future__ import annotations

import math

import pytest
import torch

from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 KDA path requires nvidia-cutlass-dsl>=4.7",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 KDA path requires SM100 or SM103",
)

D = 128
SCALE = D**-0.5
OPCHECK_UTILITIES = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
MILD_GATE = 0.05  # Decay per token in [e^-0.05, 1): the entry state still matters 4096 tokens in.


def _ops():
    from attn_gym.linear.kda.impl import mega_ops

    return mega_ops


def _make_inputs(
    lengths: list[int], *, heads: int, gate_scale: float, seed: int
) -> tuple[torch.Tensor, ...]:
    """Public Mega operands, a nonzero FP32 entry state, and an output cotangent per token."""
    q, k, value, gate, beta = make_kda_test_inputs(
        sum(lengths),
        heads=heads,
        seed=seed,
        gate_scale=gate_scale,
        sigmoid_beta=True,
        normalize_qk=True,
        dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    initial_state = torch.randn(len(lengths), heads, D, D, device="cuda", generator=generator)
    d_output = torch.randn(value.shape, device="cuda", generator=generator).to(value.dtype)
    d_final_state = torch.randn_like(initial_state)
    return q, k, value, gate, beta, initial_state, d_output, d_final_state


def _swap_state_head_value_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy `[N,H,V,K]` data into dense TMA-compatible `[N,V,H,K]` storage order."""
    _, heads, value_dim, key_dim = tensor.shape
    storage = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    return torch.as_strided(
        storage,
        tensor.shape,
        (heads * value_dim * key_dim, key_dim, heads * key_dim, 1),
    ).copy_(tensor)


@pytest.mark.parametrize("outer_strided", [False, True], ids=["compact", "outer-strided"])
def test_stateful_backward_op_registration(outer_strided: bool) -> None:
    ops = _ops()
    q, k, value, gate, beta, initial_state, d_output, d_final_state = _make_inputs(
        [65, 0, 63], heads=2, gate_scale=math.log(2.0), seed=97
    )
    cu_seqlens = cumulative_sequence_offsets([65, 0, 63])
    if outer_strided:
        initial_state = _swap_state_head_value_storage(initial_state)
        d_final_state = _swap_state_head_value_storage(d_final_state)
        assert not initial_state.is_contiguous()
    operands = (q, k, value, gate, beta, d_output, cu_seqlens)

    grads = ops.chunk_mega_packed_bwd_with_state_op(*operands, initial_state, d_final_state, SCALE)
    assert len(grads) == 6
    assert grads[5].shape == initial_state.shape and grads[5].is_contiguous()
    assert grads[5].dtype == torch.float32
    torch.library.opcheck(
        ops.chunk_mega_packed_bwd_with_state_op,
        (*operands, initial_state, d_final_state, SCALE),
        test_utils=OPCHECK_UTILITIES,
    )
    torch.library.opcheck(
        ops.chunk_mega_packed_bwd_with_state_op,
        (*operands, None, None, SCALE),
        test_utils=OPCHECK_UTILITIES,
    )
    torch.library.opcheck(
        ops.chunk_mega_packed_bwd_with_state_op,
        (*operands, initial_state, None, SCALE),
        test_utils=OPCHECK_UTILITIES,
    )


def test_stateful_backward_without_state_is_the_local_backward() -> None:
    """``None`` keeps the no-state kernel specializations, so the two ops agree bit for bit."""
    ops = _ops()
    q, k, value, gate, beta, _, d_output, _ = _make_inputs(
        [65, 0, 63], heads=2, gate_scale=math.log(2.0), seed=11
    )
    cu_seqlens = cumulative_sequence_offsets([65, 0, 63])
    *grads, d_initial_state = ops.chunk_mega_packed_bwd_with_state_op(
        q, k, value, gate, beta, d_output, cu_seqlens, None, None, SCALE
    )
    assert d_initial_state is None
    expected = ops.chunk_mega_packed_local_bwd_op(
        q, k, value, gate, beta, d_output, cu_seqlens, False, SCALE
    )
    for actual, reference in zip(grads, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def _slice(tensors: tuple[torch.Tensor, ...], start: int, stop: int) -> tuple[torch.Tensor, ...]:
    return tuple(tensor[:, start:stop].contiguous() for tensor in tensors)


def _fragments(cu_seqlens: list[int], cut: int) -> tuple[list[int], list[int], int]:
    """Split packed boundaries at ``cut``; returns both fragments' local offsets and the cut doc."""
    document = max(index for index, start in enumerate(cu_seqlens[:-1]) if start <= cut)
    assert cu_seqlens[document] < cut < cu_seqlens[document + 1], "cut must be inside a document"
    assert (cut - cu_seqlens[document]) % 16 == 0, "cut must be 16-aligned to the document"
    first = [*cu_seqlens[: document + 1], cut]
    second = [offset - cut for offset in (cut, *cu_seqlens[document + 1 :])]
    return first, second, document


@pytest.mark.parametrize(
    ("lengths", "cut", "heads", "gate_scale"),
    [
        pytest.param([4096], 2064, 64, MILD_GATE, id="h64-t4096-mild"),
        # Documents start at 0, 17, 82, 211, 468, 981; cuts are 16-aligned to their document.
        pytest.param([17, 65, 129, 257, 513, 1025], 211 + 16 * 5, 4, MILD_GATE, id="ragged-mild"),
        pytest.param([17, 65, 129, 257, 513, 1025], 468 + 16 * 3, 4, 5.0, id="ragged-strong"),
    ],
)
def test_two_aligned_fragments_with_state_handoff_match_one_native_call(
    lengths: list[int], cut: int, heads: int, gate_scale: float
) -> None:
    """A document cut at a 16-token offset and continued from the actual states is bit-exact.

    Rank 0 owns ``[0, cut)`` and rank 1 ``[cut, T)``; the cut document's exit state travels
    forward and its entry cotangent travels backward, as the context-parallel recipe does.
    """
    ops = _ops()
    q, k, value, gate, beta, initial_state, d_output, d_final_state = _make_inputs(
        lengths, heads=heads, gate_scale=gate_scale, seed=3
    )
    cu_seqlens = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    tokens = cu_seqlens[-1]
    offsets = cumulative_sequence_offsets(lengths)
    first, second, document = _fragments(cu_seqlens, cut)
    first_offsets = torch.tensor(first, dtype=torch.int32, device="cuda")
    second_offsets = torch.tensor(second, dtype=torch.int32, device="cuda")

    operands = (q, k, value, gate, beta)
    output, final_state = ops.chunk_mega_packed_fwd_with_state_op(
        *operands, initial_state, offsets, SCALE
    )
    expected = ops.chunk_mega_packed_bwd_with_state_op(
        *operands, d_output, offsets, initial_state, d_final_state, SCALE
    )

    head, tail = _slice(operands, 0, cut), _slice(operands, cut, tokens)
    head_output, head_states = ops.chunk_mega_packed_fwd_with_state_op(
        *head, initial_state[: document + 1], first_offsets, SCALE
    )
    tail_entry = torch.cat((head_states[document:], initial_state[document + 1 :]))
    tail_output, tail_states = ops.chunk_mega_packed_fwd_with_state_op(
        *tail, tail_entry, second_offsets, SCALE
    )
    torch.testing.assert_close(torch.cat((head_output, tail_output), 1), output, atol=0, rtol=0)
    torch.testing.assert_close(head_states[:document], final_state[:document], atol=0, rtol=0)
    torch.testing.assert_close(tail_states, final_state[document:], atol=0, rtol=0)

    tail_grads = ops.chunk_mega_packed_bwd_with_state_op(
        *tail,
        d_output[:, cut:].contiguous(),
        second_offsets,
        tail_entry,
        d_final_state[document:],
        SCALE,
    )
    head_exit = torch.cat((d_final_state[:document], tail_grads[5][:1]))
    head_grads = ops.chunk_mega_packed_bwd_with_state_op(
        *head,
        d_output[:, :cut].contiguous(),
        first_offsets,
        initial_state[: document + 1],
        head_exit,
        SCALE,
    )
    for name, head_grad, tail_grad, reference in zip(
        ("dq", "dk", "dv", "dgate", "dbeta"),
        head_grads[:5],
        tail_grads[:5],
        expected[:5],
        strict=True,
    ):
        torch.testing.assert_close(
            torch.cat((head_grad, tail_grad), 1), reference, atol=0, rtol=0, msg=name
        )
    d_initial_state = torch.cat((head_grads[5], tail_grads[5][1:]))
    torch.testing.assert_close(d_initial_state, expected[5], atol=0, rtol=0)


def test_staged_mega_backward_is_the_native_op() -> None:
    """``chunk_kda_prepare_backward`` on a Mega tape runs the stateful op, not the fused recompute."""
    from attn_gym.linear.kda.stages import (
        ChunkKDAMegaBackward,
        ChunkKDAMegaSaved,
        chunk_kda_prepare,
        chunk_kda_prepare_backward,
    )

    ops = _ops()
    lengths = [100, 156]
    q, k, value, gate, beta, initial_state, d_output, d_final_state = _make_inputs(
        lengths, heads=2, gate_scale=MILD_GATE, seed=5
    )
    offsets = cumulative_sequence_offsets(lengths)
    prepared = chunk_kda_prepare(
        q, k, value, gate, beta, cu_seqlens=offsets, kernel_options={"backend": "mega"}
    )
    assert isinstance(prepared.saved, ChunkKDAMegaSaved)
    prepared.run(initial_state, output_final_state=True)
    grads = chunk_kda_prepare_backward(
        prepared.saved, d_output, initial_state, scale=prepared.scale
    )
    assert isinstance(grads, ChunkKDAMegaBackward)
    expected = ops.chunk_mega_packed_bwd_with_state_op(
        q, k, value, gate, beta, d_output, offsets, initial_state, d_final_state, SCALE
    )
    for actual, reference in zip(grads.run(d_final_state), expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    # The reverse summaries still come from the fused factors, so they match the fused handle.
    fused = chunk_kda_prepare(q, k, value, gate, beta, cu_seqlens=offsets)
    fused_grads = chunk_kda_prepare_backward(
        fused.saved, d_output, initial_state, scale=fused.scale
    )
    bounds = torch.tensor([[0, 100], [100, 256]], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(
        grads.state_grad_summaries(bounds),
        fused_grads.state_grad_summaries(bounds),
        atol=0,
        rtol=0,
    )
    # Without an entry state the tape's backward keeps Mega's no-state arithmetic.
    no_state = chunk_kda_prepare_backward(prepared.saved, d_output, None, scale=prepared.scale)
    *no_state_grads, d_initial_state = no_state.run(None)
    assert d_initial_state is None
    local = ops.chunk_mega_packed_local_bwd_op(
        q, k, value, gate, beta, d_output, offsets, False, SCALE
    )
    for actual, reference in zip(no_state_grads, local, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_stateful_backward_matches_fp64_reference_including_d_initial_state() -> None:
    """All six gradients, including the entry-state cotangent, sit within the reference budget."""
    ops = _ops()
    lengths = [100, 156]
    q, k, value, gate, beta, initial_state, d_output, d_final_state = _make_inputs(
        lengths, heads=1, gate_scale=MILD_GATE, seed=7
    )
    offsets = cumulative_sequence_offsets(lengths)
    actual = ops.chunk_mega_packed_bwd_with_state_op(
        q, k, value, gate, beta, d_output, offsets, initial_state, d_final_state, SCALE
    )

    def reference(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        leaves = tuple(
            tensor.detach().to(dtype).requires_grad_()
            for tensor in (q, k, value, gate, beta, initial_state)
        )
        output, final_state = kda_reference(
            *leaves, cu_seqlens=offsets, scale=SCALE, output_final_state=True
        )
        assert final_state is not None
        return torch.autograd.grad(
            (output, final_state), leaves, (d_output.to(dtype), d_final_state.to(dtype))
        )

    high, low = reference(torch.float64), reference(torch.float32)
    names = ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")
    for name, actual_grad, high_grad, low_grad in zip(names, actual, high, low, strict=True):
        assert_matches_low_precision_reference(actual_grad, high_grad, low_grad, name)
    # The entry state must actually reach the gradient: a dead state makes the check vacuous.
    assert high[5].abs().max() > 1e-3
