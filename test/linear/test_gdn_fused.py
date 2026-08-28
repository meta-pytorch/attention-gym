import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import Impl, recurrent_gdn
from attn_gym.linear.gdn.ops import (
    recurrent_fwd_no_state_op,
    recurrent_fwd_op,
    recurrent_fwd_paged_op,
)
from attn_gym.testing import cumulative_sequence_offsets, strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused recurrent scan requires CUDA"
)


def make_inputs(
    *,
    batch: int = 2,
    tokens: int = 9,
    heads: int = 2,
    key_dim: int = 32,
    value_dim: int = 24,
    key_heads: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Create stable token-major scalar-gate inputs; ``key_heads`` enables grouped heads."""
    torch.manual_seed(0)
    q_heads = heads if key_heads is None else key_heads
    q = torch.randn(batch, tokens, q_heads, key_dim, device="cuda")
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda")
    gate = F.logsigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, value_dim, key_dim, device="cuda")
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_fused_recurrent_matches_reference(use_initial_state: bool, output_final_state: bool):
    inputs = make_inputs()
    state = inputs[-1] if use_initial_state else None
    with torch.no_grad():
        expected = recurrent_gdn(
            *inputs[:-1],
            state,
            output_final_state=output_final_state,
            impl=Impl.REFERENCE,
        )
        actual = recurrent_gdn(
            *inputs[:-1],
            state,
            output_final_state=output_final_state,
            autotune=False,
            impl=Impl.FUSED,
        )

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    if output_final_state:
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)
    else:
        assert actual[1] is expected[1] is None


@pytest.mark.parametrize("key_heads", [None, 2])
def test_fused_recurrent_matches_packed_reference(key_heads: int | None):
    q, k, v, gate, beta, _state = make_inputs(
        batch=1, tokens=8, heads=6 if key_heads else 2, key_heads=key_heads
    )
    cu_seqlens = cumulative_sequence_offsets([3, 0, 4])
    state = torch.randn(3, v.shape[2], v.shape[-1], q.shape[3], device="cuda")
    with torch.no_grad():
        expected = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl=Impl.REFERENCE,
        )
        actual = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            autotune=False,
            impl=Impl.FUSED,
        )

    # Rows beyond the terminal offset are inactive capacity and have unspecified values.
    torch.testing.assert_close(actual[0][:, :7], expected[0][:, :7], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_fused_recurrent_paged_state():
    q, k, v, gate, beta, _state = make_inputs(batch=3, tokens=5)
    slots = torch.tensor([2, 0, 4], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, False], device="cuda")
    _storage, state_cache = strided_state_pool(6, q.shape[2], q.shape[-1], v.shape[-1])
    original_cache = state_cache.clone()

    expected_output = torch.zeros_like(v)
    expected_cache = original_cache.clone()
    with torch.no_grad():
        for sequence, slot in ((0, 2), (2, 4)):
            initial_state = (
                original_cache[slot].unsqueeze(0) if has_initial_state[sequence] else None
            )
            output, final_state = recurrent_gdn(
                q[sequence : sequence + 1],
                k[sequence : sequence + 1],
                v[sequence : sequence + 1],
                gate[sequence : sequence + 1],
                beta[sequence : sequence + 1],
                initial_state,
                output_final_state=True,
                impl=Impl.REFERENCE,
            )
            expected_output[sequence] = output[0]
            expected_cache[slot] = final_state[0]

        output, final_state = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices=slots,
            has_initial_state=has_initial_state,
            impl=Impl.FUSED,
        )

    assert final_state is None
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_cache, expected_cache, rtol=1e-5, atol=1e-5)


def test_fused_recurrent_packed_paged_state():
    q, k, v, gate, beta, _state = make_inputs(batch=1, tokens=8)
    cu_seqlens = cumulative_sequence_offsets([3, 0, 4])
    slots = torch.tensor([2, 0, 4], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, False], device="cuda")
    state_cache = torch.randn(6, q.shape[2], v.shape[-1], q.shape[-1], device="cuda")
    original_cache = state_cache.clone()
    expected_output = torch.zeros_like(v)
    expected_cache = original_cache.clone()

    with torch.no_grad():
        for begin, end, slot, use_state in (
            (0, 3, 2, True),
            (3, 3, 0, False),
            (3, 7, 4, False),
        ):
            if begin == end or slot <= 0:
                continue
            span = slice(begin, end)
            initial_state = original_cache[slot].unsqueeze(0) if use_state else None
            span_output, span_state = recurrent_gdn(
                q[:, span],
                k[:, span],
                v[:, span],
                gate[:, span],
                beta[:, span],
                initial_state,
                output_final_state=True,
                impl=Impl.REFERENCE,
            )
            expected_output[:, span] = span_output
            expected_cache[slot] = span_state[0]

        output, _ = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            has_initial_state=has_initial_state,
            impl=Impl.FUSED,
        )

    torch.testing.assert_close(output[:, :7], expected_output[:, :7], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_cache, expected_cache, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tokens", [1, 9])
def test_fused_recurrent_grouped_heads_matches_reference(tokens: int):
    """Fewer q/k heads than v heads must match the expanding reference oracle."""
    q, k, v, gate, beta, state = make_inputs(tokens=tokens, heads=6, key_heads=2)
    with torch.no_grad():
        expected = recurrent_gdn(
            q, k, v, gate, beta, state, output_final_state=True, impl=Impl.REFERENCE
        )
        actual = recurrent_gdn(
            q, k, v, gate, beta, state, output_final_state=True, impl=Impl.FUSED
        )
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_fused_recurrent_grouped_heads_paged_matches_dense():
    """Paged grouped-head decode equals the dense fused path on gathered slots."""
    q, k, v, gate, beta, _state = make_inputs(batch=3, tokens=1, heads=6, key_heads=2)
    _storage, pool = strided_state_pool(5, v.shape[2], q.shape[-1], v.shape[-1])
    slots = torch.tensor([1, 3, 4], device="cuda", dtype=torch.int32)
    initial_state = pool[slots.long()].clone()

    with torch.no_grad():
        expected_output, expected_state = recurrent_gdn(
            q, k, v, gate, beta, initial_state, output_final_state=True, impl=Impl.REFERENCE
        )
        output, _ = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            pool,
            state_indices=slots,
            has_initial_state=torch.ones(3, device="cuda", dtype=torch.bool),
            impl=Impl.FUSED,
        )

    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pool[slots.long()], expected_state, rtol=1e-5, atol=1e-5)


def test_fused_recurrent_packed_paged_empty_sequences():
    """Empty packed sequences zero freshly assigned slots and preserve resumed ones."""
    q, k, v, gate, beta, _state = make_inputs(batch=1, tokens=4)
    cu_seqlens = cumulative_sequence_offsets([0, 4, 0])
    slots = torch.tensor([2, 3, 5], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False, False, True], device="cuda")
    state_cache = torch.randn(6, q.shape[2], v.shape[-1], q.shape[-1], device="cuda")
    original_cache = state_cache.clone()

    with torch.no_grad():
        recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            has_initial_state=has_initial_state,
            impl=Impl.FUSED,
        )

    # The empty fresh sequence must initialize its slot to the zero state.
    torch.testing.assert_close(state_cache[2], torch.zeros_like(state_cache[2]))
    # The empty resumed sequence and unselected slots keep their contents.
    preserved = [0, 1, 4, 5]
    torch.testing.assert_close(state_cache[preserved], original_cache[preserved])


def test_fused_recurrent_paged_registration():
    q, k, v, gate, beta, _state = make_inputs(batch=1, tokens=3)
    state_cache = torch.randn(3, q.shape[2], v.shape[-1], q.shape[-1], device="cuda")
    state_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True], device="cuda")
    torch.library.opcheck(
        recurrent_fwd_paged_op,
        (
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            has_initial_state,
            None,
            q.shape[-1] ** -0.5,
        ),
    )


def test_fused_recurrent_grouped_heads_registration():
    """Grouped-head fakes must size the final state from value heads, not query heads."""
    q, k, v, gate, beta, state = make_inputs(batch=1, tokens=3, heads=6, key_heads=2)
    torch.library.opcheck(
        recurrent_fwd_op,
        (
            q,
            k,
            v,
            gate,
            beta,
            state,
            None,
            q.shape[-1] ** -0.5,
            False,
        ),
    )


@pytest.mark.parametrize("output_final_state", [False, True])
def test_fused_recurrent_registration(output_final_state: bool):
    q, k, v, gate, beta, state = make_inputs(batch=1, tokens=3)
    op = recurrent_fwd_op if output_final_state else recurrent_fwd_no_state_op
    torch.library.opcheck(
        op,
        (
            q,
            k,
            v,
            gate,
            beta,
            state,
            None,
            q.shape[-1] ** -0.5,
            False,
        ),
    )


def test_recurrent_defaults_to_fused():
    inputs = make_inputs(batch=1, tokens=3)
    with torch.no_grad():
        output, _ = recurrent_gdn(*inputs)
    assert torch.isfinite(output).all()


def test_fused_recurrent_low_precision():
    inputs = make_inputs(batch=1, tokens=7)
    q, k, v = (tensor.bfloat16() for tensor in inputs[:3])
    with torch.no_grad():
        output, final_state = recurrent_gdn(
            q,
            k,
            v,
            *inputs[3:5],
            inputs[5],
            output_final_state=True,
            autotune=False,
            impl=Impl.FUSED,
        )
    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32
    assert torch.isfinite(output).all() and torch.isfinite(final_state).all()


def test_fused_recurrent_paged_fullgraph():
    q, k, v, gate, beta, _state = make_inputs(batch=1, tokens=3)
    state_cache = torch.randn(3, q.shape[2], v.shape[-1], q.shape[-1], device="cuda")
    expected_cache = state_cache.clone()
    state_indices = torch.tensor([2], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True], device="cuda")

    @torch.compile(fullgraph=True)
    def compiled(q, k, v, gate, beta, state_cache, state_indices, has_initial_state):
        return recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            impl=Impl.FUSED,
        )

    with torch.no_grad():
        expected = recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            expected_cache,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            impl=Impl.FUSED,
        )
        actual = compiled(q, k, v, gate, beta, state_cache, state_indices, has_initial_state)
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1] is expected[1] is None
    torch.testing.assert_close(state_cache, expected_cache)


def test_fused_recurrent_fullgraph():
    inputs = make_inputs(batch=1, tokens=3)

    @torch.compile(fullgraph=True)
    def compiled(*args):
        return recurrent_gdn(
            *args,
            output_final_state=True,
            autotune=False,
            impl=Impl.FUSED,
        )

    with torch.no_grad():
        expected = recurrent_gdn(
            *inputs,
            output_final_state=True,
            autotune=False,
            impl=Impl.FUSED,
        )
        actual = compiled(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
