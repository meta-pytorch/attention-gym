import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import Impl, recurrent_gdn
from attn_gym.linear.gdn.ops import recurrent_fwd_no_state_op, recurrent_fwd_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused recurrent scan requires CUDA"
)


def make_inputs(
    *, batch: int = 2, tokens: int = 9, heads: int = 2, key_dim: int = 32, value_dim: int = 24
) -> tuple[torch.Tensor, ...]:
    """Create stable token-major scalar-gate inputs."""
    torch.manual_seed(0)
    q = torch.randn(batch, tokens, heads, key_dim, device="cuda")
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda")
    gate = F.logsigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda")
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


def test_fused_recurrent_matches_packed_reference():
    q, k, v, gate, beta, _state = make_inputs(batch=1, tokens=8)
    cu_seqlens = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    state = torch.randn(3, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
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


@pytest.mark.parametrize("output_final_state", [False, True])
def test_fused_recurrent_registration(output_final_state: bool):
    q, k, v, gate, beta, state = make_inputs(batch=1, tokens=3)
    op = recurrent_fwd_op if output_final_state else recurrent_fwd_no_state_op
    torch.library.opcheck(
        op,
        (q, k, v, gate, beta, state, None, q.shape[-1] ** -0.5, False),
    )


def test_fused_recurrent_default_autotune():
    inputs = make_inputs(batch=1, tokens=3)
    with torch.no_grad():
        output, _ = recurrent_gdn(*inputs, impl=Impl.FUSED)
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
