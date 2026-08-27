"""Fused chunk_gdn (KDA-pipeline adapter) against the eager reference, forward and backward."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

from attn_gym.linear import chunk_gdn
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.kda import assert_matches_low_precision_reference, clone_kda_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused chunk pipeline requires SM100/SM103",
)

_KV_DIM = 128


def make_inputs(
    *,
    batch: int = 2,
    tokens: int = 96,
    heads: int = 2,
    key_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Create stable K=V=128 chunk inputs; ``key_heads`` enables grouped heads."""
    torch.manual_seed(seed)
    q_heads = heads if key_heads is None else key_heads
    q = F.normalize(torch.randn(batch, tokens, q_heads, _KV_DIM, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, tokens, q_heads, _KV_DIM, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, tokens, heads, _KV_DIM, device="cuda", dtype=dtype)
    gate = F.logsigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, _KV_DIM, _KV_DIM, device="cuda")
    tensors = (q, k, v, gate, beta, state)
    if requires_grad:
        tensors = tuple(tensor.clone().requires_grad_() for tensor in tensors)
    return tensors


@pytest.mark.parametrize("key_heads", [None, 1])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_fused_chunk_matches_reference(key_heads: int | None, use_initial_state: bool):
    q, k, v, gate, beta, state = make_inputs(key_heads=key_heads)
    state = state if use_initial_state else None
    doubles = [tensor.double() for tensor in (q, k, v, gate, beta)]
    golden = chunk_gdn(
        *doubles,
        None if state is None else state.double(),
        output_final_state=True,
        impl="reference",
    )
    expected = chunk_gdn(q, k, v, gate, beta, state, output_final_state=True, impl="reference")
    actual = chunk_gdn(q, k, v, gate, beta, state, output_final_state=True, impl="fused")

    assert actual[0].dtype == q.dtype
    assert_matches_low_precision_reference(actual[0], golden[0], expected[0], "output")
    assert_matches_low_precision_reference(actual[1], golden[1], expected[1], "final_state")


def test_fused_chunk_packed_matches_reference():
    q, k, v, gate, beta, _ = make_inputs(batch=1, tokens=160)
    cu_seqlens = cumulative_sequence_offsets([64, 0, 96])
    state = torch.randn(3, v.shape[2], _KV_DIM, _KV_DIM, device="cuda")

    golden = chunk_gdn(
        q.double(),
        k.double(),
        v.double(),
        gate.double(),
        beta.double(),
        state.double(),
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    expected = chunk_gdn(
        q, k, v, gate, beta, state, cu_seqlens=cu_seqlens, output_final_state=True
    )
    actual = chunk_gdn(
        q, k, v, gate, beta, state, cu_seqlens=cu_seqlens, output_final_state=True, impl="fused"
    )

    assert_matches_low_precision_reference(actual[0], golden[0], expected[0], "output")
    assert_matches_low_precision_reference(actual[1], golden[1], expected[1], "final_state")


@pytest.mark.parametrize(("key_heads", "scale"), [(None, None), (1, None), (1, 0.25)])
def test_fused_chunk_backward_matches_reference(key_heads: int | None, scale: float | None):
    """Autograd reduces dgate over K and dq/dk over value-head groups automatically."""
    reference_leaves = make_inputs(key_heads=key_heads, tokens=64, requires_grad=True)
    golden_leaves = clone_kda_inputs(tuple(tensor.double() for tensor in reference_leaves))
    fused_leaves = clone_kda_inputs(reference_leaves)

    def run(leaves: tuple[torch.Tensor, ...], impl: str) -> None:
        q, k, v, gate, beta, state = leaves
        output, final_state = chunk_gdn(
            q, k, v, gate, beta, state, scale=scale, output_final_state=True, impl=impl
        )
        (output.float().square().mean() + final_state.float().square().mean()).backward()

    run(golden_leaves, "reference")
    run(reference_leaves, "reference")
    run(fused_leaves, "fused")

    names = ("q", "k", "v", "gate", "beta", "initial_state")
    for name, golden, reference, fused in zip(
        names, golden_leaves, reference_leaves, fused_leaves
    ):
        assert fused.grad is not None, f"missing {name} gradient"
        assert fused.grad.shape == reference.grad.shape
        assert_matches_low_precision_reference(fused.grad, golden.grad, reference.grad, f"d{name}")


def test_fused_chunk_large_scale_override_stays_finite_in_fp16():
    """The kernels apply scale in FP32; folding ``scale * sqrt(K)`` into FP16 q would inf.

    The one-hot query row is the worst case: its unit component times ``6000 * sqrt(128)``
    exceeds the FP16 maximum, while the true scaled output stays comfortably finite.
    """
    q, k, v, gate, beta, _ = make_inputs(tokens=64, dtype=torch.float16)
    q[0, 0, 0] = 0.0
    q[0, 0, 0, 0] = 1.0
    v = v * 1e-2
    scale = 6000.0

    expected = chunk_gdn(q, k, v, gate, beta, scale=scale, impl="reference")[0]
    actual = chunk_gdn(q, k, v, gate, beta, scale=scale, impl="fused")[0]
    golden = chunk_gdn(
        q.double(), k.double(), v.double(), gate.double(), beta.double(), scale=scale
    )[0]

    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    assert_matches_low_precision_reference(
        actual, golden, expected, "output", source_dtype=torch.float16
    )


def test_fused_chunk_accepts_float64_gate_and_beta():
    """GDN permits independent floating gate/beta dtypes; the adapter casts for KDA."""
    q, k, v, gate, beta, _ = make_inputs(tokens=64)
    expected = chunk_gdn(q, k, v, gate, beta, impl="fused")[0]
    actual = chunk_gdn(q, k, v, gate.double(), beta.double(), impl="fused")[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_chunk_fullgraph_forward_and_backward():
    """The adapter is torch-only composition around registered ops; fullgraph must hold."""
    eager_inputs = make_inputs(key_heads=1, tokens=64, requires_grad=True)
    compiled_inputs = clone_kda_inputs(eager_inputs)

    def operation(*inputs):
        return chunk_gdn(*inputs, output_final_state=True, impl="fused")

    expected = operation(*eager_inputs)
    actual = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    cotangents = tuple(torch.randn_like(tensor) for tensor in expected)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    expected_grads = torch.autograd.grad(expected, eager_inputs, cotangents)
    actual_grads = torch.autograd.grad(actual, compiled_inputs, cotangents)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_fused_chunk_rejects_unsupported_head_dims():
    q, k, v, gate, beta, _ = make_inputs(tokens=64)
    with pytest.raises(ValueError, match="K=V=128"):
        chunk_gdn(q[..., :64], k[..., :64], v, gate, beta, impl="fused")
