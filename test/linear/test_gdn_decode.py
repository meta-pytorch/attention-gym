import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_gdn, recurrent_gdn_decode
from attn_gym.linear.gdn.ops import recurrent_fwd_paged_op
from attn_gym.testing import strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_gdn_decode requires CUDA"
)


def make_decode_inputs(
    *,
    batch: int = 3,
    heads: int = 4,
    key_dim: int = 32,
    value_dim: int = 24,
    key_heads: int | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Create one-token decode inputs; ``key_heads`` enables grouped heads."""
    torch.manual_seed(seed)
    q = torch.randn(batch, key_heads or heads, key_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, heads, value_dim, device="cuda", dtype=dtype)
    gate = F.logsigmoid(torch.randn(batch, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, heads, device="cuda"))
    return q, k, v, gate, beta


def fla_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize like the fused kernel: ``x / sqrt(sum(x^2) + 1e-6)``."""
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)


@pytest.mark.parametrize("key_heads", [None, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_matches_reference(key_heads: int | None, dtype: torch.dtype):
    """Fused decode equals the reference recurrence on externally normalized q/k."""
    q, k, v, gate, beta = make_decode_inputs(key_heads=key_heads, dtype=dtype)
    _storage, pool = strided_state_pool(6, v.shape[1], q.shape[-1], v.shape[-1])
    slots = torch.tensor([1, 3, 5], device="cuda", dtype=torch.int32)
    initial_state = pool[slots.long()].transpose(-1, -2).contiguous()

    with torch.no_grad():
        expected, expected_state = recurrent_gdn(
            fla_l2norm(q.float()).to(dtype).unsqueeze(1),
            fla_l2norm(k.float()).to(dtype).unsqueeze(1),
            v.unsqueeze(1),
            gate.unsqueeze(1),
            beta.unsqueeze(1),
            initial_state,
            output_final_state=True,
            impl="reference",
        )
        output = recurrent_gdn_decode(q, k, v, gate, beta, pool, slots)

    if dtype is torch.float32:
        tolerance = {"rtol": 1e-5, "atol": 1e-5}
    else:
        # The kernel normalizes raw bf16 loads in FP32 registers; the external baseline
        # must round the normalized q/k back to bf16, adding one quantization the fused
        # path never performs.
        tolerance = {"rtol": 2e-2, "atol": 2e-3}
    torch.testing.assert_close(output, expected.squeeze(1), **tolerance)
    torch.testing.assert_close(pool[slots.long()], expected_state.transpose(-1, -2), **tolerance)


def test_decode_padding_and_fresh_slots():
    """Nonpositive slots produce zero output; fresh slots start from zero and overwrite."""
    q, k, v, gate, beta = make_decode_inputs()
    _storage, pool = strided_state_pool(6, v.shape[1], q.shape[-1], v.shape[-1])
    original_pool = pool.clone()
    slots = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], device="cuda")

    with torch.no_grad():
        output = recurrent_gdn_decode(
            q, k, v, gate, beta, pool, slots, has_initial_state=has_initial_state
        )
        fresh_pool = torch.zeros_like(original_pool[:2])
        fresh_expected = recurrent_gdn_decode(
            q[1:2],
            k[1:2],
            v[1:2],
            gate[1:2],
            beta[1:2],
            fresh_pool,
            torch.tensor([1], device="cuda", dtype=torch.int32),
        )

    torch.testing.assert_close(output[0], torch.zeros_like(output[0]), rtol=0, atol=0)
    torch.testing.assert_close(output[1:2], fresh_expected)
    # The fresh sequence must also store its resulting state into the selected slot.
    torch.testing.assert_close(pool[2], fresh_pool[1], rtol=1e-5, atol=1e-5)
    preserved = [0, 1, 3, 5]
    torch.testing.assert_close(pool[preserved], original_pool[preserved], rtol=0, atol=0)


def test_decode_qk_l2norm_disabled_matches_recurrent_gdn():
    """With normalization off, decode is exactly the paged recurrent op on one token."""
    q, k, v, gate, beta = make_decode_inputs()
    _storage, pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
    # An identically strided pool keeps the kernel specialization (and thus the exact
    # floating-point schedule) the same across both calls.
    _reference_storage, reference_pool = strided_state_pool(
        5, v.shape[1], q.shape[-1], v.shape[-1]
    )
    reference_pool.copy_(pool)
    slots = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        output = recurrent_gdn_decode(q, k, v, gate, beta, pool, slots, qk_l2norm=False)
        expected = recurrent_gdn(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            gate.unsqueeze(1),
            beta.unsqueeze(1),
            reference_pool,
            state_indices=slots,
            impl="fused",
        )[0]

    torch.testing.assert_close(output, expected.squeeze(1), rtol=0, atol=0)
    torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)


def test_decode_rejects_grad_and_bad_shapes():
    q, k, v, gate, beta = make_decode_inputs()
    _storage, pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
    slots = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match=r"\[B, HK, K\]"):
        recurrent_gdn_decode(q.unsqueeze(1), k, v, gate, beta, pool, slots)
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_gdn_decode(q.clone().requires_grad_(), k, v, gate, beta, pool, slots)


def test_decode_custom_op_registration():
    """Decode lowers to the paged operator; opcheck it with the fused normalization on."""
    q, k, v, gate, beta = make_decode_inputs()
    _storage, pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
    slots = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)
    token_shaped = (tensor.unsqueeze(1) for tensor in (q, k, v, gate.float(), beta.float()))
    torch.library.opcheck(
        recurrent_fwd_paged_op,
        (*token_shaped, pool, slots, None, None, 0.25, True),
    )


def test_decode_fullgraph_and_cuda_graph():
    """The decode op compiles fullgraph and replays under CUDA graph capture."""
    q, k, v, gate, beta = make_decode_inputs()
    _storage, pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
    initial_pool = pool.clone()
    slots = torch.tensor([1, 2, 4], device="cuda", dtype=torch.int32)

    compiled = torch.compile(recurrent_gdn_decode, fullgraph=True)
    with torch.no_grad():
        _eager_storage, eager_pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
        eager_pool.copy_(initial_pool)
        expected = recurrent_gdn_decode(q, k, v, gate, beta, eager_pool, slots)
        actual = compiled(q, k, v, gate, beta, pool, slots)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, eager_pool, rtol=0, atol=0)

    _graph_storage, graph_pool = strided_state_pool(5, v.shape[1], q.shape[-1], v.shape[-1])
    with torch.no_grad():
        graph_pool.copy_(initial_pool)
        recurrent_gdn_decode(q, k, v, gate, beta, graph_pool, slots)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = recurrent_gdn_decode(q, k, v, gate, beta, graph_pool, slots)
        # Reset to the pre-decode snapshot so replay reproduces the eager call.
        graph_pool.copy_(initial_pool)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
