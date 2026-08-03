"""Tests for selected_attention primitive edge cases and semantics.

Covers: local-only, selected-block-only, joint normalization,
attention sink behavior, and repeated selections.
Tests both eager (reference) and triton backends.
"""

import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention


BACKENDS = ["eager"]
if torch.cuda.is_available():
    BACKENDS.append("triton")


def _device_for_backend(backend):
    return torch.device("cuda") if backend == "triton" else torch.device("cpu")


def _dtype_for_backend(backend):
    return torch.float32


# ---------------------------------------------------------------------------
# 1. Local-only attention (topk=0, no index blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_only_attention(backend):
    """With topk=0, output depends only on the local sliding window + sink."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 3

    torch.manual_seed(0)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, window, False, backend=backend)

    # Output should be a weighted combination of local KV only.
    # Changing index_kv should have zero effect.
    index_kv2 = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    out2 = selected_attention(
        Q, KV, index_kv2, indices, sink, None, window, False, backend=backend
    )
    torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)

    # Output shape is correct
    assert out.shape == (b, h, s, d)


@pytest.mark.parametrize("backend", BACKENDS)
def test_local_only_matches_manual_computation(backend):
    """Local-only with sink=0 should match standard causal sliding window softmax."""
    device = _device_for_backend(backend)
    dtype = torch.float64 if backend == "eager" else torch.float32
    b, h, s, d = 1, 1, 6, 8
    window = 3

    torch.manual_seed(42)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, 2, d, device=device, dtype=dtype)
    indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, window, False, backend=backend)

    # Manual: build causal sliding window mask, compute attention
    scale = d**0.5
    scores = (Q @ KV.transpose(-2, -1)) / scale  # (1,1,6,6)
    # Build mask
    pos_q = torch.arange(s, device=device)[:, None]
    pos_k = torch.arange(s, device=device)[None, :]
    valid = (pos_k <= pos_q) & (pos_k >= pos_q - window + 1)
    mask = torch.where(valid, 0.0, float("-inf")).to(dtype)
    scores = scores + mask[None, None, :, :]
    # Sink softmax with sink=0: denominator adds exp(0)=1
    max_s = scores.max(dim=-1, keepdim=True).values
    max_s = torch.maximum(max_s, torch.zeros_like(max_s))  # sink=0
    exp_scores = torch.exp(scores - max_s)
    exp_sink = torch.exp(torch.zeros_like(max_s) - max_s)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)
    expected = probs @ KV

    atol = 1e-5 if backend == "triton" else 1e-10
    torch.testing.assert_close(out, expected, atol=atol, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Selected-block-only attention (window=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_block_only(backend):
    """With window=0, output depends only on the selected index blocks + sink."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    index_seq_len = 6
    topk = 2

    torch.manual_seed(7)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, index_seq_len, device=device)
    _, indices = torch.topk(scores, k=topk, dim=-1)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, 0, False, backend=backend)

    # Changing local KV should have zero effect when window=0
    KV2 = torch.randn(b, h, s, d, device=device, dtype=dtype)
    out2 = selected_attention(Q, KV2, index_kv, indices, sink, None, 0, False, backend=backend)
    torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)

    assert out.shape == (b, h, s, d)


@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_block_only_manual(backend):
    """Selected-block-only with sink=0 should match manual gather + softmax."""
    device = _device_for_backend(backend)
    dtype = torch.float64 if backend == "eager" else torch.float32
    b, h, s, d = 1, 1, 4, 8
    index_seq_len = 6

    torch.manual_seed(11)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype)
    # Fixed indices for manual computation
    indices = torch.tensor([[[0, 2, 4], [1, 3, 5], [0, 1, 2], [3, 4, 5]]], device=device)

    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, 0, False, backend=backend)

    # Manual computation: for each query, gather the selected index_kv positions
    scale = d**0.5
    # index_kv is (1, 1, 6, 8), indices is (1, 4, 3)
    # For each query position, gather 3 vectors from index_kv
    gathered = index_kv[0, 0, indices[0]]  # (4, 3, 8)
    q = Q[0, 0]  # (4, 8)
    scores_manual = (
        torch.bmm(q.unsqueeze(1), gathered.transpose(-2, -1)).squeeze(1) / scale
    )  # (4, 3)
    # Sink softmax with sink=0
    max_s = scores_manual.max(dim=-1, keepdim=True).values
    max_s = torch.maximum(max_s, torch.zeros_like(max_s))
    exp_s = torch.exp(scores_manual - max_s)
    exp_sink = torch.exp(-max_s)
    probs = exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink)
    expected = torch.bmm(probs.unsqueeze(1), gathered).squeeze(1)  # (4, 8)

    atol = 1e-4 if backend == "triton" else 1e-10
    torch.testing.assert_close(out[0, 0], expected, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. Joint normalization (both branches share the same softmax denominator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_joint_normalization(backend):
    """When both local and index branches are active, they share normalization.

    Adding index blocks should change the output even for positions that are
    within the local window (because the denominator changes).
    """
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 4

    torch.manual_seed(55)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, 6, d, device=device, dtype=dtype)
    sink = torch.zeros(h, device=device, dtype=dtype)

    # Local-only
    indices_empty = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    out_local = selected_attention(
        Q, KV, index_kv, indices_empty, sink, None, window, False, backend=backend
    )

    # With index blocks
    scores = torch.randn(b, s, 6, device=device)
    _, indices = torch.topk(scores, k=2, dim=-1)
    out_joint = selected_attention(
        Q, KV, index_kv, indices, sink, None, window, False, backend=backend
    )

    # They should differ (joint normalization changes the local contribution)
    assert not torch.allclose(
        out_local, out_joint, atol=1e-3
    ), "Adding index blocks should change output due to shared normalization"


# ---------------------------------------------------------------------------
# 4. Attention sink behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sink_large_absorbs_probability(backend):
    """As sink → +∞, all probability goes to sink, output → 0."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 4

    torch.manual_seed(33)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, 4, device=device)
    _, indices = torch.topk(scores, k=2, dim=-1)

    # Very large positive sink
    sink = torch.full((h,), 50.0, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, window, False, backend=backend)

    # Output should be near zero (sink absorbs essentially all probability)
    assert (
        out.abs().max().item() < 0.01
    ), f"With large sink, output should be near zero, got max={out.abs().max().item()}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sink_very_negative_has_no_effect(backend):
    """With sink → -∞, the sink contributes nothing to the denominator.

    Output should match standard softmax (no sink term in denominator).
    """
    device = _device_for_backend(backend)
    dtype = torch.float64 if backend == "eager" else torch.float32
    b, h, s, d = 1, 1, 6, 8
    window = 3

    torch.manual_seed(99)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, 2, d, device=device, dtype=dtype)
    indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)

    # Very negative sink (contributes nothing)
    sink = torch.full((h,), -100.0, device=device, dtype=dtype)

    out = selected_attention(Q, KV, index_kv, indices, sink, None, window, False, backend=backend)

    # Manual standard softmax (no sink in denominator)
    scale = d**0.5
    logits = (Q @ KV.transpose(-2, -1)) / scale
    pos_q = torch.arange(s, device=device)[:, None]
    pos_k = torch.arange(s, device=device)[None, :]
    valid = (pos_k <= pos_q) & (pos_k >= pos_q - window + 1)
    mask = torch.where(valid, 0.0, float("-inf")).to(dtype)
    logits = logits + mask[None, None, :, :]
    probs = torch.softmax(logits, dim=-1)
    expected = probs @ KV

    atol = 1e-4 if backend == "triton" else 1e-6
    torch.testing.assert_close(out, expected, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------------------
# 5. Repeated selections (backends must agree)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
@pytest.mark.parametrize("num_repeats", [2, 3])
@pytest.mark.parametrize("sliding_window_size", [0, 4])
def test_repeated_indices_backends_match(num_repeats, sliding_window_size):
    """Repeated indices should produce identical results in eager and triton."""
    device = torch.device("cuda")
    dtype = torch.float32
    b, h, s, d = 1, 2, 8, 16
    index_seq_len = 6

    torch.manual_seed(88)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    # All slots repeat position 2
    indices = torch.full((b, s, num_repeats), 2, dtype=torch.long, device=device)

    out_eager = selected_attention(
        Q, KV, index_kv, indices, sink, None, sliding_window_size, False, backend="eager"
    )
    out_triton = selected_attention(
        Q, KV, index_kv, indices, sink, None, sliding_window_size, False, backend="triton"
    )

    torch.testing.assert_close(out_eager, out_triton, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
def test_mixed_repeated_and_unique_indices_backends_match():
    """Mix of repeated and unique indices should match between backends."""
    device = torch.device("cuda")
    dtype = torch.float32
    b, h, s, d = 1, 2, 6, 16
    index_seq_len = 5

    torch.manual_seed(99)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    # Mix: some rows have repeats, some are unique, some have -1 sentinels
    indices = torch.tensor(
        [[[0, 0, 1], [2, 2, 2], [0, 1, 2], [3, 3, -1], [-1, -1, -1], [4, 4, 4]]],
        dtype=torch.long,
        device=device,
    )

    out_eager = selected_attention(Q, KV, index_kv, indices, sink, None, 3, False, backend="eager")
    out_triton = selected_attention(
        Q, KV, index_kv, indices, sink, None, 3, False, backend="triton"
    )

    torch.testing.assert_close(out_eager, out_triton, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. torch.compile fullgraph compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_torch_compile_fullgraph_forward(backend):
    """selected_attention compiles with torch.compile(fullgraph=True)."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 3
    index_seq_len = 4
    topk = 2

    torch.manual_seed(0)
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    KV = torch.randn(b, h, s, d, device=device, dtype=dtype)
    index_kv = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, index_seq_len, device=device)
    _, indices = torch.topk(scores, k=topk, dim=-1)
    sink = torch.randn(h, device=device, dtype=dtype)

    def fn(Q, KV, index_kv, indices, sink):
        return selected_attention(
            Q, KV, index_kv, indices, sink, None, window, False, backend=backend
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    with torch.inference_mode():
        expected = fn(Q, KV, index_kv, indices, sink)
        actual = compiled_fn(Q, KV, index_kv, indices, sink)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_torch_compile_fullgraph_backward(backend):
    """selected_attention backward works under torch.compile(fullgraph=True)."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 3
    index_seq_len = 4
    topk = 2

    torch.manual_seed(0)
    scores = torch.randn(b, s, index_seq_len, device=device)
    _, indices = torch.topk(scores, k=topk, dim=-1)

    def fn(Q, KV, index_kv, sink):
        return selected_attention(
            Q, KV, index_kv, indices, sink, None, window, False, backend=backend
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    # Eager reference
    Q_ref = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    KV_ref = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    idx_ref = torch.randn(b, h, index_seq_len, d, device=device, dtype=dtype, requires_grad=True)
    sink_ref = torch.randn(h, device=device, dtype=dtype, requires_grad=True)

    out_ref = fn(Q_ref, KV_ref, idx_ref, sink_ref)
    out_ref.sum().backward()

    # Compiled
    Q_c = Q_ref.detach().clone().requires_grad_(True)
    KV_c = KV_ref.detach().clone().requires_grad_(True)
    idx_c = idx_ref.detach().clone().requires_grad_(True)
    sink_c = sink_ref.detach().clone().requires_grad_(True)

    out_c = compiled_fn(Q_c, KV_c, idx_c, sink_c)
    out_c.sum().backward()

    torch.testing.assert_close(out_c, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(Q_c.grad, Q_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(KV_c.grad, KV_ref.grad, atol=1e-5, rtol=1e-5)
