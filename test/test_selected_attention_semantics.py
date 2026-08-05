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
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    kv_indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # Output should be a weighted combination of local KV only.
    # Changing sparse_kv should have zero effect.
    sparse_kv2 = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    out2 = selected_attention(
        query, local_kv, sparse_kv2, kv_indices, sink, None, window, backend=backend
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
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 2, d, device=device, dtype=dtype)
    kv_indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # Manual: build causal sliding window mask, compute attention
    scale = d**0.5
    scores = (query @ local_kv.transpose(-2, -1)) / scale  # (1,1,6,6)
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
    expected = probs @ local_kv

    atol = 1e-5 if backend == "triton" else 1e-10
    torch.testing.assert_close(out, expected, atol=atol, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Selected-block-only attention (window=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_block_only(backend):
    """With window=0, output depends only on the selected sparse blocks + sink."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    sparse_seq_len = 6
    topk = 2

    torch.manual_seed(7)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, sparse_seq_len, device=device)
    _, kv_indices = torch.topk(scores, k=topk, dim=-1)
    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, 0, backend=backend
    )

    # Changing local KV should have zero effect when window=0
    local_kv2 = torch.randn(b, h, s, d, device=device, dtype=dtype)
    out2 = selected_attention(
        query, local_kv2, sparse_kv, kv_indices, sink, None, 0, backend=backend
    )
    torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)

    assert out.shape == (b, h, s, d)


@pytest.mark.parametrize("backend", BACKENDS)
def test_selected_block_only_manual(backend):
    """Selected-block-only with sink=0 should match manual gather + softmax."""
    device = _device_for_backend(backend)
    dtype = torch.float64 if backend == "eager" else torch.float32
    b, h, s, d = 1, 1, 4, 8
    sparse_seq_len = 6

    torch.manual_seed(11)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    # Fixed indices for manual computation
    kv_indices = torch.tensor([[[0, 2, 4], [1, 3, 5], [0, 1, 2], [3, 4, 5]]], device=device)

    sink = torch.zeros(h, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, 0, backend=backend
    )

    # Manual computation: for each query, gather the selected sparse_kv positions
    scale = d**0.5
    # sparse_kv is (1, 1, 6, 8), kv_indices is (1, 4, 3)
    # For each query position, gather 3 vectors from sparse_kv
    gathered = sparse_kv[0, 0, kv_indices[0]]  # (4, 3, 8)
    q = query[0, 0]  # (4, 8)
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
    """When both local and sparse branches are active, they share normalization.

    Adding sparse blocks should change the output even for positions that are
    within the local window (because the denominator changes).
    """
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 4

    torch.manual_seed(55)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 6, d, device=device, dtype=dtype)
    sink = torch.zeros(h, device=device, dtype=dtype)

    # Local-only
    kv_indices_empty = torch.zeros(b, s, 0, dtype=torch.long, device=device)
    out_local = selected_attention(
        query, local_kv, sparse_kv, kv_indices_empty, sink, None, window, backend=backend
    )

    # With sparse blocks
    scores = torch.randn(b, s, 6, device=device)
    _, kv_indices = torch.topk(scores, k=2, dim=-1)
    out_joint = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # They should differ (joint normalization changes the local contribution)
    assert not torch.allclose(
        out_local, out_joint, atol=1e-3
    ), "Adding sparse blocks should change output due to shared normalization"


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
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 4, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, 4, device=device)
    _, kv_indices = torch.topk(scores, k=2, dim=-1)

    # Very large positive sink
    sink = torch.full((h,), 50.0, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

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
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 2, d, device=device, dtype=dtype)
    kv_indices = torch.zeros(b, s, 0, dtype=torch.long, device=device)

    # Very negative sink (contributes nothing)
    sink = torch.full((h,), -100.0, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # Manual standard softmax (no sink in denominator)
    scale = d**0.5
    logits = (query @ local_kv.transpose(-2, -1)) / scale
    pos_q = torch.arange(s, device=device)[:, None]
    pos_k = torch.arange(s, device=device)[None, :]
    valid = (pos_k <= pos_q) & (pos_k >= pos_q - window + 1)
    mask = torch.where(valid, 0.0, float("-inf")).to(dtype)
    logits = logits + mask[None, None, :, :]
    probs = torch.softmax(logits, dim=-1)
    expected = probs @ local_kv

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
    sparse_seq_len = 6

    torch.manual_seed(0)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    # All slots repeat position 2
    kv_indices = torch.full((b, s, num_repeats), 2, dtype=torch.long, device=device)

    out_eager = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="eager"
    )
    out_triton = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="triton"
    )

    torch.testing.assert_close(out_eager, out_triton, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
def test_mixed_repeated_and_unique_indices_backends_match():
    """Mix of repeated and unique indices should match between backends."""
    device = torch.device("cuda")
    dtype = torch.float32
    b, h, s, d = 1, 2, 6, 16
    sparse_seq_len = 5

    torch.manual_seed(99)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    # Mix: some rows have repeats, some are unique, some have -1 sentinels
    kv_indices = torch.tensor(
        [[[0, 0, 1], [2, 2, 2], [0, 1, 2], [3, 3, -1], [-1, -1, -1], [4, 4, 4]]],
        dtype=torch.long,
        device=device,
    )

    out_eager = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, 3, backend="eager"
    )
    out_triton = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, 3, backend="triton"
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
    sparse_seq_len = 4
    topk = 2

    torch.manual_seed(0)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    scores = torch.randn(b, s, sparse_seq_len, device=device)
    _, kv_indices = torch.topk(scores, k=topk, dim=-1)
    sink = torch.randn(h, device=device, dtype=dtype)

    def fn(query, local_kv, sparse_kv, kv_indices, sink):
        return selected_attention(
            query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    with torch.inference_mode():
        expected = fn(query, local_kv, sparse_kv, kv_indices, sink)
        actual = compiled_fn(query, local_kv, sparse_kv, kv_indices, sink)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_torch_compile_fullgraph_backward(backend):
    """selected_attention backward works under torch.compile(fullgraph=True)."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 2, 8, 16
    window = 3
    sparse_seq_len = 4
    topk = 2

    torch.manual_seed(0)
    scores = torch.randn(b, s, sparse_seq_len, device=device)
    _, kv_indices = torch.topk(scores, k=topk, dim=-1)

    def fn(query, local_kv, sparse_kv, sink):
        return selected_attention(
            query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
        )

    compiled_fn = torch.compile(fn, fullgraph=True)

    # Eager reference
    query_ref = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    local_kv_ref = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    sparse_kv_ref = torch.randn(
        b, h, sparse_seq_len, d, device=device, dtype=dtype, requires_grad=True
    )
    sink_ref = torch.randn(h, device=device, dtype=dtype, requires_grad=True)

    out_ref = fn(query_ref, local_kv_ref, sparse_kv_ref, sink_ref)
    out_ref.sum().backward()

    # Compiled
    query_c = query_ref.detach().clone().requires_grad_(True)
    local_kv_c = local_kv_ref.detach().clone().requires_grad_(True)
    sparse_kv_c = sparse_kv_ref.detach().clone().requires_grad_(True)
    sink_c = sink_ref.detach().clone().requires_grad_(True)

    out_c = compiled_fn(query_c, local_kv_c, sparse_kv_c, sink_c)
    out_c.sum().backward()

    torch.testing.assert_close(out_c, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(query_c.grad, query_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local_kv_c.grad, local_kv_ref.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Repeated selections against manual computation
# ---------------------------------------------------------------------------


def test_repeated_indices_manual_forward_and_backward():
    """Verify repeated-index semantics against a manual computation.
    This test builds that manually and checks both forward and backward.
    """
    dtype = torch.float64
    b, h, s, d = 1, 1, 3, 4
    sparse_seq_len = 4
    num_topk = 3

    torch.manual_seed(77)
    query = torch.randn(b, h, s, d, dtype=dtype, requires_grad=True)
    local_kv = torch.randn(b, h, s, d, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, dtype=dtype, requires_grad=True)
    sink = torch.full((h,), float("-inf"), dtype=dtype)

    # kv_indices: query 0 selects [0,0,1] (pos 0 twice, pos 1 once),
    #             query 1 selects [2,2,2] (pos 2 three times),
    #             query 2 selects [1,3,3] (pos 1 once, pos 3 twice)
    kv_indices = torch.tensor([[[0, 0, 1], [2, 2, 2], [1, 3, 3]]])

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, 0, backend="eager"
    )

    # --- Manual forward ---
    scale = d**0.5
    q = query[0, 0, :, :]

    # If an element in indexed 3 times, it should show up in effective KV 3 times
    # Build a different effective kv for each query index
    expected = torch.empty((s, d), dtype=dtype)
    for seq in range(s):
        effective_kv = torch.empty((num_topk, d), dtype=dtype)
        for v in range(num_topk):
            effective_kv[v, :] = sparse_kv[0, 0, kv_indices[0, seq, v], :]

        scores = torch.exp((q[seq] @ effective_kv.T) / scale)
        probs = scores / scores.sum()
        expected[seq, :] = probs @ effective_kv

    torch.testing.assert_close(out[0, 0], expected, atol=1e-12, rtol=1e-12)

    # --- Backward: sum output and check gradients ---
    out.sum().backward()

    query_m = query.detach().clone().requires_grad_(True)
    sparse_kv_m = sparse_kv.detach().clone().requires_grad_(True)
    q_m = query_m[0, 0]
    expected_m = torch.zeros((s, d), dtype=dtype)
    for seq in range(s):
        effective_kv_m = torch.empty((num_topk, d), dtype=dtype)
        for v in range(num_topk):
            effective_kv_m[v, :] = sparse_kv_m[0, 0, kv_indices[0, seq, v], :]
        scores_m = torch.exp((q_m[seq] @ effective_kv_m.T) / scale)
        probs_m = scores_m / scores_m.sum()
        expected_m[seq, :] = probs_m @ effective_kv_m
    expected_m.sum().backward()

    torch.testing.assert_close(query.grad, query_m.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(sparse_kv.grad, sparse_kv_m.grad, atol=1e-12, rtol=1e-12)
