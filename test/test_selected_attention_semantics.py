"""Tests for selected_attention primitive edge cases and semantics.

Covers: local-only, selected-block-only, joint normalization,
attention sink behavior, and repeated selections.
Tests both eager (reference) and triton backends.
"""

import math

import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention

BACKENDS = ["eager"]
if torch.cuda.is_available():
    BACKENDS.append("triton")

BLACKWELL_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.fixture
def shared_kv_blackwell_inputs():
    """Create a batched, partial-head-tile shared-KV case with a float32 sink."""
    torch.manual_seed(123)
    batch, heads, seq_len, head_dim = 2, 17, 33, 32
    sparse_seq_len, topk = 79, 73
    query = torch.randn(
        batch,
        heads,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    local_kv = torch.randn(
        batch,
        1,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    sparse_kv = torch.randn(
        batch,
        1,
        sparse_seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv_indices = torch.randint(0, sparse_seq_len, (batch, seq_len, topk), device="cuda")
    kv_indices[:, ::3, -1] = -1
    kv_indices[:, 1::4, 1] = kv_indices[:, 1::4, 0]
    attention_sink = torch.randn(heads, device="cuda", dtype=torch.float32, requires_grad=True)
    doc_ids = torch.tensor(
        [[0] * 11 + [1] * 9 + [2] * 13], device="cuda", dtype=torch.int32
    ).expand(batch, -1)
    return query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids


def _device_for_backend(backend):
    return torch.device("cuda") if backend == "triton" else torch.device("cpu")


def _dtype_for_backend(backend):
    return torch.float32


def assert_matches_low_precision_eager(
    actual,
    low_precision_expected,
    high_precision_expected,
    reduction_sizes,
):
    """Bound kernel error by low-precision eager error against an FP64 measuring stick."""
    assert torch.isfinite(actual).all()
    actual_difference = (actual.double() - high_precision_expected).abs()
    eager_difference = (low_precision_expected.double() - high_precision_expected).abs()
    accumulation_eps = (
        sum(math.sqrt(size) for size in reduction_sizes) * torch.finfo(torch.float32).eps
    )
    output_rounding_eps = torch.finfo(actual.dtype).eps
    rounding_eps = accumulation_eps + output_rounding_eps
    mean_atol = rounding_eps * high_precision_expected.abs().mean().item()
    max_atol = rounding_eps * high_precision_expected.abs().max().item()
    assert actual_difference.mean().item() <= eager_difference.mean().item() + mean_atol
    assert actual_difference.max().item() <= eager_difference.max().item() + max_atol


# ---------------------------------------------------------------------------
# 1. Local-only attention (topk=0, no index blocks)
# ---------------------------------------------------------------------------


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

    sparse_kv2 = torch.randn_like(sparse_kv)
    out2 = selected_attention(
        query, local_kv, sparse_kv2, kv_indices, sink, None, window, backend=backend
    )

    atol = 1e-5 if backend == "triton" else 1e-10
    assert out.shape == (b, h, s, d)
    torch.testing.assert_close(out, expected, atol=atol, rtol=1e-5)
    torch.testing.assert_close(out2, expected, atol=atol, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Selected-block-only attention (window=0)
# ---------------------------------------------------------------------------


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

    local_kv2 = torch.randn_like(local_kv)
    out2 = selected_attention(
        query, local_kv2, sparse_kv, kv_indices, sink, None, 0, backend=backend
    )

    atol = 1e-4 if backend == "triton" else 1e-10
    assert out.shape == (b, h, s, d)
    torch.testing.assert_close(out[0, 0], expected, atol=atol, rtol=1e-4)
    torch.testing.assert_close(out2[0, 0], expected, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. Joint normalization (both branches share the same softmax denominator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_joint_normalization(backend):
    """When both local and sparse branches are active, they share normalization.

    We verify against a manual computation that runs a single softmax over the
    concatenation of sparse and local logits.
    """
    device = _device_for_backend(backend)
    dtype = torch.float64 if backend == "eager" else torch.float32
    b, h, s, d = 1, 1, 4, 8
    window = 2
    sparse_seq_len = 3

    torch.manual_seed(55)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    kv_indices = torch.tensor([[[0, 1], [1, 2], [0, 2], [1, 0]]], device=device)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # --- Manual: single softmax over gathered sparse + local window + sink ---
    scale = d**0.5
    q = query[0, 0]  # (s, d)
    lkv = local_kv[0, 0]  # (s, d)
    skv = sparse_kv[0, 0]  # (sparse_seq_len, d)
    sink_val = sink[0]

    expected = torch.zeros(s, d, device=device, dtype=dtype)
    for seq in range(s):
        # Gather sparse entries (with duplicates)
        gathered_sparse = skv[kv_indices[0, seq]]  # (num_topk, d)

        # Gather local window entries
        first = max(0, seq - window + 1)
        local_entries = lkv[first : seq + 1]  # (window_len, d)

        # Concatenate all KV entries this query attends to
        all_kv = torch.cat([gathered_sparse, local_entries], dim=0)  # (num_topk + window_len, d)

        # Compute logits over all entries + sink
        logits = (q[seq] @ all_kv.T) / scale  # (num_topk + window_len,)
        logits_with_sink = torch.cat([logits, sink_val.unsqueeze(0)])
        probs_with_sink = torch.softmax(logits_with_sink, dim=0)

        # Verify probs sum to 1 (joint normalization)
        assert torch.allclose(
            probs_with_sink.sum(), torch.ones(1, device=device, dtype=dtype), atol=1e-10
        )

        # Drop the sink prob, compute output
        probs = probs_with_sink[:-1]
        expected[seq] = probs @ all_kv

    atol = 1e-4 if backend == "triton" else 1e-10
    torch.testing.assert_close(out[0, 0], expected, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Attention sink behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_sink_large_absorbs_probability(backend):
    """As sink → +∞, all probability goes to sink, output → 0."""
    device = _device_for_backend(backend)
    dtype = _dtype_for_backend(backend)
    b, h, s, d = 1, 1, 4, 8
    window = 2

    torch.manual_seed(33)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, 3, d, device=device, dtype=dtype)
    kv_indices = torch.tensor([[[0, 1], [1, 2], [0, 2], [1, 0]]], device=device)

    # Very large positive sink
    sink = torch.full((h,), 50.0, device=device, dtype=dtype)

    out = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, window, backend=backend
    )

    # Output should be near zero (sink absorbs essentially all probability)
    assert out.abs().max().item() < 0.01, (
        f"With large sink, output should be near zero, got max={out.abs().max().item()}"
    )


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_attention_sink(backend):
    """A float32 sink is supported with lower-precision query and KV tensors."""
    device = _device_for_backend(backend)
    torch.manual_seed(101)
    query = torch.randn(1, 2, 4, 16, device=device, dtype=torch.bfloat16)
    local_kv = torch.randn(1, 2, 4, 16, device=device, dtype=torch.bfloat16)
    sparse_kv = torch.randn(1, 2, 4, 16, device=device, dtype=torch.bfloat16)
    kv_indices = torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 3]]], device=device, dtype=torch.int32)
    attention_sink = torch.randn(2, device=device, dtype=torch.float32)

    high_precision_expected = selected_attention(
        query.double(),
        local_kv.double(),
        sparse_kv.double(),
        kv_indices,
        attention_sink.double(),
        None,
        2,
        backend="eager",
    )
    low_precision_expected = selected_attention(
        query, local_kv, sparse_kv, kv_indices, attention_sink, None, 2, backend="eager"
    )
    actual = selected_attention(
        query, local_kv, sparse_kv, kv_indices, attention_sink, None, 2, backend=backend
    )

    assert actual.dtype == torch.bfloat16
    assert_matches_low_precision_eager(
        actual,
        low_precision_expected,
        high_precision_expected,
        reduction_sizes=(query.shape[-1], kv_indices.shape[-1] + 2, kv_indices.shape[-1] + 2),
    )


def test_eager_bfloat16_mixed_precision_schedule():
    """Eager mirrors FP32 QK/softmax/PV accumulation and BF16 dot operands."""
    torch.manual_seed(2027)
    batch, heads, seq_len, head_dim = 1, 2, 5, 16
    sparse_seq_len, window = 6, 3
    query = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
    local_kv = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16)
    sparse_kv = torch.randn(batch, heads, sparse_seq_len, head_dim, dtype=torch.bfloat16)
    kv_indices = torch.tensor(
        [[[0, 0, 2], [1, 3, -1], [2, 4, 5], [0, 3, 5], [1, 1, 4]]],
        dtype=torch.int64,
    )
    attention_sink = torch.randn(heads, dtype=torch.bfloat16)

    valid_indices = kv_indices >= 0
    counts = torch.zeros(batch, seq_len, sparse_seq_len, dtype=torch.float32)
    counts.scatter_add_(
        -1,
        kv_indices.clamp(min=0),
        valid_indices.to(torch.float32),
    )
    sparse_mask = torch.where(
        counts > 0,
        counts.log(),
        torch.full_like(counts, -float("inf")),
    )
    query_positions = torch.arange(seq_len)[:, None]
    key_positions = torch.arange(seq_len)[None, :]
    local_valid = (key_positions <= query_positions) & (
        key_positions >= query_positions - window + 1
    )
    local_mask = torch.zeros(seq_len, seq_len, dtype=torch.float32).masked_fill(
        ~local_valid, -float("inf")
    )
    attention_mask = torch.cat([sparse_mask, local_mask.expand(batch, -1, -1)], dim=-1).unsqueeze(
        1
    )
    attention_kv = torch.cat([sparse_kv, local_kv], dim=-2)
    logits = (query.float() @ attention_kv.float().transpose(-2, -1)) / math.sqrt(
        head_dim
    ) + attention_mask
    sink_logits = attention_sink.float()[None, :, None, None].expand(batch, -1, seq_len, 1)
    probabilities = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)[..., :-1].to(
        torch.bfloat16
    )
    expected = (probabilities.float() @ attention_kv.float()).to(torch.bfloat16)

    actual = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        None,
        window,
        backend="eager",
    )
    actual_float32_sink = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink.float(),
        None,
        window,
        backend="eager",
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_float32_sink, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 5. Repeated selections (backends must agree)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
@pytest.mark.parametrize(
    "num_repeats,sliding_window_size",
    [(2, 0), (3, 4)],
    ids=["sparse-only", "joint"],
)
def test_repeated_indices_backends_match(num_repeats, sliding_window_size):
    """Repeated indices should produce identical results in eager and triton."""
    device = torch.device("cuda")
    dtype = torch.float32
    b, h, s, d = 1, 2, 8, 16
    sparse_seq_len = 6

    torch.manual_seed(0)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype, requires_grad=True)
    sparse_kv = torch.randn(
        b, h, sparse_seq_len, d, device=device, dtype=dtype, requires_grad=True
    )
    sink = torch.randn(h, device=device, dtype=dtype, requires_grad=True)

    # All slots repeat position 2
    kv_indices = torch.full((b, s, num_repeats), 2, dtype=torch.long, device=device)

    out_eager = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="eager"
    )
    grad_output = torch.randn_like(out_eager)
    out_eager.backward(grad_output)
    grad_query_eager = query.grad.clone()
    grad_local_kv_eager = local_kv.grad.clone()
    grad_sparse_kv_eager = sparse_kv.grad.clone()
    grad_sink_eager = sink.grad.clone()

    query.grad = None
    local_kv.grad = None
    sparse_kv.grad = None
    sink.grad = None

    out_triton = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="triton"
    )
    out_triton.backward(grad_output)

    torch.testing.assert_close(out_eager, out_triton, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(query.grad, grad_query_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(local_kv.grad, grad_local_kv_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sparse_kv.grad, grad_sparse_kv_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sink.grad, grad_sink_eager, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
def test_mixed_repeated_and_unique_indices_backends_match():
    """Mix of repeated and unique indices should match between backends."""
    device = torch.device("cuda")
    dtype = torch.float32
    b, h, s, d = 1, 2, 8, 16
    sparse_seq_len = 4

    torch.manual_seed(99)
    query = torch.randn(b, h, s, d, device=device, dtype=dtype)
    local_kv = torch.randn(b, h, s, d, device=device, dtype=dtype)
    sparse_kv = torch.randn(b, h, sparse_seq_len, d, device=device, dtype=dtype)
    sink = torch.randn(h, device=device, dtype=dtype)

    # Mix: some rows have repeats, some are unique, some have -1 sentinels
    kv_indices = torch.tensor(
        [[[0, 0], [1, 1], [0, 1], [2, -1], [-1, -1], [3, 3], [1, 2], [0, -1]]],
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
# 6. Shared-KV Blackwell schedules
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
def test_shared_kv_blackwell_matches_eager(shared_kv_blackwell_inputs):
    """The default atomic shared-KV path matches eager forward and gradients."""
    query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids = shared_kv_blackwell_inputs
    kv_indices = kv_indices.to(torch.int32)
    differentiable_inputs = query, local_kv, sparse_kv, attention_sink
    high_precision_inputs = tuple(
        tensor.detach().double().requires_grad_(True) for tensor in differentiable_inputs
    )

    high_precision_expected = selected_attention(
        high_precision_inputs[0],
        high_precision_inputs[1],
        high_precision_inputs[2],
        kv_indices,
        high_precision_inputs[3],
        doc_ids,
        19,
        backend="eager",
    )
    expected = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        doc_ids,
        19,
        backend="eager",
    )
    actual = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        doc_ids,
        19,
        backend="triton",
    )
    grad_output = torch.randn_like(expected)
    high_precision_grads = torch.autograd.grad(
        high_precision_expected,
        high_precision_inputs,
        grad_output.double(),
    )
    expected_grads = torch.autograd.grad(expected, differentiable_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, differentiable_inputs, grad_output)

    assert_matches_low_precision_eager(
        actual,
        expected,
        high_precision_expected,
        reduction_sizes=(query.shape[-1], kv_indices.shape[-1] + 19),
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.06, rtol=0.03)
    assert_matches_low_precision_eager(
        actual_grads[2],
        expected_grads[2],
        high_precision_grads[2],
        reduction_sizes=(
            query.shape[-1],
            kv_indices.shape[-1] + 19,
            query.shape[1] * query.shape[2],
        ),
    )


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
def test_shared_kv_blackwell_deterministic_backward(shared_kv_blackwell_inputs):
    """Deterministic mode retains the output-owned shared-KV backward schedule."""
    query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids = shared_kv_blackwell_inputs
    differentiable_inputs = query, local_kv, sparse_kv, attention_sink
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = selected_attention(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            doc_ids,
            19,
            backend="eager",
        )
        actual = selected_attention(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            doc_ids,
            19,
            backend="triton",
        )
        grad_output = torch.randn_like(expected)
        expected_grads = torch.autograd.grad(expected, differentiable_inputs, grad_output)
        actual_grads = torch.autograd.grad(
            actual, differentiable_inputs, grad_output, retain_graph=True
        )
        repeated_grads = torch.autograd.grad(actual, differentiable_inputs, grad_output)
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)

    for actual_grad, repeated_grad, expected_grad in zip(
        actual_grads, repeated_grads, expected_grads, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.06, rtol=0.03)
        torch.testing.assert_close(repeated_grad, actual_grad, atol=0, rtol=0)


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
def test_zero_stride_unshared_kv_keeps_per_head_gradients():
    """Zero-stride shape-H KV inputs retain distinct per-head gradient semantics."""
    torch.manual_seed(456)
    batch, heads, seq_len, head_dim = 1, 16, 8, 16
    sparse_seq_len, topk, window = 6, 2, 3
    query = torch.randn(
        batch,
        heads,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    local_kv = (
        torch.randn(batch, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        .expand(-1, heads, -1, -1)
        .detach()
        .requires_grad_()
    )
    sparse_kv = (
        torch.randn(batch, 1, sparse_seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        .expand(-1, heads, -1, -1)
        .detach()
        .requires_grad_()
    )
    kv_indices = torch.randint(0, sparse_seq_len, (batch, seq_len, topk), device="cuda")
    attention_sink = torch.randn(heads, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    differentiable_inputs = query, local_kv, sparse_kv, attention_sink

    expected = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        None,
        window,
        backend="eager",
    )
    actual = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        None,
        window,
        backend="triton",
    )
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, differentiable_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, differentiable_inputs, grad_output)

    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.06, rtol=0.03)


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
@pytest.mark.parametrize(
    "heads,seq_len,head_dim,topk,window",
    [(128, 17, 128, 16, 0), (16, 160, 64, 0, 136)],
    ids=["max-head-sparse-only", "min-head-multi-tile-window-only"],
)
def test_shared_kv_blackwell_single_branch(heads, seq_len, head_dim, topk, window):
    """Boundary head tiles handle either attention branch alone."""
    torch.manual_seed(321)
    batch = 1
    sparse_seq_len = 19
    query = torch.randn(
        batch,
        heads,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    local_kv = torch.randn(
        batch, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    sparse_kv = torch.randn(
        batch,
        1,
        sparse_seq_len,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv_indices = torch.randint(0, sparse_seq_len, (batch, seq_len, topk), device="cuda")
    if topk:
        kv_indices[:, ::3, -1] = -1
        kv_indices[:, 1::4, 1] = kv_indices[:, 1::4, 0]
    attention_sink = torch.randn(heads, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    differentiable_inputs = query, local_kv, sparse_kv, attention_sink
    high_precision_inputs = tuple(
        tensor.detach().double().requires_grad_(True) for tensor in differentiable_inputs
    )

    high_precision_expected = selected_attention(
        high_precision_inputs[0],
        high_precision_inputs[1],
        high_precision_inputs[2],
        kv_indices,
        high_precision_inputs[3],
        None,
        window,
        backend="eager",
    )
    expected = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        None,
        window,
        backend="eager",
    )
    actual = selected_attention(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        None,
        window,
        backend="triton",
    )
    grad_output = torch.randn_like(expected)
    high_precision_grads = torch.autograd.grad(
        high_precision_expected, high_precision_inputs, grad_output.double()
    )
    expected_grads = torch.autograd.grad(expected, differentiable_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, differentiable_inputs, grad_output)

    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
    for actual_grad, expected_grad, high_precision_grad in zip(
        actual_grads, expected_grads, high_precision_grads, strict=True
    ):
        assert_matches_low_precision_eager(
            actual_grad,
            expected_grad,
            high_precision_grad,
            reduction_sizes=(head_dim, topk + window, heads * seq_len),
        )
    if topk == 0:
        assert actual_grads[2].count_nonzero().item() == 0


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
def test_shared_kv_blackwell_dsv4_forward():
    """The tiled shared-KV forward handles DSV4's head dimension and sparse top-k."""
    torch.manual_seed(2026)
    batch, heads, seq_len, head_dim = 1, 64, 16, 512
    sparse_seq_len, topk, window = 512, 512, 128
    query = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    local_kv = torch.randn(batch, 1, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    sparse_kv = torch.randn(
        batch, 1, sparse_seq_len, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_indices = torch.stack(
        [torch.randperm(sparse_seq_len, device="cuda")[:topk] for _ in range(seq_len)]
    ).unsqueeze(0)
    kv_indices[:, :4, -3:] = -1
    attention_sink = torch.randn(heads, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        high_precision_expected = selected_attention(
            query.double(),
            local_kv.double(),
            sparse_kv.double(),
            kv_indices,
            attention_sink.double(),
            None,
            window,
            backend="eager",
        )
        low_precision_expected = selected_attention(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            None,
            window,
            backend="eager",
        )
        actual = selected_attention(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            None,
            window,
            backend="triton",
        )

    assert_matches_low_precision_eager(
        actual,
        low_precision_expected,
        high_precision_expected,
        reduction_sizes=(head_dim, topk + window, topk + window),
    )
    with pytest.raises(NotImplementedError, match="head_dim=512 only for inference"):
        selected_attention(
            query.requires_grad_(),
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            None,
            window,
            backend="triton",
        )


# ---------------------------------------------------------------------------
# 7. torch.compile fullgraph compatibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required")
def test_shared_kv_blackwell_torch_compile_fullgraph(shared_kv_blackwell_inputs):
    """The shared-KV Triton forward and backward compile without graph breaks."""
    query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids = shared_kv_blackwell_inputs
    differentiable_inputs = query, local_kv, sparse_kv, attention_sink

    def fn(query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids):
        return selected_attention(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            doc_ids,
            19,
            backend="triton",
        )

    compiled_fn = torch.compile(fn, fullgraph=True)
    compiled_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in differentiable_inputs
    )
    compiled_query, compiled_local_kv, compiled_sparse_kv, compiled_sink = compiled_inputs
    expected = fn(query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids)
    actual = compiled_fn(
        compiled_query,
        compiled_local_kv,
        compiled_sparse_kv,
        kv_indices,
        compiled_sink,
        doc_ids,
    )
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, differentiable_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, compiled_inputs, grad_output)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], atol=0, rtol=0)
    # Inductor and eager may use different reduction trees for the BF16 shared-local dKV sum.
    torch.testing.assert_close(actual_grads[1], expected_grads[1], atol=0.01, rtol=0.01)
    # Atomic sparse-dKV accumulation order may differ across compiled and eager launches.
    torch.testing.assert_close(actual_grads[2], expected_grads[2], atol=0.06, rtol=0.03)
    torch.testing.assert_close(actual_grads[3], expected_grads[3], atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
def test_torch_compile_fullgraph_forward():
    """The Triton inference path compiles with torch.compile(fullgraph=True)."""
    backend = "triton"
    device = torch.device("cuda")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
@pytest.mark.parametrize("backend", BACKENDS)
def test_torch_compile_fullgraph_backward(backend):
    """selected_attention backward works under torch.compile(fullgraph=True)."""
    device = torch.device("cuda")
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
    torch.testing.assert_close(sparse_kv_c.grad, sparse_kv_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sink_c.grad, sink_ref.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 8. Repeated selections against manual computation
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
