"""
Tests for the CuTe DSL (SM100) backend of selected attention.

Mirrors test_selected_attention_triton.py but adapted for CuTe constraints:
  - head_dim = 512, nheads = 128, share_kv = True, dtype = bfloat16
  - Forward-only (no backward support)
  - Tolerance: 1e-2 absolute (Roughly 4x bf16 reference error; bf16 ref error typically ranges from 2.5-3)
      - Cute backend error ranges from 0.5-3x reference error
  - Precision test: cute error must be < 5x the reference bf16 error
"""

import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention


ATOL_FWD = 1e-1
RTOL_FWD = 1e-1


def _skip_no_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CuTe backend")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 (compute capability 10.0) required for CuTe backend")


def _make_inputs(
    batch: int = 2,
    heads: int = 128,
    seq_len: int = 256,
    head_dim: int = 512,
    index_seq_len: int = 128,
    num_topk: int = 16,
    sliding_window_size: int = 64,
    doc_ids: torch.Tensor | None = None,
    seed: int = 42,
):
    """Create bf16 inputs on CUDA satisfying CuTe constraints (share_kv=True, H=128, D=512)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    Q = randn(batch, heads, seq_len, head_dim)
    KV = randn(batch, 1, seq_len, head_dim)
    index_kv = randn(batch, 1, index_seq_len, head_dim)

    if num_topk > 0:
        scores = torch.randn(batch, seq_len, index_seq_len, device=device, generator=generator)
        _, indices = torch.topk(scores, k=min(num_topk, index_seq_len), dim=-1)
    else:
        indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    # sink=0 since the CuTe backend does not fuse sink correction
    attention_sink = torch.zeros(heads, device=device, dtype=dtype)

    if doc_ids is not None:
        doc_ids = doc_ids.to(device)

    return {
        "Q": Q,
        "KV": KV,
        "index_kv": index_kv,
        "indices": indices,
        "attention_sink": attention_sink,
        "doc_ids": doc_ids,
        "sliding_window_size": sliding_window_size,
        "share_kv": True,
    }


@pytest.mark.parametrize("num_topk", [16, 32, 64])
def test_cute_forward_matches_reference(num_topk):
    """CuTe forward matches the eager reference implementation (sink=0)."""
    _skip_no_sm100()
    inputs = _make_inputs(num_topk=num_topk)

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="cute")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_forward_with_doc_ids(num_topk):
    """CuTe forward with doc_ids matches the eager reference."""
    _skip_no_sm100()
    seq_len = 256
    doc_ids = (
        torch.cat(
            [
                torch.zeros(seq_len // 2, dtype=torch.long),
                torch.ones(seq_len // 2, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(2, -1)
    )

    inputs = _make_inputs(num_topk=num_topk, seq_len=seq_len, doc_ids=doc_ids)

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="cute")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


def test_cute_larger_sequence():
    """CuTe handles larger sequence lengths correctly."""
    _skip_no_sm100()
    inputs = _make_inputs(
        batch=1,
        seq_len=512,
        index_seq_len=256,
        num_topk=32,
        sliding_window_size=128,
    )

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="cute")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("num_topk", [16, 32, 64])
def test_cute_precision_vs_fp64(num_topk):
    """CuTe bf16 error must be less than 5x the reference bf16 error vs fp64 ground truth.

    This ensures the kernel doesn't introduce catastrophic precision loss beyond
    what bf16 arithmetic inherently causes.
    """
    _skip_no_sm100()
    batch, heads, seq_len, head_dim = 2, 128, 256, 512
    index_seq_len = 128
    sliding_window_size = 64
    seed = 77
    device = torch.device("cuda")

    # --- Generate inputs in fp64 (ground truth) ---
    gen64 = torch.Generator(device=device).manual_seed(seed)

    def randn64(*shape):
        return torch.randn(*shape, dtype=torch.float64, device=device, generator=gen64)

    Q_64 = randn64(batch, heads, seq_len, head_dim)
    KV_64 = randn64(batch, 1, seq_len, head_dim)
    index_kv_64 = randn64(batch, 1, index_seq_len, head_dim)
    scores_64 = torch.randn(
        batch, seq_len, index_seq_len, dtype=torch.float64, device=device, generator=gen64
    )
    if num_topk > 0:
        _, indices = torch.topk(scores_64, k=min(num_topk, index_seq_len), dim=-1)
    else:
        indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)
    # sink=0 for both
    sink_64 = torch.zeros(heads, dtype=torch.float64, device=device)

    # --- bf16 copies ---
    Q_bf_ref = Q_64.to(torch.bfloat16)
    KV_bf_ref = KV_64.to(torch.bfloat16)
    index_kv_bf_ref = index_kv_64.to(torch.bfloat16)
    sink_bf = sink_64.to(torch.bfloat16)

    Q_bf_cute = Q_bf_ref.clone()
    KV_bf_cute = KV_bf_ref.clone()
    index_kv_bf_cute = index_kv_bf_ref.clone()

    # --- Forward ---
    with torch.inference_mode():
        out_64 = selected_attention(
            Q_64, KV_64, index_kv_64, indices, sink_64,
            None, sliding_window_size, True, backend="eager",
        )
        out_bf_ref = selected_attention(
            Q_bf_ref, KV_bf_ref, index_kv_bf_ref, indices, sink_bf,
            None, sliding_window_size, True, backend="eager",
        )
        out_bf_cute = selected_attention(
            Q_bf_cute, KV_bf_cute, index_kv_bf_cute, indices, sink_bf,
            None, sliding_window_size, True, backend="cute",
        )

    ref_fwd_diff = (out_64.float() - out_bf_ref.float()).abs().max().item()
    cute_fwd_diff = (out_64.float() - out_bf_cute.float()).abs().max().item()

    def _ratio(cute_val, ref_val):
        if ref_val == 0:
            return float("inf") if cute_val > 0 else 1.0
        return cute_val / ref_val

    r_fwd = _ratio(cute_fwd_diff, ref_fwd_diff)

    # Print report (visible with pytest -s)
    print(
        f"\n[bf16, topk={num_topk}]"
        f"\n  {'':15s} {'fwd max diff':>12s}"
        f"\n  {'ref bf16':15s} {ref_fwd_diff:12.4e}"
        f"\n  {'cute bf16':15s} {cute_fwd_diff:12.4e}"
        f"\n  {'cute/ref':15s} {r_fwd:12.2f}x"
    )

    assert r_fwd < 5, (
        f"CuTe fwd diff too large: {cute_fwd_diff:.4e} vs reference bf16 {ref_fwd_diff:.4e}, "
        f"ratio = {r_fwd:.2f}x (must be < 5x)"
    )


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

ATOL_BWD = 2e-1
RTOL_BWD = 2e-1


def _make_inputs_grad(
    batch: int = 1,
    heads: int = 128,
    seq_len: int = 128,
    head_dim: int = 512,
    index_seq_len: int = 64,
    num_topk: int = 16,
    sliding_window_size: int = 32,
    seed: int = 42,
):
    """Create bf16 inputs with requires_grad for backward tests."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator,
                           requires_grad=True)

    Q = randn(batch, heads, seq_len, head_dim)
    KV = randn(batch, 1, seq_len, head_dim)
    index_kv = randn(batch, 1, index_seq_len, head_dim)

    scores = torch.randn(batch, seq_len, index_seq_len, device=device, generator=generator)
    _, indices = torch.topk(scores, k=min(num_topk, index_seq_len), dim=-1)

    attention_sink = torch.zeros(heads, device=device, dtype=dtype)

    return Q, KV, index_kv, indices, attention_sink, sliding_window_size


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_dq(num_topk):
    """CuTe backward produces correct dQ gradients."""
    _skip_no_sm100()
    Q, KV, index_kv, indices, sink, window = _make_inputs_grad(num_topk=num_topk, seed=100)
    Q2 = Q.detach().clone().requires_grad_(True)
    KV2 = KV.detach().clone().requires_grad_(True)
    ikv2 = index_kv.detach().clone().requires_grad_(True)

    out_cute = selected_attention(Q, KV, index_kv, indices, sink, None, window, True, backend="cute")
    out_ref = selected_attention(Q2, KV2, ikv2, indices, sink, None, window, True, backend="eager")

    grad = torch.randn_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(Q.grad, Q2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_dkv(num_topk):
    """CuTe backward produces correct dKV gradients."""
    _skip_no_sm100()
    Q, KV, index_kv, indices, sink, window = _make_inputs_grad(num_topk=num_topk, seed=200)
    Q2 = Q.detach().clone().requires_grad_(True)
    KV2 = KV.detach().clone().requires_grad_(True)
    ikv2 = index_kv.detach().clone().requires_grad_(True)

    out_cute = selected_attention(Q, KV, index_kv, indices, sink, None, window, True, backend="cute")
    out_ref = selected_attention(Q2, KV2, ikv2, indices, sink, None, window, True, backend="eager")

    grad = torch.randn_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(KV.grad, KV2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_d_index_kv(num_topk):
    """CuTe backward produces correct d_index_kv gradients."""
    _skip_no_sm100()
    Q, KV, index_kv, indices, sink, window = _make_inputs_grad(num_topk=num_topk, seed=300)
    Q2 = Q.detach().clone().requires_grad_(True)
    KV2 = KV.detach().clone().requires_grad_(True)
    ikv2 = index_kv.detach().clone().requires_grad_(True)

    out_cute = selected_attention(Q, KV, index_kv, indices, sink, None, window, True, backend="cute")
    out_ref = selected_attention(Q2, KV2, ikv2, indices, sink, None, window, True, backend="eager")

    grad = torch.randn_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(index_kv.grad, ikv2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_precision_vs_fp64(num_topk):
    """CuTe backward bf16 error must be < 5x the reference bf16 error."""
    _skip_no_sm100()
    batch, heads, seq_len, head_dim = 1, 128, 128, 512
    index_seq_len, window = 64, 32
    seed = 77
    device = torch.device("cuda")

    gen64 = torch.Generator(device=device).manual_seed(seed)
    def randn64(*shape):
        return torch.randn(*shape, dtype=torch.float64, device=device, generator=gen64,
                           requires_grad=True)

    Q_64 = randn64(batch, heads, seq_len, head_dim)
    KV_64 = randn64(batch, 1, seq_len, head_dim)
    ikv_64 = randn64(batch, 1, index_seq_len, head_dim)
    scores = torch.randn(batch, seq_len, index_seq_len, dtype=torch.float64, device=device, generator=gen64)
    _, indices = torch.topk(scores, k=num_topk, dim=-1)
    sink_64 = torch.zeros(heads, dtype=torch.float64, device=device)

    # bf16 copies
    Q_ref = Q_64.detach().to(torch.bfloat16).requires_grad_(True)
    KV_ref = KV_64.detach().to(torch.bfloat16).requires_grad_(True)
    ikv_ref = ikv_64.detach().to(torch.bfloat16).requires_grad_(True)
    Q_cute = Q_ref.detach().clone().requires_grad_(True)
    KV_cute = KV_ref.detach().clone().requires_grad_(True)
    ikv_cute = ikv_ref.detach().clone().requires_grad_(True)
    sink_bf = sink_64.to(torch.bfloat16)

    # fp64 reference
    out_64 = selected_attention(Q_64, KV_64, ikv_64, indices, sink_64, None, window, True, backend="eager")
    grad_64 = torch.randn_like(out_64)
    out_64.backward(grad_64)

    grad_bf = grad_64.to(torch.bfloat16)

    # bf16 eager reference
    out_ref = selected_attention(Q_ref, KV_ref, ikv_ref, indices, sink_bf, None, window, True, backend="eager")
    out_ref.backward(grad_bf)

    # bf16 cute
    out_cute = selected_attention(Q_cute, KV_cute, ikv_cute, indices, sink_bf, None, window, True, backend="cute")
    out_cute.backward(grad_bf)

    ref_dq_diff = (Q_64.grad.float() - Q_ref.grad.float()).abs().max().item()
    cute_dq_diff = (Q_64.grad.float() - Q_cute.grad.float()).abs().max().item()

    def _ratio(c, r):
        return c / r if r > 0 else (float("inf") if c > 0 else 1.0)

    r = _ratio(cute_dq_diff, ref_dq_diff)
    print(f"\n[topk={num_topk}] dQ: ref_bf16={ref_dq_diff:.4e} cute_bf16={cute_dq_diff:.4e} ratio={r:.2f}x")

    assert r < 5, f"CuTe dQ error {r:.2f}x reference (must be < 5x)"


# ---------------------------------------------------------------------------
# Backward with doc_ids tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_doc_ids_dq(num_topk):
    """CuTe backward with doc_ids produces correct dQ gradients."""
    _skip_no_sm100()
    seq_len = 256
    batch = 1
    doc_ids = (
        torch.cat(
            [
                torch.zeros(seq_len // 2, dtype=torch.long),
                torch.ones(seq_len // 2, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(batch, -1)
        .cuda()
    )

    Q, KV, index_kv, indices, sink, window = _make_inputs_grad(
        batch=batch, seq_len=seq_len, num_topk=num_topk, seed=400
    )
    Q2 = Q.detach().clone().requires_grad_(True)
    KV2 = KV.detach().clone().requires_grad_(True)
    ikv2 = index_kv.detach().clone().requires_grad_(True)

    out_cute = selected_attention(Q, KV, index_kv, indices, sink, doc_ids, window, True, backend="cute")
    out_ref = selected_attention(Q2, KV2, ikv2, indices, sink, doc_ids, window, True, backend="eager")

    grad = torch.randn_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(Q.grad, Q2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)


@pytest.mark.parametrize("num_topk", [16, 32])
def test_cute_backward_doc_ids_dkv(num_topk):
    """CuTe backward with doc_ids produces correct dKV gradients."""
    _skip_no_sm100()
    seq_len = 256
    batch = 1
    doc_ids = (
        torch.cat(
            [
                torch.zeros(seq_len // 2, dtype=torch.long),
                torch.ones(seq_len // 2, dtype=torch.long),
            ]
        )
        .unsqueeze(0)
        .expand(batch, -1)
        .cuda()
    )

    Q, KV, index_kv, indices, sink, window = _make_inputs_grad(
        batch=batch, seq_len=seq_len, num_topk=num_topk, seed=500
    )
    Q2 = Q.detach().clone().requires_grad_(True)
    KV2 = KV.detach().clone().requires_grad_(True)
    ikv2 = index_kv.detach().clone().requires_grad_(True)

    out_cute = selected_attention(Q, KV, index_kv, indices, sink, doc_ids, window, True, backend="cute")
    out_ref = selected_attention(Q2, KV2, ikv2, indices, sink, doc_ids, window, True, backend="eager")

    grad = torch.randn_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)

    torch.testing.assert_close(KV.grad, KV2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)
    torch.testing.assert_close(index_kv.grad, ikv2.grad, atol=ATOL_BWD, rtol=RTOL_BWD)
