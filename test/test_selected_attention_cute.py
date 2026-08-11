"""
Tests for the CuTe DSL (SM100) backend of selected attention.

Mirrors test_selected_attention_triton.py but adapted for CuTe constraints:
  - head_dim = 512, nheads = 128, share_kv = True, dtype = bfloat16, SM100
  - Uses assert_matches_low_precision_eager for precision checks
"""

import math

import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention

def _skip_no_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CuTe backend")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 (compute capability 10.0) required for CuTe backend")


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
    max_atol = (accumulation_eps + len(reduction_sizes) * output_rounding_eps) * (
        high_precision_expected.abs().max().item()
    )
    assert actual_difference.mean().item() <= eager_difference.mean().item() + mean_atol
    assert actual_difference.max().item() <= eager_difference.max().item() + max_atol


def _make_inputs(
    batch: int = 2,
    heads: int = 128,
    seq_len: int = 256,
    head_dim: int = 512,
    sparse_seq_len: int = 128,
    num_topk: int = 16,
    sliding_window_size: int = 64,
    doc_ids: torch.Tensor | None = None,
    requires_grad: bool = False,
    seed: int = 42,
):
    """Create bf16 inputs on CUDA satisfying CuTe constraints (share_kv=True, H=128, D=512)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, device=device, dtype=dtype, generator=generator, requires_grad=requires_grad
        )

    query = randn(batch, heads, seq_len, head_dim)
    local_kv = randn(batch, 1, seq_len, head_dim)
    sparse_kv = randn(batch, 1, sparse_seq_len, head_dim)

    if num_topk > 0:
        scores = torch.randn(batch, seq_len, sparse_seq_len, device=device, generator=generator)
        _, kv_indices = torch.topk(scores, k=min(num_topk, sparse_seq_len), dim=-1)
    else:
        kv_indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    attention_sink = None

    if doc_ids is not None:
        doc_ids = doc_ids.to(device)

    return {
        "query": query,
        "local_kv": local_kv,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
        "attention_sink": attention_sink,
        "doc_ids": doc_ids,
        "sliding_window_size": sliding_window_size,
    }



# ---------------------------------------------------------------------------
# Precision vs FP64 (measuring-stick test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_topk", [16, 32, 64])
@pytest.mark.parametrize("test_docids", [False, True], ids=["no_docids", "with_docids"])
def test_cute_precision_vs_fp64(num_topk, test_docids):
    """CuTe bf16 error bounded by low-precision eager error vs FP64.

    Inputs are generated in bf16 first, then promoted to FP64 via .double().
    This ensures eager, CuTe, and FP64 all see the same quantized values
    so that the FP64 baseline isolates arithmetic error without input-quantization noise.
    """
    _skip_no_sm100()
    batch, heads, seq_len, head_dim = 2, 128, 128, 512
    sparse_seq_len = 128
    sliding_window_size = 64
    seed = 77
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if test_docids:
        doc_ids = (
            torch.cat(
                [
                    torch.zeros(seq_len // 2, dtype=torch.long),
                    torch.ones(seq_len // 2, dtype=torch.long),
                ]
            )
            .unsqueeze(0)
            .expand(batch, -1)
            .to(device)
        )
    else:
        doc_ids = None

    # --- Generate inputs in bf16 (the "quantized" source) ---
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn_lp(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    query_lp = randn_lp(batch, heads, seq_len, head_dim)
    local_kv_lp = randn_lp(batch, 1, seq_len, head_dim)
    sparse_kv_lp = randn_lp(batch, 1, sparse_seq_len, head_dim)

    scores = torch.randn(batch, seq_len, sparse_seq_len, dtype=dtype, device=device, generator=gen)
    if num_topk > 0:
        _, kv_indices = torch.topk(scores, k=min(num_topk, sparse_seq_len), dim=-1)
    else:
        kv_indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    # sink=0 for CuTe
    sink = None

    # --- Derive FP64 inputs from the same quantized values ---
    query_64 = query_lp.double().requires_grad_(True)
    local_kv_64 = local_kv_lp.double().requires_grad_(True)
    sparse_kv_64 = sparse_kv_lp.double().requires_grad_(True)

    # --- Lower-precision copies for eager and cute ---
    query_lp_ref = query_lp.clone().requires_grad_(True)
    local_kv_lp_ref = local_kv_lp.clone().requires_grad_(True)
    sparse_kv_lp_ref = sparse_kv_lp.clone().requires_grad_(True)

    query_lp_cute = query_lp.clone().requires_grad_(True)
    local_kv_lp_cute = local_kv_lp.clone().requires_grad_(True)
    sparse_kv_lp_cute = sparse_kv_lp.clone().requires_grad_(True)

    # --- Forward ---
    out_64 = selected_attention(
        query_64,
        local_kv_64,
        sparse_kv_64,
        kv_indices,
        sink,
        doc_ids,
        sliding_window_size,
        backend="eager",
    )
    out_lp_ref = selected_attention(
        query_lp_ref,
        local_kv_lp_ref,
        sparse_kv_lp_ref,
        kv_indices,
        sink,
        doc_ids,
        sliding_window_size,
        backend="eager",
    )
    out_lp_cute = selected_attention(
        query_lp_cute,
        local_kv_lp_cute,
        sparse_kv_lp_cute,
        kv_indices,
        sink,
        doc_ids,
        sliding_window_size,
        backend="cute",
    )

    # --- Backward ---
    grad_gen = torch.Generator(device=device).manual_seed(1234)
    grad_lp = torch.randn(out_lp_ref.shape, dtype=dtype, device=device, generator=grad_gen)
    grad_64 = grad_lp.double()

    out_64.backward(grad_64)
    out_lp_ref.backward(grad_lp)
    out_lp_cute.backward(grad_lp)

    # --- Reduction sizes ---
    num_keys = sliding_window_size + num_topk
    fwd_reduction_sizes = (head_dim, num_keys, num_keys)
    dq_reduction_sizes = fwd_reduction_sizes + (head_dim,) + (num_keys,) + (batch,)
    dkv_reduction_sizes = (
        fwd_reduction_sizes + (head_dim,) + (sliding_window_size, sliding_window_size) + (batch,)
    )
    dsparse_kv_reduction_sizes = fwd_reduction_sizes + (num_topk,) + (batch,)

    # --- Forward check ---
    assert_matches_low_precision_eager(
        out_lp_cute,
        out_lp_ref,
        out_64,
        reduction_sizes=fwd_reduction_sizes,
    )

    # --- Backward checks ---
    assert_matches_low_precision_eager(
        query_lp_cute.grad,
        query_lp_ref.grad,
        query_64.grad,
        reduction_sizes=dq_reduction_sizes,
    )
    assert_matches_low_precision_eager(
        local_kv_lp_cute.grad,
        local_kv_lp_ref.grad,
        local_kv_64.grad,
        reduction_sizes=dkv_reduction_sizes,
    )
    assert_matches_low_precision_eager(
        sparse_kv_lp_cute.grad,
        sparse_kv_lp_ref.grad,
        sparse_kv_64.grad,
        reduction_sizes=dsparse_kv_reduction_sizes,
    )


# ---------------------------------------------------------------------------
# torch.compile fullgraph compatibility
# ---------------------------------------------------------------------------


def test_cute_compile_fullgraph():
    """CuTe backend works under torch.compile(fullgraph=True)."""
    _skip_no_sm100()
    inputs = _make_inputs(num_topk=16, seed=123)

    compiled = torch.compile(selected_attention, fullgraph=True)

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="cute")
        actual = compiled(**inputs, backend="cute")

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
