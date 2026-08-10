"""
To check reference bf16 vs triton bf16 diffs, run:
python -m pytest test/test_selected_attention_triton.py::test_precision_vs_fp64 -v -s

Inputs are generated in the target dtype first, then promoted to FP64 via .double()
so that eager, Triton, and FP64 all see the same quantized values. This isolates
arithmetic error from input-quantization noise.

Triton max diff is between 0.59-1.23x the reference's max diff
"""

import math

import pytest
import torch

from attn_gym.sparse.selected_attention import selected_attention

ATOL_FWD = 1e-2
RTOL_FWD = 1e-2
ATOL_BWD = 1e-2
RTOL_BWD = 1e-2


def _skip_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for triton backend")


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
    heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    sparse_seq_len: int = 16,
    num_topk: int = 3,
    sliding_window_size: int = 8,
    share_kv: bool = True,
    doc_ids: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    seed: int = 42,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    kv_heads = 1 if share_kv else heads

    def randn(*shape):
        return torch.randn(
            *shape, device=device, dtype=dtype, generator=generator, requires_grad=requires_grad
        )

    query = randn(batch, heads, seq_len, head_dim)
    local_kv = randn(batch, kv_heads, seq_len, head_dim)
    sparse_kv = randn(batch, kv_heads, sparse_seq_len, head_dim)

    if num_topk > 0:
        scores = torch.randn(batch, seq_len, sparse_seq_len, device=device, generator=generator)
        _, kv_indices = torch.topk(scores, k=min(num_topk, sparse_seq_len), dim=-1)
    else:
        kv_indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    attention_sink = torch.randn(
        heads, device=device, dtype=dtype, generator=generator, requires_grad=requires_grad
    )

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


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 1, 4])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_triton_forward_matches_reference(share_kv, num_topk, head_dim):
    """Triton forward matches the eager reference implementation."""
    _skip_no_cuda()
    inputs = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, head_dim=head_dim, dtype=torch.float32
    )

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_forward_with_doc_ids(share_kv, num_topk):
    """Triton forward with doc_ids matches the eager reference."""
    _skip_no_cuda()
    seq_len = 32
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

    inputs = _make_inputs(share_kv=share_kv, num_topk=num_topk, seq_len=seq_len, doc_ids=doc_ids)

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 1, 2, 3])
@pytest.mark.parametrize("grad_target", ["query", "local_kv", "sparse_kv", "attention_sink"])
@pytest.mark.parametrize("sliding_window_size", [0, 8])
def test_triton_backward(share_kv, num_topk, grad_target, sliding_window_size):
    """Triton backward produces correct gradients for all differentiable inputs."""
    _skip_no_cuda()

    inputs_ref = _make_inputs(
        share_kv=share_kv,
        num_topk=num_topk,
        sliding_window_size=sliding_window_size,
        requires_grad=True,
        seed=42,
    )
    inputs_tri = _make_inputs(
        share_kv=share_kv,
        num_topk=num_topk,
        sliding_window_size=sliding_window_size,
        requires_grad=True,
        seed=42,
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(7777)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri[grad_target].grad,
        inputs_ref[grad_target].grad,
        atol=ATOL_BWD,
        rtol=RTOL_BWD,
    )


@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_backward_with_doc_ids(num_topk):
    """Triton backward with doc_ids matches the reference."""
    _skip_no_cuda()
    seq_len = 32
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

    inputs_ref = _make_inputs(
        num_topk=num_topk, seq_len=seq_len, doc_ids=doc_ids, requires_grad=True, seed=999
    )
    inputs_tri = _make_inputs(
        num_topk=num_topk, seq_len=seq_len, doc_ids=doc_ids, requires_grad=True, seed=999
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(4444)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri["query"].grad, inputs_ref["query"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )
    torch.testing.assert_close(
        inputs_tri["local_kv"].grad, inputs_ref["local_kv"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton")
@pytest.mark.parametrize("sliding_window_size", [0, 4])
def test_empty_sliding_window(sliding_window_size):
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
    kv_indices = torch.full((b, s, 1), 2, dtype=torch.long, device=device)

    out_eager = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="eager"
    )
    out_triton = selected_attention(
        query, local_kv, sparse_kv, kv_indices, sink, None, sliding_window_size, backend="triton"
    )

    torch.testing.assert_close(out_eager, out_triton, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_forward_half_precision(dtype):
    """Triton works with half-precision types."""
    _skip_no_cuda()
    inputs = _make_inputs(dtype=dtype, num_topk=2)

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    # Wider tolerance for half precision
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_triton_larger_sequence():
    """Triton handles larger sequence lengths correctly."""
    _skip_no_cuda()
    inputs = _make_inputs(
        batch=1,
        heads=2,
        seq_len=256,
        head_dim=64,
        sparse_seq_len=64,
        num_topk=4,
        sliding_window_size=32,
    )

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_precision_vs_fp64(share_kv, num_topk, dtype):
    """Report max forward/backward diffs between a lower-precision dtype and fp64.

    This characterizes the numerical error introduced by the given precision so that
    tolerance thresholds for kernel tests can be set with confidence.

    Inputs are generated in the low-precision dtype first, then promoted to FP64 via
    .double(). This ensures eager, Triton, and FP64 all see the same quantized values
    so that the FP64 baseline isolates arithmetic error without input-quantization noise.
    """
    _skip_no_cuda()
    batch, heads, seq_len, head_dim = 2, 4, 32, 64
    sparse_seq_len = 16
    sliding_window_size = 8
    kv_heads = 1 if share_kv else heads
    seed = 77
    device = torch.device("cuda")
    dtype_name = str(dtype).removeprefix("torch.")

    # --- Generate inputs in the low-precision dtype (the "quantized" source) ---
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn_lp(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    query_lp = randn_lp(batch, heads, seq_len, head_dim)
    local_kv_lp = randn_lp(batch, kv_heads, seq_len, head_dim)
    sparse_kv_lp = randn_lp(batch, kv_heads, sparse_seq_len, head_dim)

    scores = torch.randn(batch, seq_len, sparse_seq_len, dtype=dtype, device=device, generator=gen)
    if num_topk > 0:
        _, kv_indices = torch.topk(scores, k=min(num_topk, sparse_seq_len), dim=-1)
    else:
        kv_indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    sink_lp = randn_lp(heads)

    # --- Derive FP64 inputs from the same quantized values ---
    query_64 = query_lp.double().requires_grad_(True)
    local_kv_64 = local_kv_lp.double().requires_grad_(True)
    sparse_kv_64 = sparse_kv_lp.double().requires_grad_(True)
    sink_64 = sink_lp.double().requires_grad_(True)

    # --- Lower-precision copies for eager and triton ---
    query_lp_ref = query_lp.clone().requires_grad_(True)
    local_kv_lp_ref = local_kv_lp.clone().requires_grad_(True)
    sparse_kv_lp_ref = sparse_kv_lp.clone().requires_grad_(True)
    sink_lp_ref = sink_lp.clone().requires_grad_(True)

    query_lp_tri = query_lp.clone().requires_grad_(True)
    local_kv_lp_tri = local_kv_lp.clone().requires_grad_(True)
    sparse_kv_lp_tri = sparse_kv_lp.clone().requires_grad_(True)
    sink_lp_tri = sink_lp.clone().requires_grad_(True)

    # --- Forward (fp64 reference as ground truth) ---
    out_64 = selected_attention(
        query_64,
        local_kv_64,
        sparse_kv_64,
        kv_indices,
        sink_64,
        None,
        sliding_window_size,
        backend="eager",
    )
    out_lp_ref = selected_attention(
        query_lp_ref,
        local_kv_lp_ref,
        sparse_kv_lp_ref,
        kv_indices,
        sink_lp_ref,
        None,
        sliding_window_size,
        backend="eager",
    )
    out_lp_tri = selected_attention(
        query_lp_tri,
        local_kv_lp_tri,
        sparse_kv_lp_tri,
        kv_indices,
        sink_lp_tri,
        None,
        sliding_window_size,
        backend="triton",
    )

    # --- Backward ---
    grad_gen = torch.Generator(device=device).manual_seed(1234)
    grad_lp = torch.randn(out_lp_ref.shape, dtype=dtype, device=device, generator=grad_gen)
    grad_64 = grad_lp.double()

    out_64.backward(grad_64)
    out_lp_ref.backward(grad_lp)
    out_lp_tri.backward(grad_lp)

    # --- Report ratios (visible with pytest -s) ---
    ref_fwd_diff = (out_64 - out_lp_ref.double()).abs().max().item()
    tri_fwd_diff = (out_64 - out_lp_tri.double()).abs().max().item()
    ref_dq = (query_64.grad - query_lp_ref.grad.double()).abs().max().item()
    tri_dq = (query_64.grad - query_lp_tri.grad.double()).abs().max().item()
    ref_dkv = (local_kv_64.grad - local_kv_lp_ref.grad.double()).abs().max().item()
    tri_dkv = (local_kv_64.grad - local_kv_lp_tri.grad.double()).abs().max().item()
    ref_didx = (sparse_kv_64.grad - sparse_kv_lp_ref.grad.double()).abs().max().item()
    tri_didx = (sparse_kv_64.grad - sparse_kv_lp_tri.grad.double()).abs().max().item()
    ref_dsink = (sink_64.grad - sink_lp_ref.grad.double()).abs().max().item()
    tri_dsink = (sink_64.grad - sink_lp_tri.grad.double()).abs().max().item()

    def _ratio(tri_val, ref_val):
        if ref_val == 0:
            return float("inf") if tri_val > 0 else 1.0
        return tri_val / ref_val

    r_fwd = _ratio(tri_fwd_diff, ref_fwd_diff)
    r_dq = _ratio(tri_dq, ref_dq)
    r_dkv = _ratio(tri_dkv, ref_dkv)
    r_didx = _ratio(tri_didx, ref_didx)
    r_dsink = _ratio(tri_dsink, ref_dsink)

    print(
        f"\n[{dtype_name}, share_kv={share_kv}, topk={num_topk}]"
        f"\n  {'':15s} {'fwd':>10s} {'dQ':>10s} {'dKV':>10s} {'dIdx':>10s} {'dSink':>10s}"
        f"\n  {f'ref {dtype_name}':15s} {ref_fwd_diff:10.4e} {ref_dq:10.4e} {ref_dkv:10.4e}"
        f" {ref_didx:10.4e} {ref_dsink:10.4e}"
        f"\n  {f'triton {dtype_name}':15s} {tri_fwd_diff:10.4e} {tri_dq:10.4e} {tri_dkv:10.4e}"
        f" {tri_didx:10.4e} {tri_dsink:10.4e}"
        f"\n  {'triton/ref':15s} {r_fwd:10.2f}x {r_dq:10.2f}x {r_dkv:10.2f}x"
        f" {r_didx:10.2f}x {r_dsink:10.2f}x"
    )

    # The reduction dimension for QK is head_dim; for the softmax/PV step it is
    # sliding_window_size + num_topk (local window tokens + selected sparse tokens).
    fwd_reduction_sizes = (
        head_dim,
        sliding_window_size + num_topk,
        sliding_window_size + num_topk,
    )

    # We reduce over head dim, effective sequence length, and batch size for backward pass
    dq_reduction_sizes = (
        fwd_reduction_sizes + (head_dim,) + (sliding_window_size + num_topk,) + (batch,)
    )
    # We reduce over head dim, probs @ grad, grad @ query, and batch
    dkv_reduction_sizes = (
        fwd_reduction_sizes
        + (head_dim,)
        + (
            sliding_window_size,
            sliding_window_size,
        )
        + (batch,)
    )
    dsparse_kv_reduction_sizes = fwd_reduction_sizes + (num_topk,) + (batch,)
    # _bwd_dq (sink part): delta (head_dim) + atomic_add over seq_len query positions.
    dsink_reduction_sizes = fwd_reduction_sizes + (head_dim, seq_len) + (batch,)

    assert_matches_low_precision_eager(
        out_lp_tri,
        out_lp_ref,
        out_64,
        reduction_sizes=fwd_reduction_sizes,
    )

    assert_matches_low_precision_eager(
        query_lp_tri.grad,
        query_lp_ref.grad,
        query_64.grad,
        reduction_sizes=dq_reduction_sizes,
    )
    assert_matches_low_precision_eager(
        local_kv_lp_tri.grad,
        local_kv_lp_ref.grad,
        local_kv_64.grad,
        reduction_sizes=dkv_reduction_sizes,
    )
    assert_matches_low_precision_eager(
        sparse_kv_lp_tri.grad,
        sparse_kv_lp_ref.grad,
        sparse_kv_64.grad,
        reduction_sizes=dsparse_kv_reduction_sizes,
    )
    assert_matches_low_precision_eager(
        sink_lp_tri.grad,
        sink_lp_ref.grad,
        sink_64.grad,
        reduction_sizes=dsink_reduction_sizes,
    )
