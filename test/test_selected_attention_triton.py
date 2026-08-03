'''
To check reference bf16 vs triton bf16 diffs, run:
python -m pytest test/test_selected_attention_triton.py::test_reference_bf16_vs_fp64_precision -v -s

Triton max diff is between 0.59-1.23x the reference's max diff
'''

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


def _make_inputs(
    batch: int = 2,
    heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    index_seq_len: int = 16,
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

    Q = randn(batch, heads, seq_len, head_dim)
    KV = randn(batch, kv_heads, seq_len, head_dim)
    index_kv = randn(batch, kv_heads, index_seq_len, head_dim)

    if num_topk > 0:
        scores = torch.randn(batch, seq_len, index_seq_len, device=device, generator=generator)
        _, indices = torch.topk(scores, k=min(num_topk, index_seq_len), dim=-1)
    else:
        indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)

    attention_sink = torch.randn(
        heads, device=device, dtype=dtype, generator=generator, requires_grad=requires_grad
    )

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
        "share_kv": share_kv,
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
    doc_ids = torch.cat([
        torch.zeros(seq_len // 2, dtype=torch.long),
        torch.ones(seq_len // 2, dtype=torch.long),
    ]).unsqueeze(0).expand(2, -1)

    inputs = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, seq_len=seq_len, doc_ids=doc_ids
    )

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_backward_dq(share_kv, num_topk):
    """Triton backward produces correct gradients for Q."""
    _skip_no_cuda()
    inputs_ref = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=123
    )
    inputs_tri = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=123
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(7777)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri["Q"].grad, inputs_ref["Q"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_backward_dlocal_kv(share_kv, num_topk):
    """Triton backward produces correct gradients for local KV."""
    _skip_no_cuda()
    inputs_ref = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=456
    )
    inputs_tri = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=456
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(8888)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri["KV"].grad, inputs_ref["KV"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [1, 3])
def test_triton_backward_dindex_kv(share_kv, num_topk):
    """Triton backward produces correct gradients for index KV."""
    _skip_no_cuda()
    inputs_ref = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=789
    )
    inputs_tri = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=789
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(9999)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri["index_kv"].grad,
        inputs_ref["index_kv"].grad,
        atol=ATOL_BWD,
        rtol=RTOL_BWD,
    )


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_backward_dsink(share_kv, num_topk):
    """Triton backward produces correct gradients for attention_sink."""
    _skip_no_cuda()
    inputs_ref = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=321
    )
    inputs_tri = _make_inputs(
        share_kv=share_kv, num_topk=num_topk, requires_grad=True, seed=321
    )

    out_ref = selected_attention(**inputs_ref, backend="eager")
    out_tri = selected_attention(**inputs_tri, backend="triton")

    grad_gen = torch.Generator(device=out_ref.device).manual_seed(5555)
    grad_output = torch.randn(out_ref.shape, device=out_ref.device, generator=grad_gen)
    out_ref.backward(grad_output)
    out_tri.backward(grad_output)

    torch.testing.assert_close(
        inputs_tri["attention_sink"].grad,
        inputs_ref["attention_sink"].grad,
        atol=ATOL_BWD,
        rtol=RTOL_BWD,
    )


@pytest.mark.parametrize("num_topk", [0, 2])
def test_triton_backward_with_doc_ids(num_topk):
    """Triton backward with doc_ids matches the reference."""
    _skip_no_cuda()
    seq_len = 32
    doc_ids = torch.cat([
        torch.zeros(seq_len // 2, dtype=torch.long),
        torch.ones(seq_len // 2, dtype=torch.long),
    ]).unsqueeze(0).expand(2, -1)

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
        inputs_tri["Q"].grad, inputs_ref["Q"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )
    torch.testing.assert_close(
        inputs_tri["KV"].grad, inputs_ref["KV"].grad, atol=ATOL_BWD, rtol=RTOL_BWD
    )


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
        batch=1, heads=2, seq_len=256, head_dim=64,
        index_seq_len=64, num_topk=4, sliding_window_size=32,
    )

    with torch.inference_mode():
        expected = selected_attention(**inputs, backend="eager")
        actual = selected_attention(**inputs, backend="triton")

    torch.testing.assert_close(actual, expected, atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk", [0, 2, 4])
def test_reference_bf16_vs_fp64_precision(share_kv, num_topk):
    """Report max forward/backward diffs between bf16 and fp64 reference runs.

    This characterizes the numerical error introduced by bf16 precision so that
    tolerance thresholds for kernel tests can be set with confidence.
    """
    _skip_no_cuda()
    batch, heads, seq_len, head_dim = 2, 4, 32, 64
    index_seq_len = 16
    sliding_window_size = 8
    kv_heads = 1 if share_kv else heads
    seed = 77
    device = torch.device("cuda")

    # --- Generate inputs in fp64 (ground truth) ---
    gen64 = torch.Generator(device=device).manual_seed(seed)

    def randn64(*shape):
        return torch.randn(
            *shape, dtype=torch.float64, device=device, generator=gen64, requires_grad=True
        )

    Q_64 = randn64(batch, heads, seq_len, head_dim)
    KV_64 = randn64(batch, kv_heads, seq_len, head_dim)
    index_kv_64 = randn64(batch, kv_heads, index_seq_len, head_dim)
    scores_64 = torch.randn(
        batch, seq_len, index_seq_len, dtype=torch.float64, device=device, generator=gen64
    )
    if num_topk > 0:
        _, indices = torch.topk(scores_64, k=min(num_topk, index_seq_len), dim=-1)
    else:
        indices = torch.zeros(batch, seq_len, 0, dtype=torch.long, device=device)
    sink_64 = torch.randn(
        heads, dtype=torch.float64, device=device, generator=gen64, requires_grad=True
    )

    # --- bf16 copies on CUDA (cast from the same fp64 values) ---
    Q_bf_ref = Q_64.detach().to(torch.bfloat16).requires_grad_(True)
    KV_bf_ref = KV_64.detach().to(torch.bfloat16).requires_grad_(True)
    index_kv_bf_ref = index_kv_64.detach().to(torch.bfloat16).requires_grad_(True)
    sink_bf_ref = sink_64.detach().to(torch.bfloat16).requires_grad_(True)

    Q_bf_tri = Q_64.detach().to(torch.bfloat16).requires_grad_(True)
    KV_bf_tri = KV_64.detach().to(torch.bfloat16).requires_grad_(True)
    index_kv_bf_tri = index_kv_64.detach().to(torch.bfloat16).requires_grad_(True)
    sink_bf_tri = sink_64.detach().to(torch.bfloat16).requires_grad_(True)

    # --- Forward (fp64 reference as ground truth) ---
    out_64 = selected_attention(
        Q_64, KV_64, index_kv_64, indices, sink_64, None,
        sliding_window_size, share_kv, backend="eager",
    )
    out_bf_ref = selected_attention(
        Q_bf_ref, KV_bf_ref, index_kv_bf_ref, indices, sink_bf_ref, None,
        sliding_window_size, share_kv, backend="eager",
    )
    out_bf_tri = selected_attention(
        Q_bf_tri, KV_bf_tri, index_kv_bf_tri, indices, sink_bf_tri, None,
        sliding_window_size, share_kv, backend="triton",
    )

    ref_fwd_diff = (out_64.float() - out_bf_ref.float()).abs().max().item()
    tri_fwd_diff = (out_64.float() - out_bf_tri.float()).abs().max().item()

    # --- Backward ---
    grad_gen = torch.Generator(device=device).manual_seed(1234)
    grad_64 = torch.randn(out_64.shape, dtype=torch.float64, device=device, generator=grad_gen)
    grad_bf = grad_64.to(torch.bfloat16)

    out_64.backward(grad_64)
    out_bf_ref.backward(grad_bf)
    out_bf_tri.backward(grad_bf)

    ref_dq = (Q_64.grad.float() - Q_bf_ref.grad.float()).abs().max().item()
    tri_dq = (Q_64.grad.float() - Q_bf_tri.grad.float()).abs().max().item()
    ref_dkv = (KV_64.grad.float() - KV_bf_ref.grad.float()).abs().max().item()
    tri_dkv = (KV_64.grad.float() - KV_bf_tri.grad.float()).abs().max().item()
    ref_didx = (index_kv_64.grad.float() - index_kv_bf_ref.grad.float()).abs().max().item()
    tri_didx = (index_kv_64.grad.float() - index_kv_bf_tri.grad.float()).abs().max().item()
    ref_dsink = (sink_64.grad.float() - sink_bf_ref.grad.float()).abs().max().item()
    tri_dsink = (sink_64.grad.float() - sink_bf_tri.grad.float()).abs().max().item()

    # Compute ratios (triton error / reference error). Ratio ~1 means triton
    # adds no precision loss beyond bf16 itself. >1 means triton is worse.
    def _ratio(tri_val, ref_val):
        if ref_val == 0:
            return float("inf") if tri_val > 0 else 1.0
        return tri_val / ref_val

    r_fwd = _ratio(tri_fwd_diff, ref_fwd_diff)
    r_dq = _ratio(tri_dq, ref_dq)
    r_dkv = _ratio(tri_dkv, ref_dkv)
    r_didx = _ratio(tri_didx, ref_didx)
    r_dsink = _ratio(tri_dsink, ref_dsink)

    # Print a report (visible with pytest -s)
    print(
        f"\n[share_kv={share_kv}, topk={num_topk}]"
        f"\n  {'':15s} {'fwd':>10s} {'dQ':>10s} {'dKV':>10s} {'dIdx':>10s} {'dSink':>10s}"
        f"\n  {'ref bf16':15s} {ref_fwd_diff:10.4e} {ref_dq:10.4e} {ref_dkv:10.4e}"
        f" {ref_didx:10.4e} {ref_dsink:10.4e}"
        f"\n  {'triton bf16':15s} {tri_fwd_diff:10.4e} {tri_dq:10.4e} {tri_dkv:10.4e}"
        f" {tri_didx:10.4e} {tri_dsink:10.4e}"
        f"\n  {'triton/ref':15s} {r_fwd:10.2f}x {r_dq:10.2f}x {r_dkv:10.2f}x"
        f" {r_didx:10.2f}x {r_dsink:10.2f}x"
    )

    # Sanity checks to make sure we don't get anything insane
    # Not really meant for checking correctness
    # This test is intended to be run with pytest -s
    assert tri_fwd_diff < 0.1, f"Triton fwd diff too large: {tri_fwd_diff}"
    assert tri_dq < 0.1, f"Triton dQ diff too large: {tri_dq}"
    assert tri_dkv < 0.1, f"Triton dKV diff too large: {tri_dkv}"
    assert tri_dsink < 0.1, f"Triton dSink diff too large: {tri_dsink}"
