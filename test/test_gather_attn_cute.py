"""
Tests for the CuTe DSL (SM100/SM103) backend of gather attention.

Validates forward and backward precision against an FP64 eager baseline.
CuTe constraints: head_dim=512, 1 <= nheads <= 128, share_kv=True, dtype=bfloat16, SM100 or SM103.
Fewer than 128 heads require FA4's sparse-MLA head-padding support.

Note: torch.compile is NOT supported for the CuTe backend (eager-only).
"""

import inspect
import math

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.sparse.gather_attn.impl.cute import SUPPORTED_CAPABILITIES
from attn_gym.testing.kda import assert_relative_rms_within


def _skip_no_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CuTe backend")
    if torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES:
        pytest.skip("SM100 or SM103 required for CuTe backend")


def skip_unsupported_cute_sink(sink_dtype: torch.dtype | None) -> None:
    """Skip only sink cases until the installed sparse-MLA kernel supports them."""
    if sink_dtype is None:
        return
    mla = pytest.importorskip("flash_attn.cute.flash_fwd_mla_sm100")
    # The public flash_attn_func already accepted sinks before the MLA kernel did.
    if (
        "learnable_sink"
        not in inspect.signature(mla.FlashAttentionMLAForwardSm100.__call__).parameters
    ):
        pytest.skip("installed FA4 sparse-MLA kernel does not support attention sinks yet")


def assert_matches_low_precision_eager(
    actual,
    low_precision_expected,
    high_precision_expected,
    reduction_sizes,
    computation_dtype=None,
):
    """Bound kernel error by low-precision eager error against an FP64 measuring stick."""
    assert torch.isfinite(actual).all()
    actual_difference = (actual.double() - high_precision_expected).abs()
    eager_difference = (low_precision_expected.double() - high_precision_expected).abs()
    accumulation_eps = (
        sum(math.sqrt(size) for size in reduction_sizes) * torch.finfo(torch.float32).eps
    )
    output_rounding_eps = torch.finfo(computation_dtype or actual.dtype).eps
    rounding_eps = accumulation_eps + output_rounding_eps
    mean_atol = rounding_eps * high_precision_expected.abs().mean().item()
    max_atol = (accumulation_eps + len(reduction_sizes) * output_rounding_eps) * (
        high_precision_expected.abs().max().item()
    )
    assert actual_difference.mean().item() <= eager_difference.mean().item() + mean_atol
    assert actual_difference.max().item() <= eager_difference.max().item() + max_atol
    # The pointwise budgets above catch outliers; two source-dtype eps bound aggregate drift.
    assert_relative_rms_within(
        actual,
        high_precision_expected,
        "gather_attn",
        max_eps=2,
        source_dtype=computation_dtype or actual.dtype,
    )


# ---------------------------------------------------------------------------
# Precision vs FP64 (measuring-stick test)
# ---------------------------------------------------------------------------


def check_cute_precision(
    *,
    batch: int,
    num_topk: int,
    packed: bool,
    seq_len: int,
    sparse_seq_len: int,
    sliding_window_size: int,
    sink_dtype: torch.dtype | None,
    scale: float | None,
    kernel_options: dict[str, str] | None,
    heads: int = 128,
) -> None:
    """CuTe bf16 error bounded by low-precision eager error vs FP64.

    Inputs are generated in bf16 first, then promoted to FP64 via .double().
    This ensures eager, CuTe, and FP64 all see the same quantized values
    so that the FP64 baseline isolates arithmetic error without input-quantization noise.
    """
    skip_unsupported_cute_sink(sink_dtype)
    head_dim = 512
    seed = 77
    device = torch.device("cuda")
    dtype = torch.bfloat16

    cu_seqlens = cu_seqlens_k = None
    if packed:
        assert batch == 1
        cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(
            [0, sparse_seq_len // 2, sparse_seq_len], dtype=torch.int32, device=device
        )

    # --- Generate inputs in bf16 (the "quantized" source) ---
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn_lp(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    query_lp = randn_lp(batch, heads, seq_len, head_dim)
    local_kv_lp = randn_lp(batch, 1, seq_len, head_dim)
    sparse_kv_lp = randn_lp(batch, 1, sparse_seq_len, head_dim)

    kv_indices = torch.full((batch, seq_len, num_topk), -1, dtype=torch.int64, device=device)
    segments = (
        [
            (0, seq_len // 2, sparse_seq_len // 2),
            (seq_len // 2, seq_len, sparse_seq_len - sparse_seq_len // 2),
        ]
        if packed
        else [(0, seq_len, sparse_seq_len)]
    )
    for start, end, pool_length in segments:
        scores = torch.randn(
            batch, end - start, pool_length, dtype=dtype, device=device, generator=gen
        )
        selected = scores.topk(min(num_topk, pool_length), dim=-1).indices
        kv_indices[:, start:end, : selected.shape[-1]] = selected

    sink = (
        torch.linspace(-2, 8, heads * 2, device=device, dtype=sink_dtype)[::2]
        if sink_dtype is not None
        else None
    )

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

    sink_64 = sink.double().requires_grad_(True) if sink is not None else None
    sink_ref = sink.detach().requires_grad_(True) if sink is not None else None
    sink_cute = sink.detach().requires_grad_(True) if sink is not None else None

    # --- Forward ---
    out_64 = gather_attn(
        query_64,
        local_kv_64,
        sparse_kv_64,
        kv_indices,
        sink_64,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        sliding_window_size=sliding_window_size,
        impl=Impl.REFERENCE,
        scale=scale,
    )
    out_lp_ref = gather_attn(
        query_lp_ref,
        local_kv_lp_ref,
        sparse_kv_lp_ref,
        kv_indices,
        sink_ref,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        sliding_window_size=sliding_window_size,
        impl=Impl.REFERENCE,
        scale=scale,
    )
    out_lp_cute = gather_attn(
        query_lp_cute,
        local_kv_lp_cute,
        sparse_kv_lp_cute,
        kv_indices,
        sink_cute,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        sliding_window_size=sliding_window_size,
        kernel_options=kernel_options,
        scale=scale,
    )

    # --- Backward ---
    grad_gen = torch.Generator(device=device).manual_seed(1234)
    grad_lp = torch.randn(out_lp_ref.shape, dtype=dtype, device=device, generator=grad_gen)
    grad_64 = grad_lp.double()

    out_64.backward(grad_64)
    out_lp_ref.backward(grad_lp)
    out_lp_cute.backward(grad_lp)

    # --- Reduction sizes ---
    effective_topk = min(num_topk, sparse_seq_len) if num_topk > 0 else 0
    num_keys = sliding_window_size + effective_topk
    fwd_reduction_sizes = (head_dim, num_keys, num_keys)
    dq_reduction_sizes = fwd_reduction_sizes + (head_dim,) + (num_keys,) + (batch,)
    dkv_reduction_sizes = (
        fwd_reduction_sizes + (head_dim,) + (sliding_window_size, sliding_window_size) + (batch,)
    )
    dsparse_kv_reduction_sizes = fwd_reduction_sizes + (effective_topk,) + (batch,)

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
    if effective_topk > 0:
        assert_matches_low_precision_eager(
            sparse_kv_lp_cute.grad,
            sparse_kv_lp_ref.grad,
            sparse_kv_64.grad,
            reduction_sizes=dsparse_kv_reduction_sizes,
        )

    if sink is not None:
        assert sink_cute.grad.dtype == sink_dtype
        assert_matches_low_precision_eager(
            sink_cute.grad,
            sink_ref.grad,
            sink_64.grad,
            reduction_sizes=fwd_reduction_sizes + (head_dim, batch * seq_len),
            # FA4 computes dsink from the BF16 output and upstream gradient,
            # even when the sink parameter and its gradient are stored in FP32.
            computation_dtype=dtype,
        )


@pytest.mark.parametrize(
    "heads,kernel_options,sink_dtype",
    [pytest.param(128, {"backend": "cute"}, torch.float32, id="forced-sink")]
    + [
        pytest.param(
            heads, None, sink_dtype, id=f"auto-{heads}-{'sink' if sink_dtype else 'no-sink'}"
        )
        for heads in (1, 24, 64, 128)
        for sink_dtype in (None, torch.float32)
    ],
)
def test_cute_dependency_smoke(heads, kernel_options, sink_dtype):
    """Exercise auto dispatch through the real FA4 forward and backward, when supported."""
    from attn_gym.sparse.gather_attn.api import _select_backend
    from attn_gym.sparse.gather_attn.impl.cute import _fa4_available

    _skip_no_sm100()
    if not _fa4_available(with_sink=sink_dtype is not None, padded_heads=heads != 128):
        pytest.skip("installed FA4 does not support the requested sparse-MLA features")
    query = torch.empty(1, heads, 8, 512, device="cuda", dtype=torch.bfloat16)
    sink = (
        torch.empty(heads * 2, device="cuda", dtype=sink_dtype)[::2]
        if sink_dtype is not None
        else None
    )
    # A singleton strided sink is unsupported by FA4's layout conversion.
    expected_backend = "triton" if heads == 1 and sink is not None else "cute"
    assert _select_backend(query, sink, True, num_keys=6) == expected_backend
    check_cute_precision(
        batch=1,
        num_topk=2,
        packed=True,
        seq_len=8,
        sparse_seq_len=8,
        sliding_window_size=4,
        sink_dtype=sink_dtype,
        scale=0.025,
        kernel_options=kernel_options,
        heads=heads,
    )


@pytest.mark.parametrize("kernel_options", [None, {"backend": "cute"}], ids=["auto", "forced"])
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn-only"])
def test_cute_determinism_enabled_after_forward(kernel_options, warn_only):
    from attn_gym.sparse.gather_attn.api import _select_backend
    from attn_gym.sparse.gather_attn.impl.cute import _fa4_available

    _skip_no_sm100()
    if not _fa4_available(with_sink=False):
        pytest.skip("FA4 is not installed")
    query = torch.randn(1, 128, 8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, 1, 8, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(1, 8, 2, device="cuda", dtype=torch.int32)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        assert _select_backend(query, None, True, num_keys=6) == "cute"
        out, aux = gather_attn(
            query,
            kv,
            kv,
            indices,
            sliding_window_size=4,
            kernel_options=kernel_options,
            return_aux=AuxRequest(lse=True),
        )
        assert not aux.lse.requires_grad
        with pytest.raises(RuntimeError, match="does not require grad"):
            aux.lse.sum().backward()
        if warn_only:
            (expected_grad,) = torch.autograd.grad(out.sum(), query, retain_graph=True)
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if warn_only:
            with pytest.warns(UserWarning, match="use kernel_options"):
                out.sum().backward()
            torch.testing.assert_close(query.grad, expected_grad)
        else:
            with pytest.raises(RuntimeError, match="use kernel_options"):
                out.sum().backward()
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


# Keep every original shape/top-k pair, crossed only with window and doc masking (64 cases).
# Every sink/scale combination occurs, and every shape and top-k sees all three sink/scale
# values. Sinks are strided here; the LSE cases below exercise contiguous sinks.
@pytest.mark.parametrize(
    "seq_len,sparse_seq_len,num_topk,sink_dtype,scale",
    [
        pytest.param(128, 128, 16, None, None, id="s128-sp128-k16-no-sink-default"),
        pytest.param(128, 128, 32, torch.bfloat16, 0.025, id="s128-sp128-k32-bf16-small"),
        pytest.param(128, 128, 64, torch.float32, 0.125, id="s128-sp128-k64-fp32-large"),
        pytest.param(128, 128, 128, None, None, id="s128-sp128-k128-no-sink-default"),
        pytest.param(256, 128, 16, torch.bfloat16, 0.125, id="s256-sp128-k16-bf16-large"),
        pytest.param(256, 128, 32, torch.float32, None, id="s256-sp128-k32-fp32-default"),
        pytest.param(256, 128, 64, None, 0.025, id="s256-sp128-k64-no-sink-small"),
        pytest.param(256, 128, 128, torch.float32, 0.025, id="s256-sp128-k128-fp32-small"),
        pytest.param(256, 256, 16, torch.float32, 0.025, id="s256-sp256-k16-fp32-small"),
        pytest.param(256, 256, 32, None, 0.125, id="s256-sp256-k32-no-sink-large"),
        pytest.param(256, 256, 64, torch.bfloat16, None, id="s256-sp256-k64-bf16-default"),
        pytest.param(256, 256, 128, torch.bfloat16, 0.025, id="s256-sp256-k128-bf16-small"),
        pytest.param(128, 256, 16, None, 0.025, id="s128-sp256-k16-no-sink-small"),
        pytest.param(128, 256, 32, torch.bfloat16, None, id="s128-sp256-k32-bf16-default"),
        pytest.param(128, 256, 64, torch.float32, None, id="s128-sp256-k64-fp32-default"),
        pytest.param(128, 256, 128, None, 0.125, id="s128-sp256-k128-no-sink-large"),
    ],
)
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("sliding_window_size", [64, 128])
def test_cute_precision_vs_fp64(
    num_topk, packed, seq_len, sparse_seq_len, sliding_window_size, sink_dtype, scale
):
    """Cover the original shape grid with representative sink/scale forward and gradients."""
    _skip_no_sm100()
    pytest.importorskip("flash_attn.cute", reason="gather-attention CuTe tests require FA4")
    check_cute_precision(
        batch=1 if packed else 2,
        num_topk=num_topk,
        packed=packed,
        seq_len=seq_len,
        sparse_seq_len=sparse_seq_len,
        sliding_window_size=sliding_window_size,
        sink_dtype=sink_dtype,
        scale=scale,
        kernel_options={"backend": "cute"},
    )


# ---------------------------------------------------------------------------
# LSE correctness
# ---------------------------------------------------------------------------


# Default plus one custom scale per sink; the precision matrix covers all nine pairs.
@pytest.mark.parametrize(
    "sink_dtype,scale",
    [
        pytest.param(None, None, id="no-sink-default"),
        pytest.param(None, 0.025, id="no-sink-small"),
        pytest.param(torch.bfloat16, None, id="bf16-default"),
        pytest.param(torch.bfloat16, 0.125, id="bf16-large"),
        pytest.param(torch.float32, None, id="fp32-default"),
        pytest.param(torch.float32, 0.025, id="fp32-small"),
    ],
)
def test_cute_lse_matches_manual_computation(sink_dtype, scale):
    """Returned LSE from CuTe backend matches manual logsumexp over all logits."""
    _skip_no_sm100()
    pytest.importorskip("flash_attn.cute", reason="gather-attention CuTe tests require FA4")
    skip_unsupported_cute_sink(sink_dtype)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, heads, seq_len, head_dim = 1, 128, 64, 512
    sparse_seq_len = 32
    num_topk = 16
    window = 32

    torch.manual_seed(99)
    query = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    local_kv = torch.randn(batch, 1, seq_len, head_dim, device=device, dtype=dtype)
    sparse_kv = torch.randn(batch, 1, sparse_seq_len, head_dim, device=device, dtype=dtype)
    kv_indices = torch.randint(0, sparse_seq_len, (batch, seq_len, num_topk), device=device)
    sink = (
        torch.linspace(-2, 8, heads, device=device, dtype=sink_dtype)
        if sink_dtype is not None
        else None
    )

    _, aux_cute = gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        sink,
        sliding_window_size=window,
        kernel_options={"backend": "cute"},
        scale=scale,
        return_aux=AuxRequest(lse=True),
    )
    lse_cute = aux_cute.lse

    # Use the eager backend on the same inputs (promoted to fp64) as ground truth.
    _, aux_eager = gather_attn(
        query.double(),
        local_kv.double(),
        sparse_kv.double(),
        kv_indices,
        sink.double() if sink is not None else None,
        sliding_window_size=window,
        impl=Impl.REFERENCE,
        scale=scale,
        return_aux=AuxRequest(lse=True),
    )
    lse_eager = aux_eager.lse

    assert lse_cute.shape == (batch, heads, seq_len)
    assert lse_eager.shape == (batch, heads, seq_len)
    # CuTe operates in bf16/fp32 so allow tolerance comparable to the forward output.
    torch.testing.assert_close(lse_cute.double(), lse_eager, atol=0.05, rtol=0.02)
