# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU unit tests for the optimized (Triton) KDA leaf kernels."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

# These imports intentionally follow the optional-dependency check above.
from attn_gym._backends.triton.utils import can_use_tma
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.bwd.triton.l2norm_bwd import l2norm_bwd_kernel
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import (
    chunk_gla_fwd_kernel_o,
    chunk_gla_fwd_o_gk,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop,
)
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import (
    _l2norm_bwd_op,
    _l2norm_fwd_op,
    _l2norm_launch_config,
    l2norm,
    l2norm_fwd_kernel,
)
from attn_gym.linear.kda.naive import l2norm_bwd_ref, l2norm_fwd_ref
from attn_gym.linear.kda.utils import IS_GATHER_SUPPORTED
from attn_gym.testing.kda import (
    bwd_daqk_reference,
)
from attn_gym.testing.kda import (
    bwd_intra_reference as _bwd_intra_ref,
)
from attn_gym.testing.kda import (
    bwd_wy_dqkg_reference as _bwd_wy_dqkg_ref,
)

IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
    (10, 0),
    (10, 3),
)

try:
    from attn_gym.linear.kda.bwd.cute import chunk_delta_h_bwd as delta_h_bwd
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
        blackwell_delta_h_bwd_dhu_dv_fused,
    )
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
        chunk_kda_bwd_intra as chunk_kda_bwd_intra_cute,
    )
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
        chunk_kda_bwd_wy_dqkg as chunk_kda_bwd_wy_dqkg_fused_cute,
    )
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import (
        chunk_kda_fwd_intra as chunk_kda_fwd_intra_cute,
    )
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd

    HAS_CUTE = True
    CUTE_IMPORT_ERR = ""
except Exception as e:
    optional_dep_missing = isinstance(e, ModuleNotFoundError) and (e.name or "").split(".")[0] in {
        "cutlass",
        "cuda",
    }
    if IS_SM100 and not optional_dep_missing:
        raise
    HAS_CUTE = False
    CUTE_IMPORT_ERR = f"{type(e).__name__}: {e}"

requires_cute = pytest.mark.skipif(
    not (IS_SM100 and HAS_CUTE),
    reason=(
        "CuTe DSL kernels require SM100 (Blackwell) and an importable cutlass"
        + (f"; import failed with {CUTE_IMPORT_ERR}" if IS_SM100 and not HAS_CUTE else "")
    ),
)

DEV = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KDA Triton kernels require a CUDA device"
)

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_DTYPE_IDS = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def _case_id(value):
    """Keep generated case IDs concise and readable."""
    if value in _DTYPE_IDS:
        return _DTYPE_IDS[value]
    if isinstance(value, tuple):
        return "docs-" + "-".join(map(str, value))
    return None


def _autotuner(kernel):
    """Return the Triton Autotuner below optional heuristic wrappers."""
    while not hasattr(kernel, "configs"):
        kernel = kernel.fn
    return kernel


def _representative_config(kernel, *, num_warps, num_stages=None, **kwargs):
    """Select one explicit production config for a correctness test."""
    autotuner = _autotuner(kernel)
    matches = [
        config
        for config in autotuner.configs
        if config.num_warps == num_warps
        and (num_stages is None or config.num_stages == num_stages)
        and all(config.kwargs.get(name) == value for name, value in kwargs.items())
    ]
    assert len(matches) == 1
    return autotuner, matches[0]


@pytest.fixture(scope="module", autouse=True)
def use_one_autotune_config_for_correctness_tests():
    """Avoid benchmarking configs whose outputs these tests never inspect."""
    selections = (_representative_config(l2norm_bwd_kernel, BT=16, num_warps=4),)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
        for autotuner, config in selections:
            monkeypatch.setattr(autotuner, "configs", [config])
        yield


# Golden / plus-minus comparison helper
def assert_golden(
    actual: torch.Tensor,
    golden: torch.Tensor,
    ref: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> None:
    golden64 = golden.to(torch.float64)
    band = torch.finfo(dtype).eps * golden64.abs().max().item()
    a_err = (actual.to(torch.float64) - golden64).abs().max().item()
    r_err = (ref.to(torch.float64) - golden64).abs().max().item()
    budget = 2.0 * r_err + 2.0 * band
    assert a_err <= budget, (
        f"{name}: kernel err {a_err:.3e} exceeds budget {budget:.3e} "
        f"(ref err {r_err:.3e}, band {band:.3e}, dtype {dtype})"
    )


def _empty_strided_like(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate an output view with the same shape and a non-unit innermost stride."""
    storage_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    return torch.empty(storage_shape, dtype=tensor.dtype, device=tensor.device)[..., ::2]


def _strided_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor into a view with a non-unit innermost stride."""
    view = _empty_strided_like(tensor)
    view.copy_(tensor)
    return view


# Shapes chosen to straddle chunk boundaries: single token, one-below / exact /
# one-above the internal block, and multi-block. One existing fp32 case combines
# a non-power-of-two feature dimension with strided inputs and outputs, avoiding
# an additional parameterized case and its compilation cost.
L2NORM_SHAPES = (
    (1, 128, False),
    (7, 96, True),
    (16, 128, False),
    (17, 64, False),
    (64, 128, False),
    (130, 32, False),
)
_L2NORM_CASES = [(torch.float32, T, D, strided) for T, D, strided in L2NORM_SHAPES] + [
    (dtype, 17, 64, False) for dtype in DTYPES[1:]
]


# l2norm forward
@pytest.mark.parametrize(
    ("rows", "tuned_major", "expected"),
    [
        (1, 9, (1, 1)),
        (2, 9, (1, 1)),
        (9, 9, (1, 1)),
        (10, 9, (4, 2)),
        (2048, 9, (4, 2)),
        (2049, 9, (16, 4)),
        (8, 10, (1, 1)),
        (10, 10, (1, 4)),
        (32, 10, (1, 4)),
        (34, 10, (4, 4)),
        (512, 10, (4, 4)),
        (514, 10, (8, 4)),
        (128, None, (16, 4)),
    ],
)
def test_l2norm_launch_config(
    rows: int,
    tuned_major: int | None,
    expected: tuple[int, int],
):
    assert _l2norm_launch_config(rows, tuned_major) == expected


@pytest.mark.parametrize("dtype,T,D,strided", _L2NORM_CASES, ids=_case_id)
def test_l2norm_fwd(dtype, T, D, strided):
    torch.manual_seed(0)
    eps = 1e-6
    x64 = torch.randn(T, D, device=DEV, dtype=torch.float64)
    x = x64.to(dtype)
    if strided:
        x = _strided_copy(x)

    golden = l2norm_fwd_ref(x64, eps)
    ref = l2norm_fwd_ref(x, eps)

    rstd_golden = torch.rsqrt((x64 * x64).sum(-1) + eps)
    rstd_ref = torch.rsqrt((x.float() ** 2).sum(-1) + eps)

    BD = triton.next_power_of_2(D)

    # block kernel
    y = _empty_strided_like(x) if strided else torch.empty_like(x)
    rstd_template = torch.empty(T, device=DEV, dtype=torch.float32)
    rstd = _empty_strided_like(rstd_template) if strided else rstd_template
    l2norm_fwd_kernel[(triton.cdiv(T, 16),)](
        x,
        y,
        rstd,
        eps,
        T,
        None,
        X_STRIDES=(0, x.stride(0), 0, x.stride(1)),
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        T=T,
        H=1,
        D=D,
        BD=BD,
        NUM_SEQUENCES=0,
        IS_VARLEN=False,
        BT=16,
        num_warps=4,
        num_stages=3,
    )
    assert_golden(y, golden, ref, dtype, f"l2norm_fwd_kernel T={T} D={D}")
    assert_golden(rstd, rstd_golden, rstd_ref, dtype, f"l2norm_fwd_kernel rstd T={T} D={D}")


@pytest.mark.parametrize("dtype", DTYPES, ids=_case_id)
def test_l2norm_autograd_wrapper(dtype):
    """Check every supported dtype at the fused-backend normalization boundary."""
    torch.manual_seed(1)
    x = torch.randn(2, 17, 3, 128, device=DEV, dtype=dtype, requires_grad=True)
    d_output = torch.randn_like(x)

    output = l2norm(x)
    actual_gradient = torch.autograd.grad(output, x, d_output)[0]
    expected = l2norm_fwd_ref(x.float())
    expected_gradient = torch.autograd.grad(expected, x, d_output.float())[0]
    rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (2e-2, 2e-3)

    assert output.dtype == dtype
    torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        actual_gradient.float(), expected_gradient.float(), rtol=rtol, atol=atol
    )


def test_l2norm_op_registration():
    x = torch.randn(1, 17, 3, 128, device=DEV, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 9, 17, 17], device=DEV, dtype=torch.int32)
    torch.library.opcheck(_l2norm_fwd_op, (x, 1e-6))
    torch.library.opcheck(_l2norm_fwd_op, (x, 1e-6, cu_seqlens))

    output, rstd = _l2norm_fwd_op(x, 1e-6)
    d_output = torch.randn_like(output)
    torch.library.opcheck(
        _l2norm_bwd_op,
        (output.view(-1, output.shape[-1]), rstd, d_output),
    )
    ragged_output, ragged_rstd = _l2norm_fwd_op(x, 1e-6, cu_seqlens)
    torch.library.opcheck(
        _l2norm_bwd_op,
        (
            ragged_output.view(-1, ragged_output.shape[-1]),
            ragged_rstd,
            d_output,
            cu_seqlens,
        ),
    )


def test_l2norm_blackwell_kda_decode_matches_reference_and_replays():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell-specific decode schedule")
    torch.manual_seed(2)
    tokens, heads = 32, 4
    qkv = torch.randn(1, tokens, 3, heads, 128, device=DEV, dtype=torch.bfloat16)
    x = qkv[:, :, 0]
    cu_seqlens = torch.arange(tokens + 1, device=DEV, dtype=torch.int32)

    output = l2norm(x, cu_seqlens=cu_seqlens)
    compiled_output = torch.compile(l2norm, fullgraph=True)(x, cu_seqlens=cu_seqlens)
    expected = l2norm_fwd_ref(x.float())
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(compiled_output, output, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = l2norm(x, cu_seqlens=cu_seqlens)

    x.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    expected = l2norm(x, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


def test_l2norm_ragged_matches_exact_active_prefix():
    """Packed normalization ignores NaN-poisoned physical capacity in both passes."""
    torch.manual_seed(2)
    tokens, active_tokens, heads, head_dim = 128, 65, 3, 128
    x = torch.randn(
        1,
        tokens,
        heads,
        head_dim,
        device=DEV,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    d_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 33, 65, 65], device=DEV, dtype=torch.int32)
    with torch.no_grad():
        x[:, active_tokens:].fill_(float("nan"))
        d_output[:, active_tokens:].fill_(float("nan"))

    output = l2norm(x, cu_seqlens=cu_seqlens)
    gradient = torch.autograd.grad(output, x, d_output)[0]
    exact_x = x[:, :active_tokens].detach().clone().requires_grad_()
    exact_d_output = d_output[:, :active_tokens].detach().clone()
    exact_output = l2norm(exact_x)
    exact_gradient = torch.autograd.grad(exact_output, exact_x, exact_d_output)[0]

    torch.testing.assert_close(output[:, :active_tokens], exact_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(gradient[:, :active_tokens], exact_gradient, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(
    "layout",
    (
        "compact",
        "packed-qkv-view",
        "arbitrary-head-inner-strides",
        "arbitrary-batch-token-strides",
    ),
)
def test_l2norm_compile_matches_eager(layout):
    """Keep compiled normalization equivalent across direct and fallback layouts."""
    torch.manual_seed(2)
    if layout == "packed-qkv-view":
        x = torch.randn(1, 1024, 3, 2, 128, device=DEV, dtype=torch.bfloat16)[:, :, 0]
        assert x.stride() == (1024 * 3 * 2 * 128, 3 * 2 * 128, 128, 1)
    elif layout == "arbitrary-head-inner-strides":
        x = torch.randn(1, 1024, 128, 2, device=DEV, dtype=torch.bfloat16).transpose(-1, -2)
        assert x.stride() == (1024 * 2 * 128, 2 * 128, 1, 2)
    elif layout == "arbitrary-batch-token-strides":
        x = torch.randn(128, 2, 2, 128, device=DEV, dtype=torch.bfloat16).transpose(0, 1)
        assert x.stride() == (2 * 128, 2 * 2 * 128, 128, 1)
    else:
        x = torch.randn(1, 1024, 2, 128, device=DEV, dtype=torch.bfloat16)

    x.requires_grad_()
    d_output = torch.randn_like(x)
    expected = l2norm(x)
    actual = torch.compile(l2norm, fullgraph=True, dynamic=True)(x)
    expected_gradient = torch.autograd.grad(expected, x, d_output)[0]
    actual_gradient = torch.autograd.grad(actual, x, d_output)[0]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    lower = torch.nextafter(expected_gradient, torch.full_like(expected_gradient, -torch.inf))
    upper = torch.nextafter(expected_gradient, torch.full_like(expected_gradient, torch.inf))
    assert ((actual_gradient >= lower) & (actual_gradient <= upper)).all()


def test_l2norm_ragged_compile_matches_eager():
    """Compile the public packed forward and backward without reading its endpoint."""
    torch.manual_seed(3)
    tokens, active_tokens = 128, 65
    x = torch.randn(1, tokens, 2, 128, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    d_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 33, 65, 65], device=DEV, dtype=torch.int32)
    with torch.no_grad():
        x[:, active_tokens:].fill_(float("nan"))
        d_output[:, active_tokens:].fill_(float("nan"))

    def operation(x, cu_seqlens):
        return l2norm(x, cu_seqlens=cu_seqlens)

    expected = operation(x, cu_seqlens)
    actual = torch.compile(operation, fullgraph=True, dynamic=True)(x, cu_seqlens)
    expected_gradient = torch.autograd.grad(expected, x, d_output)[0]
    actual_gradient = torch.autograd.grad(actual, x, d_output)[0]

    torch.testing.assert_close(
        actual[:, :active_tokens], expected[:, :active_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_gradient[:, :active_tokens],
        expected_gradient[:, :active_tokens],
        rtol=0,
        atol=0,
    )


def test_l2norm_ragged_cuda_graph_replays_smaller_endpoint():
    """Replay updates the active row bound from static metadata."""
    torch.manual_seed(4)
    tokens, active_tokens = 1024, 65
    x = torch.randn(1, tokens, 3, 128, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    d_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 512, 1024], device=DEV, dtype=torch.int32)

    def operation():
        output = l2norm(x, cu_seqlens=cu_seqlens)
        return output, torch.autograd.grad(output, x, d_output)[0]

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        operation()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output, gradient = operation()

    with torch.no_grad():
        x[:, active_tokens:].fill_(float("nan"))
        d_output[:, active_tokens:].fill_(float("nan"))
        cu_seqlens.copy_(torch.tensor([0, 33, 65], device=DEV, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    exact_x = x[:, :active_tokens].detach().clone().requires_grad_()
    exact_d_output = d_output[:, :active_tokens].detach().clone()
    exact_output = l2norm(exact_x)
    exact_gradient = torch.autograd.grad(exact_output, exact_x, exact_d_output)[0]
    torch.testing.assert_close(output[:, :active_tokens], exact_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(gradient[:, :active_tokens], exact_gradient, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32], ids=["fp64", "int32"])
def test_l2norm_rejects_unsupported_dtype(dtype):
    x = torch.ones(2, 8, device=DEV, dtype=dtype)
    with pytest.raises(TypeError, match="float16, bfloat16, or float32"):
        l2norm(x)


def test_l2norm_rejects_non_kda_shape():
    with pytest.raises(ValueError, match=r"shape \[B, T, H, D\]"):
        l2norm(torch.empty(2, 128, device=DEV))


def test_l2norm_rejects_empty_outer_dimension():
    with pytest.raises(ValueError, match="at least one row"):
        l2norm(torch.empty(1, 0, 2, 128, device=DEV))


# l2norm backward   dx = rstd * (dy - y * <dy, y>)
@pytest.mark.parametrize("dtype,T,D,strided", _L2NORM_CASES, ids=_case_id)
def test_l2norm_bwd(dtype, T, D, strided):
    torch.manual_seed(1)
    eps = 1e-6
    x64 = torch.randn(T, D, device=DEV, dtype=torch.float64)
    dy64 = torch.randn(T, D, device=DEV, dtype=torch.float64)
    rstd64 = 1.0 / torch.sqrt((x64 * x64).sum(-1, keepdim=True) + eps)
    y64 = x64 * rstd64

    golden = l2norm_bwd_ref(y64, rstd64, dy64)

    # kernel inputs are the low-precision y/rstd/dy
    y = y64.to(dtype)
    dy = dy64.to(dtype)
    rstd = rstd64.squeeze(-1).to(torch.float32)
    if strided:
        y, dy, rstd = map(_strided_copy, (y, dy, rstd))

    ref = l2norm_bwd_ref(y, rstd.unsqueeze(-1), dy)

    BD = triton.next_power_of_2(D)

    dx = _empty_strided_like(y) if strided else torch.empty_like(y)
    grid = lambda meta: (triton.cdiv(T, meta["BT"]),)
    # Present physical [T, D] storage to the kernel as logical [1, T, 1, D].
    dy_strides = (0, dy.stride(0), 0, dy.stride(1))
    l2norm_bwd_kernel[grid](
        y,
        rstd,
        dy,
        dx,
        T,
        None,
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=dy_strides,
        DX_STRIDES=dx.stride(),
        TOKENS=T,
        HEADS=1,
        D=D,
        BD=BD,
        NB=triton.cdiv(T, 16),
        NUM_SEQUENCES=0,
        IS_VARLEN=False,
    )
    assert_golden(dx, golden, ref, dtype, f"l2norm_bwd_kernel T={T} D={D}")


@pytest.mark.parametrize("tokens,heads", [(64, 1), (128, 2)])
def test_chunk_kda_bwd_daqk(tokens, heads):
    torch.manual_seed(10)
    shape = (1, tokens, heads, 128)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    d_output = torch.randn_like(v)
    scale = 0.5

    actual = chunk_kda_bwd_daqk(v, d_output, scale)
    expected = bwd_daqk_reference(v, d_output, [tokens], scale)
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)


def _fwd_o_ref(q, g, h, A, v, scale, chunk_size=64, use_exp2=True):
    """Reference for ``chunk_gla_fwd_kernel_o``: ``o = scale*(q*2^g) @ h + tril(A) @ v``.

    ``h`` is the per-chunk state stack ``(B*num_chunks, H, K, V)``; ``A`` is the causal intra-chunk
    matrix (lower-triangular incl. diag); ``v`` carries the intra-chunk pseudo-values.
    """
    B, T, H, _ = q.shape
    V = v.shape[-1]
    num_chunks = (T + chunk_size - 1) // chunk_size
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qc, gc, vc, Ac, hc = q.to(acc), g.to(acc), v.to(acc), A.to(acc), h.to(acc)
    decay = gc.exp2() if use_exp2 else gc.exp()
    qg = qc * decay
    o = torch.zeros(B, T, H, V, dtype=acc, device=q.device)
    hcr = hc.reshape(B, num_chunks, H, hc.shape[-2], V)  # (b, chunk, h, k, v)
    full_tril = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))
    for it in range(num_chunks):
        s = it * chunk_size
        e = min(s + chunk_size, T)
        L = e - s
        inter = torch.einsum("blhk,bhkv->blhv", qg[:, s:e], hcr[:, it]) * scale  # q@h
        Ablk = Ac[:, s:e, :, :L] * full_tril[:L, :L][None, :, None, :]  # (b, l, h, m)
        o[:, s:e] = inter + torch.einsum("blhm,bmhv->blhv", Ablk, vc[:, s:e])  # + tril(A)@v
    return o


@pytest.mark.parametrize(
    "dtype,T,H,K,V",
    [
        (torch.float16, 64, 1, 64, 64),
        (torch.bfloat16, 128, 2, 128, 128),
        (torch.bfloat16, 130, 2, 128, 128),
    ],
    ids=["single-fp16", "multi-bf16", "tail-bf16"],
)
def test_chunk_gla_fwd_o(dtype, T, H, K, V):
    torch.manual_seed(11)
    B = 1
    num_chunks = triton.cdiv(T, 64)
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1  # small so 2^g is scaled
    h = torch.randn(B * num_chunks, H, K, V, device="cuda", dtype=dtype)
    A = torch.randn(B, T, H, 64, device="cuda", dtype=dtype)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)

    golden = _fwd_o_ref(q.double(), g.double(), h.double(), A.double(), v.double(), scale)
    ref = _fwd_o_ref(q, g, h, A, v, scale)

    o = torch.empty(B, T, H, V, device="cuda", dtype=dtype)
    grid = lambda meta: (triton.cdiv(V, meta["BV"]), num_chunks, B * H)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=None,
        chunk_offsets=None,
        scale=scale,
        T=T,
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        H=H,
        K=K,
        V=V,
        BT=64,
        num_sequences=0,
        USE_EXP2=True,
    )
    assert_golden(o, golden, ref, dtype, f"gla_fwd_o T={T} H={H} K={K} V={V}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="chunk_gla_fwd_o_gk TMA path requires SM90+ tensor descriptors",
)
@pytest.mark.parametrize("dtype,T", [(torch.bfloat16, 128)], ids=["production"])
def test_chunk_gla_fwd_o_gk_tma(dtype, T):
    """Cover the composed launcher and its TMA kernel at the fixed K=V=128, chunk_size=64 shape."""
    torch.manual_seed(11)
    B, H, K, V = 1, 2, 128, 128
    num_chunks = T // 64
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    # g stays fp32 to match the production gate ABI (see test_chunk_gla_fwd_o).
    g = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    h = torch.randn(B, num_chunks, H, K, V, device="cuda", dtype=dtype)
    A = torch.randn(B, T, H, 64, device="cuda", dtype=dtype)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)

    # _fwd_o_ref indexes h as (B*num_chunks, H, K, V); the launcher takes (B, chunks, H, K, V).
    golden = _fwd_o_ref(
        q.double(),
        g.double(),
        h.double().reshape(B * num_chunks, H, K, V),
        A.double(),
        v.double(),
        scale,
    )
    ref = _fwd_o_ref(q, g, h.reshape(B * num_chunks, H, K, V), A, v, scale)

    assert can_use_tma(v), "expected TMA-eligible tensors so the launcher takes its TMA path"
    o = chunk_gla_fwd_o_gk(q, v, g, A, h, scale)
    assert_golden(o, golden, ref, dtype, f"gla_fwd_o_gk_tma T={T}")


def _fwd_h_ref(k, w, v, gk, h0, chunk_size=64):
    """Reference for ``chunk_gated_delta_rule_fwd_h``.

    Per chunk, with state ``S`` in ``[K, V]``::

        h[chunk] = S
        v_new    = v_chunk - w_chunk @ S       (saved *ungated*)
        S        = S * 2^{gk_last}             (per key channel; gk_last is the chunk's last row)
        S        = S + k_chunk^T @ v_new
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    num_chunks = (T + chunk_size - 1) // chunk_size
    acc = torch.float64 if k.dtype == torch.float64 else torch.float32
    kc, wc, vc, gkc = k.to(acc), w.to(acc), v.to(acc), gk.to(acc)
    h = torch.zeros(B, num_chunks, H, K, V, dtype=acc, device=k.device)
    v_new = torch.zeros(B, T, H, V, dtype=acc, device=k.device)
    # State recurrence is sequential across chunks; batch (B, H) into the matmuls.
    state = (
        torch.zeros(B, H, K, V, dtype=acc, device=k.device)
        if h0 is None
        else h0.transpose(-1, -2).to(acc)
    )
    for it in range(num_chunks):
        s = it * chunk_size
        e = min(s + chunk_size, T)
        h[:, it] = state
        # The kernel casts the corrected v_new back to the input dtype before storing it and
        # feeding it into the k^T @ v_new state update, so quantize here to match.
        vn = (vc[:, s:e] - torch.einsum("blhk,bhkv->blhv", wc[:, s:e], state)).to(
            k.dtype
        )  # (B, L, H, V)
        v_new[:, s:e] = vn
        state = state * gkc[:, e - 1].exp2()[..., None]
        state = state + torch.einsum("blhk,blhv->bhkv", kc[:, s:e], vn.to(acc))
    return h.reshape(B * num_chunks, H, K, V), v_new, state.transpose(-1, -2)


@pytest.mark.parametrize(
    "dtype,T,use_h0",
    [
        (torch.float16, 64, False),
        (torch.bfloat16, 128, True),
        (torch.bfloat16, 130, False),
    ],
    ids=["single-fp16", "multi-state-bf16", "tail-bf16"],
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the KDA recurrence requires CUDA capability 8.0",
)
def test_chunk_delta_h_fwd(dtype, T, use_h0):
    torch.manual_seed(12)
    B, H, K, V = 1, 2, 128, 128
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    w = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    # Since prod keeps initial and final state in fp32, do that here to match.
    h0 = torch.randn(B, H, V, K, device="cuda", dtype=torch.float32) if use_h0 else None

    gh, gvn, ght = _fwd_h_ref(
        k.double(), w.double(), v.double(), gk.double(), h0.double() if use_h0 else None
    )
    rh, rvn, rht = _fwd_h_ref(k, w, v, gk, h0)

    # The dense path requires complete chunks; partial tails go through the
    # packed schedule like production ragged batches do.
    metadata = None
    if T % 64:
        cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, T, 64)
    h, v_new, ht = chunk_gated_delta_rule_fwd_h(k, w, v, gk, h0, metadata=metadata)

    num_chunks = triton.cdiv(T, 64)
    tag = f"T={T} h0={use_h0}"
    assert_golden(
        h[:, :num_chunks].reshape(B * num_chunks, H, K, V), gh, rh, dtype, f"delta_h h {tag}"
    )
    assert_golden(v_new, gvn, rvn, dtype, f"delta_h v_new {tag}")
    assert_golden(ht, ght, rht, dtype, f"delta_h ht {tag}")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the KDA recurrence requires CUDA capability 8.0",
)
def test_chunk_delta_h_fwd_fp32_path():
    """FP32 takes the non-warp-specialized BV=32 launch; dots still run in TF32."""
    torch.manual_seed(12)
    B, T, H, K, V = 1, 66, 2, 128, 128
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.float32)
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5

    gh, gvn, ght = _fwd_h_ref(k.double(), w.double(), v.double(), gk.double(), None)
    cu_seqlens = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, T, 64)
    h, v_new, ht = chunk_gated_delta_rule_fwd_h(k, w, v, gk, None, metadata=metadata)

    num_chunks = triton.cdiv(T, 64)
    # TF32 mantissa error compounds through the sequential state updates, so
    # this is a path-coverage check at TF32 tolerances, not a precision
    # guarantee (the dots ran in TF32 before this kernel, too).
    tf32 = {"atol": 1e-1, "rtol": 5e-2}
    torch.testing.assert_close(
        h[:, :num_chunks].reshape(B * num_chunks, H, K, V).double(), gh, **tf32
    )
    torch.testing.assert_close(v_new.double(), gvn, **tf32)
    torch.testing.assert_close(ht.double(), ght, **tf32)


def _fwd_intra_ref(q, k, gk, beta, scale, chunk_size=64, BC=16, causal_normref=True):
    """Reference for ``chunk_kda_fwd_kernel_intra_sub_chunk_forloop``.

    Per 16-token sub-chunk, gates are rebased against a *normref* row before ``2^{g - g_norm}``
    gating is folded into q/k::

        Aqk[l, m] = scale * <q_l 2^{gm_l}, k_m 2^{-gm_m}>    (l >= m, incl. diag)
        L[l, m]   = beta_l * <k_l 2^{gm_l}, k_m 2^{-gm_m}>   (l >  m, strict)
        Akk_block = (I + L)^{-1}                             (unit lower-triangular inverse)

    Only the diagonal 16x16 blocks of ``Aqk`` are written by this kernel so
    keep zero off-diagonal ``Aqk`` entries
    """
    B, T, H, _ = q.shape
    sub_chunks = chunk_size // BC
    num_chunks = (T + chunk_size - 1) // chunk_size
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qc, kc, gkc, bc = q.to(acc), k.to(acc), gk.to(acc), beta.to(acc)
    Aqk = torch.zeros(B, T, H, chunk_size, dtype=acc, device=q.device)
    Akk = torch.zeros(B, T, H, BC, dtype=acc, device=q.device)
    eye = torch.eye(BC, dtype=acc, device=q.device)
    full_tril = torch.tril(torch.ones(BC, BC, dtype=torch.bool, device=q.device))
    for it in range(num_chunks):
        for ii in range(sub_chunks):
            i_ti = it * chunk_size + ii * BC
            if i_ti >= T:
                continue
            e = min(i_ti + BC, T)
            L = e - i_ti
            normref_idx = 0 if causal_normref else min(BC // 2, L - 1)
            gm = gkc[:, i_ti:e] - gkc[:, i_ti + normref_idx : i_ti + normref_idx + 1]
            gq = gm.exp2()  # (b, L, H, K)
            qg = qc[:, i_ti:e] * gq
            kg = kc[:, i_ti:e] * (-gm).exp2()
            kgq = kc[:, i_ti:e] * gq
            tril = full_tril[:L, :L]
            aqk = scale * torch.einsum("blhk,bmhk->blhm", qg, kg)
            Aqk[:, i_ti:e, :, ii * BC : ii * BC + L] = aqk * tril[None, :, None, :]
            raw = torch.einsum("blhk,bmhk->blhm", kgq, kg) * bc[:, i_ti:e][..., None]
            m_lower = raw.permute(0, 2, 1, 3) * torch.tril(tril, -1)  # (b, H, l, m), strict
            eyeL = eye[:L, :L].expand(B, H, L, L)
            inv = torch.linalg.solve_triangular(eyeL + m_lower, eyeL, upper=False)
            Akk[:, i_ti:e, :, :L] = inv.permute(0, 2, 1, 3)
    return Aqk, Akk


@pytest.mark.parametrize(
    "dtype,T,H,K,causal_normref",
    [
        (torch.float16, 64, 1, 64, True),
        (torch.bfloat16, 130, 2, 128, False),
    ],
    ids=["legacy-causal", "production-tail"],
)
def test_chunk_kda_fwd_intra(dtype, T, H, K, causal_normref):
    torch.manual_seed(13)
    B = 1
    BC = 16
    num_chunks = triton.cdiv(T, 64)
    sub_chunks = 64 // BC
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    # gk (cumulative gate) and beta are fp32 in production; the kernel exp2s gk directly.
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.1

    gAqk, gAkk = _fwd_intra_ref(
        q.double(), k.double(), gk.double(), beta.double(), scale, causal_normref=causal_normref
    )
    rAqk, rAkk = _fwd_intra_ref(q, k, gk, beta, scale, causal_normref=causal_normref)

    # Aqk is zero-initialized: the kernel writes only diagonal blocks, leaving the rest zero.
    Aqk = torch.zeros(B, T, H, 64, device="cuda", dtype=dtype)
    Akk = torch.zeros(B, T, H, BC, device="cuda", dtype=torch.float32)
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop[(num_chunks, sub_chunks, B * H)](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=None,
        chunk_offsets=None,
        T=T,
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        k_stride_t=k.stride(1),
        k_stride_h=k.stride(2),
        H=H,
        K=K,
        BT=64,
        BC=BC,
        BK=triton.next_power_of_2(K),
        num_sequences=0,
        USE_GATHER=IS_GATHER_SUPPORTED,
        CAUSAL_NORMREF=causal_normref,
        GRID_NT=num_chunks,
        MAX_NT=num_chunks,
    )
    tag = f"T={T} H={H} K={K} causal_normref={causal_normref}"
    assert_golden(Aqk, gAqk, rAqk, dtype, f"fwd_intra Aqk {tag}")
    assert_golden(Akk, gAkk, rAkk, dtype, f"fwd_intra Akk {tag}")


def _recompute_wu_ref(k, v, beta, A, gk, q, chunk_size=64):
    """Reference for ``recompute_w_u_fwd`` (per-chunk 64-token block, ``gk`` in log2 units)::

    w  = A @ (k * beta * 2^gk)
    u  = A @ (v * beta)
    qg = q * 2^gk
    kg = k * 2^(gk_last - gk)      (gk_last = gate at the chunk's last row)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    num_chunks = T // chunk_size
    acc = torch.float64 if k.dtype == torch.float64 else torch.float32
    kf, vf, bf = k.to(acc), v.to(acc), beta.to(acc)
    Af, gf, qf = A.to(acc), gk.to(acc), q.to(acc)
    # Block-diagonal per chunk: reshape the token axis to (chunk, in-chunk row) and contract
    # the in-chunk column axis of A against each chunk independently.
    gr = gf.reshape(B, num_chunks, chunk_size, H, K)
    kr = kf.reshape(gr.shape)
    Ac = Af.reshape(B, num_chunks, chunk_size, H, chunk_size)
    kb = (kr * bf.reshape(B, num_chunks, chunk_size, H)[..., None]) * gr.exp2()
    vb = (vf * bf[..., None]).reshape(B, num_chunks, chunk_size, H, V)
    W = torch.einsum("bnihj,bnjhk->bnihk", Ac, kb).reshape(B, T, H, K)
    U = torch.einsum("bnihj,bnjhv->bnihv", Ac, vb).reshape(B, T, H, V)
    KG = (kr * (gr[:, :, -1:] - gr).exp2()).reshape(B, T, H, K)
    QG = qf * gf.exp2()
    return W, U, QG, KG


@requires_cute
def test_recompute_w_u_fwd_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(20)
    B, K, V = 1, 128, 128
    T = num_chunks * 64
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    # A is the block-lower-triangular (I - Akk)^-1; the kernel reads it triangularly, so
    # mask the random draw to the per-chunk lower triangle (incl. diagonal).
    A = torch.randn(B, T, H, 64, device="cuda", dtype=torch.float32) * 0.1
    A = A * _chunk_tril_mask(T, 64, "cuda")[None, :, None, :]
    # gk is the cumulative per-channel log2-decay (<= 0), so 2^gk <= 1.
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)

    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        metadata=None,
        q=q,
        gk=gk,
        chunk_size=64,
    )

    gW, gU, gQG, gKG = _recompute_wu_ref(
        k.double(), v.double(), beta.double(), A.double(), gk.double(), q.double()
    )
    rW, rU, rQG, rKG = _recompute_wu_ref(k, v, beta, A, gk, q)
    tag = f"NT={num_chunks} H={H}"
    assert_golden(w, gW, rW, torch.bfloat16, f"recompute_w_u w {tag}")
    assert_golden(u, gU, rU, torch.bfloat16, f"recompute_w_u u {tag}")
    assert_golden(qg, gQG, rQG, torch.bfloat16, f"recompute_w_u qg {tag}")
    assert_golden(kg, gKG, rKG, torch.bfloat16, f"recompute_w_u kg {tag}")


def _delta_h_bwd_dhu_ref(q, k, w, do, dv, gk, h0, dht, scale, chunk_size=64):
    """Reference for the backward of the delta-rule
    h-recurrence (see ``_fwd_h_ref``), iterated over chunks in reverse::

        dh_out[c] = dh                                      (snapshot before update)
        dv2[c]    = k_c @ dh + dv_in_c
        dh        = 2^{gk_last_c} * dh + scale*(q_c^T @ do_c) - w_c^T @ dv2_c
        dh0       = dh   (final, when h0 is provided)

    ``dh`` is seeded with ``dht`` when given, else zero. ``gk_last_c`` is the gate at the
    chunk's last row. match the kernel's stated ops exactly (no gate on ``k @ dh``,
    per-channel decay ``2^{gk_last}`` only — matching the forward cuteDSL config).
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    num_chunks = (T + chunk_size - 1) // chunk_size
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, wf, dof, dvf, gf = (t.to(acc) for t in (q, k, w, do, dv, gk))
    dh_out = torch.zeros(B, num_chunks, H, K, V, dtype=acc, device=q.device)
    dv2 = torch.zeros(B, T, H, V, dtype=acc, device=q.device)
    # Reverse recurrence is sequential across chunks; batch (B, H) into the matmuls.
    D = (
        dht.transpose(-1, -2).to(acc).clone()
        if dht is not None
        else torch.zeros(B, H, K, V, dtype=acc, device=q.device)
    )
    for c in reversed(range(num_chunks)):
        s = c * chunk_size
        e = min(s + chunk_size, T)
        dh_out[:, c] = D
        dv2c = torch.einsum("blhk,bhkv->blhv", kf[:, s:e], D) + dvf[:, s:e]  # (B, L, H, V)
        dv2[:, s:e] = dv2c
        D = (
            gf[:, e - 1].exp2()[..., None] * D
            + scale * torch.einsum("blhk,blhv->bhkv", qf[:, s:e], dof[:, s:e])
            - torch.einsum("blhk,blhv->bhkv", wf[:, s:e], dv2c)
        )
    dh0 = D.transpose(-1, -2) if h0 is not None else None
    return dh_out, dh0, dv2


@requires_cute
@pytest.mark.parametrize(
    "bv,num_chunks,use_h0,use_dht",
    [(16, 4, False, False), (32, 5, True, False), (16, 4, True, True)],
    ids=["bv16", "bv32-state", "final-state-gradient"],
)
def test_delta_h_bwd_dhu_cute(bv, num_chunks, use_h0, use_dht):
    # Non-varlen path: B=1, K=V=128, chunk_size=64. Covers the dht (final-state grad)
    # and h0 (initial-state grad -> dh0) input paths, and both BV=16/BV=32 SS-mode tiles.
    torch.manual_seed(21)
    B, H, K, V = 1, 2, 128, 128
    T = num_chunks * 64
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32), dim=-1
    )
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) * 0.1
    do = torch.randn(B, T, H, V, device="cuda", dtype=torch.float32)
    aqk = torch.randn(B, T, H, 64, device="cuda", dtype=torch.float32) * 0.1
    aqk = aqk * _chunk_tril_mask(T, 64, "cuda")[None, :, None, :]
    # gk is the cumulative per-channel log2-decay (<= 0), so 2^{gk_last} <= 1.
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    h0 = torch.randn(B, H, V, K, device="cuda", dtype=torch.float32) if use_h0 else None
    dht = torch.randn(B, H, V, K, device="cuda", dtype=torch.float32) if use_dht else None

    qb, kb, wb, dob, aqkb = (t.to(torch.bfloat16) for t in (q, k, w, do, aqk))
    dh, dh0, dv2 = blackwell_delta_h_bwd_dhu_dv_fused(
        qb, kb, wb, dob, aqkb, gk=gk, h0=h0, dht=dht, scale=scale, chunk_size=64, bv=bv
    )
    dvb = torch.empty(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    for start in range(0, T, 64):
        dvb[:, start : start + 64] = torch.einsum(
            "blhm,blhv->bmhv",
            aqkb[:, start : start + 64],
            dob[:, start : start + 64],
        )

    h0d = h0.double() if use_h0 else None
    dhtd = dht.double() if use_dht else None
    gDH, gDH0, gDV2 = _delta_h_bwd_dhu_ref(
        qb.double(),
        kb.double(),
        wb.double(),
        dob.double(),
        dvb.double(),
        gk.double(),
        h0d,
        dhtd,
        scale,
    )
    rDH, rDH0, rDV2 = _delta_h_bwd_dhu_ref(qb, kb, wb, dob, dvb, gk, h0, dht, scale)

    tag = f"NT={num_chunks} h0={use_h0} dht={use_dht} bv={bv}"
    assert_golden(dh, gDH, rDH, torch.bfloat16, f"delta_h_bwd dh {tag}")
    assert_golden(dv2, gDV2, rDV2, torch.bfloat16, f"delta_h_bwd dv2 {tag}")
    if use_h0:
        assert_golden(dh0, gDH0, rDH0, torch.bfloat16, f"delta_h_bwd dh0 {tag}")


@requires_cute
@pytest.mark.parametrize("sm_count,expected_bv", [(1024, 16), (8, 32)])
def test_delta_h_bwd_dhu_dispatch_selects_bv(monkeypatch, sm_count, expected_bv):
    """The BV-dispatch wrapper picks BV=32 when V-tiles exceed the SM count, else BV=16.

    The leaf kernel is mocked so this isolates the selection heuristic (an integer compare)
    without a kernel launch; numerics for both BV tiles are covered by the leaf test above.
    """
    B, H, K, V = 1, 2, 128, 128
    q = torch.empty(B, 64, H, K, device="cuda")
    zeros = torch.empty(B, 64, H, V, device="cuda")

    captured = {}

    def fake_fused(*args, bv, **kwargs):
        captured["bv"] = bv
        return zeros, None, zeros

    class _Props:
        multi_processor_count = sm_count

    monkeypatch.setattr(delta_h_bwd, "get_device_properties", lambda device: _Props())
    monkeypatch.setattr(delta_h_bwd, "blackwell_delta_h_bwd_dhu_dv_fused", fake_fused)

    delta_h_bwd.blackwell_delta_h_bwd_dhu_dv_fused_dispatch(q, q, q, zeros, zeros)
    assert captured["bv"] == expected_bv


def _inter_solve_ref(q, k, gk, beta, scale, chunk_size=64):
    """Reference for the composed CuTe forward-intra inter-solve (K3b off-diagonal Aqk +
    K4b 64x64 block inverse), per 64-token chunk with ``gk`` the cumulative gate::

        Aqk[l, m] = scale * <q_l 2^{gk_l}, k_m 2^{-gk_m}>          (l >= m, incl. diag)
        raw[l, m] = beta_l * <k_l 2^{gk_l}, k_m 2^{-gk_m}>         (l >  m, strict)
        Akk       = (I + raw)^-1                                   (unit lower-triangular)
    """
    B, T, H, K = q.shape
    num_chunks = T // chunk_size
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, gf, bf = (t.to(acc) for t in (q, k, gk, beta))
    # Reshape the token axis to (chunk, in-chunk row); each 64x64 block solves independently
    # with (B, num_chunks, H) as batch dims.
    gr = gf.reshape(B, num_chunks, chunk_size, H, K)
    qg = qf.reshape(B, num_chunks, chunk_size, H, K) * gr.exp2()
    kgi = kf.reshape(B, num_chunks, chunk_size, H, K) * (-gr).exp2()
    kb = (
        kf.reshape(B, num_chunks, chunk_size, H, K)
        * bf.reshape(B, num_chunks, chunk_size, H)[..., None]
        * gr.exp2()
    )
    eye = torch.eye(chunk_size, dtype=acc, device=q.device)
    ones = torch.ones(chunk_size, chunk_size, dtype=acc, device=q.device)
    aqk = scale * torch.einsum("bnlhk,bnmhk->bnhlm", qg, kgi) * torch.tril(ones)
    raw = torch.einsum("bnlhk,bnmhk->bnhlm", kb, kgi) * torch.tril(ones, -1)
    akk = torch.linalg.solve_triangular(eye + raw, eye.expand_as(raw), upper=False)
    # (B, num_chunks, H, l, m) -> (B, num_chunks, l, H, m) -> (B, T, H, chunk_size)
    Aqk = aqk.permute(0, 1, 3, 2, 4).reshape(B, T, H, chunk_size)
    Akk = akk.permute(0, 1, 3, 2, 4).reshape(B, T, H, chunk_size)
    return Aqk, Akk


def _chunk_tril_mask(T: int, BT: int, device) -> torch.Tensor:
    """(T, BT) bool mask keeping each row's block-lower-triangular columns (incl. diagonal)."""
    row_local = (torch.arange(T, device=device) % BT)[:, None]
    return torch.arange(BT, device=device)[None, :] <= row_local


@requires_cute
def test_chunk_kda_fwd_intra_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(22)
    B, K, V = 1, 128, 128
    T = num_chunks * 64
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g_inc = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.05
    gk = g_inc.view(B, num_chunks, 64, H, K).cumsum(2).view(B, T, H, K)
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))

    w, u, _kg, Aqk, Akk = chunk_kda_fwd_intra_cute(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        None,
        chunk_size=64,
    )

    gAqk, gAkk = _inter_solve_ref(q.double(), k.double(), gk.double(), beta.double(), scale)
    rAqk, rAkk = _inter_solve_ref(q, k, gk, beta, scale)
    gW, gU = _recompute_wu_ref(
        k.double(), v.double(), beta.double(), gAkk, gk.double(), q.double()
    )[:2]
    rW, rU = _recompute_wu_ref(k, v, beta, rAkk, gk, q)[:2]

    tag = f"NT={num_chunks} H={H}"
    mask = _chunk_tril_mask(T, 64, "cuda")[None, :, None, :]
    Aqk_tril = torch.where(mask, Aqk, torch.zeros_like(Aqk))
    assert_golden(Aqk_tril, gAqk, rAqk, torch.bfloat16, f"fwd_intra_cute Aqk {tag}")
    assert_golden(Akk, gAkk, rAkk, torch.bfloat16, f"fwd_intra_cute Akk {tag}")
    assert_golden(w, gW, rW, torch.bfloat16, f"fwd_intra_cute w {tag}")
    assert_golden(u, gU, rU, torch.bfloat16, f"fwd_intra_cute u {tag}")


def _dense_reference_routes(num_chunks):
    """Build the mathematical reference's single-document chunk work list."""
    tokens = num_chunks * 64
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    chunk_indices = torch.stack(
        (
            torch.zeros(num_chunks, dtype=torch.int32, device="cuda"),
            torch.arange(num_chunks, dtype=torch.int32, device="cuda"),
        ),
        dim=1,
    )
    return cu_seqlens, chunk_indices


def _perchunk_gate(cu, chunk_indices, T, H, K, gate_inc=0.5):
    """Per-chunk cumulative log2-gate (reset at every chunk boundary), filled only over each
    chunk's valid rows so a partial tail never differences gates across a document boundary."""
    g = torch.zeros(1, T, H, K, device="cuda", dtype=torch.float32)
    for seq_idx, chunk_idx in chunk_indices.tolist():
        bos, eos = cu[seq_idx].item(), cu[seq_idx + 1].item()
        rs = bos + chunk_idx * 64
        valid = min(eos - rs, 64)
        inc = -torch.rand(valid, H, K, device="cuda", dtype=torch.float32) * gate_inc
        g[0, rs : rs + valid] = inc.cumsum(0)
    return g


def _perchunk_unit_lower(cu, chunk_indices, T, H):
    """Per-chunk unit-lower-triangular ``A`` (a plausible (I - Akk)^-1), filled only over each
    chunk's valid rows/cols; tail rows of a partial chunk stay zero."""
    A = torch.zeros(1, T, H, 64, device="cuda", dtype=torch.float32)
    for seq_idx, chunk_idx in chunk_indices.tolist():
        bos, eos = cu[seq_idx].item(), cu[seq_idx + 1].item()
        rs = bos + chunk_idx * 64
        valid = min(eos - rs, 64)
        tri = torch.tril(torch.ones(valid, 64, device="cuda", dtype=torch.bool), diagonal=-1)
        blk = torch.randn(valid, H, 64, device="cuda") * 0.05 * tri[:, None, :]
        idx = torch.arange(valid, device="cuda")
        blk[idx, :, idx] = 1.0  # unit diagonal on the valid sub-block
        A[0, rs : rs + valid] = blk
    return A.to(torch.bfloat16)


@requires_cute
def test_chunk_kda_bwd_intra_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(3)
    B, K = 1, 128
    T = num_chunks * 64
    cu, chunk_indices = _dense_reference_routes(num_chunks)
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    g = _perchunk_gate(cu, chunk_indices, T, H, K, gate_inc=0.05)
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    # Incoming grads: dense random (the kernel masks to the non-strict causal triangle).
    dAqk = torch.randn(B, T, H, 64, device="cuda", dtype=torch.float32) * 0.1
    dAkk = torch.randn(B, T, H, 64, device="cuda", dtype=torch.float32) * 0.1
    zq = torch.zeros(B, T, H, K, device="cuda", dtype=torch.float32)
    zb = torch.zeros(B, T, H, device="cuda", dtype=torch.float32)

    dq, dk, dg, db = chunk_kda_bwd_intra_cute(
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        zq.clone(),
        zq.clone(),
        zb.clone(),
        zq.clone(),
        None,
    )

    gdq, gdk, gdb, gdg = _bwd_intra_ref(
        q.double(),
        k.double(),
        g.double(),
        beta.double(),
        dAqk.double(),
        dAkk.double(),
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    rdq, rdk, rdb, rdg = _bwd_intra_ref(
        q, k, g, beta, dAqk, dAkk, cu_seqlens=cu, chunk_indices=chunk_indices
    )

    tag = f"NT={num_chunks} H={H}"
    assert_golden(dq, gdq, rdq, torch.bfloat16, f"bwd_intra_cute dq {tag}")
    assert_golden(dk, gdk, rdk, torch.bfloat16, f"bwd_intra_cute dk {tag}")
    assert_golden(db, gdb, rdb, torch.bfloat16, f"bwd_intra_cute db {tag}")
    assert_golden(dg, gdg, rdg, torch.bfloat16, f"bwd_intra_cute dg {tag}")


@requires_cute
def test_chunk_kda_bwd_wy_dqkg_fused_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(4)
    B, K, V = 1, 128, 128
    HV = H
    T = num_chunks * 64
    scale = K**-0.5
    cu, chunk_indices = _dense_reference_routes(num_chunks)

    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.2
    v_new = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.2
    do = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.2
    dv = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.2
    h = torch.randn(B, num_chunks, HV, K, V, device="cuda", dtype=torch.bfloat16) * 0.1
    dh = torch.randn(B, num_chunks, HV, K, V, device="cuda", dtype=torch.bfloat16) * 0.1

    # g: per-chunk cumulative decreasing log2-gate -> exp2 stays in (0, 1], no overflow.
    g = _perchunk_gate(cu, chunk_indices, T, HV, K, gate_inc=0.5)
    beta = torch.sigmoid(torch.randn(B, T, HV, device="cuda", dtype=torch.float32))
    # A: per-chunk unit-lower-triangular (a plausible (I - Akk)^-1).
    A = _perchunk_unit_lower(cu, chunk_indices, T, HV)

    dq, dk, dv2, dg, db, dA = chunk_kda_bwd_wy_dqkg_fused_cute(
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        do,
        dh,
        dv,
        None,
        scale=scale,
        chunk_size=64,
    )

    ref_args = (v_new, g, beta, A, h, do, dh, dv)
    gdq, gdk, gdv2, gdb, gdg, gdA = _bwd_wy_dqkg_ref(
        q.double(),
        k.double(),
        v.double(),
        *(t.double() for t in ref_args),
        scale,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
    )
    rdq, rdk, rdv2, rdb, rdg, rdA = _bwd_wy_dqkg_ref(
        q, k, v, *ref_args, scale, cu_seqlens=cu, chunk_indices=chunk_indices
    )

    tag = f"NT={num_chunks} H={H}"
    assert_golden(dq, gdq, rdq, torch.bfloat16, f"bwd_wy_cute dq {tag}")
    assert_golden(dk, gdk, rdk, torch.bfloat16, f"bwd_wy_cute dk {tag}")
    assert_golden(dv2, gdv2, rdv2, torch.bfloat16, f"bwd_wy_cute dv2 {tag}")
    assert_golden(db, gdb, rdb, torch.bfloat16, f"bwd_wy_cute db {tag}")
    assert_golden(dg, gdg, rdg, torch.bfloat16, f"bwd_wy_cute dg {tag}")
    assert_golden(dA, gdA, rdA, torch.bfloat16, f"bwd_wy_cute dA {tag}")
