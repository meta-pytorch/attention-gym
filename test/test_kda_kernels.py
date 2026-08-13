# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU unit tests for the optimized (Triton) KDA leaf kernels."""

from __future__ import annotations

import math
from itertools import product

import pytest
import torch

triton = pytest.importorskip("triton")

# These imports intentionally follow the optional-dependency check above.
import attn_gym.linear.kda.bwd.triton.cumsum as cumsum_module
from attn_gym._backends.triton.utils import can_use_tma
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav import chunk_kda_bwd_kernel_dAv
from attn_gym.linear.kda.bwd.triton.cumsum import (
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_scalar_kernel,
    chunk_local_cumsum_vector_kernel,
)
from attn_gym.linear.kda.bwd.triton.gate_bwd import kda_gate_bwd_kernel
from attn_gym.linear.kda.bwd.triton.l2norm_bwd import (
    l2norm_bwd_kernel,
    l2norm_bwd_kernel1,
)
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_forloop,
)
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import (
    chunk_gla_fwd_kernel_o,
    chunk_gla_fwd_o_gk,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop,
)
from attn_gym.linear.kda.fwd.triton.gate_fwd import (
    _requires_int64_offsets,
    kda_gate_chunk_cumsum,
    kda_gate_chunk_cumsum_vector_kernel,
    kda_gate_chunk_cumsum_vector_kernel_forloop,
)
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import (
    _l2norm_bwd_op,
    _l2norm_fwd_op,
    l2norm,
    l2norm_fwd_kernel,
    l2norm_fwd_kernel1,
)
from attn_gym.linear.kda.naive import (
    chunk_cumsum_ref,
    gate_bwd_ref,
    gate_fwd_ref,
    l2norm_bwd_ref,
    l2norm_fwd_ref,
)
from attn_gym.linear.kda.utils import IS_GATHER_SUPPORTED, ChunkMetadata, prepare_chunk_indices

IS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

try:
    from attn_gym.linear.kda.bwd.cute import chunk_delta_h_bwd_v1_dispatch as dispatch_mod
    from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1 import blackwell_delta_h_bwd_dhu_v1
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
    selections = (
        _representative_config(l2norm_bwd_kernel, BT=16, num_warps=4),
        _representative_config(l2norm_bwd_kernel1, num_warps=4),
        _representative_config(kda_gate_bwd_kernel, num_warps=4, num_stages=3),
        _representative_config(chunk_local_cumsum_scalar_kernel, num_warps=4),
        _representative_config(
            chunk_local_cumsum_vector_kernel,
            BS=32,
            num_warps=4,
        ),
    )
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


def _cu_seqlens(doc_lens: tuple[int, ...]) -> torch.Tensor:
    cu = torch.zeros(len(doc_lens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(doc_lens, dtype=torch.int32, device=DEV).cumsum(0)
    return cu


def _optional_matrix_strides(tensor: torch.Tensor | None) -> tuple[int, ...]:
    """Return matrix strides or inert strides for a disabled optional input."""
    return (0, 0) if tensor is None else tensor.stride()


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
GATE_TS = (1, 15, 16, 17, 33, 64)
VARLEN_DOCS = ((16,), (16, 32, 16), (7, 16, 24, 1), (64, 3))

_L2NORM_CASES = [(torch.float32, T, D, strided) for T, D, strided in L2NORM_SHAPES] + [
    (dtype, 17, 64, False) for dtype in DTYPES[1:]
]


# l2norm forward
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
    grid = lambda meta: (triton.cdiv(T, meta["BT"]),)
    l2norm_fwd_kernel[grid](
        x,
        y,
        rstd,
        eps,
        T,
        X_STRIDES=(0, x.stride(0), 0, x.stride(1)),
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        T=T,
        H=1,
        D=D,
        BD=BD,
        NB=triton.cdiv(T, 16),
    )
    assert_golden(y, golden, ref, dtype, f"l2norm_fwd_kernel T={T} D={D}")
    assert_golden(rstd, rstd_golden, rstd_ref, dtype, f"l2norm_fwd_kernel rstd T={T} D={D}")

    # single-row kernel
    y1 = _empty_strided_like(x) if strided else torch.empty_like(x)
    rstd1 = _empty_strided_like(rstd_template) if strided else torch.empty_like(rstd_template)
    l2norm_fwd_kernel1[(T,)](
        x,
        y1,
        rstd1,
        eps,
        X_STRIDES=(0, x.stride(0), 0, x.stride(1)),
        Y_STRIDES=y1.stride(),
        RSTD_STRIDES=rstd1.stride(),
        T=T,
        H=1,
        D=D,
        BD=BD,
    )
    assert_golden(y1, golden, ref, dtype, f"l2norm_fwd_kernel1 T={T} D={D}")
    assert_golden(rstd1, rstd_golden, rstd_ref, dtype, f"l2norm_fwd_kernel1 rstd T={T} D={D}")


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
    torch.library.opcheck(_l2norm_fwd_op, (x, 1e-6))

    output, rstd = _l2norm_fwd_op(x, 1e-6)
    d_output = torch.randn_like(output)
    torch.library.opcheck(
        _l2norm_bwd_op,
        (output.view(-1, output.shape[-1]), rstd, d_output),
    )


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
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=dy_strides,
        DX_STRIDES=dx.stride(),
        TOKENS=T,
        HEADS=1,
        D=D,
        BD=BD,
        NB=triton.cdiv(T, 16),
    )
    assert_golden(dx, golden, ref, dtype, f"l2norm_bwd_kernel T={T} D={D}")

    dx1 = _empty_strided_like(y) if strided else torch.empty_like(y)
    l2norm_bwd_kernel1[(T,)](
        y,
        rstd,
        dy,
        dx1,
        D,
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=dy_strides,
        DX_STRIDES=dx1.stride(),
        TOKENS=T,
        HEADS=1,
        BD=BD,
    )
    assert_golden(dx1, golden, ref, dtype, f"l2norm_bwd_kernel1 T={T} D={D}")


# gate forward (gate map + chunk-local cumsum)
def _run_gate_fwd(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    lower_bound,
    chunk_size,
    cu_seqlens,
    reverse=False,
    *,
    grid_nt=None,
):
    """Launch the direct or persistent gate kernel with logical tensor strides."""
    # o carries the full output dim S; g may carry a reduced dim S_in = S / F_REPEAT.
    B, T, H, S = o.shape
    S_in = g.shape[-1]
    F_REPEAT = S // S_in
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = chunk_indices.shape[0]
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)

    kernel = kda_gate_chunk_cumsum_vector_kernel
    kernel_kwargs = {}
    if grid_nt is not None:
        kernel = kda_gate_chunk_cumsum_vector_kernel_forloop
        kernel_kwargs = {"GRID_NT": grid_nt, "MAX_NT": NT}
    # BS is autotuned; the HAS_*/USE_*/IS_VARLEN/USE_REPEAT flags come from @triton.heuristics.
    grid = lambda meta: (grid_nt or NT, B * H, triton.cdiv(S, meta["BS"]))
    kernel[grid](
        g,
        A_log,
        dt_bias,
        o,
        scale,
        cu_seqlens,
        chunk_indices,
        lower_bound,
        T,
        None,
        g.stride(0),
        o.stride(0),
        S_STRIDES=g.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=_optional_matrix_strides(dt_bias),
        O_STRIDES=o.stride()[1:],
        B=B,
        H=H,
        S=S,
        S_in=S_in,
        F_REPEAT=F_REPEAT,
        BT=chunk_size,
        REVERSE=reverse,
        **kernel_kwargs,
    )


_GATE_FWD_CASES = (
    [
        (torch.float32, 17, lower_bound, has_bias, scale, reverse)
        for lower_bound, has_bias, scale, reverse in product(
            (None, 0.5), (False, True), (None, 2.0), (False, True)
        )
    ]
    + [(torch.float32, T, None, False, None, False) for T in GATE_TS if T != 17]
    + [(dtype, 17, None, False, None, False) for dtype in DTYPES[1:]]
)


@pytest.mark.parametrize(
    "dtype,T,lower_bound,has_bias,scale,reverse",
    _GATE_FWD_CASES,
    ids=_case_id,
)
def test_gate_fwd(dtype, T, lower_bound, has_bias, scale, reverse):
    torch.manual_seed(2)
    B, H, S, chunk = 2, 2, 16, 16
    g64 = torch.randn(B, T, H, S, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    bias64 = torch.randn(H, S, device=DEV, dtype=torch.float64) if has_bias else None

    golden = gate_fwd_ref(g64, A_log64, bias64, lower_bound, scale, reverse, chunk, None)

    g = g64.to(dtype)
    A_log = A_log64.to(torch.float32)
    bias = bias64.to(dtype) if has_bias else None
    ref = gate_fwd_ref(g, A_log, bias, lower_bound, scale, reverse, chunk, None)

    o = kda_gate_chunk_cumsum(
        g,
        A_log,
        bias,
        chunk_size=chunk,
        lower_bound=lower_bound,
        scale=scale,
        reverse=reverse,
    )
    tag = f"T={T} lb={lower_bound} bias={has_bias} scale={scale} rev={reverse}"
    assert_golden(o, golden, ref, dtype, f"gate_fwd {tag}")


@pytest.mark.parametrize(
    "shape",
    [(0, 17, 2, 32), (1, 0, 2, 32), (1, 17, 0, 32), (1, 17, 2, 0)],
)
def test_gate_fwd_wrapper_rejects_empty_dimensions(shape):
    g = torch.empty(shape, device=DEV)
    A_log = torch.empty(shape[2], device=DEV)
    dt_bias = torch.empty(shape[2:], device=DEV)
    with pytest.raises(ValueError, match="no empty dimensions"):
        kda_gate_chunk_cumsum(g, A_log, dt_bias)


_GATE_FWD_VARLEN_CASES = [
    (torch.float32, docs, lower_bound) for docs, lower_bound in product(VARLEN_DOCS, (None, 0.5))
] + [(dtype, VARLEN_DOCS[2], None) for dtype in DTYPES[1:]]


@pytest.mark.parametrize(
    "dtype,docs,lower_bound",
    _GATE_FWD_VARLEN_CASES,
    ids=_case_id,
)
def test_gate_fwd_varlen(dtype, docs, lower_bound):
    torch.manual_seed(3)
    H, S, chunk = 2, 16, 16
    total = sum(docs)
    cu = _cu_seqlens(docs)
    g64 = torch.randn(1, total, H, S, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    bias64 = torch.randn(H, S, device=DEV, dtype=torch.float64)

    golden = gate_fwd_ref(g64, A_log64, bias64, lower_bound, None, False, chunk, cu)

    g = g64.to(dtype)
    A_log = A_log64.to(torch.float32)
    bias = bias64.to(dtype)
    ref = gate_fwd_ref(g, A_log, bias, lower_bound, None, False, chunk, cu)

    generated = kda_gate_chunk_cumsum(
        g,
        A_log,
        bias,
        chunk_size=chunk,
        lower_bound=lower_bound,
        cu_seqlens=cu,
    )
    chunk_indices = prepare_chunk_indices(cu, chunk)
    explicit = kda_gate_chunk_cumsum(
        g,
        A_log,
        bias,
        chunk_size=chunk,
        lower_bound=lower_bound,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        num_chunks=torch.tensor(len(chunk_indices), dtype=torch.int32, device=DEV),
    )
    tag = f"gate_fwd_varlen docs={docs} lb={lower_bound}"
    assert_golden(generated, golden, ref, dtype, f"{tag} generated")
    assert_golden(explicit, golden, ref, dtype, f"{tag} explicit")


# gate forward with a reduced input dim S_in < S (USE_REPEAT branch): the kernel maps
# output column j to input column j // F_REPEAT, i.e. a repeat_interleave over channels.
@pytest.mark.parametrize("dtype", DTYPES, ids=_case_id)
@pytest.mark.parametrize("f_repeat", [2, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_gate_fwd_repeat(dtype, f_repeat, reverse):
    torch.manual_seed(8)
    B, T, H, chunk = 1, 33, 2, 16  # partial trailing chunk (33 = 2*16 + 1)
    S_in = 8
    S = S_in * f_repeat
    g64 = torch.randn(B, T, H, S_in, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    bias64 = torch.randn(H, S, device=DEV, dtype=torch.float64)  # bias is in full dim S

    g64_full = g64.repeat_interleave(f_repeat, dim=-1)
    golden = gate_fwd_ref(g64_full, A_log64, bias64, None, None, reverse, chunk, None)

    g = g64.to(dtype)
    A_log = A_log64.to(torch.float32)
    bias = bias64.to(dtype)
    ref = gate_fwd_ref(
        g.repeat_interleave(f_repeat, dim=-1), A_log, bias, None, None, reverse, chunk, None
    )

    o = torch.empty(B, T, H, S, device=DEV, dtype=dtype)
    _run_gate_fwd(g, A_log, bias, o, None, None, chunk, None, reverse=reverse)
    assert_golden(o, golden, ref, dtype, f"gate_fwd_repeat f_repeat={f_repeat} rev={reverse}")


def test_gate_fwd_varlen_repeat():
    """Cover packed varlen, repeated channels, and non-contiguous strides together."""
    torch.manual_seed(9)
    dtype = torch.float32
    docs = (7, 17)
    total, H, S_in, f_repeat, chunk = sum(docs), 2, 8, 2, 16
    S = S_in * f_repeat
    cu = _cu_seqlens(docs)
    g64 = torch.randn(1, total, H, S_in, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    bias64 = torch.randn(H, S, device=DEV, dtype=torch.float64)

    g64_full = g64.repeat_interleave(f_repeat, dim=-1)
    golden = gate_fwd_ref(g64_full, A_log64, bias64, 0.5, None, True, chunk, cu)

    g = _strided_copy(g64.to(dtype))
    A_log = _strided_copy(A_log64.to(torch.float32))
    bias = _strided_copy(bias64.to(dtype))
    ref = gate_fwd_ref(
        g.repeat_interleave(f_repeat, dim=-1), A_log, bias, 0.5, None, True, chunk, cu
    )
    output_template = torch.empty(1, total, H, S, device=DEV, dtype=dtype)
    o = _empty_strided_like(output_template)

    _run_gate_fwd(g, A_log, bias, o, None, 0.5, chunk, cu, reverse=True)
    assert_golden(o, golden, ref, dtype, "gate_fwd_varlen_repeat")

    o_forloop = _empty_strided_like(output_template)
    _run_gate_fwd(
        g,
        A_log,
        bias,
        o_forloop,
        None,
        0.5,
        chunk,
        cu,
        reverse=True,
        grid_nt=2,
    )
    assert_golden(o_forloop, golden, ref, dtype, "gate_fwd_varlen_repeat_forloop")


def test_gate_fwd_forloop_fixed_batch():
    """Cover the persistent kernel's fixed-length and nonzero-batch offsets."""
    torch.manual_seed(11)
    dtype = torch.float32
    B, T, H, S, chunk = 2, 33, 2, 16, 16
    g64 = torch.randn(B, T, H, S, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    golden = gate_fwd_ref(g64, A_log64, None, None, None, False, chunk, None)

    g = g64.to(dtype)
    A_log = A_log64.to(torch.float32)
    ref = gate_fwd_ref(g, A_log, None, None, None, False, chunk, None)
    o = torch.empty_like(g)

    _run_gate_fwd(g, A_log, None, o, None, None, chunk, None, grid_nt=2)
    assert_golden(o, golden, ref, dtype, "gate_fwd_forloop_fixed_batch")


def test_gate_fwd_int64_offsets():
    """Force the 64-bit offset specialization without allocating a huge tensor."""
    torch.manual_seed(12)
    g = torch.randn(1, 1, 1, 1, device=DEV)
    A_log = torch.randn(1, device=DEV)
    o = torch.empty_like(g)
    grid = lambda meta: (1, 1, triton.cdiv(1, meta["BS"]))

    # B=2 makes the synthetic batch span exceed int32, while the one-program
    # grid accesses only batch zero and therefore remains within the tiny allocation.
    assert _requires_int64_offsets(
        {
            "B": 2,
            "T": 1,
            "H": 1,
            "S": 1,
            "S_in": 1,
            "s_batch_stride": 1 << 31,
            "o_batch_stride": 1 << 31,
            "S_STRIDES": g.stride()[1:],
            "A_LOG_STRIDES": A_log.stride(),
            "DT_BIAS_STRIDES": (0, 0),
            "O_STRIDES": o.stride()[1:],
        }
    )
    kda_gate_chunk_cumsum_vector_kernel[grid](
        g,
        A_log,
        None,
        o,
        None,
        None,
        None,
        None,
        1,
        None,
        1 << 31,
        1 << 31,
        S_STRIDES=g.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=(0, 0),
        O_STRIDES=o.stride()[1:],
        B=2,
        H=1,
        S=1,
        S_in=1,
        F_REPEAT=1,
        BT=1,
        REVERSE=False,
    )
    ref = gate_fwd_ref(g, A_log, None, None, None, False, 1, None)
    torch.testing.assert_close(o, ref)


# gate backward (pointwise gate map only -- no cumsum)
_GATE_BWD_CASES = (
    [
        (
            torch.float32,
            17,
            lower_bound,
            has_bias,
            12 if lower_bound is None and has_bias else 16,
            lower_bound is None and has_bias,
        )
        for lower_bound, has_bias in product((None, 0.5), (False, True))
    ]
    + [(torch.float32, T, None, False, 16, False) for T in GATE_TS if T != 17]
    + [(dtype, 17, None, False, 16, False) for dtype in DTYPES[1:]]
)


@pytest.mark.parametrize(
    "dtype,T,lower_bound,has_bias,D,strided",
    _GATE_BWD_CASES,
    ids=_case_id,
)
def test_gate_bwd(dtype, T, lower_bound, has_bias, D, strided):
    torch.manual_seed(4)
    H, BT = 2, 16
    g64 = torch.randn(T, H, D, device=DEV, dtype=torch.float64)
    A_log64 = torch.randn(H, device=DEV, dtype=torch.float64)
    bias64 = torch.randn(H, D, device=DEV, dtype=torch.float64) if has_bias else None
    dyg64 = torch.randn(T, H, D, device=DEV, dtype=torch.float64)

    # gate_bwd math is applied per (T, H, D) row; add B=1 for the ref helper.
    def as_bthd(x):
        return None if x is None else x.unsqueeze(0)

    gold_dg, gold_dA, gold_db = gate_bwd_ref(
        as_bthd(g64), A_log64, bias64, as_bthd(dyg64), lower_bound
    )

    g = g64.to(dtype)
    A_log = A_log64.to(torch.float32)
    bias = bias64.to(dtype) if has_bias else None
    dyg = dyg64.to(dtype)
    if strided:
        assert bias is not None
        g, A_log, dyg = map(_strided_copy, (g, A_log, dyg))
        bias = _strided_copy(bias)
    ref_dg, ref_dA, ref_db = gate_bwd_ref(as_bthd(g), A_log, bias, as_bthd(dyg), lower_bound)

    NT = triton.cdiv(T, BT)
    BD = triton.next_power_of_2(D)
    dg = _empty_strided_like(g) if strided else torch.empty_like(g)
    dA_template = torch.zeros(NT, H, device=DEV, dtype=torch.float32)
    dA = _strided_copy(dA_template) if strided else dA_template
    kda_gate_bwd_kernel[(NT, H)](
        g,
        A_log,
        bias,
        dyg,
        dg,
        dA,
        lower_bound,
        T,
        G_STRIDES=g.stride(),
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=_optional_matrix_strides(bias),
        DYG_STRIDES=dyg.stride(),
        DG_STRIDES=dg.stride(),
        DA_STRIDES=dA.stride(),
        H=H,
        D=D,
        BT=BT,
        BD=BD,
    )

    tag = f"T={T} D={D} lb={lower_bound} bias={has_bias} strided={strided}"
    assert_golden(
        dg, gold_dg.reshape(T, H, D), ref_dg.reshape(T, H, D), dtype, f"gate_bwd dg {tag}"
    )
    assert_golden(dA.sum(0), gold_dA, ref_dA, dtype, f"gate_bwd dA_log {tag}")
    if has_bias:
        assert_golden(dg.sum(0), gold_db, ref_db.reshape(H, D), dtype, f"gate_bwd dt_bias {tag}")


# Plain chunk-local cumsum (scalar + vector), used by the reverse-cumsum adjoint.
# Exercise every static branch combination once, every runtime boundary once,
# and every dtype once instead of taking their full Cartesian product.
_CUMSUM_CASES = (
    [(torch.float32, 17, reverse, scale) for reverse, scale in product((False, True), (None, 2.0))]
    + [(torch.float32, T, False, None) for T in GATE_TS if T != 17]
    + [(dtype, 17, False, None) for dtype in DTYPES[1:]]
)


@pytest.mark.parametrize("dtype,T,reverse,scale", _CUMSUM_CASES, ids=_case_id)
def test_cumsum_scalar(dtype, T, reverse, scale):
    torch.manual_seed(5)
    B, H, BT = 2, 2, 16
    s64 = torch.randn(B, T, H, device=DEV, dtype=torch.float64)
    golden = chunk_cumsum_ref(s64, BT, reverse, scale, None)
    s = s64.to(dtype)
    ref = chunk_cumsum_ref(s, BT, reverse, scale, None)

    o = torch.empty_like(s)
    NT = triton.cdiv(T, BT)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s,
        o,
        scale,
        None,
        None,
        T,
        s.stride(0),
        o.stride(0),
        S_STRIDES=s.stride()[1:],
        O_STRIDES=o.stride()[1:],
        B=B,
        H=H,
        BT=BT,
        REVERSE=reverse,
    )
    assert_golden(o, golden, ref, dtype, f"cumsum_scalar T={T} rev={reverse} scale={scale}")


class _KernelRecorder:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.name, args, kwargs))

        return launch


def test_cumsum_vector_autotune_is_batch_invariant():
    assert "B" not in _autotuner(chunk_local_cumsum_vector_kernel).keys


def test_cumsum_scalar_dispatches_by_layout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cumsum_module,
        "chunk_local_cumsum_scalar_kernel",
        _KernelRecorder("scalar", calls),
    )
    monkeypatch.setattr(
        cumsum_module,
        "chunk_local_cumsum_vector_kernel",
        _KernelRecorder("vector", calls),
    )

    token_major = torch.randn(2, 65, 96, device=DEV)
    output = chunk_local_cumsum_scalar(token_major, chunk_size=64)
    name, args, kwargs = calls.pop()
    assert name == "vector"
    assert args[0].shape == (2, 65, 1, 96)
    assert args[0].data_ptr() == token_major.data_ptr()
    assert args[1].data_ptr() == output.data_ptr()
    assert kwargs["S_STRIDES"] == args[0].stride()[1:]
    assert kwargs["H"] == 1
    assert kwargs["S"] == 96

    strided = _strided_copy(token_major)
    chunk_local_cumsum_scalar(strided, chunk_size=64)
    name, args, kwargs = calls.pop()
    assert name == "vector"
    assert args[0].data_ptr() == strided.data_ptr()
    assert args[0].stride(-1) == 2
    assert kwargs["S_STRIDES"] == args[0].stride()[1:]

    head_first = token_major.transpose(1, 2).contiguous()
    chunk_local_cumsum_scalar(head_first, chunk_size=64, head_first=True)
    assert calls.pop()[0] == "scalar"

    small = torch.randn(2, 65, 15, device=DEV)
    chunk_local_cumsum_scalar(small, chunk_size=64)
    assert calls.pop()[0] == "scalar"


@pytest.mark.parametrize(("reverse", "scale"), ((False, None), (True, 1.25)))
def test_cumsum_scalar_vector_view_fixed(reverse, scale):
    torch.manual_seed(51)
    B, T, H, BT = 3, 129, 96, 64
    source = torch.randn(B, T, H, device=DEV)
    expected = chunk_cumsum_ref(source, BT, reverse=reverse, scale=scale)

    actual = chunk_local_cumsum_scalar(source, BT, reverse=reverse, scale=scale)
    torch.testing.assert_close(actual, expected)

    strided = _strided_copy(source)
    strided_actual = chunk_local_cumsum_scalar(strided, BT, reverse=reverse, scale=scale)
    torch.testing.assert_close(strided_actual, expected)

    single = chunk_local_cumsum_scalar(source[1:2], BT, reverse=reverse, scale=scale)
    torch.testing.assert_close(actual[1], single[0], rtol=0, atol=0)

    head_first = source.transpose(1, 2).contiguous()
    head_first_actual = chunk_local_cumsum_scalar(
        head_first,
        BT,
        reverse=reverse,
        scale=scale,
        head_first=True,
    )
    torch.testing.assert_close(head_first_actual, expected.transpose(1, 2))


@pytest.mark.parametrize("dtype,T,reverse,scale", _CUMSUM_CASES, ids=_case_id)
def test_cumsum_vector(dtype, T, reverse, scale):
    torch.manual_seed(6)
    B, H, S, BT = 2, 2, 16, 16
    s64 = torch.randn(B, T, H, S, device=DEV, dtype=torch.float64)
    golden = chunk_cumsum_ref(s64, BT, reverse, scale, None)
    s = s64.to(dtype)
    ref = chunk_cumsum_ref(s, BT, reverse, scale, None)

    o = torch.empty_like(s)
    NT = triton.cdiv(T, BT)
    grid = lambda meta: (triton.cdiv(S, meta["BS"]), NT, B * H)
    chunk_local_cumsum_vector_kernel[grid](
        s,
        o,
        scale,
        None,
        None,
        T,
        s.stride(0),
        o.stride(0),
        S_STRIDES=s.stride()[1:],
        O_STRIDES=o.stride()[1:],
        B=B,
        H=H,
        S=S,
        BT=BT,
        REVERSE=reverse,
    )
    assert_golden(o, golden, ref, dtype, f"cumsum_vector T={T} rev={reverse} scale={scale}")


_CUMSUM_VARLEN_CASES = [
    (torch.float32, docs, reverse) for docs, reverse in product(VARLEN_DOCS, (False, True))
] + [(dtype, VARLEN_DOCS[2], False) for dtype in DTYPES[1:]]


@pytest.mark.parametrize(
    "dtype,docs,reverse",
    _CUMSUM_VARLEN_CASES,
    ids=_case_id,
)
def test_cumsum_vector_varlen(dtype, docs, reverse):
    torch.manual_seed(7)
    H, S, BT = 2, 16, 16
    total = sum(docs)
    cu = _cu_seqlens(docs)
    chunk_indices = prepare_chunk_indices(cu, BT)
    s64 = torch.randn(1, total, H, S, device=DEV, dtype=torch.float64)
    golden = chunk_cumsum_ref(s64, BT, reverse, None, cu)
    s = s64.to(dtype)
    ref = chunk_cumsum_ref(s, BT, reverse, None, cu)

    o = torch.empty_like(s)
    NT = chunk_indices.shape[0]
    grid = lambda meta: (triton.cdiv(S, meta["BS"]), NT, H)
    chunk_local_cumsum_vector_kernel[grid](
        s,
        o,
        None,
        cu,
        chunk_indices,
        total,
        s.stride(0),
        o.stride(0),
        S_STRIDES=s.stride()[1:],
        O_STRIDES=o.stride()[1:],
        B=1,
        H=H,
        S=S,
        BT=BT,
        REVERSE=reverse,
    )
    assert_golden(o, golden, ref, dtype, f"cumsum_vector_varlen docs={docs} rev={reverse}")


def test_cumsum_varlen_head_first():
    """Exercise packed documents stored physically as (B, H, T[, S])."""
    torch.manual_seed(10)
    dtype = torch.float32
    docs = (7, 17)
    total, H, S, BT = sum(docs), 2, 16, 16
    cu = _cu_seqlens(docs)
    chunk_indices = prepare_chunk_indices(cu, BT)
    NT = chunk_indices.shape[0]

    scalar64 = torch.randn(1, total, H, device=DEV, dtype=torch.float64)
    scalar = scalar64.to(dtype).permute(0, 2, 1).contiguous().permute(0, 2, 1)
    scalar_out = torch.empty(1, H, total, device=DEV, dtype=dtype).permute(0, 2, 1)
    chunk_local_cumsum_scalar_kernel[(NT, H)](
        scalar,
        scalar_out,
        None,
        cu,
        chunk_indices,
        total,
        scalar.stride(0),
        scalar_out.stride(0),
        S_STRIDES=scalar.stride()[1:],
        O_STRIDES=scalar_out.stride()[1:],
        B=1,
        H=H,
        BT=BT,
        REVERSE=True,
    )
    scalar_golden = chunk_cumsum_ref(scalar64, BT, True, None, cu)
    scalar_ref = chunk_cumsum_ref(scalar, BT, True, None, cu)
    assert_golden(
        scalar_out,
        scalar_golden,
        scalar_ref,
        dtype,
        "cumsum_scalar_varlen_head_first",
    )

    vector64 = torch.randn(1, total, H, S, device=DEV, dtype=torch.float64)
    vector = vector64.to(dtype).permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    vector_out = torch.empty(1, H, total, S, device=DEV, dtype=dtype).permute(0, 2, 1, 3)
    grid = lambda meta: (triton.cdiv(S, meta["BS"]), NT, H)
    chunk_local_cumsum_vector_kernel[grid](
        vector,
        vector_out,
        None,
        cu,
        chunk_indices,
        total,
        vector.stride(0),
        vector_out.stride(0),
        S_STRIDES=vector.stride()[1:],
        O_STRIDES=vector_out.stride()[1:],
        B=1,
        H=H,
        S=S,
        BT=BT,
        REVERSE=True,
    )
    vector_golden = chunk_cumsum_ref(vector64, BT, True, None, cu)
    vector_ref = chunk_cumsum_ref(vector, BT, True, None, cu)
    assert_golden(
        vector_out,
        vector_golden,
        vector_ref,
        dtype,
        "cumsum_vector_varlen_head_first",
    )


def test_cumsum_scalar_vector_view_varlen():
    torch.manual_seed(52)
    docs = (31, 65, 79)
    H, BT = 96, 64
    cu = _cu_seqlens(docs)
    source = torch.randn(1, sum(docs), H, device=DEV)
    expected = chunk_cumsum_ref(source, BT, reverse=True, scale=0.75, cu_seqlens=cu)

    actual = chunk_local_cumsum_scalar(
        source,
        BT,
        reverse=True,
        scale=0.75,
        cu_seqlens=cu,
    )
    torch.testing.assert_close(actual, expected)

    head_first = source.transpose(1, 2).contiguous()
    head_first_actual = chunk_local_cumsum_scalar(
        head_first,
        BT,
        reverse=True,
        scale=0.75,
        cu_seqlens=cu,
        head_first=True,
    )
    torch.testing.assert_close(head_first_actual, expected.transpose(1, 2))

    start, end = docs[0], docs[0] + docs[1]
    single = chunk_local_cumsum_scalar(
        source[:, start:end],
        BT,
        reverse=True,
        scale=0.75,
        cu_seqlens=_cu_seqlens((docs[1],)),
    )
    torch.testing.assert_close(actual[:, start:end], single, rtol=0, atol=0)


def _bwd_dav_ref(v, A, do, scale, chunk_size=64):
    """Reference for ``chunk_kda_bwd_kernel_dAv``.

    The forward adds ``o += tril(A) @ v_new`` per chunk (``A`` lower-triangular incl. diag),
    so the adjoints are ``dv = tril(A)^T @ do`` and ``dA = tril(do @ v^T) * scale``.
    """
    B, T, H, V = v.shape
    num_chunks = (T + chunk_size - 1) // chunk_size
    acc = torch.float64 if v.dtype == torch.float64 else torch.float32
    vc, Ac, doc = v.to(acc), A.to(acc), do.to(acc)
    dv = torch.zeros(B, T, H, V, dtype=acc, device=v.device)
    dA = torch.zeros(B, T, H, chunk_size, dtype=acc, device=v.device)
    full_tril = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=v.device))
    for it in range(num_chunks):
        s = it * chunk_size
        e = min(s + chunk_size, T)
        L = e - s
        tril = full_tril[:L, :L][None, :, None, :]  # (1, l, 1, m), l >= m
        dob = doc[:, s:e]  # (b, l, h, v)
        dv[:, s:e] = torch.einsum("blhm,blhv->bmhv", Ac[:, s:e, :, :L] * tril, dob)
        dA[:, s:e, :, :L] = torch.einsum("blhv,bmhv->blhm", dob, vc[:, s:e]) * scale * tril
    return dv, dA


@pytest.mark.parametrize(
    "dtype,T,H",
    [
        (torch.float16, 64, 1),
        (torch.bfloat16, 128, 2),
        (torch.bfloat16, 130, 2),
    ],
    ids=["single-fp16", "multi-bf16", "tail-bf16"],
)
def test_chunk_kda_bwd_dav(dtype, T, H):
    torch.manual_seed(10)
    B, V = 1, 64
    scale = 0.5
    # Build the low-precision inputs first, then upcast the *same* values for the fp64 measuring
    # stick so the reference error reflects only compute precision, not input quantization.
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    A = torch.randn(B, T, H, 64, device="cuda", dtype=dtype)

    gdv, gdA = _bwd_dav_ref(v.double(), A.double(), do.double(), scale)
    rdv, rdA = _bwd_dav_ref(v, A, do, scale)

    dv = torch.empty(B, T, H, V, device="cuda", dtype=dtype)
    dA = torch.zeros(B, T, H, 64, device="cuda", dtype=dtype)
    chunk_kda_bwd_kernel_dAv[(triton.cdiv(T, 64), B * H)](
        v,
        A,
        do,
        dv,
        dA,
        None,
        None,
        None,
        scale,
        T,
        H=H,
        V=V,
        BT=64,
        BV=V,
    )
    tag = f"T={T} H={H}"
    assert_golden(dv, gdv, rdv, dtype, f"bwd_dav dv {tag}")
    assert_golden(dA, gdA, rdA, dtype, f"bwd_dav dA {tag}")


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
        q,
        v,
        g,
        h,
        o,
        A,
        None,
        None,
        None,
        scale,
        T,
        q.stride(1),
        q.stride(2),
        H=H,
        K=K,
        V=V,
        BT=64,
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
    """Reference for ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` (USE_GK + USE_EXP2).

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
    state = torch.zeros(B, H, K, V, dtype=acc, device=k.device) if h0 is None else h0.to(acc)
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
    return h.reshape(B * num_chunks, H, K, V), v_new, state


@pytest.mark.parametrize(
    "dtype,T,use_h0",
    [
        (torch.float16, 64, False),
        (torch.bfloat16, 128, True),
        (torch.bfloat16, 130, False),
    ],
    ids=["single-fp16", "multi-state-bf16", "tail-bf16"],
)
def test_chunk_delta_h_fwd(dtype, T, use_h0):
    torch.manual_seed(12)
    B, H, K, V = 1, 2, 128, 128
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    w = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    # Since prod keeps initial and final state in fp32, do that here to match.
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) if use_h0 else None

    gh, gvn, ght = _fwd_h_ref(
        k.double(), w.double(), v.double(), gk.double(), h0.double() if use_h0 else None
    )
    rh, rvn, rht = _fwd_h_ref(k, w, v, gk, h0)

    h = torch.empty(B * triton.cdiv(T, 64), H, K, V, device="cuda", dtype=dtype)
    v_new = torch.empty(B, T, H, V, device="cuda", dtype=dtype)
    ht = torch.empty(B, H, K, V, device="cuda", dtype=torch.float32)
    grid = lambda meta: (B * H, triton.cdiv(V, meta["BV"]))
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k,
        v,
        w,
        v_new,
        None,
        gk,
        h,
        h0,
        ht,
        None,
        None,
        None,
        T,
        H=H,
        K=K,
        V=V,
        BT=64,
        USE_EXP2=True,
    )
    tag = f"T={T} h0={use_h0}"
    assert_golden(h, gh, rh, dtype, f"delta_h h {tag}")
    assert_golden(v_new, gvn, rvn, dtype, f"delta_h v_new {tag}")
    assert_golden(ht, ght, rht, dtype, f"delta_h ht {tag}")


@pytest.mark.parametrize("use_h0", [False, True], ids=["no-state", "state"])
def test_chunk_delta_h_fwd_forloop(use_h0):
    """Cover the persistent-grid delta-h variant that walks multiple sequences per program."""
    dtype, T = torch.bfloat16, 128
    torch.manual_seed(12)
    B, H, K, V = 2, 2, 128, 128
    BV = 64
    num_chunks = triton.cdiv(T, 64)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    w = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    # gk is the fused gate/cumsum output and is always fp32 (the kernel exp2s it directly).
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    # State (initial + final) is fp32 in production; h and v_new stay in the input dtype. See the
    # note in test_chunk_delta_h_fwd.
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) if use_h0 else None

    gh, gvn, ght = _fwd_h_ref(
        k.double(), w.double(), v.double(), gk.double(), h0.double() if use_h0 else None
    )
    rh, rvn, rht = _fwd_h_ref(k, w, v, gk, h0)

    h = torch.empty(B * num_chunks, H, K, V, device="cuda", dtype=dtype)
    v_new = torch.empty(B, T, H, V, device="cuda", dtype=dtype)
    ht = torch.empty(B, H, K, V, device="cuda", dtype=torch.float32)
    # GRID_N=1 forces each program to walk all MAX_N=B sequences through the persistent loop.
    grid_n = 1
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_forloop[(grid_n * H, triton.cdiv(V, BV))](
        k,
        v,
        w,
        v_new,
        None,
        gk,
        h,
        h0,
        ht,
        None,
        None,
        None,
        T,
        H=H,
        K=K,
        V=V,
        BT=64,
        BV=BV,
        USE_EXP2=True,
        GRID_N=grid_n,
        MAX_N=B,
        num_warps=4,
        num_stages=2,
    )
    tag = f"T={T} h0={use_h0}"
    assert_golden(h, gh, rh, dtype, f"delta_h_forloop h {tag}")
    assert_golden(v_new, gvn, rvn, dtype, f"delta_h_forloop v_new {tag}")
    assert_golden(ht, ght, rht, dtype, f"delta_h_forloop ht {tag}")


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
        chunk_indices=None,
        chunk_offsets=None,
        num_chunks=None,
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

    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    metadata = ChunkMetadata(
        cu,
        prepare_chunk_indices(cu, 64).to(torch.int32),
        torch.tensor(num_chunks, device="cuda", dtype=torch.int32),
    )
    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        metadata=metadata,
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
    """Reference for ``blackwell_delta_h_bwd_dhu_v1`` — the backward of the delta-rule
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
        dht.to(acc).clone()
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
    dh0 = D if h0 is not None else None
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
    dv = torch.randn(B, T, H, V, device="cuda", dtype=torch.float32)
    # gk is the cumulative per-channel log2-decay (<= 0), so 2^{gk_last} <= 1.
    gk = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 0.5
    h0 = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) if use_h0 else None
    dht = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32) if use_dht else None

    qb, kb, wb, dob, dvb = (t.to(torch.bfloat16) for t in (q, k, w, do, dv))
    dh, dh0, dv2 = blackwell_delta_h_bwd_dhu_v1(
        qb, kb, wb, dob, dvb, gk=gk, h0=h0, dht=dht, scale=scale, chunk_size=64, bv=bv
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

    def fake_v1(*args, bv, **kwargs):
        captured["bv"] = bv
        return zeros, None, zeros

    class _Props:
        multi_processor_count = sm_count

    monkeypatch.setattr(dispatch_mod, "get_device_properties", lambda device: _Props())
    monkeypatch.setattr(dispatch_mod, "blackwell_delta_h_bwd_dhu_v1", fake_v1)

    dispatch_mod.blackwell_delta_h_bwd_dhu_dispatch(q, q, q, zeros, zeros)
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

    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    metadata = ChunkMetadata(
        cu,
        prepare_chunk_indices(cu, 64).to(torch.int32),
        torch.tensor(num_chunks, device="cuda", dtype=torch.int32),
    )
    w, u, _kg, Aqk, Akk = chunk_kda_fwd_intra_cute(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        metadata,
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


def _fixed_meta(num_chunks):
    """Build single-document (cu_seqlens, chunk_indices, num_chunks) metadata for T=num_chunks*64.

    The composed CuTe backward kernels compile a single-document work-list, so the leaf-kernel
    tests exercise one packed document of complete 64-token chunks.
    """
    T = num_chunks * 64
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    chunk_indices = prepare_chunk_indices(cu, 64).to(torch.int32)
    nc = torch.tensor([chunk_indices.shape[0]], dtype=torch.int32, device="cuda")
    return cu, chunk_indices, nc


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


def _default_worklist(T, chunk_size, device):
    """Single-doc [0, T] fallback: cu_seqlens + one (seq_idx=0, chunk_idx) row per full chunk."""
    n = T // chunk_size
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    chunk_indices = torch.stack(
        [torch.zeros(n, dtype=torch.long, device=device), torch.arange(n, device=device)], dim=1
    )
    return cu, chunk_indices


def _bwd_intra_ref(q, k, g, beta, dAqk, dAkk, chunk_size=64, cu_seqlens=None, chunk_indices=None):
    """fp64 oracle for ``chunk_kda_bwd_intra`` (the intra-chunk backward of the K3b/K4b
    forward).

    Given the incoming grads ``dAqk``/``dAkk`` (grad of loss w.r.t. the forward Aqk/Akk),
    returns this stage's intra contributions (dq, dk, db, dg) with the running grads set to
    zero. Per 64-token chunk, indexing i=query/row, j=key/col, ``exp2 = 2^{g_i - g_j}``::

        Aqk path:  dq = sum_j dAqk*exp2*k_j        dk += sum_i dAqk*exp2*q_i
                   dg  = sum_j dAqk*exp2*q_i*k_j - sum_i (same)
        Akk path:  dk += sum_j dAkk*exp2*beta_i*k_j + sum_i dAkk*exp2*beta_i*k_i
                   db  = sum_j dAkk * <exp2*k_i, k_j>
                   dg += sum_j dAkk*exp2*beta_i*k_i*k_j - sum_i (same)

    Both dAqk and dAkk are masked with a NON-strict causal mask (i>=j, incl. diagonal) to
    match fla upstream. There is no ``scale`` because it is folded upstream into dAqk.
    The final ``dg`` writer applies ``ln(2)`` for the derivative of ``2**g``.
    """
    B, T, H, Kd = q.shape
    device = q.device
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, gf, bf = (t.to(acc) for t in (q, k, g, beta))
    daqk, dakk = dAqk.to(acc), dAkk.to(acc)
    dq = torch.zeros(B, T, H, Kd, dtype=acc, device=device)
    dk = torch.zeros_like(dq)
    dg = torch.zeros_like(dq)
    db = torch.zeros(B, T, H, dtype=acc, device=device)
    # Varlen work-list iteration (see ``_bwd_wy_dqkg_ref``): a partial last chunk uses only its
    # ``valid`` rows/cols so the non-strict causal mask never straddles a document boundary.
    if chunk_indices is None:
        cu_seqlens, chunk_indices = _default_worklist(T, chunk_size, device)
    cu = cu_seqlens.tolist()
    for b in range(B):
        for seq_idx, chunk_idx in chunk_indices.tolist():
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            row_start = bos + chunk_idx * chunk_size
            cl = min(eos - row_start, chunk_size)
            s = slice(row_start, row_start + cl)
            mask = torch.tril(torch.ones(cl, cl, dtype=torch.bool, device=device))[:, :, None]
            q_i, k_i, k_j = qf[b, s][:, None], kf[b, s][:, None], kf[b, s][None, :]
            beta_i = bf[b, s][:, None, :, None]
            exp2 = torch.exp2(gf[b, s][:, None] - gf[b, s][None, :])  # 2^{g_i - g_j}
            aq = torch.where(mask, daqk[b, s, :, :cl].permute(0, 2, 1), 0.0)  # (i, j, H)
            ak = torch.where(mask, dakk[b, s, :, :cl].permute(0, 2, 1), 0.0)

            aqk = aq[..., None] * exp2  # (i, j, H, K)
            t_aqk = aqk * q_i * k_j
            dq[b, s] = (aqk * k_j).sum(1)
            dk[b, s] = (aqk * q_i).sum(0)
            dg[b, s] = t_aqk.sum(1) - t_aqk.sum(0)

            akk = ak[..., None] * exp2 * beta_i  # (i, j, H, K)
            t_akk = akk * k_i * k_j
            dk[b, s] += (akk * k_j).sum(1) + (akk * k_i).sum(0)
            db[b, s] = (ak * (exp2 * k_i * k_j).sum(-1)).sum(1)
            dg[b, s] += t_akk.sum(1) - t_akk.sum(0)
    return dq, dk, db, dg * math.log(2.0)


@requires_cute
def test_chunk_kda_bwd_intra_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(3)
    B, K = 1, 128
    T = num_chunks * 64
    cu, chunk_indices, nc = _fixed_meta(num_chunks)
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
        ChunkMetadata(cu, chunk_indices, nc),
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


def _bwd_wy_dqkg_ref(
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
    scale,
    chunk_size=64,
    cu_seqlens=None,
    chunk_indices=None,
):
    """fp64 oracle for ``chunk_kda_bwd_wy_dqkg_fused`` — the fused WY / ``(I-Akk)^-1``
    chunk-level backward.

    Given the recomputed forward intermediates (``A`` = Akk inverse, ``h`` = chunk-start
    hidden state, ``v_new``) and the incoming grads (``do``, ``dh``, ``dv``), produces the
    six per-token grads. All six are FRESH (write-only). Conventions: ``scale`` (=K**-0.5)
    multiplies ``dq`` only; ``beta`` enters raw; gate grads differentiate ``2^g`` directly
    (no ``ln2``); ``dA`` is masked with a STRICT-lower mask (diagonal=-1)::

        dq  = (do @ h) * 2^g * scale
        dv2 = (A^T @ dv) * beta ; dk = (v_new @ dh) * 2^{g_last-g} + (A^T @ dw) * 2^g * beta
        dA  = -A^T @ striL(beta * (dv@v^T + dw@kg^T)) @ A     (dw = -dv @ h, kg = k*2^g)

    Per 64-token chunk; ``A``/``dA`` sliced to ``[:chunk_len]``; ``h``/``dh`` indexed by
    the chunk id on the num_chunks axis.
    """
    B, T, H, K = q.shape
    V = v.shape[3]
    device = q.device
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    dq = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dk = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dv2 = torch.zeros(B, T, H, V, dtype=acc, device=device)
    db = torch.zeros(B, T, H, dtype=acc, device=device)
    dg = torch.zeros(B, T, H, K, dtype=acc, device=device)
    dA = torch.zeros(B, T, H, chunk_size, dtype=acc, device=device)
    if chunk_indices is None:
        cu_seqlens, chunk_indices = _default_worklist(T, chunk_size, device)
    cu = cu_seqlens.tolist()
    for b in range(B):
        for flat, (seq_idx, chunk_idx) in enumerate(chunk_indices.tolist()):
            bos, eos = cu[seq_idx], cu[seq_idx + 1]
            row_start = bos + chunk_idx * chunk_size
            cl = min(eos - row_start, chunk_size)
            s = slice(row_start, row_start + cl)
            q_f, k_f, v_f = q[b, s].to(acc), k[b, s].to(acc), v[b, s].to(acc)
            v_new_f, g_f = v_new[b, s].to(acc), g[b, s].to(acc)
            beta_f = beta[b, s].to(acc)
            A_f = A[b, s, :, :cl].to(acc)
            h_f, dh_f = h[b, flat].to(acc), dh[b, flat].to(acc)
            do_f, dv_f = do[b, s].to(acc), dv[b, s].to(acc)

            strict = torch.tril(torch.ones(cl, cl, device=device, dtype=torch.bool), -1)
            exp2_g = torch.exp2(g_f)
            beta_k = beta_f.unsqueeze(-1)
            kg = k_f * exp2_g  # k * 2^g
            A_t = A_f.permute(1, 2, 0)  # (H, col, row): A_f transposed on (row, col)

            rev_decay = torch.exp2(g_f[-1:].float() - g_f)  # 2^{g_last - g}, per key channel
            dq_chunk = torch.einsum("thv,hkv->thk", do_f, h_f) * exp2_g * scale
            dk_state = torch.einsum("thv,hkv->thk", v_new_f, dh_f) * rev_decay
            dw = -torch.einsum("thv,hkv->thk", dv_f, h_f)  # (row, H, K)
            dvb = torch.einsum("ths,thv->shv", A_f, dv_f)  # (A^T @ dv)  (col, H, V)
            dkgb = torch.einsum("ths,thk->shk", A_f, dw)  # (A^T @ dw)  (col, H, K)

            dv2_chunk = dvb * beta_k
            db_chunk = (dvb * v_f).sum(-1) + (dkgb * kg).sum(-1)
            dk_chunk = dk_state + dkgb * exp2_g * beta_k
            kdk_state = k_f * dk_state
            dg_chunk = q_f * dq_chunk - kdk_state + kg * dkgb * beta_k
            dg_chunk[-1] += (h_f * dh_f).sum(-1) * torch.exp2(g_f[-1]) + kdk_state.sum(0)

            dA_repr = torch.einsum("thv,shv->hts", dv_f, v_f) + torch.einsum(
                "thk,shk->hts", dw, kg
            )
            dA_repr = torch.where(strict, dA_repr, 0.0) * beta_f.transpose(0, 1).unsqueeze(1)
            dA_raw = torch.where(strict, -torch.bmm(A_t, torch.bmm(dA_repr, A_t)), 0.0)

            dq[b, s], dk[b, s] = dq_chunk, dk_chunk
            dv2[b, s], db[b, s], dg[b, s] = dv2_chunk, db_chunk, dg_chunk
            dA[b, s, :, :cl] = dA_raw.permute(1, 0, 2)
    return dq, dk, dv2, db, dg, dA


@requires_cute
def test_chunk_kda_bwd_wy_dqkg_fused_cute():
    num_chunks, H = 4, 4
    torch.manual_seed(4)
    B, K, V = 1, 128, 128
    HV = H
    T = num_chunks * 64
    scale = K**-0.5
    cu, chunk_indices, num_chunks_tensor = _fixed_meta(num_chunks)

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
        ChunkMetadata(cu, chunk_indices, num_chunks_tensor),
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
