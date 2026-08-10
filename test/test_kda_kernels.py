# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU unit tests for the optimized (Triton) KDA leaf kernels."""

from __future__ import annotations

from itertools import product

import pytest
import torch

triton = pytest.importorskip("triton")

# These imports intentionally follow the optional-dependency check above.
import attn_gym.linear.kda.bwd.triton.cumsum as cumsum_module
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
from attn_gym.linear.kda.fwd.triton.gate_fwd import (
    _requires_int64_offsets,
    kda_gate_chunk_cumsum_vector_kernel,
    kda_gate_chunk_cumsum_vector_kernel_forloop,
)
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import (
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
from attn_gym.linear.kda.utils import prepare_chunk_indices

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
        X_STRIDES=x.stride(),
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
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
        X_STRIDES=x.stride(),
        Y_STRIDES=y1.stride(),
        RSTD_STRIDES=rstd1.stride(),
        D=D,
        BD=BD,
    )
    assert_golden(y1, golden, ref, dtype, f"l2norm_fwd_kernel1 T={T} D={D}")
    assert_golden(rstd1, rstd_golden, rstd_ref, dtype, f"l2norm_fwd_kernel1 rstd T={T} D={D}")


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
    l2norm_bwd_kernel[grid](
        y,
        rstd,
        dy,
        dx,
        eps,
        T,
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=dy.stride(),
        DX_STRIDES=dx.stride(),
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
        eps,
        D,
        Y_STRIDES=y.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=dy.stride(),
        DX_STRIDES=dx1.stride(),
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

    o = torch.empty_like(g)
    _run_gate_fwd(g, A_log, bias, o, scale, lower_bound, chunk, None, reverse=reverse)
    tag = f"T={T} lb={lower_bound} bias={has_bias} scale={scale} rev={reverse}"
    assert_golden(o, golden, ref, dtype, f"gate_fwd {tag}")


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

    o = torch.empty_like(g)
    _run_gate_fwd(g, A_log, bias, o, None, lower_bound, chunk, cu)
    assert_golden(o, golden, ref, dtype, f"gate_fwd_varlen docs={docs} lb={lower_bound}")


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
