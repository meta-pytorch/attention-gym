"""Warpdrive protocol and dynamic-ABI coverage for native affine summaries."""

import pytest
import torch

pytest.importorskip(
    "attn_gym.linear._delta_rule.cute.affine_summary_fwd",
    reason="legacy CuTe kernel is not installed",
)

pytest.importorskip("cutlass")

from test_delta_affine_summary import (
    make_summary_inputs,
    state_grad_summary_reference,
    state_summary_reference,
)

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear._delta_rule.cute import affine_summary_fwd as forward
from attn_gym.linear._delta_rule.cute import affine_summary_rev as reverse
from attn_gym.warpdrive.cache import detect_compile_target, get_compile_target, set_compile_target

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="native affine summaries require Blackwell",
)


def test_reverse_summary_compiles_once_across_runtime_shapes():
    reverse._compile_affine_summary_rev.cache_clear()
    shapes = ((1, 1), (16, 16), (17, 17), (32, 2), (1023, 3), (1024, 16), (1025, 1))
    try:
        for tokens, rows in shapes:
            qg, kg, w, _u, dout, aqk, gate = make_summary_inputs(73, tokens=tokens)
            bounds = torch.tensor([[0, tokens]] * rows, dtype=torch.int32, device="cuda")
            out = torch.empty((rows, 1, 256, 128), device="cuda")
            fn = reverse._compile_affine_summary_rev("bf16", 1, False, True, False)
            workspace, _max_ctas = reverse._LAUNCH_WORK.launch_args(out.device)
            fn(qg, kg, w, dout, aqk, gate, bounds, out, 128**-0.5, workspace, 2)
            expected = state_grad_summary_reference(qg, kg, w, dout, aqk, gate, 128**-0.5)
            torch.testing.assert_close(out[0], expected, atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(out, out[:1].expand_as(out), atol=0, rtol=0)
        info = reverse._compile_affine_summary_rev.cache_info()
        assert info.currsize == 1
        assert info.hits + info.misses == len(shapes)
    finally:
        reverse._compile_affine_summary_rev.cache_clear()


def test_reverse_summary_persistent_items_include_empty_and_short_ranges():
    qg, kg, w, _u, dout, aqk, gate = make_summary_inputs(87, tokens=641)
    inputs = (qg, kg, w, dout, aqk, gate)
    ranges = ((0, 641), (1, 302), (302, 302), (302, 303))
    bounds = torch.tensor(ranges, dtype=torch.int32, device="cuda")
    out = torch.empty((len(ranges), 1, 256, 128), device="cuda")
    workspace, _max_ctas = reverse._LAUNCH_WORK.launch_args(out.device)
    fn = reverse._compile_affine_summary_rev("bf16", 1, False, True, False)
    fn(*inputs, bounds, out, 128**-0.5, workspace, 2)
    for row, (start, stop) in enumerate(ranges):
        expected = state_grad_summary_reference(*(t[:, start:stop] for t in inputs), 128**-0.5)
        tolerance = 0 if start == stop else 2e-4
        torch.testing.assert_close(out[row], expected, atol=tolerance, rtol=tolerance)


def test_reverse_summary_verifies_all_tma_rings(monkeypatch, traced_compiles):
    entries = []
    compile_tvm_ffi = reverse.compile_tvm_ffi

    def compile_and_keep(op, *args, **kwargs):
        entries.append(op)
        return compile_tvm_ffi(op, *args, **kwargs)

    monkeypatch.setattr(reverse, "compile_tvm_ffi", compile_and_keep)
    previous = get_compile_target()
    reverse._compile_affine_summary_rev.cache_clear()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        reverse._compile_affine_summary_rev("bf16", 1, False, True, False)
    finally:
        set_compile_target(previous)
        reverse._compile_affine_summary_rev.cache_clear()
    (op,) = entries
    tma = [c.tma for c in op.checks().channels.values() if c.tma is not None]
    assert tma == ["verified"] * 6, op.describe()
    assert list(op.checks().aliases.values()) == ["no hazard found"] * 2, op.describe()


@pytest.mark.parametrize("bn", [32, 64])
@pytest.mark.parametrize("whole_ranges", [True, False], ids=["bounds", "split-table"])
def test_forward_summary_compiles_once_across_runtime_shapes(bn, whole_ranges):
    forward._compile_affine_summary.cache_clear()
    shapes = ((1, 1), (16, 16), (17, 17), (32, 2), (1023, 3), (1024, 16), (1025, 1))
    try:
        for tokens, rows in shapes:
            _qg, kg, w, u, _dout, _aqk, gate = make_summary_inputs(83, tokens=tokens, heads=2)
            ranges = [(0, tokens)] * rows
            if rows > 1:
                ranges[-1] = (1, tokens)
            table = ranges
            if not whole_ranges:
                table = [
                    (start, 0, (stop - start + 63) // 64, stop - start) for start, stop in ranges
                ]
            work = torch.tensor(table, dtype=torch.int32, device="cuda")
            out = torch.empty((rows, 2, 256, 128), device="cuda")
            fn = forward._compile_affine_summary("bf16", 2, bn, False, whole_ranges, False)
            fn(kg, w, u, gate, work, out)
            expected = state_summary_reference(kg, w, u, gate)
            torch.testing.assert_close(out[0], expected, atol=2e-4, rtol=2e-4)
            if rows > 1:
                torch.testing.assert_close(out[:-1], out[:1].expand_as(out[:-1]), atol=0, rtol=0)
                expected = state_summary_reference(*(t[:, 1:tokens] for t in (kg, w, u, gate)))
                torch.testing.assert_close(out[-1], expected, atol=2e-4, rtol=2e-4)
        info = forward._compile_affine_summary.cache_info()
        assert info.currsize == 1
        assert info.hits + info.misses == len(shapes)
    finally:
        forward._compile_affine_summary.cache_clear()


@pytest.mark.parametrize("bn", [32, 64])
def test_forward_summary_verifies_all_tma_rings(monkeypatch, traced_compiles, bn):
    entries = []
    compile_tvm_ffi = forward.compile_tvm_ffi

    def compile_and_keep(op, *args, **kwargs):
        entries.append(op)
        return compile_tvm_ffi(op, *args, **kwargs)

    monkeypatch.setattr(forward, "compile_tvm_ffi", compile_and_keep)
    previous = get_compile_target()
    forward._compile_affine_summary.cache_clear()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        forward._compile_affine_summary("bf16", 1, bn, False, True, False)
    finally:
        set_compile_target(previous)
        forward._compile_affine_summary.cache_clear()
    (op,) = entries
    tma = [c.tma for c in op.checks().channels.values() if c.tma is not None]
    assert tma == ["verified"] * 4, op.describe()
    assert list(op.checks().aliases.values()) == ["no hazard found"] * 2, op.describe()
