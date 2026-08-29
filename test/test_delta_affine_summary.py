"""Tests for the shared CuTeDSL delta-rule affine-summary recurrence."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

from attn_gym.linear._delta_rule.cute import build_state_grad_summary, build_state_summary

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL affine summary requires CUDA capability 10.0 or newer",
)


def int64_stride_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a B=1 tensor into a contiguous view whose unreachable batch stride needs int64."""
    result = torch.empty_strided(
        tensor.shape,
        (2**31, *tensor.stride()[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result.copy_(tensor)
    assert result.is_contiguous()
    return result


def misaligned_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into contiguous storage beginning one element past an aligned allocation."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous() and result.data_ptr() % 16 != 0
    return result


def state_summary_reference(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the augmented affine recurrence in eager FP32."""
    _, tokens, heads, key_dim = kg.shape
    value_dim = u.shape[-1]
    pad = (-tokens) % 64
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        kg, w, u = (F.pad(tensor, padding) for tensor in (kg, w, u))
        cumulative_gate = torch.cat(
            (cumulative_gate, cumulative_gate[:, -1:].expand(-1, pad, -1, -1)),
            dim=1,
        )
        tokens += pad
    identity = torch.eye(key_dim, dtype=torch.float32, device=kg.device)
    state = torch.cat(
        (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float32, device=kg.device),
            identity.expand(heads, key_dim, key_dim),
        ),
        dim=-1,
    )
    for start in range(0, tokens, 64):
        stop = start + 64
        chunk_w = w[0, start:stop].transpose(0, 1).float()
        chunk_kg = kg[0, start:stop].transpose(0, 1).float()
        tmp = chunk_w @ state
        tmp[..., :value_dim] = u[0, start:stop].transpose(0, 1).float() - tmp[..., :value_dim]
        tmp[..., value_dim:].neg_()
        decay = cumulative_gate[0, stop - 1].exp2().unsqueeze(-1)
        state = state * decay + chunk_kg.transpose(-2, -1) @ tmp
    return state.transpose(-2, -1).contiguous()


def state_grad_summary_reference(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate the augmented reverse-affine recurrence in eager FP32."""
    _, tokens, heads, key_dim = qg.shape
    value_dim = dout.shape[-1]
    pad = (-tokens) % 64
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        qg, kg, w, dout, aqk = (F.pad(tensor, padding) for tensor in (qg, kg, w, dout, aqk))
        cumulative_gate = torch.cat(
            (cumulative_gate, cumulative_gate[:, -1:].expand(-1, pad, -1, -1)),
            dim=1,
        )
        tokens += pad
    identity = torch.eye(key_dim, dtype=torch.float32, device=qg.device)
    state = torch.cat(
        (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float32, device=qg.device),
            identity.expand(heads, key_dim, key_dim),
        ),
        dim=-1,
    )
    for start in range(tokens - 64, -1, -64):
        stop = start + 64
        chunk_kg = kg[0, start:stop].transpose(0, 1).float()
        chunk_w = w[0, start:stop].transpose(0, 1).float()
        chunk_qg = qg[0, start:stop].transpose(0, 1).float()
        chunk_aqk = aqk[0, start:stop].transpose(0, 1).float()
        dout_augmented = torch.zeros(
            heads, 64, value_dim + key_dim, dtype=torch.float32, device=qg.device
        )
        dout_augmented[..., :value_dim] = dout[0, start:stop].transpose(0, 1).float()
        corrected = chunk_kg @ state + chunk_aqk.transpose(-2, -1) @ dout_augmented
        decay = cumulative_gate[0, stop - 1].exp2().unsqueeze(-1)
        state = (
            state * decay
            + scale * chunk_qg.transpose(-2, -1) @ dout_augmented
            - chunk_w.transpose(-2, -1) @ corrected
        )
    return state.transpose(-2, -1).contiguous()


def test_affine_summaries_support_int64_tensor_strides():
    """Compile wide-address kernels for realistic long-sequence batch strides."""
    torch.manual_seed(19)
    shape = (1, 64, 1, 128)
    qg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    u = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, 64, 1, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    expected_forward = build_state_summary(kg, w, u, cumulative_gate)
    expected_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )
    actual_forward = build_state_summary(
        *(int64_stride_copy(tensor) for tensor in (kg, w, u, cumulative_gate))
    )
    actual_reverse = build_state_grad_summary(
        *(int64_stride_copy(tensor) for tensor in (qg, kg, w, dout, aqk, cumulative_gate)),
        128**-0.5,
    )

    torch.testing.assert_close(actual_forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(actual_reverse, expected_reverse, rtol=0, atol=0)


def test_affine_summaries_accept_misaligned_contiguous_storage():
    """Normalize ordinary storage-offset views before constructing TMA descriptors."""
    torch.manual_seed(21)
    shape = (1, 64, 1, 128)
    qg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    u = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, 64, 1, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    expected_forward = build_state_summary(kg, w, u, cumulative_gate)
    expected_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )
    actual_forward = build_state_summary(
        *(misaligned_copy(tensor) for tensor in (kg, w, u, cumulative_gate))
    )
    actual_reverse = build_state_grad_summary(
        *(misaligned_copy(tensor) for tensor in (qg, kg, w, dout, aqk, cumulative_gate)),
        128**-0.5,
    )

    torch.testing.assert_close(actual_forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(actual_reverse, expected_reverse, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,heads", [(64, 1), (65, 2), (256, 3)])
def test_affine_summary_matches_fp32_reference(dtype, tokens, heads):
    torch.manual_seed(23)
    shape = (1, tokens, heads, 128)
    kg = torch.randn(shape, dtype=dtype, device="cuda") / 16
    w = torch.randn_like(kg) / 16
    u = torch.randn_like(kg) / 16
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    actual = build_state_summary(kg, w, u, cumulative_gate)
    expected = state_summary_reference(kg, w, u, cumulative_gate)

    assert actual.shape == (heads, 256, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,heads", [(64, 1), (65, 2), (256, 3)])
def test_build_state_grad_summary_matches_fp32_reference(dtype, tokens, heads):
    torch.manual_seed(27)
    shape = (1, tokens, heads, 128)
    qg = torch.randn(shape, dtype=dtype, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, tokens, heads, 64, dtype=dtype, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8
    scale = 128**-0.5

    actual = build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)
    expected = state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, scale)

    assert actual.shape == (heads, 256, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_affine_summary_selector_variants_and_persistent_work():
    torch.manual_seed(28)
    shape = (1, 64, 32, 128)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    w = torch.randn_like(kg) / 32
    u = torch.randn_like(kg) / 32
    qg = torch.randn_like(kg) / 32
    dout = torch.randn_like(kg) / 32
    aqk = torch.randn(1, 64, 32, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    torch.testing.assert_close(
        build_state_summary(kg, w, u, cumulative_gate),
        state_summary_reference(kg, w, u, cumulative_gate),
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5),
        state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5),
        atol=2e-4,
        rtol=2e-4,
    )


def test_affine_summaries_reject_torch_export_instead_of_emitting_empty_outputs():
    """Fail capture explicitly until the launchers become registered graph operators."""

    class ForwardSummary(torch.nn.Module):
        def forward(self, kg, w, u, cumulative_gate):
            return build_state_summary(kg, w, u, cumulative_gate)

    class ReverseSummary(torch.nn.Module):
        def forward(self, qg, kg, w, dout, aqk, cumulative_gate):
            return build_state_grad_summary(
                qg,
                kg,
                w,
                dout,
                aqk,
                cumulative_gate,
                128**-0.5,
            )

    shape = (1, 64, 1, 128)
    x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(shape, dtype=torch.float32, device="cuda")
    aqk = torch.zeros(1, 64, 1, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="build_state_summary does not support torch.export"):
        torch.export.export(ForwardSummary(), (x, x, x, gate), strict=False)
    with pytest.raises(TypeError, match="build_state_grad_summary does not support torch.export"):
        torch.export.export(ReverseSummary(), (x, x, x, x, aqk, gate), strict=False)


def test_build_state_summary_cuda_graph_replay():
    torch.manual_seed(29)
    shape = (1, 128, 2, 128)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 16
    w = torch.randn_like(kg) / 16
    u = torch.randn_like(kg) / 16
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    build_state_summary(kg, w, u, cumulative_gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_state_summary(kg, w, u, cumulative_gate)

    u.mul_(0.5)
    expected = state_summary_reference(kg, w, u, cumulative_gate)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_build_state_grad_summary_cuda_graph_replay():
    torch.manual_seed(31)
    shape = (1, 128, 2, 128)
    qg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8
    scale = 128**-0.5

    build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)

    dout.mul_(0.5)
    expected = state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
