"""D=512 training coverage for portable and Blackwell shared-KV Triton schedules."""

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch
from test_selected_attention_triton import (
    assert_matches_low_precision_eager,
    assert_sink_gradient_fp32,
)

from attn_gym.sparse.selected_attention import (
    AuxRequest,
    SelectedAttentionAux,
    selected_attention,
)

pytestmark = pytest.mark.usefixtures("selected_attention_single_config")


class AttentionInputs(NamedTuple):
    query: torch.Tensor
    local_kv: torch.Tensor
    sparse_kv: torch.Tensor
    kv_indices: torch.Tensor
    attention_sink: torch.Tensor
    doc_ids: torch.Tensor | None


def make_inputs(
    heads: int,
    share_kv: bool,
    dtype: torch.dtype,
    seq_len: int = 33,
    topk: int = 19,
    with_docs: bool = True,
    batch: int = 1,
) -> AttentionInputs:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    if heads >= 16 and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Blackwell required for the shared-KV schedule")

    generator = torch.Generator(device="cuda").manual_seed(2026)
    kv_heads = 1 if share_kv else heads
    sparse_seq_len = max(23, topk)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device="cuda", dtype=dtype, generator=generator, requires_grad=True
        )

    query = randn(batch, heads, seq_len, 512)
    local_kv = randn(batch, kv_heads, seq_len, 512)
    sparse_kv = randn(batch, kv_heads, sparse_seq_len, 512)
    kv_indices = torch.randint(
        sparse_seq_len, (batch, seq_len, topk), device="cuda", generator=generator
    )
    if topk:
        kv_indices[:, :, 1] = kv_indices[:, :, 0]
        kv_indices[:, ::3, -1] = -1
        kv_indices[:, 0, :] = -1
    # Keep the sink and its gradient in FP32: BF16 casts would hide delta-reduction errors.
    attention_sink = torch.randn(
        heads, device="cuda", dtype=torch.float32, generator=generator, requires_grad=True
    )
    doc_ids = (
        (torch.arange(seq_len, device="cuda", dtype=torch.int32) // 11)
        .unsqueeze(0)
        .expand(batch, -1)
        if with_docs
        else None
    )
    return AttentionInputs(query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids)


def check_training(
    inputs: AttentionInputs,
    window: int,
    operation: Callable[..., tuple[torch.Tensor, SelectedAttentionAux]] = selected_attention,
    *,
    repeat_backward: bool = False,
) -> None:
    from attn_gym.sparse.selected_attention.impl.triton.primitives import (
        can_use_shared_kv_schedule,
    )

    query, local_kv, sparse_kv, kv_indices, sink, _ = inputs
    heads, seq_len, head_dim = query.shape[1:]
    shared = local_kv.shape[1] == 1
    assert can_use_shared_kv_schedule(
        query,
        sparse_kv.expand(-1, heads, -1, -1),
        local_kv.expand(-1, heads, -1, -1),
        window,
    ) == (heads >= 16)
    differentiable = query, local_kv, sparse_kv, sink
    high_precision_tensors = tuple(
        tensor.detach().double().requires_grad_() for tensor in differentiable
    )
    high_precision_inputs = inputs._replace(
        query=high_precision_tensors[0],
        local_kv=high_precision_tensors[1],
        sparse_kv=high_precision_tensors[2],
        attention_sink=high_precision_tensors[3],
    )
    high_precision_output = selected_attention(*high_precision_inputs, window, backend="eager")
    low_precision_output = selected_attention(*inputs, window, backend="eager")
    actual, aux = operation(*inputs, window, backend="triton", return_aux=AuxRequest(lse=True))
    generator = torch.Generator(device="cuda").manual_seed(1234)
    grad_output = torch.randn(
        query.shape, device=query.device, dtype=query.dtype, generator=generator
    )
    high_precision_grads = torch.autograd.grad(
        high_precision_output, high_precision_tensors, grad_output.double()
    )
    low_precision_grads = torch.autograd.grad(low_precision_output, differentiable, grad_output)

    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if repeat_backward:
            # Forward already ran: switching modes must still use the output-owned fallback.
            torch.use_deterministic_algorithms(True)
        actual_grads = torch.autograd.grad(
            actual, differentiable, grad_output, retain_graph=repeat_backward
        )
        if repeat_backward:
            for repeat in range(3):
                repeated_grads = torch.autograd.grad(
                    actual, differentiable, grad_output, retain_graph=repeat < 2
                )
                for repeated, first in zip(repeated_grads, actual_grads, strict=True):
                    torch.testing.assert_close(repeated, first, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)

    topk = kv_indices.shape[-1]
    forward_reductions = (head_dim, topk + window, topk + window)
    assert_matches_low_precision_eager(
        actual, low_precision_output, high_precision_output, forward_reductions
    )
    gradient_reductions = (
        forward_reductions + (head_dim, topk + window),
        forward_reductions + (head_dim, window, heads if shared else 1),
        forward_reductions + (head_dim, seq_len, heads if shared else 1),
    )
    for index, reductions in enumerate(gradient_reductions):
        assert_matches_low_precision_eager(
            actual_grads[index],
            low_precision_grads[index],
            high_precision_grads[index],
            reductions,
        )
    assert_matches_low_precision_eager(
        actual_grads[3],
        low_precision_grads[3],
        high_precision_grads[3],
        forward_reductions + (head_dim, seq_len),
        quantized_intermediates=2,
        intermediate_dtype=query.dtype,
    )

    assert aux.lse is not None
    assert_sink_gradient_fp32(sink, aux.lse, actual, grad_output, actual_grads[3])
    if topk == 0:
        assert actual_grads[2].count_nonzero() == 0
    if window == 0:
        assert actual_grads[1].count_nonzero() == 0
    if topk == window == 0:
        assert actual.count_nonzero() == 0
        assert all(gradient.count_nonzero() == 0 for gradient in actual_grads)


@pytest.mark.parametrize(
    "heads,share_kv,dtype,seq_len,topk,window,with_docs",
    [
        pytest.param(17, True, torch.bfloat16, 33, 19, 19, True, id="shared-bf16"),
        pytest.param(2, True, torch.bfloat16, 257, 32, 33, True, id="generic-bf16-shared"),
        pytest.param(2, False, torch.bfloat16, 17, 19, 19, True, id="generic-bf16-unshared"),
        pytest.param(2, True, torch.float16, 17, 19, 19, True, id="generic-fp16-shared"),
        pytest.param(2, False, torch.float16, 33, 19, 19, True, id="generic-fp16-unshared"),
        pytest.param(2, True, torch.float32, 33, 19, 19, True, id="generic-fp32-shared"),
        pytest.param(2, False, torch.float32, 17, 19, 19, True, id="generic-fp32-unshared"),
        pytest.param(2, True, torch.bfloat16, 17, 19, 0, False, id="selected-only"),
        pytest.param(17, True, torch.bfloat16, 33, 0, 19, True, id="local-only"),
        pytest.param(2, False, torch.float32, 17, 0, 0, False, id="sink-only"),
        pytest.param(16, True, torch.bfloat16, 17, 512, 128, False, id="dsv4-topk512"),
    ],
)
def test_d512_training(heads, share_kv, dtype, seq_len, topk, window, with_docs):
    inputs = make_inputs(heads, share_kv, dtype, seq_len, topk, with_docs)
    check_training(inputs, window)


@pytest.mark.parametrize("heads,share_kv", [(2, False), (17, True)], ids=["generic", "shared"])
def test_d512_deterministic_backward(heads, share_kv):
    inputs = make_inputs(heads, share_kv, torch.bfloat16, batch=2)
    check_training(inputs, 19, repeat_backward=True)


@pytest.mark.parametrize("heads,share_kv", [(2, False), (17, True)], ids=["generic", "shared"])
def test_d512_torch_compile_fullgraph(heads, share_kv):
    # Stride tuples are Triton constexpr arguments, so each shape specializes independently.
    compiled = torch.compile(selected_attention, fullgraph=True, dynamic=False)
    for seq_len in (17, 33):
        inputs = make_inputs(heads, share_kv, torch.bfloat16, seq_len=seq_len)
        check_training(inputs, 19, operation=compiled)


@pytest.mark.parametrize("head_dim", [16, 64, 128, 144, 256, 496, 512, 528])
def test_shared_schedule_head_dimensions(monkeypatch, head_dim):
    pytest.importorskip("triton")
    from attn_gym.sparse.selected_attention.impl.triton.primitives import (
        can_use_shared_kv_schedule,
    )

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    query = torch.empty(1, 16, 1, head_dim, dtype=torch.bfloat16, device="meta")
    kv = torch.empty(1, 1, 1, head_dim, dtype=query.dtype, device="meta").expand(-1, 16, -1, -1)
    assert can_use_shared_kv_schedule(query, kv, kv, 128) == (head_dim in (16, 64, 128, 512))


@pytest.mark.full_autotune
@pytest.mark.parametrize("kernel_name", ["dq_shared", "dsparse_shared_atomic", "dsparse_generic"])
def test_d512_backward_config_pruning(kernel_name):
    pytest.importorskip("triton")
    from attn_gym.sparse.selected_attention.impl.triton import backward, shared_backward

    kernels = {
        "dq_shared": shared_backward._selected_attention_bwd_dq_shared,
        "dsparse_shared_atomic": shared_backward._selected_attention_bwd_dsparse_kv_shared_atomic,
        "dsparse_generic": backward._selected_attention_bwd_dsparse_kv,
    }
    kernel = kernels[kernel_name]
    wide_configs = kernel.early_config_prune(kernel.configs, {}, D=512)
    assert len(wide_configs) == 1
    config = wide_configs[0]
    assert config.num_warps == 4
    assert config.num_stages == 1
    assert all(size == 16 for size in config.kwargs.values())
    narrow_configs = kernel.early_config_prune(kernel.configs, {}, D=128)
    assert len(narrow_configs) > 1
    if kernel_name == "dq_shared":
        assert {config.kwargs["BLOCK_N"] for config in narrow_configs} == {64, 128}
    else:
        assert narrow_configs == kernel.configs
