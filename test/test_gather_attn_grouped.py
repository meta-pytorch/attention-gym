"""Triton grouped-schedule forward: separate values and grouped KV heads on Blackwell."""

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import (
    assert_matches_low_precision_eager,
    make_gather_attn_test_inputs,
    make_qsa_block_indices,
)

BLACKWELL_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

pytestmark = [
    pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required"),
    pytest.mark.usefixtures("gather_attn_single_config"),
]


def _assert_forward_matches_reference(inputs, **options):
    """Bound Triton output and LSE error by bfloat16 reference error against FP64."""
    high_precision_inputs = {
        name: tensor.double() if tensor is not None and tensor.is_floating_point() else tensor
        for name, tensor in inputs.items()
    }
    high_precision, high_precision_aux = gather_attn(
        **high_precision_inputs, **options, impl=Impl.REFERENCE, return_aux=AuxRequest(lse=True)
    )
    expected, expected_aux = gather_attn(
        **inputs, **options, impl=Impl.REFERENCE, return_aux=AuxRequest(lse=True)
    )
    actual, actual_aux = gather_attn(
        **inputs, **options, kernel_options={"backend": "triton"}, return_aux=AuxRequest(lse=True)
    )
    reduction_sizes = (
        inputs["query"].shape[-1],
        inputs["kv_indices"].shape[-1] + options["sliding_window_size"],
    )
    assert_matches_low_precision_eager(actual, expected, high_precision, reduction_sizes)
    assert_matches_low_precision_eager(
        actual_aux.lse, expected_aux.lse, high_precision_aux.lse, reduction_sizes
    )


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,separate_values,packed,window,topk",
    [
        (32, 4, 128, False, False, 19, 37),
        (24, 2, 256, True, True, 19, 37),
        (16, 1, 64, True, True, 19, 37),
        (4, 4, 64, True, False, 7, 5),
        (24, 2, 256, True, False, 0, 2051),
    ],
    ids=[
        "gqa-shared-values",
        "gqa-group12-d256-packed",
        "mqa-separate-values-packed",
        "mha-separate-values",
        "long-topk-sparse-only",
    ],
)
def test_grouped_forward_matches_reference(
    heads, kv_heads, head_dim, separate_values, packed, window, topk
):
    """Each KV head's rows are gathered once for its query-head group, including masked tiles."""
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        tokens=33,
        sparse_tokens=79,
        topk=topk,
        separate_values=separate_values,
        dtype=torch.bfloat16,
        batch=1 if packed else 2,
    )
    options = {"sliding_window_size": window}
    if packed:
        options["cu_seqlens"] = cumulative_sequence_offsets([11, 9, 13])
        options["cu_seqlens_k"] = cumulative_sequence_offsets([23, 21, 35])
    _assert_forward_matches_reference(inputs, **options)


def test_qsa_block_selection_matches_reference():
    """QSA shape: 24 query heads over 2 KV heads, D=256, 4-token blocks plus tail, window 0."""
    heads, kv_heads, head_dim = 24, 2, 256
    document_lengths = [70, 58]
    tokens = sum(document_lengths)
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to("cuda", torch.bfloat16)

    keys = randn(1, kv_heads, tokens, head_dim)
    values = randn(1, kv_heads, tokens, head_dim)
    offsets = cumulative_sequence_offsets(document_lengths)
    inputs = {
        "query": randn(1, heads, tokens, head_dim),
        "local_kv": keys,
        "sparse_kv": keys,
        "kv_indices": make_qsa_block_indices(
            document_lengths, block_size=4, budget_blocks=4, generator=generator
        ),
        "attention_sink": None,
        "local_v": values,
        "sparse_v": values,
    }
    _assert_forward_matches_reference(
        inputs, sliding_window_size=0, cu_seqlens=offsets, cu_seqlens_k=offsets
    )


def test_existing_inputs_keep_their_schedule():
    """Shared-value MQA and MHA inputs never move onto the grouped schedule."""
    from attn_gym.sparse.gather_attn.impl.triton.forward import forward_schedule

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, device="cuda", dtype=dtype)

    shared_kv = empty(1, 1, 8, 512).expand(-1, 64, -1, -1)
    query = empty(1, 64, 8, 512)
    assert (
        forward_schedule(
            query, shared_kv, shared_kv, None, None, kv_heads=1, sliding_window_size=128
        )
        == "shared"
    )
    for dtype in (torch.bfloat16, torch.float32):
        mha_kv = empty(1, 4, 8, 64, dtype=dtype)
        schedule = forward_schedule(
            empty(1, 4, 8, 64, dtype=dtype),
            mha_kv,
            mha_kv,
            None,
            None,
            kv_heads=4,
            sliding_window_size=7,
        )
        assert schedule in ("tma", "generic")


@pytest.mark.parametrize(
    "dtype,head_dim,window",
    [(torch.float32, 256, 0), (torch.bfloat16, 512, 0), (torch.bfloat16, 256, 4096)],
    ids=["fp32", "d512", "wide-window"],
)
def test_ineligible_grouped_inputs_raise(dtype, head_dim, window):
    """Inputs the grouped schedule cannot run raise instead of ignoring values or groups."""
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=head_dim,
        tokens=8,
        sparse_tokens=8,
        topk=2,
        separate_values=True,
        dtype=dtype,
    )
    with pytest.raises(ValueError, match="supports separate values and grouped KV heads only"):
        gather_attn(**inputs, sliding_window_size=window, kernel_options={"backend": "triton"})


def test_grouped_backward_raises():
    """Requesting gradients fails before a backward that would treat keys as values runs."""
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=64,
        tokens=8,
        sparse_tokens=8,
        topk=2,
        separate_values=True,
        dtype=torch.bfloat16,
    )
    inputs["sparse_v"].requires_grad_()
    with pytest.raises(ValueError, match="backward does not support separate values"):
        gather_attn(**inputs, sliding_window_size=3, kernel_options={"backend": "triton"})


@pytest.mark.usefixtures("fresh_compile_cache")
def test_grouped_forward_fullgraph():
    """The op's optional value tensors trace through fullgraph compile."""
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=64,
        tokens=16,
        sparse_tokens=16,
        topk=4,
        separate_values=True,
        dtype=torch.bfloat16,
    )
    expected = gather_attn(**inputs, sliding_window_size=3, kernel_options={"backend": "triton"})
    compiled = torch.compile(gather_attn, fullgraph=True)
    actual = compiled(**inputs, sliding_window_size=3, kernel_options={"backend": "triton"})
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
