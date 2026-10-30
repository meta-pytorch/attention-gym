import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import (
    gather_attn_oracle,
    make_gather_attn_test_inputs,
    make_qsa_block_indices,
)


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk_blocks", [0, 1, 3])
def test_batch_invariance(share_kv, num_topk_blocks):
    """Outputs should be identical regardless of batch position."""
    b, h, s, head_dim = 3, 5, 14, 17
    kv_heads = 1 if share_kv else h
    sparse_seq_len = s // 2

    generator = torch.Generator().manual_seed(42)

    query = torch.randn(b, h, s, head_dim, generator=generator)
    local_kv = torch.randn(b, kv_heads, s, head_dim, generator=generator)
    sparse_kv = torch.randn(b, kv_heads, sparse_seq_len, head_dim, generator=generator)

    if num_topk_blocks > 0:
        _, kv_indices = torch.topk(
            torch.randn(b, s, sparse_seq_len, generator=generator),
            k=min(num_topk_blocks, sparse_seq_len),
            dim=-1,
        )
    else:
        kv_indices = torch.zeros(b, s, 0, dtype=torch.long)

    attention_sink = torch.randn(h, generator=generator)
    sliding_window_size = 3

    # Run full batch
    full_out = gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        sliding_window_size=sliding_window_size,
        impl=Impl.REFERENCE,
    )

    # Run each batch element independently and compare
    for i in range(b):
        single_out = gather_attn(
            query[i : i + 1],
            local_kv[i : i + 1],
            sparse_kv[i : i + 1],
            kv_indices[i : i + 1],
            attention_sink,
            sliding_window_size=sliding_window_size,
            impl=Impl.REFERENCE,
        )
        torch.testing.assert_close(
            full_out[i : i + 1],
            single_out,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Batch element {i} differs when run independently vs. in a batch",
        )


def _assert_matches_oracle(inputs, *, sliding_window_size, cu_seqlens=None, cu_seqlens_k=None):
    """Compare reference outputs, LSE, and every input gradient against the FP64 oracle."""
    leaves = {
        name: tensor
        for name, tensor in inputs.items()
        if tensor is not None and tensor.is_floating_point()
    }
    for tensor in leaves.values():
        tensor.requires_grad_()
    actual, aux = gather_attn(
        **inputs,
        sliding_window_size=sliding_window_size,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        impl=Impl.REFERENCE,
        return_aux=AuxRequest(lse=True),
    )
    expected, expected_lse = gather_attn_oracle(
        **inputs,
        sliding_window_size=sliding_window_size,
        scale=inputs["query"].shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(aux.lse, expected_lse)
    grad_output = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, tuple(leaves.values()), grad_output)
    expected_grads = torch.autograd.grad(expected, tuple(leaves.values()), grad_output)
    for name, actual_grad, expected_grad in zip(leaves, actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, msg=f"gradient of {name}")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("separate_values", [False, True])
@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (6, 2), (6, 3), (6, 6)])
def test_grouped_heads_and_separate_values_match_oracle(heads, kv_heads, separate_values, packed):
    """MQA, GQA, and MHA layouts read KV head h // group, with values defaulting to keys."""
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=8,
        tokens=12,
        sparse_tokens=6,
        topk=3,
        separate_values=separate_values,
        dtype=torch.float64,
        batch=1 if packed else 2,
        device="cpu",
    )
    offsets = {}
    if packed:
        offsets = {
            "cu_seqlens": cumulative_sequence_offsets([5, 7], device="cpu"),
            "cu_seqlens_k": cumulative_sequence_offsets([2, 4], device="cpu"),
        }
    _assert_matches_oracle(inputs, sliding_window_size=3, **offsets)


def test_qsa_block_selection_matches_oracle():
    """QSA usage: raw-token pools, separate K/V, GQA, and expanded 4-token blocks plus tail."""
    heads, kv_heads, head_dim, block_size, budget_blocks = 24, 2, 256, 4, 2
    document_lengths = [21, 16]
    tokens = sum(document_lengths)
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(1, kv_heads, tokens, head_dim, dtype=torch.float64, generator=generator)
    values = torch.randn(1, kv_heads, tokens, head_dim, dtype=torch.float64, generator=generator)
    kv_indices = make_qsa_block_indices(
        document_lengths,
        block_size=block_size,
        budget_blocks=budget_blocks,
        generator=generator,
        device="cpu",
    )
    offsets = cumulative_sequence_offsets(document_lengths, device="cpu")
    inputs = {
        "query": torch.randn(1, heads, tokens, head_dim, dtype=torch.float64, generator=generator),
        "local_kv": keys,
        "sparse_kv": keys,
        "kv_indices": kv_indices,
        "attention_sink": None,
        "local_v": values,
        "sparse_v": values,
    }
    _assert_matches_oracle(inputs, sliding_window_size=0, cu_seqlens=offsets, cu_seqlens_k=offsets)
