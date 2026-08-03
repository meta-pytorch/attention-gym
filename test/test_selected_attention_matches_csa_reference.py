import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.compressed_sparse_attention.reference import CSA as reference_CSA
from attn_gym.sparse.selected_attention import selected_attention
from examples.compressed_sparse_attention import apply_rope, compress, make_block_mask

ATOL = 1e-8
RTOL = 1e-5


def _selected_attention_with_causal_blocks(
    Q,
    KV,
    compressed_kv,
    topk_blocks,
    indexer_mask,
    attention_sink,
    sliding_window_size,
):
    """Call selected_attention while preserving the completed-block constraint."""
    if topk_blocks.shape[-1] == 0:
        return selected_attention(
            Q,
            KV,
            compressed_kv,
            topk_blocks,
            attention_sink,
            None,
            sliding_window_size,
            False,
        )

    valid_blocks = torch.isfinite(indexer_mask).unsqueeze(0).expand(Q.shape[0], -1, -1)
    selected_is_valid = valid_blocks.gather(dim=-1, index=topk_blocks)

    # scatter() treats duplicate indices as one selection. Replacing an invalid
    # top-k entry with any valid entry therefore reproduces the original additive mask.
    first_valid_slot = selected_is_valid.to(torch.int64).argmax(dim=-1, keepdim=True)
    first_valid_block = topk_blocks.gather(dim=-1, index=first_valid_slot)
    causal_topk_blocks = torch.where(selected_is_valid, topk_blocks, first_valid_block)

    selected_output = selected_attention(
        Q,
        KV,
        compressed_kv,
        causal_topk_blocks,
        attention_sink,
        None,
        sliding_window_size,
        False,
    )

    # Before the first compressed block is complete there is no valid replacement.
    # An empty index tensor keeps the compressed branch present but fully masked.
    local_output = selected_attention(
        Q,
        KV,
        compressed_kv,
        topk_blocks[..., :0],
        attention_sink,
        None,
        sliding_window_size,
        False,
    )
    has_selected_block = selected_is_valid.any(dim=-1)[:, None, :, None]
    return torch.where(has_selected_block, selected_output, local_output)


def _csa_with_selected_attention(
    Q,
    Q_I,
    KV,
    C_a,
    C_b,
    Z_a,
    Z_b,
    B_a,
    B_b,
    W_I,
    K_Ia,
    K_Ib,
    Z_Ia,
    Z_Ib,
    B_Ia,
    B_Ib,
    KV_norm_weight,
    compressed_indices_norm_weight,
    compressed_kv_norm_weight,
    attention_sink,
    compression_rate,
    num_topk_blocks,
    sliding_window_size,
    rope_dims: int,
    share_kv: bool,
):
    device = Q.device
    dtype = Q.dtype
    _, num_heads, sequence_length, _ = Q.shape
    _, num_index_heads, _, index_head_dim = Q_I.shape
    if share_kv:
        KV = KV.expand(-1, num_heads, -1, -1)
        C_a = C_a.expand(-1, num_heads, -1, -1)
        C_b = C_b.expand(-1, num_heads, -1, -1)
        Z_a = Z_a.expand(-1, num_heads, -1, -1)
        Z_b = Z_b.expand(-1, num_heads, -1, -1)

        K_Ia = K_Ia.expand(-1, num_index_heads, -1, -1)
        K_Ib = K_Ib.expand(-1, num_index_heads, -1, -1)
        Z_Ia = Z_Ia.expand(-1, num_index_heads, -1, -1)
        Z_Ib = Z_Ib.expand(-1, num_index_heads, -1, -1)

    compressed_kv = compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_total_blocks = compressed_kv.shape[-2]

    Q = torch.cat([Q[..., :-rope_dims], apply_rope(Q[..., -rope_dims:])], dim=-1)
    Q_I = torch.cat([Q_I[..., :-rope_dims], apply_rope(Q_I[..., -rope_dims:])], dim=-1)
    KV = torch.cat([KV[..., :-rope_dims], apply_rope(KV[..., -rope_dims:])], dim=-1)

    compressed_positions = torch.arange(num_total_blocks, device=device) * compression_rate
    compressed_indices = torch.cat(
        [
            compressed_indices[..., :-rope_dims],
            apply_rope(
                compressed_indices[..., -rope_dims:],
                positions=compressed_positions,
            ),
        ],
        dim=-1,
    )
    compressed_kv = torch.cat(
        [
            compressed_kv[..., :-rope_dims],
            apply_rope(
                compressed_kv[..., -rope_dims:],
                positions=compressed_positions,
            ),
        ],
        dim=-1,
    )

    indexer_mask = make_block_mask(
        sequence_length,
        num_total_blocks,
        compression_rate,
        device,
        dtype,
    )
    indexer_scale = math.sqrt(index_head_dim * num_index_heads)
    scores = F.relu(Q_I @ compressed_indices.transpose(-2, -1)) / indexer_scale
    index_head_weights = W_I.transpose(1, 2).unsqueeze(-1)
    scores = torch.sum(index_head_weights * scores, dim=1) + indexer_mask

    topk_blocks = torch.topk(
        scores,
        k=min(num_topk_blocks, num_total_blocks),
        dim=-1,
    ).indices
    attention_output = _selected_attention_with_causal_blocks(
        Q,
        KV,
        compressed_kv,
        topk_blocks,
        indexer_mask,
        attention_sink,
        sliding_window_size,
    )

    return torch.cat(
        [
            attention_output[..., :-rope_dims],
            apply_rope(attention_output[..., -rope_dims:], inverse=True),
        ],
        dim=-1,
    )


def _make_inputs(
    share_kv: bool,
    num_topk_blocks: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    generator = torch.Generator(device=device).manual_seed(0)
    batch_size = 1
    num_heads = 2
    num_index_heads = 1
    sequence_length = 5
    head_dim = 4
    index_head_dim = 4
    compression_rate = 2
    sliding_window_size = 2
    rope_dims = 4
    num_blocks = (sequence_length + compression_rate - 1) // compression_rate

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

    kv_heads = 1 if share_kv else num_heads
    index_kv_heads = 1 if share_kv else num_index_heads
    return (
        randn(batch_size, num_heads, sequence_length, head_dim),
        randn(batch_size, num_index_heads, sequence_length, index_head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, head_dim),
        randn(batch_size, kv_heads, sequence_length, 1),
        randn(batch_size, kv_heads, sequence_length, 1),
        randn(1, num_heads, num_blocks, 1, 1),
        randn(1, num_heads, num_blocks, 1, 1),
        randn(batch_size, sequence_length, num_index_heads),
        randn(batch_size, index_kv_heads, sequence_length, index_head_dim),
        randn(batch_size, index_kv_heads, sequence_length, index_head_dim),
        randn(batch_size, index_kv_heads, sequence_length, 1),
        randn(batch_size, index_kv_heads, sequence_length, 1),
        randn(1, num_index_heads, num_blocks, 1, 1),
        randn(1, num_index_heads, num_blocks, 1, 1),
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.ones(index_head_dim, dtype=dtype, device=device),
        torch.ones(head_dim, dtype=dtype, device=device),
        torch.zeros(num_heads, dtype=dtype, device=device),
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("num_topk_blocks", [0, 1])
def test_selected_attention_matches_csa_reference_fp64(share_kv, num_topk_blocks):
    inputs = _make_inputs(
        share_kv,
        num_topk_blocks,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    floating_inputs = inputs[:20]
    assert all(tensor.dtype == torch.float64 for tensor in floating_inputs)

    with torch.inference_mode():
        expected = reference_CSA(*inputs)
        actual = _csa_with_selected_attention(*inputs)

    assert actual.dtype == expected.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_attention_matches_csa_reference_cuda_fp32():
    inputs = _make_inputs(
        True,
        1,
        dtype=torch.float32,
        device=torch.device("cuda"),
    )

    with torch.inference_mode():
        expected = reference_CSA(*inputs)
        actual = _csa_with_selected_attention(*inputs)

    assert actual.device.type == expected.device.type == "cuda"
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
