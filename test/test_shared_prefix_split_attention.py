import torch
import pytest
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

from attn_gym.masks.causal import causal_mask
from attn_gym.utils import merge_attention


def _causal_block_mask(seq_len: int, device: torch.device):
    return create_block_mask(causal_mask, 1, 1, seq_len, seq_len, device=device, BLOCK_SIZE=128)


def _flex_with_lse(q, k, v, block_mask=None):
    return flex_attention(q, k, v, block_mask=block_mask, return_aux=AuxRequest(lse=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_split_prefix_response_attention_matches_duplicated_causal_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(1, 2, 640, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, 640, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 2, 640, 64, device=device, dtype=torch.float16)

    prefix_slice = slice(0, 128)
    response_slice = slice(256, 384)
    prefix_q = q[:, :, prefix_slice, :]
    prefix_k = k[:, :, prefix_slice, :]
    prefix_v = v[:, :, prefix_slice, :]
    response_q = q[:, :, response_slice, :]
    response_k = k[:, :, response_slice, :]
    response_v = v[:, :, response_slice, :]

    prefix_out, _ = _flex_with_lse(
        prefix_q, prefix_k, prefix_v, _causal_block_mask(prefix_q.shape[-2], device)
    )
    response_prefix_out, response_prefix_aux = _flex_with_lse(response_q, prefix_k, prefix_v)
    response_self_out, response_self_aux = _flex_with_lse(
        response_q, response_k, response_v, _causal_block_mask(response_q.shape[-2], device)
    )
    response_out, _ = merge_attention(
        response_prefix_out,
        response_prefix_aux.lse,
        response_self_out,
        response_self_aux.lse,
    )

    split_out = torch.cat([prefix_out, response_out], dim=-2)
    duplicated_indices = list(range(prefix_slice.start, prefix_slice.stop)) + list(
        range(response_slice.start, response_slice.stop)
    )
    duplicated_out, _ = _flex_with_lse(
        q[:, :, duplicated_indices, :],
        k[:, :, duplicated_indices, :],
        v[:, :, duplicated_indices, :],
        _causal_block_mask(len(duplicated_indices), device),
    )

    torch.testing.assert_close(split_out, duplicated_out.float(), atol=3e-3, rtol=3e-3)
