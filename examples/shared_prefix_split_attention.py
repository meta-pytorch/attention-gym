"""Demonstrates split attention for shared prefixes."""

import torch
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

from attn_gym.masks.causal import causal_mask
from attn_gym.utils import merge_attention


def split_prefix_response_attention(q, k, v, prefix_slice: slice, response_slice: slice):
    prefix_q = q[:, :, prefix_slice, :]
    prefix_k = k[:, :, prefix_slice, :]
    prefix_v = v[:, :, prefix_slice, :]
    response_q = q[:, :, response_slice, :]
    response_k = k[:, :, response_slice, :]
    response_v = v[:, :, response_slice, :]

    prefix_out, _ = flex_attention(
        prefix_q,
        prefix_k,
        prefix_v,
        block_mask=create_block_mask(
            causal_mask,
            1,
            1,
            prefix_q.shape[-2],
            prefix_q.shape[-2],
            device=q.device,
            BLOCK_SIZE=128,
        ),
        return_aux=AuxRequest(lse=True),
    )
    response_prefix_out, response_prefix_aux = flex_attention(
        response_q, prefix_k, prefix_v, return_aux=AuxRequest(lse=True)
    )
    response_self_out, response_self_aux = flex_attention(
        response_q,
        response_k,
        response_v,
        block_mask=create_block_mask(
            causal_mask,
            1,
            1,
            response_q.shape[-2],
            response_q.shape[-2],
            device=q.device,
            BLOCK_SIZE=128,
        ),
        return_aux=AuxRequest(lse=True),
    )
    response_out, _ = merge_attention(
        response_prefix_out,
        response_prefix_aux.lse,
        response_self_out,
        response_self_aux.lse,
    )
    return torch.cat([prefix_out, response_out], dim=-2)


def duplicated_causal_attention(q, k, v, token_indices: list[int]):
    logical_q = q[:, :, token_indices, :]
    logical_k = k[:, :, token_indices, :]
    logical_v = v[:, :, token_indices, :]
    out, _ = flex_attention(
        logical_q,
        logical_k,
        logical_v,
        block_mask=create_block_mask(
            causal_mask,
            1,
            1,
            len(token_indices),
            len(token_indices),
            device=q.device,
            BLOCK_SIZE=128,
        ),
        return_aux=AuxRequest(lse=True),
    )
    return out


def main(device: str = "cpu"):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 9, 16, device=device)
    k = torch.randn(1, 2, 9, 16, device=device)
    v = torch.randn(1, 2, 9, 16, device=device)

    prefix_slice = slice(0, 3)
    response_slices = [slice(3, 5), slice(5, 7), slice(7, 9)]

    for response_slice in response_slices:
        split_out = split_prefix_response_attention(q, k, v, prefix_slice, response_slice)
        duplicated_indices = list(range(0, 3)) + list(
            range(response_slice.start, response_slice.stop)
        )
        duplicated_out = duplicated_causal_attention(q, k, v, duplicated_indices)
        torch.testing.assert_close(split_out, duplicated_out, atol=1e-5, rtol=1e-5)

    print("split attention matches duplicated causal attention")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
