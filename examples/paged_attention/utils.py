import torch
from torch.nn.attention.flex_attention import (
    _identity,
    BlockMask,
)
from paged_attention import PagedAttention


def batch_reserve(paged_attention: PagedAttention, target_seq_len: torch.Tensor):
    """Reserves pages for each sequence in the batch.

    Args:
        paged_attention: PagedAttention instance.
        target_seq_len: Tensor of shape (B,) containing the length of each sequence in the batch.
    """
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


def random_init_paged_attention(n_pages: int, page_size: int, bsz: int, max_seq_len: int):
    """Allocate physical pages across batches in a round-robin fashion to simulate a use case
    where multiple batches run in parallel. This is for testing and benchmarking only.

    Args:
        n_pages: Number of pages.
        page_size: Size of each page.
        bsz: Batch size.
        max_seq_len: Maximum sequence length.
    """
    paged_attention = PagedAttention(n_pages, page_size, bsz)

    repeat = bsz // 4
    sequence_lengths = [
        [max_seq_len // 4, max_seq_len // 2, max_seq_len // 4, max_seq_len // 3] * repeat,
        [max_seq_len // 4, max_seq_len // 2, max_seq_len // 2, max_seq_len // 2] * repeat,
        [max_seq_len // 4, max_seq_len // 2, max_seq_len // 2, max_seq_len // 2] * repeat,
        [max_seq_len // 2, max_seq_len, max_seq_len // 2, max_seq_len] * repeat,
        [max_seq_len, max_seq_len, max_seq_len, max_seq_len] * repeat,
    ]

    for seq_len in sequence_lengths:
        batch_reserve(
            paged_attention,
            torch.tensor(seq_len, device="cuda"),
        )

    return paged_attention


def gen_offset(off: torch.Tensor):
    """Generates an offset function.

    Args:
        off: Offset tensor.
    """

    def offset(b, h, m, n):
        return m + off[b] >= n

    return offset


def generate_score_mod(attn_type: str):
    """Generates a score modification function.

    Args:
        attn_type: Attention type.
    """

    def relative_bias(score, b, h, m, n):
        return score + (m - n)

    def head_bias(score, b, h, m, n):
        return score + 2 * h

    function_dict = {
        "noop": _identity,
        "causal": _identity,
        "rel": relative_bias,
        "head_bias": head_bias,
    }
    return function_dict[attn_type]


def _adjust_num_blocks_and_indices(
    num_blocks: torch.Tensor,
    indices: torch.Tensor,
    batch_idx: int,
    new_num_rows: int,
    new_num_cols: int,
):
    """Adjust the number of blocks and indices based on the new number of rows and columns.

    Args:
        num_blocks: KV Num Blocks.
        indices: KV indices.
        batch_idx: Batch index.
        new_num_rows: New number of rows.
        new_num_cols: New number of columns.
    """
    indices = indices[[batch_idx], :, :new_num_rows, :new_num_cols]
    num_blocks = num_blocks[[batch_idx], :, :new_num_rows]
    num_blocks = torch.where(num_blocks < new_num_cols, num_blocks, new_num_cols)
    num_blocks = torch.sum(indices < num_blocks[:, :, :, None], dim=-1).to(torch.int32)
    return num_blocks.clone(), indices.clone()


def slice_block_mask(
    block_mask: BlockMask, batch_idx: int, new_q_len: int, new_kv_len: int
) -> BlockMask:
    """Slice the block mask based on the new query and key/value lengths.

    Args:
        block_mask: Block mask.
        batch_idx: Batch index.
        new_q_len: New query length.
        new_kv_len: New key/value length.
    """
    new_num_rows = (new_q_len + block_mask.BLOCK_SIZE[0] - 1) // block_mask.BLOCK_SIZE[0]
    new_num_cols = (new_kv_len + block_mask.BLOCK_SIZE[1] - 1) // block_mask.BLOCK_SIZE[1]
    new_kv_num_blocks, new_kv_indices = _adjust_num_blocks_and_indices(
        block_mask.kv_num_blocks, block_mask.kv_indices, batch_idx, new_num_rows, new_num_cols
    )
    if block_mask.full_kv_num_blocks is not None:
        assert block_mask.full_kv_indices is not None
        (
            new_full_kv_num_blocks,
            new_full_kv_indices,
        ) = _adjust_num_blocks_and_indices(
            block_mask.full_kv_num_blocks,
            block_mask.full_kv_indices,
            batch_idx,
            new_num_rows,
            new_num_cols,
        )
    else:
        new_full_kv_num_blocks = None
        new_full_kv_indices = None
    new_block_mask = block_mask.from_kv_blocks(
        new_kv_num_blocks,
        new_kv_indices,
        new_full_kv_num_blocks,
        new_full_kv_indices,
        block_mask.BLOCK_SIZE,
        block_mask.mask_mod,
    )
    new_block_mask.seq_lengths = (new_q_len, new_kv_len)
    return new_block_mask
