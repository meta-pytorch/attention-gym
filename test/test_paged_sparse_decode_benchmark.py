"""Selection packing for the native vLLM benchmark comparison."""

import torch

from benchmarks.sparse.paged_sparse_decode_benchmark import vllm_operands


def test_vllm_operands_compacts_partial_blocks_after_valid_selections():
    inputs = {
        "index_block_size": 4,
        "indices": torch.tensor([[-1, -1, -1], [1, -1, -1], [2, 0, 1]], dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, 1, 3], dtype=torch.int32),
        "seqused_k": torch.tensor([3, 15], dtype=torch.int32),
        "topk_length": torch.tensor([0, 1, 3], dtype=torch.int32),
    }

    packed, token_to_req = vllm_operands(inputs)

    # The first request has only an open block. The second has two query rows
    # with different sparse budgets and causal tails.
    expected = [
        [0, 1, 2],
        [4, 5, 6, 7, 12, 13],
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14],
    ]
    assert token_to_req.tolist() == [0, 1, 1]
    assert packed.dtype == torch.int32
    assert packed.shape == (3, 16)
    for row, tokens in enumerate(expected):
        assert packed[row, -1].item() == len(tokens)
        assert packed[row, : len(tokens)].tolist() == tokens
        assert (packed[row, len(tokens) : -1] == -1).all()
