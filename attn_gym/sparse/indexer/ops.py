"""Torch-only private operator contract for the CuTeDSL indexer.

Registration exists before graph capture; optional backend imports and runtime
layout checks happen only when the CUDA implementation executes.
"""

import torch
from torch import Tensor

torch.library.define(
    "attn_gym::_indexer_cute",
    "(Tensor q, Tensor k, Tensor weights, int topk, bool causal) -> Tensor",
)


def _indexer_cute_cuda(q: Tensor, k: Tensor, weights: Tensor, topk: int, causal: bool) -> Tensor:
    from .impl.cute.prefill import index as launch

    return launch(q, k, weights, topk, causal)


torch.library.impl("attn_gym::_indexer_cute", "CUDA", _indexer_cute_cuda)


@torch.library.register_fake("attn_gym::_indexer_cute")
def _indexer_cute_fake(q: Tensor, k: Tensor, weights: Tensor, topk: int, causal: bool) -> Tensor:
    return q.new_empty((q.shape[0], q.shape[1], topk), dtype=torch.int32)


_indexer_cute_op = torch.ops.attn_gym._indexer_cute.default
