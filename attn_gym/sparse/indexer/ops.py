"""Private operator boundary shared by the fused indexer backends.

Dispatch happens on real CUDA tensors, outside Dynamo tracing. This keeps CuTeDSL
compilation and Triton's host TMA descriptors behind one fake-tensor contract.
"""

import torch
from torch import Tensor

torch.library.define(
    "attn_gym::_indexer",
    "(Tensor q, Tensor k, Tensor weights, int topk, bool causal, int compress_ratio, "
    "str backend) -> Tensor",
)


def _indexer_cuda(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    backend: str,
) -> Tensor:
    """Select a launcher on the input device without tracing device queries."""
    if backend == "auto":
        backend = "cute" if torch.cuda.get_device_capability(q.device) == (10, 0) else "triton"
    match backend:
        case "cute":
            from .impl.cute import launch
        case "triton":
            from .impl.triton import launch
        case _:
            raise ValueError(f"unknown indexer backend {backend!r}")
    return launch(q, k, weights, topk, causal, compress_ratio)


torch.library.impl("attn_gym::_indexer", "CUDA", _indexer_cuda)


@torch.library.register_fake("attn_gym::_indexer")
def _indexer_fake(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    backend: str,
) -> Tensor:
    """Describe the common contiguous, nondifferentiable index output."""
    return q.new_empty((q.shape[0], q.shape[1], topk), dtype=torch.int32)


_indexer_op = torch.ops.attn_gym._indexer.default
