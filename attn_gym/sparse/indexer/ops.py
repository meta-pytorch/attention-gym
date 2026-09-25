"""Private operator boundary shared by the fused indexer backends.

Dispatch happens on real CUDA tensors, outside Dynamo tracing. This keeps CuTeDSL
compilation and Triton's host TMA descriptors behind one fake-tensor contract.
"""

import torch
from torch import Tensor

torch.library.define(
    "attn_gym::_indexer",
    "(Tensor q, Tensor k, Tensor weights, int topk, bool causal, int compress_ratio, "
    "str backend, Tensor? cu_seqlens=None, Tensor? cu_seqlens_k=None) -> Tensor",
)


def _indexer_cuda(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    backend: str,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> Tensor:
    """Select a launcher on the input device without tracing device queries.

    Deterministic mode is read when this implementation executes, so compiled
    calls see the current setting; CUDA Graph replay keeps the kernel captured.
    """
    if backend == "auto":
        capability = torch.cuda.get_device_capability(q.device)
        backend = "cute" if capability in ((10, 0), (10, 3)) else "triton"
    match backend:
        case "cute":
            from .impl.cute import launch
        case "triton":
            from .impl.triton import launch
        case _:
            raise ValueError(f"unknown indexer backend {backend!r}")
    if backend == "triton":
        return launch(q, k, weights, topk, causal, compress_ratio, cu_seqlens, cu_seqlens_k)
    candidate_bounds = None
    if cu_seqlens is not None:
        from .impl.triton import prepare_candidate_bounds

        candidate_bounds = prepare_candidate_bounds(
            cu_seqlens, cu_seqlens_k, q.shape[1], causal, compress_ratio
        )
    deterministic = torch.are_deterministic_algorithms_enabled()
    return launch(q, k, weights, topk, causal, compress_ratio, deterministic, candidate_bounds)


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
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> Tensor:
    """Describe the common contiguous, nondifferentiable index output."""
    return q.new_empty((q.shape[0], q.shape[1], topk), dtype=torch.int32)


_indexer_op = torch.ops.attn_gym._indexer.default
