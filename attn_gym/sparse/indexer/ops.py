"""Private operator boundary shared by the fused indexer backends.

Dispatch happens on real CUDA tensors, outside Dynamo tracing. This keeps CuTeDSL
compilation and Triton's host TMA descriptors behind one fake-tensor contract.
"""

import torch
from torch import Tensor

from attn_gym._backends.arch import runs_sm100_kernels

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
        backend = "cute" if runs_sm100_kernels(capability) else "triton"
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


torch.library.define(
    "attn_gym::_paged_indexer_key_update",
    "(Tensor raw_keys, Tensor(a!) key_cache, Tensor(b!) raw_cache, Tensor block_table, "
    "Tensor raw_block_table, Tensor cu_seqlens_q, Tensor seqused_k, Tensor norm_weight, "
    "Tensor rope_cache, int compress_ratio, float eps, int? max_seqlen_q) -> ()",
)


def _paged_indexer_key_update_cuda(
    raw_keys,
    key_cache,
    raw_cache,
    block_table,
    raw_block_table,
    cu_seqlens_q,
    seqused_k,
    norm_weight,
    rope_cache,
    compress_ratio,
    eps,
    max_seqlen_q,
):
    from .impl.key_update_triton import launch

    launch(
        raw_keys,
        key_cache,
        raw_cache,
        block_table,
        raw_block_table,
        cu_seqlens_q,
        seqused_k,
        norm_weight,
        rope_cache,
        compress_ratio,
        eps,
        max_seqlen_q,
    )


torch.library.impl("attn_gym::_paged_indexer_key_update", "CUDA", _paged_indexer_key_update_cuda)


@torch.library.register_fake("attn_gym::_paged_indexer_key_update")
def _paged_indexer_key_update_fake(
    raw_keys,
    key_cache,
    raw_cache,
    block_table,
    raw_block_table,
    cu_seqlens_q,
    seqused_k,
    norm_weight,
    rope_cache,
    compress_ratio,
    eps,
    max_seqlen_q,
):
    return None


_paged_indexer_key_update_op = torch.ops.attn_gym._paged_indexer_key_update.default
