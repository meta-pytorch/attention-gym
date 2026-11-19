"""Private operator boundary shared by the fused indexer backends.

Dispatch happens on real CUDA tensors, outside Dynamo tracing. This keeps CuTeDSL
compilation and Triton's host TMA descriptors behind one fake-tensor contract.
"""

import torch
from torch import Tensor

from attn_gym._backends.arch import runs_sm100_kernels

from .validation import validate_precision

torch.library.define(
    "attn_gym::_indexer",
    "(Tensor q, Tensor k, Tensor weights, int topk, bool causal, int compress_ratio, "
    'str backend, Tensor? cu_seqlens=None, Tensor? cu_seqlens_k=None, str selector="default", '
    "Tensor? q_scale=None, Tensor? k_scale=None, *, bool batch_invariant=False) -> Tensor",
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
    selector: str = "default",
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
    *,
    batch_invariant: bool = False,
) -> Tensor:
    """Select a launcher on the input device without tracing device queries.

    Deterministic mode is read when this implementation executes, so compiled
    calls see the current setting; CUDA Graph replay keeps the kernel captured.
    """
    if q.dtype == torch.float8_e4m3fn and (cu_seqlens is not None or cu_seqlens_k is not None):
        raise ValueError("MXFP8 indexing does not support packed cu_seqlens.")
    if backend == "auto":
        capability = torch.cuda.get_device_capability(q.device)
        backend = "cute" if runs_sm100_kernels(capability) else "triton"
    match backend, selector:
        case "cute" | "triton", "auto" | "default" | "gvr2" if batch_invariant:
            from .impl.batch_invariant import launch

            validate_precision(q, k, weights, q_scale, k_scale, batch_invariant=True)
            if selector == "gvr2":
                raise ValueError(
                    "GVR2 is not deterministic; use selector='auto' with batch_invariant=True."
                )
            return launch(
                q,
                k,
                weights,
                topk,
                causal,
                compress_ratio,
                cu_seqlens,
                cu_seqlens_k,
                backend=backend,
            )
        case "cute", _:
            from .impl.cute import launch
        case "triton", _ if q.dtype == torch.float8_e4m3fn:
            raise ValueError("fused MXFP8 scoring requires the CuTe backend (SM100 or SM103)")
        case "triton", "gvr2":
            raise ValueError("the GVR2 selector requires the CuTe backend")
        case "triton", "auto" | "default":
            from .impl.triton import launch

            validate_precision(q, k, weights, q_scale, k_scale)
            return launch(q, k, weights, topk, causal, compress_ratio, cu_seqlens, cu_seqlens_k)
        case _:
            raise ValueError(f"unknown indexer backend {backend!r} or selector {selector!r}")
    candidate_bounds = None
    if cu_seqlens is not None:
        from .impl.triton import prepare_candidate_bounds

        candidate_bounds = prepare_candidate_bounds(
            cu_seqlens, cu_seqlens_k, q.shape[1], causal, compress_ratio
        )
    deterministic = torch.are_deterministic_algorithms_enabled()
    return launch(
        q,
        k,
        weights,
        topk,
        causal,
        compress_ratio,
        deterministic=deterministic,
        candidate_bounds=candidate_bounds,
        selector=selector,
        q_scale=q_scale,
        k_scale=k_scale,
    )


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
    selector: str = "default",
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
    *,
    batch_invariant: bool = False,
) -> Tensor:
    """Describe the common contiguous, nondifferentiable index output."""
    return q.new_empty((q.shape[0], q.shape[1], topk), dtype=torch.int32)


_indexer_op = torch.ops.attn_gym._indexer.default


torch.library.define(
    "attn_gym::_paged_indexer",
    "(Tensor query, Tensor key_cache, Tensor weights, Tensor block_table, "
    "Tensor cu_seqlens_q, Tensor seqused_k, int topk, int compress_ratio, "
    "Tensor? query_positions, str backend, Tensor? q_scale=None, Tensor? k_scale=None, "
    "*, bool batch_invariant=False) -> Tensor",
)


def _paged_indexer_cuda(
    query,
    key_cache,
    weights,
    block_table,
    cu_seqlens_q,
    seqused_k,
    topk,
    compress_ratio,
    query_positions,
    backend,
    q_scale=None,
    k_scale=None,
    *,
    batch_invariant=False,
):
    if backend == "auto":
        backend = (
            "cute"
            if runs_sm100_kernels(torch.cuda.get_device_capability(query.device))
            else "triton"
        )
    from .impl.paged_triton import launch

    return launch(
        query,
        key_cache,
        weights,
        block_table,
        cu_seqlens_q,
        seqused_k,
        topk,
        compress_ratio,
        query_positions,
        backend,
        q_scale,
        k_scale,
        batch_invariant=batch_invariant,
    )


torch.library.impl("attn_gym::_paged_indexer", "CUDA", _paged_indexer_cuda)


@torch.library.register_fake("attn_gym::_paged_indexer")
def _paged_indexer_fake(
    query,
    key_cache,
    weights,
    block_table,
    cu_seqlens_q,
    seqused_k,
    topk,
    compress_ratio,
    query_positions,
    backend,
    q_scale=None,
    k_scale=None,
    *,
    batch_invariant=False,
):
    return query.new_empty((query.shape[0], topk), dtype=torch.int32)


_paged_indexer_op = torch.ops.attn_gym._paged_indexer.default
