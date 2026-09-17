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
    "Tensor? q_scale=None, Tensor? k_scale=None) -> Tensor",
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
) -> Tensor:
    """Describe the common contiguous, nondifferentiable index output."""
    return q.new_empty((q.shape[0], q.shape[1], topk), dtype=torch.int32)


_indexer_op = torch.ops.attn_gym._indexer.default
