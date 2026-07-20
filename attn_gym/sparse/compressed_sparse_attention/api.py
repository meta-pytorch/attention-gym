"""Public API and backend dispatch for compressed sparse attention."""

from collections.abc import Callable
import importlib
import math
from typing import Literal

import torch

Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]


def _validate_inputs(
    tensors: tuple[tuple[str, object], ...],
    compression_rate: object,
    num_topk_blocks: object,
    sliding_window_size: object,
    rope_dims: object,
    share_kv: object,
) -> None:
    """Validate the backend-independent compressed sparse attention contract."""
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    integer_arguments = {
        "compression_rate": compression_rate,
        "num_topk_blocks": num_topk_blocks,
        "sliding_window_size": sliding_window_size,
        "rope_dims": rope_dims,
    }
    for name, value in integer_arguments.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int, got {type(value).__name__}.")
    if type(share_kv) is not bool:
        raise TypeError(f"share_kv must be a Python bool, got {type(share_kv).__name__}.")

    by_name = dict(tensors)
    query = by_name["Q"]
    assert isinstance(query, torch.Tensor)
    if query.ndim != 4:
        raise ValueError("Q must have shape [batch, heads, sequence, head_dim].")
    batch, heads, sequence_length, head_dim = query.shape
    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("Q dimensions must all be positive.")
    if not query.is_floating_point():
        raise TypeError("Compressed sparse attention inputs must have a floating-point dtype.")

    index_query = by_name["Q_I"]
    assert isinstance(index_query, torch.Tensor)
    if (
        index_query.ndim != 4
        or index_query.shape[0] != batch
        or index_query.shape[2] != sequence_length
    ):
        raise ValueError("Q_I must have shape [batch, index_heads, sequence, index_dim].")
    index_heads, index_dim = index_query.shape[1], index_query.shape[3]
    if index_heads <= 0 or index_dim <= 0:
        raise ValueError("Q_I index_heads and index_dim must be positive.")

    assert isinstance(compression_rate, int)
    assert isinstance(num_topk_blocks, int)
    assert isinstance(sliding_window_size, int)
    assert isinstance(rope_dims, int)
    assert isinstance(share_kv, bool)
    if compression_rate <= 0:
        raise ValueError("compression_rate must be positive.")
    if num_topk_blocks < 0 or sliding_window_size < 0:
        raise ValueError("num_topk_blocks and sliding_window_size must be non-negative.")
    if rope_dims <= 0 or rope_dims % 2 or rope_dims > min(head_dim, index_dim):
        raise ValueError(
            "rope_dims must be positive, even, and no larger than either head dimension."
        )

    for name, tensor in tensors:
        assert isinstance(tensor, torch.Tensor)
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {tensor.device}.")
        if tensor.dtype != query.dtype:
            raise TypeError(f"{name} must have dtype {query.dtype}, got {tensor.dtype}.")

    expected_kv_heads = (1, heads) if share_kv else (heads,)
    for name in ("KV", "C_a", "C_b", "Z_a", "Z_b"):
        tensor = by_name[name]
        assert isinstance(tensor, torch.Tensor)
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[1] not in expected_kv_heads
            or tensor.shape[2:] != (sequence_length, head_dim)
        ):
            expected_heads = "1 or heads" if share_kv else "heads"
            raise ValueError(
                f"{name} must have shape [batch, {expected_heads}, sequence, head_dim]."
            )

    expected_index_heads = (1, index_heads) if share_kv else (index_heads,)
    for name in ("K_Ia", "K_Ib", "Z_Ia", "Z_Ib"):
        tensor = by_name[name]
        assert isinstance(tensor, torch.Tensor)
        if (
            tensor.ndim != 4
            or tensor.shape[0] != batch
            or tensor.shape[1] not in expected_index_heads
            or tensor.shape[2:] != (sequence_length, index_dim)
        ):
            expected_heads = "1 or index_heads" if share_kv else "index_heads"
            raise ValueError(
                f"{name} must have shape "
                f"[batch, {expected_heads}, sequence, index_dim]."
            )

    expected_shapes = {
        "B_a": (compression_rate, head_dim),
        "B_b": (compression_rate, head_dim),
        "W_I": (batch, sequence_length, index_heads),
        "B_Ia": (compression_rate, index_dim),
        "B_Ib": (compression_rate, index_dim),
        "KV_norm_weight": (head_dim,),
        "compressed_indices_norm_weight": (index_dim,),
        "compressed_kv_norm_weight": (head_dim,),
        "attention_sink": (heads,),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = by_name[name]
        assert isinstance(tensor, torch.Tensor)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )


def _load_eager_implementation() -> Callable[..., torch.Tensor]:
    from . import reference

    # The checked-in oracle uses ``math`` in RoPE without importing it. Keep the reference file
    # untouched while making its public eager entry point runnable.
    reference.math = math
    return reference.CSA


def _load_triton_implementation() -> Callable[..., torch.Tensor]:
    try:
        from .triton import compressed_sparse_attention as implementation
    except ModuleNotFoundError as error:
        if error.name in (f"{__package__}.triton", "triton"):
            raise RuntimeError(
                "The Triton backend for compressed sparse attention is not available; "
                "install Triton and ensure the backend module is packaged."
            ) from error
        raise

    return implementation


def _load_cute_implementation() -> Callable[..., torch.Tensor]:
    try:
        # Probe roots explicitly even when an earlier CuTe import left its internal
        # modules cached.  This keeps missing optional dependencies deterministic.
        for dependency in ("cuda", "cutlass", "quack", "flash_attn"):
            importlib.import_module(dependency)
        from .cute import compressed_sparse_attention as implementation
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module == f"{__package__}.cute" or missing_module.split(".")[0] in (
            "flash_attn",
            "cutlass",
            "cuda",
            "quack",
        ):
            raise RuntimeError(
                "The CuTe DSL backend for compressed sparse attention is not available; "
                "install nvidia-cutlass-dsl, flash-attn-4, and quack-kernels and ensure "
                "the backend module is packaged."
            ) from error
        raise

    return implementation


def compressed_sparse_attention(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    mode: Mode = "auto",
    backend: Backend = "eager",
) -> torch.Tensor:
    '''
    Naming of args uses convention from Deepseek v4 paper

    Shape notation: B=batch size, H=attention heads, H_I=indexer heads, S=sequence length,
    D=attention dimension, D_I=index dimension, R=compression_rate,
    H_KV=H and H_IKV=H_I when share_kv=False. When share_kv=True, those dimensions
    may instead be 1; the CuTe backend requires them to be 1.

    Rope and normalization are applied within the function for KV vectors, but not for Q or Q_I
    This is because Q and Q_I are expected to be projected from the same normalized and pre-rotated latent
    Q: Query vector for attention; expected to be normalized beforehand; expected shape: (B, H, S, D)
    Q_I: Query vector for indexing; expected to be normalized beforehand; expected shape: (B, H_I, S, D_I)

    KV: Projection from residual stream; expected shape: (B, H_KV, S, D)
    C_a, C_b: Projections from the residual stream that will be attended to; each expected shape: (B, H_KV, S, D)
    Z_a, Z_b: Projections from the residual stream that weight C_a and C_b; each expected shape: (B, H_KV, S, D)
    B_a, B_b: Biases for C_a * Z_a + B_a computation; each expected shape: (R, D)

    W_I: Projection from the residual stream, per-head weight on indexer scores (Batch, sequence, num_heads); expected shape: (B, S, H_I)
    K_Ia, K_Ib: Projections from the residual stream for computing indexing; each expected shape: (B, H_IKV, S, D_I)
    Z_Ia, Z_Ib: Projections from the residual stream, performs similar role to Z_a and Z_b for indexing; each expected shape: (B, H_IKV, S, D_I)
    B_Ia, B_Ib: Similar role to B_a, but for the indexing branch; each expected shape: (R, D_I)

    KV_norm_weight: RMS norm weights for KV; expected shape: (D,)
    compressed_indices_norm_weight: RMS weights for compressed indices; expected shape: (D_I,)
    compressed_kv_norm_weight: RMS norm weights for compressed blocks; expected shape: (D,)

    attention_sink: Learned weight in shape of (num_heads, ), functions as attention sink; expected shape: (H,)

    compression_rate: size of each compressed block / 2 (due to block interleaving)
    num_topk_blocks: number of blocks to attend to per query
    sliding_window_size: size of sliding window for SWA
    rope_dims: number of dimensions to apply rope to
    share_kv: True if all query heads attented to one kv head; SM100 backend only supports share_kv = true
    mode: Currently only auto/chunked are supported (implementations are prefill only as of now)
    backend: One of eager, triton, or cute. Cute is supported for SM100 only
    '''
    if backend not in ("eager", "triton", "cute"):
        raise ValueError(
            f"Unsupported compressed sparse attention backend {backend!r}; "
            "expected 'eager', 'triton', or 'cute'."
        )
    if mode not in ("auto", "chunked", "recurrent"):
        raise ValueError( f"Unsupported compressed sparse attention mode {mode!r}; "
            "expected 'auto', 'chunked', or 'recurrent'."
        )
    if mode == "recurrent":
        raise ValueError(f"Recurrent backed is currently unsupported")

    tensors = (
        ("Q", Q),
        ("Q_I", Q_I),
        ("KV", KV),
        ("C_a", C_a),
        ("C_b", C_b),
        ("Z_a", Z_a),
        ("Z_b", Z_b),
        ("B_a", B_a),
        ("B_b", B_b),
        ("W_I", W_I),
        ("K_Ia", K_Ia),
        ("K_Ib", K_Ib),
        ("Z_Ia", Z_Ia),
        ("Z_Ib", Z_Ib),
        ("B_Ia", B_Ia),
        ("B_Ib", B_Ib),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_indices_norm_weight", compressed_indices_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    )
    _validate_inputs(
        tensors,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )

    if backend == "eager":
        implementation = _load_eager_implementation()
    elif backend == "triton":
        implementation = _load_triton_implementation()
    else:
        implementation = _load_cute_implementation()

    return implementation(
        Q,
        Q_I,
        KV,
        C_a,
        C_b,
        Z_a,
        Z_b,
        B_a,
        B_b,
        W_I,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_Ia,
        B_Ib,
        KV_norm_weight,
        compressed_indices_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )


__all__ = ["compressed_sparse_attention"]
