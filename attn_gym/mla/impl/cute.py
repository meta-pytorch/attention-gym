"""CuTe DSL (SM100/SM103) backend for paged MLA decode.

Delegates to FlashAttention-4's absorbed MLA path (``qv``): FA4 sums
``q_rope @ rope^T`` and ``q_latent @ latent^T`` into one score accumulator and uses the
latent rows as values, which is the 576-wide latent dot product split into its rope and
latent parts. FA4 owns compilation caching, split-KV, and the paged gather.

This backend is eager-only.

Constraints
-----------
- latent_dim = 512, rope_dim = 64, heads divides 128
- dtype = bfloat16, SM100 or SM103 (compute capability 10.0 or 10.3)
"""

from __future__ import annotations

import torch

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _constraint_violation(q_latent: torch.Tensor, q_rope: torch.Tensor) -> Exception | None:
    """Return the error for metadata this backend cannot run, or None when it qualifies."""
    if q_latent.device.type != "cuda":
        return ValueError("CuTe paged MLA decode requires CUDA tensors.")
    if torch.cuda.get_device_capability(q_latent.device) not in SUPPORTED_CAPABILITIES:
        return ValueError("CuTe paged MLA decode requires SM100 or SM103.")
    if q_latent.dtype != torch.bfloat16:
        return TypeError(f"CuTe paged MLA decode requires bfloat16, got {q_latent.dtype}.")
    if q_latent.shape[-1] != 512 or q_rope.shape[-1] != 64:
        return ValueError(
            "CuTe paged MLA decode requires latent_dim=512 and rope_dim=64, "
            f"got {q_latent.shape[-1]} and {q_rope.shape[-1]}."
        )
    if 128 % q_latent.shape[1] != 0:
        return ValueError(
            "CuTe paged MLA decode requires a query head count that divides 128, "
            f"got {q_latent.shape[1]}."
        )
    return None


def paged_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    scale: float,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FA4's absorbed MLA forward over the paged latent cache."""
    if (error := _constraint_violation(q_latent, q_rope)) is not None:
        raise error
    from flash_attn.cute.interface import flash_attn_varlen_func

    # TODO: call FA4's inner forward instead of the public wrapper so kernel configs can be
    # specified and autotuned, then register a custom op so this works under torch.compile.
    return flash_attn_varlen_func(
        q_rope,
        rope_cache.unsqueeze(2),
        latent_cache.unsqueeze(2),
        qv=q_latent,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        page_table=block_table,
        softmax_scale=scale,
        causal=True,
        pack_gqa=True,
        num_splits=0,
        return_lse=return_lse,
    )
