"""FlashAttention-3 (SM90) backend for paged MLA decode.

Delegates to FlashAttention-3's absorbed MLA path (``qv``), the same contract as the
FlashAttention-4 backend: the rope query scores against the rope keys, the latent query
against the latent rows, and the latent rows are the values. FA3 owns split-KV and the
paged gather.

Constraints
-----------
- latent_dim = 512, rope_dim = 64 (FA3's ``64_512`` head-dimension build)
- dtype = float16 or bfloat16, SM90 (compute capability 9.0)
"""

from __future__ import annotations

import torch

SUPPORTED_CAPABILITIES = ((9, 0),)


def _validate_inputs(q_latent: torch.Tensor, q_rope: torch.Tensor) -> None:
    """Raise for metadata this backend cannot run."""
    if q_latent.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"FA3 paged MLA decode requires float16 or bfloat16, got {q_latent.dtype}."
        )
    if q_latent.shape[-1] != 512 or q_rope.shape[-1] != 64:
        raise ValueError(
            "FA3 paged MLA decode requires latent_dim=512 and rope_dim=64, "
            f"got {q_latent.shape[-1]} and {q_rope.shape[-1]}."
        )


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
    """Run FA3's absorbed MLA forward over the paged latent cache."""
    _validate_inputs(q_latent, q_rope)
    from flash_attn_interface import flash_attn_with_kvcache

    result = flash_attn_with_kvcache(
        q_rope,
        rope_cache.unsqueeze(2),
        latent_cache.unsqueeze(2),
        qv=q_latent,
        cache_seqlens=seqused_k,
        page_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        return_softmax_lse=return_lse,
    )
    if not return_lse:
        return result, None
    out, lse = result
    # Varlen FA3 returns LSE as [heads, tokens]; the API returns [tokens, heads].
    return out, lse.t().contiguous()
