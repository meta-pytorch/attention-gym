"""Private operators for the fused paged MLA decode.

Schemas and fakes register at import, so ``torch.compile`` traces each FlashAttention launch
as one opaque op. The CUDA kernel picks FlashAttention-4 or FlashAttention-3 by device and
imports it on first call.
"""

from __future__ import annotations

import torch

from .impl import cute, fa3

# Two fixed-arity ops instead of an optional LSE output.
torch.library.define(
    "attn_gym::paged_mla_decode",
    "(Tensor q_latent, Tensor q_rope, Tensor latent_cache, Tensor rope_cache,"
    " Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, SymInt max_seqlen_q,"
    " SymInt? max_seqlen_k, float scale, bool autotune) -> Tensor",
)
torch.library.define(
    "attn_gym::paged_mla_decode_with_lse",
    "(Tensor q_latent, Tensor q_rope, Tensor latent_cache, Tensor rope_cache,"
    " Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, SymInt max_seqlen_q,"
    " SymInt? max_seqlen_k, float scale, bool autotune) -> (Tensor, Tensor)",
)


def _decode_cuda(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    scale: float,
    autotune: bool,
    *,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    args = (q_latent, q_rope, latent_cache, rope_cache, block_table, cu_seqlens_q, seqused_k)
    options = {"max_seqlen_q": max_seqlen_q, "scale": scale, "return_lse": return_lse}
    capability = torch.cuda.get_device_capability(q_latent.device)
    if capability in cute.SUPPORTED_CAPABILITIES:
        # Autotuning allocates and captures graphs on the current device.
        with torch.cuda.device(q_latent.device):
            return cute.paged_mla_decode(
                *args, **options, max_seqlen_k=max_seqlen_k, autotune=autotune
            )
    if capability in fa3.SUPPORTED_CAPABILITIES:
        # FlashAttention-3 takes no kernel config or key-length hint.
        return fa3.paged_mla_decode(*args, **options)
    raise ValueError(
        "The fused paged_mla_decode requires SM90 (FlashAttention-3) or "
        f"SM100/SM103 (FlashAttention-4), got SM{capability[0]}{capability[1]}."
    )


torch.library.impl(
    "attn_gym::paged_mla_decode",
    "CUDA",
    lambda *args: _decode_cuda(*args, return_lse=False)[0],
)
torch.library.impl(
    "attn_gym::paged_mla_decode_with_lse",
    "CUDA",
    lambda *args: _decode_cuda(*args, return_lse=True),
)


@torch.library.register_fake("attn_gym::paged_mla_decode")
def _decode_fake(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    scale: float,
    autotune: bool,
) -> torch.Tensor:
    del q_rope, latent_cache, rope_cache, block_table, cu_seqlens_q, seqused_k
    del max_seqlen_q, max_seqlen_k, scale, autotune
    # Both backends return contiguous outputs regardless of query strides.
    return q_latent.new_empty(q_latent.shape)


@torch.library.register_fake("attn_gym::paged_mla_decode_with_lse")
def _decode_with_lse_fake(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    scale: float,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_rope, latent_cache, rope_cache, block_table, cu_seqlens_q, seqused_k
    del max_seqlen_q, max_seqlen_k, scale, autotune
    return q_latent.new_empty(q_latent.shape), q_latent.new_empty(
        q_latent.shape[:2], dtype=torch.float32
    )
