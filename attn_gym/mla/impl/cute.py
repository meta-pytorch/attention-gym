"""CuTe DSL (SM100/SM103) backend for paged MLA decode.

Delegates to FlashAttention-4's absorbed MLA path (``qv``): FA4 sums
``q_rope @ rope^T`` and ``q_latent @ latent^T`` into one score accumulator and uses the
latent rows as values, which is the 576-wide latent dot product split into its rope and
latent parts. FA4 owns compilation caching, split-KV, and the paged gather.

The kernel runs behind the ``attn_gym::paged_mla_decode_fwd`` operator, so the call
compiles with ``torch.compile(fullgraph=True)``.

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


torch.library.define(
    "attn_gym::paged_mla_decode_fwd",
    "(Tensor q_latent, Tensor q_rope, Tensor latent_cache, Tensor rope_cache, "
    "Tensor block_table, Tensor cu_seqlens_q, Tensor seqused_k, int max_seqlen_q, "
    "float scale) -> (Tensor, Tensor)",
)


def _paged_mla_decode_fwd_cuda(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    # TODO: expose FA4's MLA kernel choice (1CTA/2CTA, split count) per call and autotune it.
    out, lse = _flash_attn_fwd(
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
        return_lse=True,
    )[:2]
    return out, lse


torch.library.impl("attn_gym::paged_mla_decode_fwd", "CUDA", _paged_mla_decode_fwd_cuda)


@torch.library.register_fake("attn_gym::paged_mla_decode_fwd")
def _paged_mla_decode_fwd_fake(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q_latent.new_empty(q_latent.shape), q_latent.new_empty(
        q_latent.shape[:2], dtype=torch.float32
    )


_paged_mla_decode_fwd = torch.ops.attn_gym.paged_mla_decode_fwd.default


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FA4's absorbed MLA forward over the paged latent cache; LSE is ``[tokens, heads]``."""
    if (error := _constraint_violation(q_latent, q_rope)) is not None:
        raise error
    return _paged_mla_decode_fwd(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        block_table,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_q,
        scale,
    )
