"""Gated delta rule attention."""

from attn_gym.linear._lazy import lazy_exports
from attn_gym.linear.gdn.api import (
    KernelOptions,
    chunk_gdn,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)

# Note: Lazy Imports (see attn_gym/linear/__init__.py)
_BACKEND_EXPORTS = {
    "ChunkGDNSaved": "attn_gym.linear.gdn.stages",
    "chunk_gdn_prepare": "attn_gym.linear.gdn.stages",
    "chunk_gdn_prepare_backward": "attn_gym.linear.gdn.stages",
}


__getattr__ = lazy_exports(__name__, _BACKEND_EXPORTS, requirement="CUDA kernel backends")


__all__ = sorted(  # noqa: PLE0605 -- backend exports resolve lazily
    [
        "KernelOptions",
        "chunk_gdn",
        "paged_chunk_gdn",
        "recurrent_gdn",
        "recurrent_gdn_decode",
        *_BACKEND_EXPORTS,
    ]
)
