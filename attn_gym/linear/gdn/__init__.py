"""Gated delta rule attention."""

import importlib

from attn_gym.linear.gdn.api import (
    KernelOptions,
    chunk_gdn,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)

# Note: Lazy Imports (see attn_gym/linear/__init__.py). Staged primitives around the affine
# state boundary (stages.py).
_BACKEND_EXPORTS = {
    "ChunkGDNSaved": "attn_gym.linear.gdn.stages",
    "chunk_gdn_prepare": "attn_gym.linear.gdn.stages",
    "chunk_gdn_prepare_backward": "attn_gym.linear.gdn.stages",
}


def __getattr__(name: str):
    module_name = _BACKEND_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{name} requires the optional CUDA kernel backends: pip install attn-gym[linear]"
        ) from error
    return getattr(module, name)


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
