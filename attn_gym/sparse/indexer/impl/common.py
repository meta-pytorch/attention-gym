"""Shared score-slab geometry and stride metadata for fused indexer backends."""

from torch import Tensor

from attn_gym.utils import cdiv

_MAX_SCORE_PAIRS = 512
_SCORE_WORKSPACE_BYTES = 32 * 1024 * 1024


def score_workspace_pairs(batch: int, tokens: int, candidates: int) -> int:
    """Bound positive, validated inputs to 1024 score rows and 32 MiB."""
    return min(
        _MAX_SCORE_PAIRS, batch * cdiv(tokens, 2), _SCORE_WORKSPACE_BYTES // (8 * candidates)
    )


def score_strides(tensor: Tensor) -> tuple[int, ...]:
    """Discard unreachable wide singleton strides before Triton casts constexpr literals."""
    return tuple(
        0 if size == 1 and stride > 2**31 - 1 else stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
