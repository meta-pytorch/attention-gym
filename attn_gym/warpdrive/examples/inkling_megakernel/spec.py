"""Inkling 2 Mini's targets of the speculative step (``examples.megakernel.spec.SpecStep``):
``TorchTargetBlock``, the capturable torch step (``fast.py``'s dataflow over ``k`` tokens). With
``dspark.DSparkDrafter(static=True)`` the step's output is ``dspark.speculative_generate``'s for
the same verify forward and drafter (its host loop).

Torch only; no CuTeDSL. Rejected positions need no undo: caches are indexed by position, so the
next block rewrites them, and the target's convolution rings, which keep ``CONV_RING``
positions, still hold the ``kernel - 1`` rows before the next anchor while a block writes at most
``CONV_RING - kernel + 1`` (13) positions (``max_block_tokens``). The chat's over-run past a stop
adds no rewind: a replay feeds at most ``SYNC_EVERY`` positions, rejected rows included.
"""

from collections.abc import Iterable

import torch

from attn_gym.warpdrive.examples.inkling_megakernel.dspark import DSparkConfig
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
    hidden_states,
    logits_of,
)


def max_block_tokens(config: InklingConfig) -> int:
    """The longest verify block whose rejected positions the convolution rings can rewind: a
    block writes its positions into rings of ``CONV_RING`` rows, and the next block reads the
    ``kernel - 1`` rows before its anchor, which follows the block's first position."""
    return CONV_RING - config.conv_kernel + 1


def block_forward(
    weights: InklingWeights,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    cache: InklingCache,
    visits: Iterable[int] = DSparkConfig.target_visits,
    kernels: Kernels = FAST,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``examples.megakernel.spec.TargetBlock``'s result for ``tokens`` at device ``positions``:
    ``ref.hidden_states`` with ``start=None`` (static shapes) on ``kernels`` and the inputs of
    ``visits`` (``config.num_visits``: the final normalized states)."""
    visits = tuple(visits)
    captured: dict[int, torch.Tensor] = {}

    def observe(visit: int, h: torch.Tensor):
        if visit in visits:
            captured[visit] = h

    h = hidden_states(weights, tokens, positions, cache, None, kernels, observe)
    return logits_of(weights, h), torch.stack([captured[v] for v in visits], dim=1)


class TorchTargetBlock:
    """``block_forward`` as a ``examples.megakernel.spec.TargetBlock`` of ``tokens`` tokens into
    ``cache``: the capturable torch step (``fast.decode_step``'s kernels by default) over a
    block."""

    def __init__(
        self,
        weights: InklingWeights,
        cache: InklingCache,
        tokens: int,
        visits: Iterable[int] = DSparkConfig.target_visits,
        kernels: Kernels = FAST,
    ):
        if not 1 <= tokens <= max_block_tokens(weights.config):
            raise ValueError(
                f"blocks of 1 .. {max_block_tokens(weights.config)} tokens fit the convolution "
                f"rings' rewind, got {tokens}"
            )
        self.weights, self.cache, self.tokens = weights, cache, tokens
        self.visits, self.kernels = tuple(visits), kernels
        self.offsets = torch.arange(tokens, device=cache.k.device)

    def __call__(
        self, tokens: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = pos.reshape(1).long() + self.offsets
        return block_forward(
            self.weights, tokens, positions, self.cache, self.visits, self.kernels
        )
