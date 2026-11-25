"""Inkling 2 Mini's targets of the speculative step (``examples.megakernel.spec.SpecStep``):
``TorchTargetBlock`` (the capturable torch step, ``fast.py``'s dataflow over ``k`` tokens) or
``DecoderTargetBlock`` (the multi-token megakernel decoder's block step), with the drafter
``dspark.DSparkDrafter(static=True)`` or the draft launches (``draft.DSparkMegakernel``). The
step's output is ``dspark.speculative_generate``'s for the same verify forward and drafter (its
host loop).

Torch only; no CuTeDSL. Rejected positions need no undo: caches are indexed by position, so the
next block rewrites them, and the target's convolution rings, which keep ``CONV_RING``
positions, still hold the ``kernel - 1`` rows before the next anchor while a block writes at most
``CONV_RING - kernel + 1`` (13) positions (``max_block_tokens``). The chat's over-run past a stop
adds no rewind: a replay feeds at most ``SYNC_EVERY`` positions, rejected rows included.
"""

from collections.abc import Iterable

import torch

from attn_gym.warpdrive.examples.inkling_megakernel.dspark import DSparkConfig, target_forward
from attn_gym.warpdrive.examples.inkling_megakernel.fast import FAST
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    InklingConfig,
    InklingWeights,
    Kernels,
)

# The blocks ``megakernel.InklingDecoder`` (``DecoderTargetBlock``) runs; the chat's default (4)
# and 8 are tested against the oracle (test_warpdrive_inkling_megakernel_multi.py).
BLOCK_TOKENS = (2, 4, 8)


def max_block_tokens(config: InklingConfig) -> int:
    """The longest verify block whose rejected positions the convolution rings can rewind: a
    block writes its positions into rings of ``CONV_RING`` rows, and the next block reads the
    ``kernel - 1`` rows before its anchor, which follows the block's first position."""
    return CONV_RING - config.conv_kernel + 1


class TorchTargetBlock:
    """An ``examples.megakernel.spec.TargetBlock`` of ``tokens`` tokens into ``cache``: the
    capturable torch step (``fast.decode_step``'s kernels by default) over a block, logits and
    the inputs of ``visits`` (``dspark.target_forward`` at a device position)."""

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

    def __call__(
        self, tokens: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return target_forward(
            self.weights, tokens, pos, self.cache, self.visits, True, self.kernels
        )


class DecoderTargetBlock:
    """``megakernel.InklingDecoder.decode_block`` as a ``TargetBlock`` into ``cache``:
    ``decoder``, built with ``block_tokens``, gives the block's logits and features (its
    ``FEATURE_VISITS`` inputs and final normalized states, ``DSparkConfig.target_visits``).
    Its kernels keep the blocks within ``max_block_tokens`` (their rings must hold a block and
    the convolution history)."""

    def __init__(self, decoder, cache: InklingCache):
        if decoder.block_tokens is None:
            raise ValueError("the decoder was built without block_tokens")
        self.decoder, self.cache, self.tokens = decoder, cache, decoder.block_tokens

    def __call__(
        self, tokens: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.decode_block(tokens, pos, self.cache, features=True)
