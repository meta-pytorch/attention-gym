"""Greedy speculative decoding as one CUDA-graph capturable step: draft, verify and accept on the
device (``SpecStep``), over a model's target and drafter (``TargetBlock``, ``DraftModel``).

Torch only; no CuTeDSL. A step at the chat's position ``pos`` (the last fed position; the anchor,
the last committed token, sits at ``pos + 1`` and is not fed yet):

1. the drafter proposes ``k - 1`` drafts after the anchor (``DraftModel.propose``);
2. the target runs the block ``[anchor, d1 .. d(k-1)]`` at ``pos + 1 .. pos + k``
   (``TargetBlock``), giving each row's logits and the drafter's features;
3. accept: ``a`` = the longest prefix of drafts equal to the target's argmax, so the step commits
   ``a + 1`` tokens (the accepted drafts and the target's next token, the bonus and new anchor)
   and feeds ``a + 1`` (the anchor and the accepted drafts); ``pos`` advances by ``a + 1``, and
   the drafter receives the block's features with ``a`` marking the most recent committed row.

Nothing reads a device value on the host, so a CUDA graph can hold any number of steps; the
chat reads each step's count and tokens after a replay (``chat.MultiTokenBackend``). The output
is the target's greedy decode in the rounding of a ``k``-token forward, whatever the drafts.

Rejected positions need no undo when the target's caches are indexed by position: the next
block rewrites them. A target with state that is not (short-convolution rings, recurrent state)
bounds ``k`` by how far that state can rewind; ``inkling_megakernel.spec`` is the example.
"""

from typing import Protocol

import torch


class DraftModel(Protocol):
    """A drafter on the device (``inkling_megakernel.dspark.DSparkDrafter(static=True)``).

    ``num_drafts``: the drafts ``propose`` returns, ``k - 1`` for blocks of ``k`` tokens.
    ``commit`` stores the target's ``features`` (one row per position, in whatever form the
    drafter reads: Inkling's are bf16 ``(rows, taps, hidden)`` hidden states) of consecutive
    int64 device ``positions``, row ``last`` (a one-element int64 device index; None: the last
    row) the most recent committed position, whose features also feed the next proposal; rows
    past ``last`` belong to rejected positions, which the next ``propose`` rewrites before
    reading.
    ``propose`` returns int64 ``(num_drafts,)`` drafts after the one-element int64 ``anchor`` at
    the one-element device position ``pos``. Neither syncs with the host."""

    num_drafts: int

    def commit(
        self, features: torch.Tensor, positions: torch.Tensor, last: torch.Tensor | None = None
    ) -> None: ...

    def propose(self, anchor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor: ...


class TargetBlock(Protocol):
    """The target over a block of ``tokens`` consecutive tokens
    (``inkling_megakernel.spec.TorchTargetBlock``): ``block(tokens, pos)`` feeds int64
    ``(tokens,)`` at ``pos .. pos + tokens - 1`` (``pos`` a one-element int64 or int32 device
    position), writes the caches there, and returns fp32 ``(tokens, vocab)`` logits and the
    drafter's features (``DraftModel.commit``'s; ``SpecStep`` passes them on unread), row ``t``
    token ``t``'s, causal within the block. No host sync."""

    tokens: int

    def __call__(
        self, tokens: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def accepted_drafts(drafts: torch.Tensor, verified: torch.Tensor) -> torch.Tensor:
    """The length (a 0-dim int64 device tensor) of the longest prefix of ``drafts`` equal to the
    target's argmaxes ``verified[:-1]`` (row ``i`` predicts the token after row ``i``)."""
    return (drafts == verified[:-1]).long().cumprod(0).sum()


class SpecStep:
    """One greedy speculative step over ``target`` and ``drafter`` (see the module docstring),
    with ``chat.MultiTokenBackend.decode``'s contract: ``step(token, pos, out)`` feeds the
    anchor ``token`` at ``pos + 1``, writes ``[count, committed tokens ...]`` into ``out``
    (``(1 + tokens_per_step,)`` int64; entries past ``count`` are the rejected rows' argmaxes),
    the new anchor into ``token`` and advances ``pos`` by ``count``. No host sync."""

    def __init__(self, target: TargetBlock, drafter: DraftModel, device="cuda"):
        if target.tokens < 2 or drafter.num_drafts != target.tokens - 1:
            raise ValueError(
                f"a block of {target.tokens} tokens needs >= 2 tokens and {target.tokens - 1} "
                f"drafts, got {drafter.num_drafts}"
            )
        self.target, self.drafter = target, drafter
        self.tokens_per_step = target.tokens
        self.offsets = torch.arange(target.tokens, device=device)

    def __call__(self, token: torch.Tensor, pos: torch.Tensor, out: torch.Tensor) -> None:
        anchor_pos = pos.reshape(1).long() + 1
        positions = anchor_pos + self.offsets
        drafts = self.drafter.propose(token, anchor_pos)
        logits, features = self.target(torch.cat([token.reshape(1), drafts]), anchor_pos)
        verified = logits.argmax(-1)
        accepted = accepted_drafts(drafts, verified)
        self.drafter.commit(features, positions, last=accepted)
        out[0].copy_(accepted + 1)
        out[1:].copy_(verified)
        token.copy_(verified.index_select(0, accepted.reshape(1)))
        pos.add_(accepted + 1)
