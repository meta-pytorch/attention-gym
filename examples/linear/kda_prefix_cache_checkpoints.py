"""Reuse a cached KDA prompt prefix with state checkpoints and replay tails.

A serving engine prefills request A in a paged state slot. ``paged_chunk_kda`` writes the
state at the last full chunk of the shared prefix into a destination slot of the same pool.
Before prefill advances A, the engine copies the prefix's replay tail to that destination.
Request B takes ownership of the completed slot and prefills only its suffix. The result
matches a from-scratch prefill bit for bit.
"""

from __future__ import annotations

import argparse

import torch

from attn_gym.linear import ReplayState, StateCheckpoints, copy_replay_tails, paged_chunk_kda

CHUNK_SIZE = 64
HEADS = 2
HEAD_DIM = 128  # Required by fused paged_chunk_kda.
TOKENS = 200  # Each request's prompt length.
SLOT_A, SLOT_B, SLOT_REFERENCE = 1, 2, 3  # Slot 0 is the null slot.
# Checkpoint writes require Triton. Pin the backend and its launch configurations on every
# call so the bitwise comparison does not mix kernels.
OPTIONS = {"backend": "fused", "autotune": False}


def make_prompt(seed: int, tokens: int) -> tuple[torch.Tensor, ...]:
    """Create projected q, k, v, per-token log gate and beta for one prompt."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, HEADS, HEAD_DIM)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(*size, device="cuda", generator=generator)

    # The delta rule expects unit-norm queries and keys, as after the model's L2 norm.
    q, k = (torch.nn.functional.normalize(randn(*shape), dim=-1).bfloat16() for _ in range(2))
    v = randn(*shape).bfloat16()
    gate = -torch.rand(shape, device="cuda", generator=generator) * 0.1
    beta = torch.rand(shape[:3], device="cuda", generator=generator)
    return q, k, v, gate, beta


def prefill(slot: int, prompt, state_cache, replay, *, fresh: bool):
    """Advance one slot's recurrent state and replay tail over a prompt segment."""
    return paged_chunk_kda(
        *prompt,
        state_cache,
        torch.tensor([slot], device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor([not fresh], device="cuda"),
        replay_state=replay,
        kernel_options=OPTIONS,
    )


def restore_shared_prefix(prefix: int) -> int:
    """Serve request B from request A's checkpoint; return the restored state boundary."""
    if not CHUNK_SIZE <= prefix < TOKENS:
        raise ValueError(f"the shared prefix must hold a full chunk and leave a suffix: {prefix}")
    restored = prefix // CHUNK_SIZE * CHUNK_SIZE  # The last chunk boundary of the shared prefix.
    prompt_a = make_prompt(seed=1, tokens=TOKENS)
    suffix_b = make_prompt(seed=2, tokens=TOKENS - prefix)
    prompt_b = tuple(
        torch.cat((a[:, :prefix], b), dim=1) for a, b in zip(prompt_a, suffix_b, strict=True)
    )

    slots = SLOT_REFERENCE + 1
    state_cache = torch.zeros(slots, HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
    replay = ReplayState(
        *(
            torch.zeros(slots, *shape, device="cuda", dtype=dtype)
            for shape, dtype in ReplayState.slot_layout(HEADS, HEAD_DIM)
        )
    )
    # B's slot was last used by another request: restoring must not depend on clearing it.
    for cache in replay[:-1]:
        cache[SLOT_B].normal_()
    replay.count[SLOT_B] = 37

    # Both operations use the scheduler's original prefix position and slot. Copy the replay
    # inputs first: the prefill below changes the source slot's count and tail.
    checkpoints = StateCheckpoints(
        state_cache,
        checkpoint_positions=torch.tensor([prefix], device="cuda", dtype=torch.int32),
        checkpoint_slots=torch.tensor([SLOT_B], device="cuda", dtype=torch.int32),
    )
    cu_seqlens = torch.tensor([0, TOKENS], device="cuda", dtype=torch.int32)
    source_slots = torch.tensor([SLOT_A], device="cuda", dtype=torch.int32)
    resumed = torch.tensor([False], device="cuda")
    copy_replay_tails(
        replay,
        prompt_a,
        cu_seqlens=cu_seqlens,
        state_indices=source_slots,
        has_initial_state=resumed,
        checkpoint_positions=checkpoints.checkpoint_positions,
        checkpoint_slots=checkpoints.checkpoint_slots,
    )
    paged_chunk_kda(
        *prompt_a,
        state_cache,
        source_slots,
        cu_seqlens=cu_seqlens,
        replay_state=replay,
        has_initial_state=resumed,
        state_checkpoints=checkpoints,
        kernel_options=OPTIONS,
    )
    # B now owns this slot exclusively. An engine retaining a shared cache entry would fork it
    # before advancing B, rather than mutate the shared entry.
    reused = prefill(SLOT_B, suffix_b, state_cache, replay, fresh=False)

    # Reference: prefill B's whole prompt from scratch in another slot.
    reference = prefill(SLOT_REFERENCE, prompt_b, state_cache, replay, fresh=True)

    if not torch.equal(reused, reference[:, prefix:]):
        raise AssertionError("the reused suffix output differs from a full prefill")
    if not torch.equal(state_cache[SLOT_B], state_cache[SLOT_REFERENCE]):
        raise AssertionError("the reused recurrent state differs from a full prefill")
    # Replay buffers are meaningful only up to each slot's count.
    count = int(replay.count[SLOT_REFERENCE])
    if int(replay.count[SLOT_B]) != count or not all(
        torch.equal(cache[SLOT_B, :count], cache[SLOT_REFERENCE, :count]) for cache in replay[:-1]
    ):
        raise AssertionError("the reused replay tail differs from a full prefill")
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", type=int, default=150, help="shared prompt tokens")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit("paged_chunk_kda requires CUDA capability 8.0 or newer")

    boundary = restore_shared_prefix(args.prefix)
    print("suffix output, state, and replay tail match a full prefill bitwise")
    print(
        f"state restored at token {boundary}; request B prefills only {TOKENS - args.prefix} tokens"
    )


if __name__ == "__main__":
    main()
