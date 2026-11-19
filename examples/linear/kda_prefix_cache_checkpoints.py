"""Reuse a cached KDA prompt prefix with state checkpoints and replay tails.

A serving engine prefills request A in a paged state slot and asks ``paged_chunk_kda`` for
periodic FP32 state checkpoints. Request B shares A's first ``prefix`` tokens. Instead of
recomputing them, B's slot is seeded with the checkpoint at the last full chunk before the
prefix end, plus the prefix tokens after that chunk as B's replay tail. B then prefills only
its own suffix, and its output matches a from-scratch prefill bit for bit.

"""

from __future__ import annotations

import argparse

import torch

from attn_gym.linear import ReplayState, StateCheckpoints, paged_chunk_kda

CHUNK_SIZE = 64
HEADS = 2
HEAD_DIM = 128  # Required by fused paged_chunk_kda.
TOKENS = 200  # Each request's prompt length.
INTERVAL = 64  # Checkpoint every chunk.
SLOT_A, SLOT_B, SLOT_REFERENCE = 1, 2, 3  # Slot 0 is the null slot.
OPTIONS = {"autotune": False}  # Fixed heuristics keep kernel selection repeatable.


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


def prefill(slot: int, prompt, state_cache, replay, *, fresh: bool, checkpoints=None):
    """Advance one slot's recurrent state and replay tail over a prompt segment."""
    return paged_chunk_kda(
        *prompt,
        state_cache,
        torch.tensor([slot], device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor([not fresh], device="cuda"),
        replay_state=replay,
        state_checkpoints=checkpoints,
        kernel_options=OPTIONS,
    )


def restore_shared_prefix(prefix: int) -> int:
    """Serve request B from request A's checkpoint; return the prefill tokens it skips."""
    restored = prefix // INTERVAL * INTERVAL  # The last checkpoint inside the shared prefix.
    tail = prefix - restored
    if not 0 < restored or tail >= CHUNK_SIZE:
        raise ValueError("the shared prefix must contain a checkpoint within one chunk")
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

    # Request A: a fresh slot replays nothing, so it reserves TOKENS // INTERVAL rows.
    checkpoints = StateCheckpoints(
        torch.empty(TOKENS // INTERVAL, HEADS, HEAD_DIM, HEAD_DIM, device="cuda"),
        torch.tensor([0, TOKENS // INTERVAL], device="cuda", dtype=torch.int32),
        INTERVAL,
    )
    prefill(SLOT_A, prompt_a, state_cache, replay, fresh=True, checkpoints=checkpoints)

    # Request B: copy the checkpoint and the prefix tokens after it into B's slot. A prefix
    # ending exactly on a chunk boundary copies no tokens, but must still set the count to 0.
    state_cache[SLOT_B].copy_(checkpoints.states[restored // INTERVAL - 1])
    for cache, tokens in zip(replay[:-1], prompt_a, strict=True):
        cache[SLOT_B, :tail].copy_(tokens[0, restored:prefix])
    replay.count[SLOT_B] = tail
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

    skipped = restore_shared_prefix(args.prefix)
    print("suffix output, state, and replay tail match a full prefill bitwise")
    print(f"prefill tokens skipped for request B: {skipped} of {TOKENS}")


if __name__ == "__main__":
    main()
