"""Mid-sequence state checkpoints of the paged GDN and KDA prefill operators."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch

from attn_gym.linear import paged_chunk_gdn, recurrent_gdn
from attn_gym.linear.kda import ReplayState, paged_chunk_kda
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_PINNED = {"autotune": False}
# Slot 0 is the padding slot; sequences use 1..N and checkpoints 16..16+N.
_NUM_SLOTS = 32
_CHECKPOINT_BASE = 16


def _int32(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _strided_pool(
    heads: int, dim: int, dtype: torch.dtype, num_slots: int = _NUM_SLOTS
) -> torch.Tensor:
    """A random pool whose slots are padded, so the kernels see a non-compact layout."""
    elements = heads * dim * dim
    storage = torch.randn(num_slots, elements + 17, device="cuda") * 0.1
    return storage.to(dtype)[:, :elements].view(num_slots, heads, dim, dim)


def _segment(tensors: Sequence[torch.Tensor], lens: Sequence[int], index: int, a: int, b: int):
    start = sum(lens[:index])
    return tuple(t[:, start + a : start + b].contiguous() for t in tensors)


Run = Callable[..., torch.Tensor]


def _assert_checkpoints(
    run: Run,
    inputs: Sequence[torch.Tensor],
    pool: torch.Tensor,
    lens: Sequence[int],
    checkpoint_lens: Sequence[int],
    stored: Sequence[bool],
    has_initial_state: torch.Tensor,
    extra_state: Callable[[], tuple[torch.Tensor, ...]] = tuple,
    checkpoint_base: int = _CHECKPOINT_BASE,
) -> None:
    """Check a call with checkpoints against the same call without them, against calls
    truncated at each checkpoint, and against resuming from each checkpoint."""
    num_sequences = len(lens)
    state_indices = torch.arange(1, num_sequences + 1, device="cuda", dtype=torch.int32)
    checkpoint_indices = torch.arange(
        checkpoint_base, checkpoint_base + num_sequences, device="cuda", dtype=torch.int32
    )
    base_extra = extra_state()

    def clone_state():
        return pool.clone(), tuple(t.clone() for t in base_extra)

    actual_pool, actual_extra = clone_state()
    actual = run(
        inputs, actual_pool, actual_extra, state_indices, lens, has_initial_state,
        checkpoint_indices, _int32(checkpoint_lens),
    )  # fmt: skip
    plain_pool, plain_extra = clone_state()
    plain = run(inputs, plain_pool, plain_extra, state_indices, lens, has_initial_state)
    torch.testing.assert_close(actual, plain, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_pool[1 : num_sequences + 1], plain_pool[1 : num_sequences + 1], rtol=0, atol=0
    )

    for index, (length, checkpoint_len) in enumerate(zip(lens, checkpoint_lens, strict=True)):
        checkpoint = int(checkpoint_indices[index])
        if not stored[index]:
            torch.testing.assert_close(actual_pool[checkpoint], pool[checkpoint], rtol=0, atol=0)
            continue
        truncated_pool, truncated_extra = clone_state()
        run(
            _segment(inputs, lens, index, 0, checkpoint_len), truncated_pool, truncated_extra,
            state_indices[index : index + 1], [checkpoint_len],
            has_initial_state[index : index + 1],
        )  # fmt: skip
        torch.testing.assert_close(
            actual_pool[checkpoint], truncated_pool[index + 1], rtol=0, atol=0
        )
        if actual_extra:
            # Replay state: the count and the tokens it covers, not the stale rows past it.
            count = int(truncated_extra[-1][index + 1].reshape(-1)[0])
            torch.testing.assert_close(
                actual_extra[-1][checkpoint], truncated_extra[-1][index + 1], rtol=0, atol=0
            )
            for actual_tensor, truncated_tensor in zip(
                actual_extra[:-1], truncated_extra[:-1], strict=True
            ):
                torch.testing.assert_close(
                    actual_tensor[checkpoint, :count],
                    truncated_tensor[index + 1, :count],
                    rtol=0,
                    atol=0,
                )
        if checkpoint_len == length or pool.dtype != torch.float32:
            # A BF16 pool rounds the checkpoint, while the full call continues from its FP32
            # registers, so only FP32 pools resume bitwise.
            continue
        resumed_pool = actual_pool.clone()
        resumed_extra = tuple(t.clone() for t in actual_extra)
        resumed = run(
            _segment(inputs, lens, index, checkpoint_len, length), resumed_pool, resumed_extra,
            checkpoint_indices[index : index + 1], [length - checkpoint_len],
            torch.ones(1, device="cuda", dtype=torch.bool),
        )  # fmt: skip
        start = sum(lens[:index])
        torch.testing.assert_close(
            resumed, actual[:, start + checkpoint_len : start + length], rtol=0, atol=0
        )
        torch.testing.assert_close(
            resumed_pool[checkpoint], actual_pool[index + 1], rtol=0, atol=0
        )


def _gdn_runner(recurrent: bool, batch_invariant: bool = False) -> Run:
    def run(inputs, pool, extra, state_indices, lens, has_initial_state, *checkpoints):
        del extra
        q, k, v, gate, beta = inputs
        kwargs = {}
        if checkpoints:
            kwargs = {"checkpoint_indices": checkpoints[0], "checkpoint_lens": checkpoints[1]}
        if recurrent:
            return recurrent_gdn(
                q, k, v, gate, beta, pool, cu_seqlens=cumulative_sequence_offsets(lens),
                state_indices=state_indices, has_initial_state=has_initial_state,
                kernel_options={**_PINNED, "batch_invariant": batch_invariant}, **kwargs,
            )[0]  # fmt: skip
        return paged_chunk_gdn(
            q, k, v, gate, beta, pool, state_indices,
            cu_seqlens=cumulative_sequence_offsets(lens),
            has_initial_state=has_initial_state, kernel_options=_PINNED, **kwargs,
        )  # fmt: skip

    return run


# Lengths with full and partial tail chunks, a one-chunk sequence and an empty request.
_LENS = (200, 64, 330, 129, 70)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_paged_chunk_gdn_checkpoints(head_dim: int, state_dtype: torch.dtype):
    """A checkpoint holds the state of a call truncated there, and resuming from it
    reproduces the rest of the call; entries that are not chunk boundaries store nothing."""
    q, k, v, gate, beta, _state, _cu = make_gdn_test_inputs(
        list(_LENS), key_heads=2, value_heads=4, key_dim=head_dim
    )
    pool = _strided_pool(4, head_dim, state_dtype)
    has_initial_state = torch.tensor([True, False, True, True, True], device="cuda")
    with torch.no_grad():
        _assert_checkpoints(
            _gdn_runner(recurrent=False),
            (q, k, v, gate, beta),
            pool,
            _LENS,
            # 100 is not a chunk boundary and 0 stores nothing.
            checkpoint_lens=(128, 64, 320, 100, 0),
            stored=(True, True, True, False, False),
            has_initial_state=has_initial_state,
        )


def test_paged_chunk_gdn_checkpoints_persistent_schedule():
    """Many short sequences select the persistent recurrence schedule."""
    lens = [64 + 64 * (i % 3) + (i % 5) for i in range(40)]
    q, k, v, gate, beta, _state, _cu = make_gdn_test_inputs(
        lens, key_heads=4, value_heads=8, key_dim=64
    )
    pool = _strided_pool(8, 64, torch.float32, num_slots=96)
    checkpoint_lens = [64 * (length // 128 + 1) for length in lens]
    with torch.no_grad():
        _assert_checkpoints(
            _gdn_runner(recurrent=False),
            (q, k, v, gate, beta),
            pool,
            lens,
            checkpoint_lens=checkpoint_lens,
            stored=[True] * len(lens),
            has_initial_state=torch.ones(len(lens), device="cuda", dtype=torch.bool),
            checkpoint_base=48,
        )


def test_paged_recurrent_gdn_checkpoints():
    """The paged scan stores a checkpoint after any token count within the sequence.

    Batch-invariant mode, since the truncated and resumed calls run other batch shapes, from
    which the scan otherwise picks its configuration.
    """
    q, k, v, gate, beta, _state, _cu = make_gdn_test_inputs(
        list(_LENS), key_heads=2, value_heads=4, key_dim=128
    )
    pool = _strided_pool(4, 128, torch.float32)
    has_initial_state = torch.tensor([True, False, True, True, True], device="cuda")
    with torch.no_grad():
        _assert_checkpoints(
            _gdn_runner(recurrent=True, batch_invariant=True),
            (q, k, v, gate, beta),
            pool,
            _LENS,
            # 71 is past the end of its 70-token sequence.
            checkpoint_lens=(137, 64, 320, 1, 71),
            stored=(True, True, True, True, False),
            has_initial_state=has_initial_state,
        )


def _kda_runner(replay: bool) -> Run:
    def run(inputs, pool, extra, state_indices, lens, has_initial_state, *checkpoints):
        kwargs = {}
        if checkpoints:
            kwargs = {"checkpoint_indices": checkpoints[0], "checkpoint_lens": checkpoints[1]}
        return paged_chunk_kda(
            *inputs, pool, state_indices, cu_seqlens=cumulative_sequence_offsets(lens),
            has_initial_state=has_initial_state, kernel_options=_PINNED,
            replay_state=ReplayState(*extra) if replay else None, **kwargs,
        )  # fmt: skip

    return run


def _replay_state(heads: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.zeros(_NUM_SLOTS, *shape, device="cuda", dtype=dtype)
        for shape, dtype in ReplayState.slot_layout(heads, 128, value_dim=128)
    )


@pytest.mark.parametrize("replay", [False, True])
def test_paged_chunk_kda_checkpoints(replay: bool):
    """Checkpoints at chunk boundaries, with and without replay state."""
    lens = (200, 64, 330)
    inputs = make_kda_test_inputs(sum(lens), heads=4, normalize_qk=True)
    pool = _strided_pool(4, 128, torch.float32)
    replay_state = _replay_state(4) if replay else ()
    with torch.no_grad():
        _assert_checkpoints(
            _kda_runner(replay),
            inputs,
            pool,
            lens,
            checkpoint_lens=(128, 64, 320),
            stored=(True, True, True),
            has_initial_state=torch.tensor([True, False, True], device="cuda"),
            extra_state=lambda: replay_state,
        )


@pytest.mark.parametrize(
    "checkpoint_lens",
    [
        # 58 and 54 new tokens reach the 128- and 64-token boundaries: no replay tail.
        (58, 54),
        # Off the chunk grid: replay tails of 36 and 30 tokens, and in the second
        # sequence 20 tokens before its first boundary (the slot's starting state).
        (94, 20),
    ],
)
def test_paged_chunk_kda_replay_checkpoints(checkpoint_lens: tuple[int, int]):
    """With replayed tokens in the slots, a checkpoint is the logical state: the state
    at its last complete chunk plus the tokens after it as replay state."""
    warm_lens, lens = (70, 10), (200, 130)
    pool = _strided_pool(4, 128, torch.float32)
    replay_state = _replay_state(4)
    state_indices = _int32([1, 2])
    with torch.no_grad():
        # Leave 6 and 10 replayed tokens in slots 1 and 2.
        paged_chunk_kda(
            *make_kda_test_inputs(sum(warm_lens), heads=4, normalize_qk=True, seed=7),
            pool, state_indices, cu_seqlens=cumulative_sequence_offsets(warm_lens),
            has_initial_state=torch.zeros(2, device="cuda", dtype=torch.bool),
            kernel_options=_PINNED, replay_state=ReplayState(*replay_state),
        )  # fmt: skip
        assert replay_state[-1].reshape(_NUM_SLOTS, -1)[1:3, 0].tolist() == [6, 10]
        _assert_checkpoints(
            _kda_runner(replay=True),
            make_kda_test_inputs(sum(lens), heads=4, normalize_qk=True),
            pool,
            lens,
            checkpoint_lens=checkpoint_lens,
            stored=(True, True),
            has_initial_state=torch.ones(2, device="cuda", dtype=torch.bool),
            extra_state=lambda: replay_state,
        )


def test_checkpoint_lens_of_unused_entries_are_ignored():
    """Entries without a checkpoint slot may hold any length, as serving engines leave
    them (SGLang's untracked rows)."""
    lens = (100, 100)
    inputs = make_kda_test_inputs(sum(lens), heads=4, normalize_qk=True)
    replay_state = _replay_state(4)
    pool = _strided_pool(4, 128, torch.float32)
    expected_pool = pool.clone()
    expected_replay = tuple(t.clone() for t in replay_state)
    run = _kda_runner(replay=True)
    state_indices, has_initial_state = (
        _int32([1, 2]),
        torch.ones(2, device="cuda", dtype=torch.bool),
    )
    with torch.no_grad():
        actual = run(
            inputs, pool, replay_state, state_indices, lens, has_initial_state,
            _int32([0, -1]), _int32([-(2**30), 2**30]),
        )  # fmt: skip
        expected = run(
            inputs, expected_pool, expected_replay, state_indices, lens, has_initial_state
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(replay_state, expected_replay, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_checkpoint_argument_validation():
    q, k, v, gate, beta, _state, cu_seqlens = make_gdn_test_inputs([64, 64], key_dim=64)
    pool = torch.zeros(4, 2, 64, 64, device="cuda")
    state_indices = _int32([1, 2])
    with torch.no_grad(), pytest.raises(ValueError, match="passed together"):
        paged_chunk_gdn(
            q, k, v, gate, beta, pool, state_indices, cu_seqlens=cu_seqlens,
            checkpoint_indices=_int32([3, 0]),
        )  # fmt: skip
    with torch.no_grad(), pytest.raises(ValueError, match="checkpoint_lens must be contiguous"):
        paged_chunk_gdn(
            q, k, v, gate, beta, pool, state_indices, cu_seqlens=cu_seqlens,
            checkpoint_indices=_int32([3, 0]), checkpoint_lens=_int32([64]),
        )  # fmt: skip
    with torch.no_grad(), pytest.raises(ValueError, match="requires a paged state pool"):
        recurrent_gdn(
            q, k, v, gate, beta, cu_seqlens=cu_seqlens,
            checkpoint_indices=_int32([3, 0]), checkpoint_lens=_int32([64, 0]),
        )  # fmt: skip
    with torch.no_grad(), pytest.raises(ValueError, match="warpdrive backend"):
        paged_chunk_gdn(
            q, k, v, gate, beta, pool, state_indices, cu_seqlens=cu_seqlens,
            kernel_options={"backend": "warpdrive"},
            checkpoint_indices=_int32([3, 0]), checkpoint_lens=_int32([64, 0]),
        )  # fmt: skip
