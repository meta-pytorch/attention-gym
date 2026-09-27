"""Direct contracts of the v1.30 shared scheduling helpers."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.linear._delta_rule.cudnn_fe.common import split_k

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="cuDNN v1.30 helpers require Blackwell",
)


@pytest.mark.parametrize("bounds", [[0, 32, 32, 48, 48], [0, 0, 0, 0]])
def test_split_table_omits_zero_chunk_sequences(bounds):
    """Interior, trailing and all-empty intervals must not enter the persistent work queue."""
    heads = 2
    sequences = len(bounds) - 1
    gate = torch.zeros(bounds[-1], heads, device="cuda")
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    items = torch.empty(
        sequences * heads * 32, split_k.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda"
    )
    staging = torch.empty_like(items)
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    chunks = torch.zeros(
        split_k.chunk_scratch_rows(bounds[-1], sequences, 16), heads, device="cuda"
    )
    sched = torch.zeros(2, dtype=torch.int32, device="cuda")
    split_k.build_split_table(
        gate,
        cu,
        items,
        count,
        ideal_chunks=32,
        n_tiles=sequences * heads,
        num_sms=torch.cuda.get_device_properties(gate.device).multi_processor_count,
        b_t=16,
        chunk_scratch=chunks,
        item_scratch=staging,
        log_gate=True,
        scheduler_counter=sched,
        split=True,
        opt_level=2,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    expected_sequences = [i for i in range(sequences) if bounds[i] < bounds[i + 1]]
    assert count.item() == len(expected_sequences) * heads
    rows = staging[: count.item()].cpu()
    assert sorted(rows[:, 0].tolist()) == sorted(expected_sequences * heads)
    assert torch.all(rows[:, 5] > rows[:, 4])


def test_scalar_split_scan_respects_head_stride():
    """Inactive interleaved heads must not contribute to the forgetting-horizon scan."""
    tokens, heads = 4096, 2
    storage = torch.full((tokens, heads * 2), 37.0, device="cuda")
    gate = storage[:, ::2]
    gate[:, 0] = -0.1
    gate[:, 1] = -0.2
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    scans = []
    for source in (gate.contiguous(), gate):
        items = torch.empty(256, split_k.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
        chunks = torch.zeros(split_k.chunk_scratch_rows(tokens, 1, 16), heads, device="cuda")
        split_k.build_split_table(
            source,
            cu,
            items,
            torch.empty(1, dtype=torch.int32, device="cuda"),
            ideal_chunks=32,
            n_tiles=heads,
            num_sms=torch.cuda.get_device_properties(source.device).multi_processor_count,
            b_t=16,
            chunk_scratch=chunks,
            item_scratch=torch.empty_like(items),
            log_gate=True,
            scheduler_counter=torch.zeros(2, dtype=torch.int32, device="cuda"),
            split=True,
            opt_level=2,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        scans.append(chunks)
    assert torch.count_nonzero(scans[0][:-1]) > 0
    torch.testing.assert_close(scans[1], scans[0], atol=0, rtol=0)


def test_split_table_replay_preserves_absent_scheduler_abi():
    """A caller's unrelated counter must not become a tensor in a compiled None slot."""
    gate = torch.zeros(32, 2, device="cuda")
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    items = torch.empty(64, split_k.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
    staging = torch.empty_like(items)
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    chunks = torch.zeros(split_k.chunk_scratch_rows(32, 1, 16), 2, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    recipe = split_k.build_split_table(
        gate,
        cu,
        items,
        count,
        ideal_chunks=32,
        n_tiles=2,
        num_sms=torch.cuda.get_device_properties(gate.device).multi_processor_count,
        b_t=16,
        chunk_scratch=chunks,
        item_scratch=staging,
        log_gate=True,
        scheduler_counter=None,
        split=True,
        opt_level=2,
        stream=stream,
    )
    unrelated_counter = torch.full((2,), 37, dtype=torch.int32, device="cuda")
    split_k.run_table(
        recipe, gate, None, None, cu, chunks, staging, items, count, unrelated_counter, stream
    )
    assert count.item() == 2
    assert unrelated_counter.tolist() == [37, 37]


@pytest.mark.parametrize("invalid", ["not_float16", "float32_extra", "torch.bfloat16_suffix"])
def test_cudnn_dtype_names_are_exact(invalid):
    """Dtype names match exactly; substrings of a supported name are rejected."""
    import cutlass

    from attn_gym.linear._delta_rule.cudnn_fe.common.host import get_dtype

    assert get_dtype(torch.float16) is cutlass.Float16
    assert get_dtype("half") is cutlass.Float16
    with pytest.raises(ValueError, match="Unsupported dtype"):
        get_dtype(invalid)
