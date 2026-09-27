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
