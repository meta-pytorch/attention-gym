"""v1.30 emits zero-chunk split work. Patched no-state callers opt into omission.

Run the original without arguments; run the patched package with --skip-empty.
Uses the upstream raw scheduler because the graph API does not expose work tables.
"""

import argparse

import torch
from cudnn.linear_attention.frost.common import split_k

parser = argparse.ArgumentParser()
parser.add_argument("--skip-empty", action="store_true")
args = parser.parse_args()
print("scheduler:", split_k.__file__, flush=True)
for bounds in ([0, 32, 32, 48, 48], [0, 0, 0, 0]):
    heads, sequences = 2, len(bounds) - 1
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
    kwargs = {"skip_empty": True} if args.skip_empty else {}
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
        scheduler_counter=torch.zeros(2, dtype=torch.int32, device="cuda"),
        split=True,
        opt_level=2,
        stream=torch.cuda.current_stream().cuda_stream,
        **kwargs,
    )
    expected = [i for i in range(sequences) if bounds[i] < bounds[i + 1]]
    actual = count.item()
    print(f"bounds={bounds}, work_items={actual}, expected={len(expected) * heads}", flush=True)
    assert actual == len(expected) * heads
    rows = staging[:actual].cpu()
    assert sorted(rows[:, 0].tolist()) == sorted(expected * heads)
    assert bool(torch.all(rows[:, 5] > rows[:, 4]))
print("PASS")
