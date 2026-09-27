"""Scope check: vanilla v1.30 requires a scheduler tensor; AG's None ABI is absent.

No upstream patch is proposed for this AG-specific replay bug.
"""

import torch
from cudnn.linear_attention.frost.common import split_k

print("scheduler:", split_k.__file__, flush=True)
gate = torch.zeros(32, 2, device="cuda")
cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
items = torch.empty(64, split_k.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
staging = torch.empty_like(items)
count = torch.empty(1, dtype=torch.int32, device="cuda")
chunks = torch.zeros(split_k.chunk_scratch_rows(32, 1, 16), 2, device="cuda")
stream = torch.cuda.current_stream().cuda_stream
kwargs = {
    "ideal_chunks": 32,
    "n_tiles": 2,
    "num_sms": 1,
    "b_t": 16,
    "chunk_scratch": chunks,
    "item_scratch": staging,
    "log_gate": True,
    "split": True,
    "opt_level": 2,
    "stream": stream,
}
try:
    split_k.build_split_table(gate, cu, items, count, scheduler_counter=None, **kwargs)
except (TypeError, AttributeError) as error:
    print("None is unsupported at build time:", type(error).__name__, str(error), flush=True)
else:
    raise AssertionError("Upstream API changed: reconsider replay-ABI applicability")
split_k.compiled_cache.clear()
scheduler = torch.zeros(2, dtype=torch.int32, device="cuda")
recipe = split_k.build_split_table(gate, cu, items, count, scheduler_counter=scheduler, **kwargs)
split_k.run_table(recipe, gate, None, None, cu, chunks, staging, items, count, scheduler, stream)
assert count.item() == 2
print("PASS: required-tensor replay works; absent-scheduler ABI is not an upstream feature")
