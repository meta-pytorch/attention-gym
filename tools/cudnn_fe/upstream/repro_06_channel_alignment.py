"""Reject inputs violating the upstream channel scan's vector-load alignment.

This checks the pre-dispatch contract, not a claimed deterministic device fault.
"""

import torch
from cudnn.linear_attention.frost.common import split_k

print("scheduler:", split_k.__file__, flush=True)
cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    gates = {
        "head stride": torch.empty(32, 2, 129, dtype=dtype, device="cuda")[:, :, :128],
        "row stride": torch.empty(32, 257, dtype=dtype, device="cuda")[:, :256].view(32, 2, 128),
        "base offset": torch.empty(32 * 256 + 2, dtype=dtype, device="cuda")[2:].view(32, 2, 128),
        "aligned": torch.empty(32, 2, 128, dtype=dtype, device="cuda"),
    }
    for name, gate in gates.items():
        rejected = False
        try:
            split_k.split_table_facts(
                gate,
                cu,
                split=True,
                n_tiles=2,
                ideal_chunks=32,
                num_sms=1,
                b_t=16,
                log2_threshold=None,
                log_gate=True,
                safe_gate=False,
                gate_lower_bound=None,
                expand_num=1,
            )
        except ValueError as error:
            assert "aligned" in str(error), error
            rejected = True
        print(dtype, name, "rejected=" + str(rejected), flush=True)
        assert rejected == (name != "aligned"), (dtype, name)
print("PASS")
