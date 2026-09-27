"""Compare scalar split scans of identical compact and stride-two gates."""

import argparse

import cutlass
import torch
from cuda.bindings import driver as cuda
from cudnn.linear_attention.frost.common import split_k
from cutlass import cute
from cutlass.cute.runtime import from_dlpack, make_fake_tensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw",
    action="store_true",
    help="isolate device address arithmetic using an explicit dynamic gate ABI",
)
args = parser.parse_args()

print("scheduler:", split_k.__file__, flush=True)
tokens, heads = 4096, 2
storage = torch.full((tokens, heads * 2), 37.0, device="cuda")
gate = storage[:, ::2]
gate[:, 0], gate[:, 1] = -0.1, -0.2
cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
scans = []
for source in (gate.contiguous(), gate):
    items = torch.empty(256, split_k.WORK_ITEM_FIELDS, dtype=torch.int32, device="cuda")
    chunks = torch.zeros(split_k.chunk_scratch_rows(tokens, 1, 16), heads, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    staging = torch.empty_like(items)
    scheduler = torch.zeros(2, dtype=torch.int32, device="cuda")
    sms = torch.cuda.get_device_properties(source.device).multi_processor_count
    stream = torch.cuda.current_stream().cuda_stream
    if args.raw:
        facts = split_k.split_table_facts(
            source,
            cu,
            split=True,
            n_tiles=heads,
            ideal_chunks=32,
            num_sms=sms,
            b_t=16,
            log2_threshold=None,
            log_gate=True,
            safe_gate=False,
            gate_lower_bound=None,
            expand_num=1,
        )
        runtime = (
            heads,
            heads,
            32,
            1,
            facts.log2_threshold,
            0.0,
            source,
            None,
            None,
            cu,
            chunks,
            staging,
            items,
            count,
            scheduler,
            facts.n_scan_ctas,
            facts.n_scan_blocks,
            facts.n_walk_ctas,
            cuda.CUstream(stream),
        )
        tensor_args = [
            from_dlpack(x, assumed_align=4).mark_layout_dynamic()
            for x in (cu, chunks, staging, items, count, scheduler)
        ]
        compiled = cute.compile(
            split_k.launch,
            True,
            16,
            facts.scan_rows,
            True,
            False,
            0,
            facts.overhead_chunks,
            1,
            facts.warmup_cap,
            facts.full_scan,
            cutlass.Int32(heads),
            sms,
            cutlass.Int32(heads),
            cutlass.Int32(32),
            cutlass.Int32(1),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Float32(0.0),
            make_fake_tensor(
                cutlass.Float32,
                (cute.sym_int(), cute.sym_int()),
                (cute.sym_int(), cute.sym_int()),
                assumed_align=4,
            ),
            None,
            None,
            *tensor_args,
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_scan_blocks),
            cutlass.Int32(facts.n_walk_ctas),
            cuda.CUstream(stream),
            options="--enable-tvm-ffi --opt-level 2",
        )
        compiled(*runtime)
    else:
        split_k.build_split_table(
            source,
            cu,
            items,
            count,
            ideal_chunks=32,
            n_tiles=heads,
            num_sms=sms,
            b_t=16,
            chunk_scratch=chunks,
            item_scratch=staging,
            log_gate=True,
            scheduler_counter=scheduler,
            split=True,
            opt_level=2,
            stream=stream,
        )
    torch.cuda.synchronize()
    scans.append(chunks)
assert torch.count_nonzero(scans[0][:-1]).item() > 0
print("max scan error:", (scans[1] - scans[0]).abs().max().item(), flush=True)
torch.testing.assert_close(scans[1], scans[0], atol=0, rtol=0)
print("PASS")
