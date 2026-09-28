"""C08: compact uncut no-state work; preserve empty state/cotangent writes.

Run with a python whose env has nvidia-cudnn-frontend installed:
    python repro_03_empty_unsplit.py [/path/to/export]
The optional export (a directory holding python/cudnn, e.g. a patched source checkout) supplies the
pure-Python kernels; the installed cudnn supplies its extension. Default: the installed package.
Vanilla v1.30 must fail the no-state work-count checks, not state-value checks.
This tests raw upstream helpers/launchers, not Attention Gym adapters. No timings.
"""

import argparse
import importlib
import inspect
from pathlib import Path

import cudnn
import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, nargs="?")
args = parser.parse_args()
if args.source is None:
    source = Path(cudnn.__path__[0])
else:
    source = args.source.resolve() / "python/cudnn"
    assert source.is_dir(), source
    cudnn.__path__.insert(0, str(source))
from cudnn.linear_attention.frost.common import split_k

assert Path(split_k.__file__).is_relative_to(source), split_k.__file__
SUPPORTS_COMPACTION = "skip_empty" in inspect.signature(split_k.order_body).parameters


@cute.kernel
def order_kernel(
    cu,
    count,
    items,
    sched,
    heads: cutlass.Int32,
    skip_empty: cutlass.Constexpr[bool],
    ct_as: cutlass.Constexpr[int],
    expand: cutlass.Constexpr[int],
):
    keys = cutlass.Array(
        cutlass.Int32, split_k.ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
    )
    indices = cutlass.Array(
        cutlass.Int32, split_k.ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
    )
    spread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
    if cutlass.const_expr(SUPPORTS_COMPACTION):
        split_k.order_body(
            True,
            16,
            split_k.ORDER_THREADS,
            split_k.ORDER_ELEMENTS,
            cutlass.Int32(cute.arch.thread_idx()[0]),
            heads,
            heads * (cu.shape[0] - 1),
            cu,
            None,
            count,
            items,
            sched,
            keys,
            indices,
            spread,
            num_ctas=ct_as,
            expand_num=expand,
            skip_empty=skip_empty,
        )
    else:
        split_k.order_body(
            True,
            16,
            split_k.ORDER_THREADS,
            split_k.ORDER_ELEMENTS,
            cutlass.Int32(cute.arch.thread_idx()[0]),
            heads,
            heads * (cu.shape[0] - 1),
            cu,
            None,
            count,
            items,
            sched,
            keys,
            indices,
            spread,
            num_ctas=ct_as,
            expand_num=expand,
        )


order_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def order_launch(
    cu,
    count,
    items,
    sched,
    heads: cutlass.Int32,
    skip_empty: cutlass.Constexpr[bool],
    ct_as: cutlass.Constexpr[int],
    expand: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    order_kernel(cu, count, items, sched, heads, skip_empty, ct_as, expand).launch(
        grid=(1, 1, 1), block=(split_k.ORDER_THREADS, 1, 1), stream=stream
    )


def cu_tensor(lengths):
    return torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0).int()


def order_check(lengths, heads=2, ct_as=0, expand=1, skip=True):
    cu = cu_tensor(lengths)
    count = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    items = torch.full(
        (max(1, len(lengths) * heads), split_k.WORK_ITEM_FIELDS),
        -99,
        device="cuda",
        dtype=torch.int32,
    )
    sched = torch.full((16,), -1, device="cuda", dtype=torch.int32)
    tensors = [from_dlpack(x, assumed_align=4) for x in (cu, count, items, sched)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        order_launch,
        *tensors,
        cutlass.Int32(heads),
        skip,
        ct_as,
        expand,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )
    compiled(cu, count, items, sched, heads, stream)
    actual_count = count.item()
    retained = [
        b
        for b, length in enumerate(lengths)
        if length or not skip or len(lengths) > split_k.ORDER_CAPACITY
    ]
    assert actual_count == len(retained) * heads, (actual_count, len(retained) * heads)
    rows = items[:actual_count].cpu().tolist()
    expected = []
    start = 0
    for b, length in enumerate(lengths):
        end = start + length * expand
        chunks = (length * expand + 15) // 16
        if b in retained:
            expected.extend([b, h, 0, chunks, 0, chunks, start, end, b, b] for h in range(heads))
        start = end
    assert sorted(rows) == sorted(expected), "original sequence/head IDs or destinations lost"
    assert torch.equal(sched, torch.zeros_like(sched))


def family_module(family, suffix):
    module = importlib.import_module(f"cudnn.linear_attention.frost.kernel.{family}_{suffix}_f16")
    assert Path(module.__file__).is_relative_to(source), module.__file__
    return module


def invoke(fn, values, *positional):
    names = inspect.signature(fn).parameters
    return fn(*positional, **{key: value for key, value in values.items() if key in names})


def forward(family, lengths, state_kind, q, k, v, gate, beta):
    module = family_module(family, "warmup_forward")
    batch, heads, dim = len(lengths), q.shape[1], q.shape[2]
    state_in = state_out = indices = None
    if state_kind in ("seeded", "indexed"):
        state_in = torch.randn((batch, heads, dim, dim), device="cuda", dtype=torch.float32)
    if state_kind != "none":
        state_out = torch.full((batch, heads, dim, dim), float("nan"), device="cuda")
    if state_kind == "indexed":
        indices = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    cu = cu_tensor(lengths)
    values = {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "a_log": None,
        "dt_bias": None,
        "o": torch.full_like(v, float("nan")),
        "cu_seqlens": cu,
        "state_in": state_in,
        "state_out": state_out,
        "seed_indices": indices,
        "final_indices": indices,
        "checkpoints": None,
        "work_items": torch.empty(
            (max(1, batch * heads), split_k.WORK_ITEM_FIELDS), device="cuda", dtype=torch.int32
        ),
        "work_count": torch.full((1,), -1, device="cuda", dtype=torch.int32),
        "item_scratch": None,
        "chunk_scratch": None,
        "scheduler": torch.empty((64,), device="cuda", dtype=torch.int32),
        "workspace": torch.full((6 * batch * 16,), -1, device="cuda", dtype=torch.int64),
        "split": False,
        "n_tiles": batch * heads,
        "ideal_chunks": None,
        "num_sm": torch.cuda.get_device_properties(0).multi_processor_count,
        "b_t": 16 if family == "kda" else 64,
        "log_gate": True,
        "safe_gate": False,
        "gate_lower_bound": -5.0,
        "use_qk_l2norm": False,
        "use_beta_sigmoid": False,
        "allow_neg_eigval": False,
        "expand_num": 1,
        "checkpoint_every_n_tokens": 0,
        "scale": dim**-0.5,
        "device": 0,
        "stream": torch.cuda.current_stream().cuda_stream,
    }
    compiled, facts = invoke(module.build_warmup_forward, values)
    invoke(module.run_warmup_forward, values, compiled, facts)
    torch.cuda.synchronize()
    if state_out is not None:
        empty = torch.tensor(
            [b for b, length in enumerate(lengths) if length == 0],
            device="cuda",
            dtype=torch.int64,
        )
        if indices is not None:
            empty = indices[empty].long()
        expected = torch.zeros_like(state_out[empty]) if state_in is None else state_in[empty]
        torch.testing.assert_close(state_out[empty], expected, rtol=0, atol=0)
        assert values["work_count"].item() == batch * heads, "state work was dropped"
    return values["o"], values["work_count"].item()


def forward_check(family, state_kind):
    lengths, heads, dim = [16, 0, 32, 0], 2, 128
    torch.manual_seed(308)
    q, k, v = [
        torch.randn((48, heads, dim), device="cuda", dtype=torch.bfloat16) * 0.05 for _ in range(3)
    ]
    gate_shape = q.shape if family == "kda" else q.shape[:2]
    gate = torch.full(gate_shape, -0.05, device="cuda", dtype=torch.float32)
    beta = torch.full(q.shape[:2], 0.25, device="cuda", dtype=torch.float32)
    output, count = forward(family, lengths, state_kind, q, k, v, gate, beta)
    assert torch.isfinite(output).all(), "nonfinite active output"
    if state_kind == "none":
        compact, compact_count = forward(family, [16, 32], state_kind, q, k, v, gate, beta)
        torch.testing.assert_close(output, compact, rtol=0, atol=0)
        assert count == compact_count == 4, (count, compact_count)


def empty_backward_check(family, exit_cotangent):
    # Real raw backward kernels consume empty items: no checkpoint/TMA data is read.
    batch, heads, dim, tokens = 3, 2, 128, 16
    q = torch.zeros((tokens, heads, dim), device="cuda", dtype=torch.bfloat16)
    gate_shape = q.shape if family == "kda" else q.shape[:2]
    gate = torch.zeros(gate_shape, device="cuda", dtype=torch.float32)
    beta = torch.full(q.shape[:2], 0.25, device="cuda", dtype=torch.float32)
    dstate = torch.randn((batch, heads, dim, dim), device="cuda") if exit_cotangent else None
    result = torch.full((batch, heads, dim, dim), float("nan"), device="cuda")
    count = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    items = torch.empty(
        (batch * heads, split_k.WORK_ITEM_FIELDS), device="cuda", dtype=torch.int32
    )
    sched = torch.empty((64,), device="cuda", dtype=torch.int32)
    words = torch.full((10 * batch * 16,), -1, device="cuda", dtype=torch.int64)
    checkpoints = torch.zeros((1, heads, dim, dim), device="cuda", dtype=torch.bfloat16)
    values = {
        "q": q,
        "k": q,
        "v": q,
        "gate": gate,
        "beta": beta,
        "do": q,
        "dq": torch.empty_like(q),
        "dk": torch.empty_like(q),
        "dv": torch.empty_like(q),
        "dgate": torch.empty_like(gate),
        "dbeta": torch.empty_like(beta),
        "cu_seqlens": cu_tensor([0] * batch),
        "scale": dim**-0.5,
        "use_initial_state": True,
        "d_initial_state": result,
        "d_final_state": dstate,
        "work_items": items,
        "work_count": count,
        "scheduler_counter": sched,
        "scheduler_all": sched,
        "order_in_prologue": True,
        "log_gate": True,
        "workspace": words,
        "state_checkpoints": checkpoints,
        "device": 0,
        "num_sm": torch.cuda.get_device_properties(0).multi_processor_count,
        "stream": torch.cuda.current_stream().cuda_stream,
    }
    if family == "gdn":
        module = family_module(family, "bprop")
        invoke(module.chunk_gdn_bwd, values)
    else:
        module = family_module(family, "warmup_backward")
        values.update(
            a_log=None,
            dt_bias=None,
            checkpoints=checkpoints,
            seed_checkpoints=None,
            state_in=torch.zeros_like(result),
            dstate0=result,
            dstate_in=dstate,
            series_items=None,
            series_count=None,
            item_scratch=None,
            chunk_scratch=None,
            scheduler_recompute=sched,
            scheduler_bwd=sched,
            recompute_words=None,
            bprop_words=words,
            split=False,
            n_tiles=batch * heads,
            ideal_chunks=None,
            b_t=16,
            recompute=False,
            recompute_orders=False,
            coarse=False,
            bwd_orders=True,
            seed_span_tokens=0,
            seed_every_n_tokens=16,
            safe_gate=False,
            gate_lower_bound=-5.0,
            use_qk_l2norm=False,
            use_beta_sigmoid=False,
            allow_neg_eigval=False,
        )
        compiled, facts = invoke(module.build_warmup_backward, values)
        invoke(module.run_warmup_backward, values, compiled, facts)
    torch.cuda.synchronize()
    expected = torch.zeros_like(result) if dstate is None else dstate
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert count.item() == batch * heads, "backward state work was dropped"


def main():
    import importlib.metadata

    print("source:", source, flush=True)
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    print(
        "torch:",
        torch.__version__,
        "CuTeDSL:",
        importlib.metadata.version("nvidia-cutlass-dsl"),
        "TVM-FFI:",
        importlib.metadata.version("apache-tvm-ffi"),
        flush=True,
    )
    cases = [
        ("sorted IDs and expansion", lambda: order_check([0, 17, 0, 49, 0], expand=2)),
        ("uniform keys", lambda: order_check([16, 0, 16, 0])),
        ("one item per CTA", lambda: order_check([16, 0, 32, 0], ct_as=132)),
        ("multiple scan rounds", lambda: order_check([16] + [0] * 1065 + [32])),
        ("compacted items exceed capacity", lambda: order_check([16] * 2200 + [0], heads=2)),
        ("sequence capacity fallback", lambda: order_check([16] + [0] * 4096)),
        ("all empty", lambda: order_check([0, 0, 0])),
        ("default retains state work", lambda: order_check([16, 0, 16, 0], skip=False)),
    ]
    for family in ("gdn", "kda"):
        for kind in ("none", "zero", "seeded", "indexed"):
            cases.append(
                (f"{family} forward {kind}", lambda f=family, k=kind: forward_check(f, k))
            )
        for exit_cotangent in (False, True):
            cases.append(
                (
                    f"{family} backward exit={exit_cotangent}",
                    lambda f=family, e=exit_cotangent: empty_backward_check(f, e),
                )
            )
    failures = []
    for name, test in cases:
        print("RUN", name, flush=True)
        try:
            test()
        except AssertionError as error:
            failures.append(name)
            print("FAIL", name, str(error), flush=True)
        else:
            print("PASS", name, flush=True)
    print(
        f"RESULT {len(cases) - len(failures)} passed, {len(failures)} failed: {failures}",
        flush=True,
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
