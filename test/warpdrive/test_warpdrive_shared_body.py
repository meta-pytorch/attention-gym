"""Test shared-body roles: one traced body for every instance, with the index at run time.

The toy runs three round-robin consumer instances of four warps each against one producer
warp, on an mbarrier ring and a named-barrier ring whose stages and phases wrap several
times. Each instance also records its ``Local`` indices and ``tmem_row``, syncs on its own
``NamedBarriers`` entry, and holds the TMEM allocator warp, so teardown runs from the shared
body. The static form (one body per instance) must produce the same output.
"""

import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support

from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Local,
    NamedBarriers,
    Producer,
    Ring,
    RoleConfig,
    Sharing,
    Slots,
    Threads,
    Tmem,
    TmemPlan,
    Turn,
    Turnstile,
    WarpSpecialized,
    role,
    tracing,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.trace_session import trace_session

RR = Sharing.ROUND_ROBIN
K = 3  # consumer instances
STAGES = 3  # a multiple of K: each instance sees its own stages' previous epochs
REPS = 5  # epochs per instance: 15 epochs wrap the 3-stage rings five times
EPOCHS = K * REPS
THREADS = 128  # per consumer instance


# placement -> (cons instance warps in declared order, prod warp). "reversed" and "gapped"
# take the select paths of the instance index; "reversed" also puts the TMEM allocator warp
# (warp 1) in the last instance.
PLACEMENTS = {
    "ascending": ([range(4), range(4, 8), range(8, 12)], 12),
    "reversed": ([range(8, 12), range(4, 8), range(4)], 12),
    "gapped": ([range(4), range(4, 8), range(12, 16)], 8),
}


class SharedToy(WarpSpecialized):
    """prod fills ``ring`` and ``named`` with ``1000 * epoch + lane``; instance j of ``cons``
    takes epochs j, j + K, ... of both and stores what it read plus its index, then its
    ``Local`` indices and a value exchanged under ``ex[j]``."""

    def __init__(self, shared: bool, placement: str = "ascending", barriers: int = K):
        self.shared, self.placement = shared, placement
        cons, prod = PLACEMENTS[placement]
        self.roles = {
            "cons": RoleConfig(warps=cons, shared_body=shared),
            "prod": RoleConfig(warps=prod),
        }
        self.ring = Ring(STAGES, Threads, Threads, consumer_sharing=RR, payload=(Int32, (32,)))
        self.named = Ring(
            STAGES, Threads, Threads, consumer_sharing=RR, sync="named", payload=(Int32, (32,))
        )
        self.ex = NamedBarriers(barriers, threads=THREADS)
        self.tmem_plan = TmemPlan().region("acc", 32)

    @role(warps=12)  # placed by self.roles
    @cute.jit
    def prod(self, ctx, *, ring: Producer, named: Producer, me: Local):
        for e in cutlass.range(EPOCHS, unroll=1):
            with ring.produce() as h:
                h.data[me.lane] = e * 1000 + me.lane
            with named.produce() as h:
                h.data[me.lane] = e * 1000 + me.lane + 500
        return ring, named

    @role(warps=[range(4)])  # placed by self.roles
    @cute.jit
    def cons(
        self, ctx, *, stage: Instance, ring: Consumer, named: Consumer, tmem: Tmem, me: Local
    ):
        tid = me.tid
        for r in cutlass.range(REPS, unroll=1):
            e = stage + r * K
            with ring.consume() as h:
                value = h.data[tid % 32]
            if tid == 0:
                ctx.out_ring[e] = value + 100 * stage
            with named.consume() as h:
                value = h.data[tid % 32]
            if tid == 0:
                ctx.out_named[e] = value + 100 * stage
        row = stage * THREADS + tid
        ctx.out_local[row, 0] = tid
        ctx.out_local[row, 1] = me.warp
        ctx.out_local[row, 2] = me.tmem_row
        ctx.scratch[row] = 7 * row
        self.ex[stage].sync()  # orders the instance's scratch writes before its reads
        ctx.out_local[row, 3] = ctx.scratch[stage * THREADS + (tid + 1) % THREADS]
        return ring, named

    @cute.kernel
    def kernel(self, out_ring, out_named, out_local, trace):
        smem = self.begin_roles()
        scratch = smem.allocate_tensor(Int32, cute.make_layout(K * THREADS), 16)
        ctx = SimpleNamespace(
            out_ring=out_ring, out_named=out_named, out_local=out_local, scratch=scratch
        )
        self.run_roles(ctx, trace_buffer=trace)

    @cute.jit
    def __call__(self, out_ring: cute.Tensor, out_named: cute.Tensor, out_local, trace, stream):
        self.kernel.set_name_prefix(f"warpdrive_shared_toy_{int(self.shared)}_{self.placement}")
        self.kernel(out_ring, out_named, out_local, trace).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_toy(op):
    ring = make_fake_compact_tensor(Int32, (EPOCHS,), assumed_align=4)
    named = make_fake_compact_tensor(Int32, (EPOCHS,), assumed_align=4)
    local = make_fake_compact_tensor(Int32, (K * THREADS, 4), assumed_align=4)
    trace = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),))
    name = f"shared_toy_{int(op.shared)}_{op.placement}"
    return compile_tvm_ffi(op, ring, named, local, trace, name=name)


def _outputs():
    out_ring = torch.full((EPOCHS,), -1, dtype=torch.int32, device="cuda")
    out_local = torch.full((K * THREADS, 4), -1, dtype=torch.int32, device="cuda")
    return out_ring, torch.full_like(out_ring, -1), out_local


def _expected():
    e = torch.arange(EPOCHS, dtype=torch.int32)
    owner = e % K
    row = torch.arange(K * THREADS, dtype=torch.int32)
    tid = row % THREADS
    neighbor = (row - tid) + (tid + 1) % THREADS
    local = torch.stack([tid, tid // 32, tid, 7 * neighbor], dim=1)
    return e * 1000 + 100 * owner, e * 1000 + 500 + 100 * owner, local


@pytest.mark.parametrize("placement", list(PLACEMENTS))
@pytest.mark.parametrize("shared", [True, False])
def test_shared_body_matches_static_instances(requires_blackwell, shared, placement):
    """Round-robin turns, named-channel seeding, Local indices, an instance-indexed named
    barrier and TMEM teardown give the same result from one body as from three, for any
    placement of the declared instances; repeated launches show TMEM is freed."""
    op = SharedToy(shared, placement)
    fn = compile_toy(op)
    ring_ref, named_ref, local_ref = _expected()
    unused = torch.zeros(1, dtype=torch.int64, device="cuda")
    for _ in range(3):
        out_ring, out_named, out_local = _outputs()
        fn(out_ring, out_named, out_local, unused)
        torch.cuda.synchronize()
        assert torch.equal(out_ring.cpu(), ring_ref)
        assert torch.equal(out_named.cpu(), named_ref)
        assert torch.equal(out_local.cpu(), local_ref)


def test_shared_body_compiles_one_body(requires_blackwell, tmp_path):
    """The shared form compiles the consumer body once, so its ring wait sites appear once
    instead of once per instance, while describe() and the checks still name every instance."""
    disassembler = shutil.which("nvdisasm")
    if disassembler is None:
        pytest.skip("the SASS check requires nvdisasm")
    waits = {}
    for shared in (True, False):
        op = SharedToy(shared)
        compile_toy(op)
        text = op.describe()
        assert all(f"cons[{j}]" in text for j in range(K))
        users = op._ws_named_users
        assert [users[f"ex[{j}]"]["sync"] for j in range(K)] == [[f"cons[{j}]"] for j in range(K)]
        ring = make_fake_compact_tensor(Int32, (EPOCHS,), assumed_align=4)
        local = make_fake_compact_tensor(Int32, (K * THREADS, 4), assumed_align=4)
        trace = make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),))
        compiled = cute.compile(SharedToy(shared), ring, ring, local, trace, cuda.CUstream(0))
        obj = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        cubin = tmp_path / f"toy{int(shared)}.cubin"
        cubin.write_bytes(obj[[m.start() for m in re.finditer(b"\x7fELF", obj)][1] :])
        sass = subprocess.check_output([disassembler, str(cubin)], text=True)
        waits[shared] = sass.count("SYNCS.PHASECHK.TRANS64.TRYWAIT")
    assert waits[True] < waits[False]


@pytest.mark.parametrize("placement", ["ascending", "gapped"])
def test_shared_body_globaltimer_ranges_per_instance(requires_blackwell, placement):
    """Each instance records its own role range and wait ranges from its first warp."""
    op = SharedToy(True, placement).enable_trace("globaltimer")
    fn = compile_toy(op)
    out_ring, out_named, out_local = _outputs()
    with trace_session(op, 1) as session:
        fn(out_ring, out_named, out_local, session.prof.tensor)
    assert torch.equal(out_ring.cpu(), _expected()[0])
    by_name = {}
    for r in session.ranges:
        by_name.setdefault(r.name, []).append(r.warp)
    firsts = [r.start for r in PLACEMENTS[placement][0]]
    assert [by_name[f"cons[{j}]"] for j in range(K)] == [[w] for w in firsts]
    assert sorted(by_name["ring.wait"]) == sorted(firsts * REPS)
    assert "cons[*]" not in by_name


def test_shared_body_iket_ranges_per_instance(monkeypatch, traced_compiles):
    """IKET names are static strings: the shared body pushes each instance's name under its
    own branch."""
    pushed = []
    real = tracing.iket

    class Recorder:
        def range_push(self, name, *payload):
            pushed.append(name)
            return real.range_push(name, *payload)

        def range_pop(self):
            return real.range_pop()

    monkeypatch.setattr(tracing, "iket", Recorder())
    compile_toy.__wrapped__(SharedToy(True).enable_iket())
    assert {f"cons[{j}]" for j in range(K)} <= set(pushed)
    assert "cons[*]" not in pushed


class _Bad(WarpSpecialized):
    def __init__(self, warps, **_):
        self.roles = {"cons": RoleConfig(warps=warps, shared_body=True)}
        self.ring = Ring(2, Threads, Threads, consumer_sharing=RR)

    @role(warps=4)
    @cute.jit
    def prod(self, ctx, *, ring: Producer):
        return ring

    @role(warps=[range(1)])
    @cute.jit
    def cons(self, ctx, *, ring: Consumer):
        return ring


class _BadSlot(_Bad):
    def __init__(self):
        self.roles = {"cons": RoleConfig(warps=[range(1), range(1, 2)], shared_body=True)}
        self.slots = Slots(2, Threads, Threads)

    @role(warps=4)
    @cute.jit
    def prod(self, ctx, *, slots: Producer):
        return slots

    @role(warps=[range(1)])
    @cute.jit
    def cons(self, ctx, *, slots: Consumer[Instance]):
        return slots


class _BadTurn(_Bad):
    def __init__(self):
        self.roles = {"cons": RoleConfig(warps=[range(1), range(1, 2)], shared_body=True)}
        self.ring = Ring(2, Threads, Threads, consumer_sharing=RR)
        self.ts = Turnstile()

    @role(warps=[range(1)])
    @cute.jit
    def cons(self, ctx, *, ring: Consumer, ts: Turn):
        return ring, ts


class _BadBudget(_Bad):
    """cons[0] and cons[1] sit in warpgroups 0 and 2; prod's budget splits them."""

    def __init__(self):
        self.roles = {
            "cons": RoleConfig(warps=[range(4), range(8, 12)], regs=160, shared_body=True)
        }
        self.ring = Ring(2, Threads, Threads, consumer_sharing=RR)

    @role(warps=range(4, 8), regs=56)
    @cute.jit
    def prod(self, ctx, *, ring: Producer):
        return ring


@pytest.mark.parametrize(
    "make, error, kind",
    [
        (lambda: _Bad([range(1), range(1, 3)]), "equal warp counts", ValueError),
        (_BadSlot, "slot-bound slots", None),
        (_BadTurn, "turnstile seat ts", None),
        (_BadBudget, "different register-budget dispatch branches", ValueError),
    ],
)
def test_shared_body_rejections(make, error, kind):
    if kind is None:
        _raises_protocol(error, lambda: make().describe())
    else:
        with pytest.raises(kind, match=error):
            make().describe()


def test_shared_body_flag_is_keyed_and_validated():
    with pytest.raises(ValueError, match="shared_body must be a bool"):
        RoleConfig(warps=1, shared_body=1)
    assert SharedToy(True).role_config_key() != SharedToy(False).role_config_key()


def test_named_barriers_indexed_by_instance_need_one_per_instance():
    _raises_protocol(
        r"NamedBarriers ex has 2 barriers, but the shared body indexing it by Instance runs 3",
        lambda: compile_toy.__wrapped__(SharedToy(True, barriers=2)),
    )


def test_named_barriers_declare_consecutive_barriers():
    op = SharedToy(True)
    op.describe()
    ids = [op.ex[j].id for j in range(K)]
    assert ids == list(range(ids[0], ids[0] + K))
    assert [op.ex[j].name for j in range(K)] == [f"ex[{j}]" for j in range(K)]
    with pytest.raises(ValueError, match="positive int"):
        NamedBarriers(0, threads=32)
