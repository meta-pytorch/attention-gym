"""tcgen05 thread-sync fences: the PTX checker, and the fences the layer emits.

PTX orders tcgen05 operations of different threads through a synchronization only with
``tcgen05.fence::before_thread_sync`` before the signal and ``::after_thread_sync`` after the
wait (ISA 9.7.18.6.3). The checker (``ptx_check``) is exercised on hand-written PTX, then run on
the PTX of compiled kernels: the layer fences every managed wait and signal of a role that
issues tcgen05 operations, except where a call skips a fence, and the TMEM teardown.
"""

import re
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.arch import inline_ptx

from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Local,
    NamedBarrier,
    Producer,
    Ring,
    Threads,
    Tmem,
    TmemPlan,
    Warps,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.named import TMEM_TEARDOWN_BARRIER_ID
from attn_gym.warpdrive.protocol import fence_skip
from attn_gym.warpdrive.ptx_check import capture_ptx, check_tcgen05_sync, read_ptx

# ---- the checker on hand-written PTX ----------------------------------------------------


def _entry(body: str) -> str:
    return f".visible .entry k(\n\t.param .u64 p\n)\n{{\n\t.reg .b32 %r<9>;\n{body}\n\tret;\n}}\n"


WAIT = """\t{
\t.reg .pred P1;
\tLAB_WAIT:
\tmbarrier.try_wait.parity.shared.b64 P1, [%r1], %r2, 10000000;
\t@P1 bra.uni DONE;
\tbra.uni LAB_WAIT;
\tDONE:
\t}"""
LD = "\ttcgen05.ld.sync.aligned.32x32b.x2.b32 {%r3, %r4}, [%r5];"
ARRIVE = "\tmbarrier.arrive.shared.b64 %rd1, [%r1];"
AFTER = "\ttcgen05.fence::after_thread_sync;"
BEFORE = "\ttcgen05.fence::before_thread_sync;"
MMA = "\t{ .reg .pred p; setp.ne.b32 p, %r6, 0; tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd2, %rd3, %r4, p; }"
COMMIT = "\ttcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r1];"


def _rules(*lines: str, **kwargs) -> list[str]:
    return [f.rule for f in check_tcgen05_sync(_entry("\n".join(lines)), **kwargs)]


@pytest.mark.parametrize(
    "lines, rules",
    [
        ((WAIT, AFTER, LD, "\ttcgen05.wait::ld.sync.aligned;", BEFORE, ARRIVE), []),
        ((WAIT, LD), ["after"]),  # 9.7.18.6.4.4: the waiter fences before its tcgen05.ld
        ((LD, ARRIVE), ["before"]),  # the signaller fences after its tcgen05.ld
        ((LD, "\ttcgen05.wait::ld.sync.aligned;", ARRIVE), ["before"]),  # completion is not order
        ((MMA, COMMIT, ARRIVE), []),  # tcgen05.commit fences the MMAs it tracks
        ((LD, COMMIT, ARRIVE), ["before"]),  # but not tcgen05.ld/st
        ((WAIT, AFTER, MMA), []),
        ((WAIT, MMA), ["after"]),
        ((LD, BEFORE, "\tbar.warp.sync -1;", ARRIVE), []),  # the Warps signal
        ((LD, "\tbarrier.sync 3, 128;"), ["before"]),  # a barrier sync is a signal
        (("\tbar.sync 3, 128;", LD), ["after"]),  # and a wait
        (("\tbar.arrive 3, 128;", LD), []),  # an arrive alone is not a wait
        ((LD, "\tst.release.gpu.global.b32 [%rd1], %r1;"), ["before"]),
        (
            ("\tbarrier.sync 2, 128;", "\ttcgen05.dealloc.cta_group::1.sync.aligned.b32 %r1, 32;"),
            ["after"],
        ),
        # A guarded fence or commit may not run: the path past a false predicate is unfenced.
        ((LD, "\t@%p1 " + BEFORE.strip(), ARRIVE), ["before"]),
        ((WAIT, "\t@%p1 " + AFTER.strip(), LD), ["after"]),
        ((MMA, "\t@%p1 " + COMMIT.strip(), ARRIVE), ["before"]),
        ((WAIT, AFTER, "\t@%p1 " + LD.strip(), BEFORE, ARRIVE), []),
        # An unknown tcgen05 instruction counts as an asynchronous operation, and a commit
        # does not fence it.
        (("\ttcgen05.future.b32 [%r1];", ARRIVE), ["before"]),
        (("\ttcgen05.future.b32 [%r1];", COMMIT, ARRIVE), ["before"]),
    ],
    ids=[
        "fenced-handoff",
        "wait-then-ld",
        "ld-then-arrive",
        "waited-ld-then-arrive",
        "mma-commit",
        "ld-commit-arrive",
        "wait-fence-mma",
        "wait-mma",
        "warps-signal",
        "ld-then-barrier-sync",
        "barrier-sync-then-ld",
        "barrier-arrive-then-ld",
        "ld-then-release-store",
        "barrier-then-dealloc",
        "guarded-before-fence",
        "guarded-after-fence",
        "guarded-commit",
        "guarded-ld-fenced",
        "unknown-tcgen05",
        "unknown-tcgen05-commit",
    ],
)
def test_checker_rules(lines, rules):
    assert _rules(*lines) == rules


def test_checker_follows_branches_and_loops():
    """A fence on one arm of a branch leaves the other path unfenced; a loop's back edge carries
    the state of its last iteration into the first."""
    branch = ["\t@%p1 bra $L__BB0_1;", AFTER, "$L__BB0_1:", LD]
    assert _rules(WAIT, *branch) == ["after"]
    assert _rules(WAIT, AFTER, *branch) == []
    loop = ["$L__BB0_2:", LD, ARRIVE, "\t@%p1 bra $L__BB0_2;"]
    assert _rules(BEFORE, *loop) == ["before"]
    loop = ["$L__BB0_2:", WAIT, AFTER, LD, BEFORE, ARRIVE, "\t@%p1 bra $L__BB0_2;"]
    assert _rules(*loop) == []


def test_checker_drain_at_the_teardown_barrier():
    teardown = f"\tbarrier.sync {TMEM_TEARDOWN_BARRIER_ID}, 128;"
    st = "\ttcgen05.st.sync.aligned.32x32b.x1.b32 [%r5], {%r3};"
    assert _rules(LD, BEFORE, teardown, teardown_barrier=2) == ["drain"]
    assert _rules(st, BEFORE, teardown, teardown_barrier=2) == ["drain"]
    waits = ("\ttcgen05.wait::ld.sync.aligned;", "\ttcgen05.wait::st.sync.aligned;")
    assert _rules(LD, st, *waits, BEFORE, teardown, teardown_barrier=2) == []
    assert _rules(LD, BEFORE, teardown) == []  # no teardown barrier named: not checked
    guarded = "\t@%p1 tcgen05.wait::ld.sync.aligned;"  # may not run: the load may be in flight
    assert _rules(LD, guarded, BEFORE, teardown, teardown_barrier=2) == ["drain"]


def test_checker_rejects_calls():
    """Without call summaries a callee's synchronization is invisible: fail, never pass. A
    failed device assert does not return."""
    with pytest.raises(ValueError, match="calls are not supported"):
        _rules(WAIT, "\tcall.uni (%r1), callee, (%r2);", LD)
    with pytest.raises(ValueError, match="calls are not supported"):
        _rules("\tcall.uni;")
    assert _rules(WAIT, "\tcall.uni __assertfail, (%r2);", LD) == []
    assert _rules(WAIT, "\t@%p1 call.uni __assertfail, (%r2);", LD) == ["after"]


def test_checker_names_source_lines():
    """``.loc`` directives (``CUTE_DSL_LINEINFO=1``) name each finding's source lines."""
    body = f"{WAIT}\n\t.loc\t1 42 3\n$L__BB0_3:\n\t.loc\t1 43 3\n{LD}"
    ptx = '\t.file\t1 "/src/kernel.py"\n' + _entry(body)
    (finding,) = check_tcgen05_sync(ptx)
    assert (finding.rule, finding.source) == ("after", "kernel.py:43")


def test_checker_scopes_inline_labels():
    """Inlined spin waits reuse their label names in sibling brace scopes."""
    assert _rules(WAIT, AFTER, LD, BEFORE, WAIT, LD) == ["after"]


@pytest.mark.parametrize(
    "ptx",
    [
        ".visible .entry k(\n\t.param .u64 p\n)\n{\n\tbar.sync 3, 128;\n{LD}\n\tret;\n}\n",
        ".visible .entry k(.param .u64 p) { bar.sync 3, 128;\n{LD}\n\tret; }\n",
        (
            ".visible .entry k\n(\n\t.param .u64 p\n)\n.maxntid 128, 1, 1\n"
            "{\n\tbar.sync 3, 128;\n{LD}\n\tret;\n}"
        ),
        (
            '.file 1 "a{b}.py" // }\n.global .b32 g[2] = {0, 1};\n'
            ".visible .entry k(.param .u64 p) {bar.sync 3, 128; {LD} ret;}"
        ),
    ],
    ids=["multi-line", "body-on-brace-lines", "paren-on-next-line", "one-line-after-braces"],
)
def test_checker_reads_kernels_in_any_layout(ptx):
    """Kernel bodies are found wherever lines break, past braces in strings, comments and
    initializers: the same unfenced bar.sync -> tcgen05.ld gives the same finding."""
    found = check_tcgen05_sync(ptx.replace("{LD}", LD))
    assert [(f.kernel, f.rule, f.text, f.origin_text) for f in found] == [
        ("k", "after", LD.strip().rstrip(";"), "bar.sync 3, 128")
    ]


@pytest.mark.parametrize(
    "ptx, match",
    [
        (".version 8.6\n.target sm_100a\n", "no .entry with a body"),
        (".extern .func f(.param .b32 a);\n", "no .entry with a body"),
        (".visible .entry (.param .u64 p)\n{\n\tret;\n}\n", "unparsed function header"),
        (".visible .entry k()\n{\n\tret;\n", "unclosed brace"),
    ],
    ids=["empty", "declaration-only", "unnamed", "unclosed"],
)
def test_checker_rejects_modules_it_cannot_read(ptx, match):
    with pytest.raises(ValueError, match=match):
        check_tcgen05_sync(ptx)


# ---- a TMEM handoff over every kind of managed synchronization ---------------------------

EPOCHS = 6
THREADS = 128


@cute.jit
def _st(taddr, value):
    inline_ptx(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1};", read_only_args=[Int32(taddr), value]
    )


@cute.jit
def _ld(taddr):
    return inline_ptx(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1];",
        write_only_types=[Int32],
        read_only_args=[Int32(taddr)],
    )


@cute.jit
def _addr(tmem, name, me):
    """This warp's lane quadrant of TMEM region ``name``, column 0."""
    return tmem.ptr(name).toint() + ((me.warp % 4) * 32 << 16)


class SyncToy(WarpSpecialized):
    """``a`` stores ``1000 * e + tid`` into TMEM and hands it to ``b`` over ``fwd`` (an mbarrier
    ring, ``b`` probing it first); ``b`` loads it, stores it plus 7 and hands it back over
    ``back`` (a named-barrier ring); ``a`` loads that. After the loop each stores a value,
    meets the other at a ``NamedBarrier`` and loads the other's. Only the layer orders these
    TMEM accesses across threads."""

    def __init__(self, skip: str | None = None):
        self.skip = skip  # b's fwd consume skips this fence (a needed one)
        self.fwd = Ring(2, Threads, Warps, commit_fence="tmem_store", release_fence="tmem_load")
        self.back = Ring(
            1, Threads, Threads, sync="named", commit_fence="tmem_store", release_fence="tmem_load"
        )
        self.done = NamedBarrier(threads=2 * THREADS)
        self.tmem_plan = TmemPlan().region("x", 32, slots=2).region("y", 32).region("z", 32)

    @role(warps=range(4))
    @cute.jit
    def a(self, ctx, *, fwd: Producer, back: Consumer, tmem: Tmem, me: Local):
        for e in cutlass.range(EPOCHS, unroll=1):
            with fwd.produce() as h:
                _st(_addr(tmem, "x", me) + h.index * 32, e * 1000 + me.tid)
            with back.consume():
                ctx.out[e, me.tid] = _ld(_addr(tmem, "y", me))
        _st(_addr(tmem, "z", me), Int32(-1) - me.tid)
        cute.arch.fence_view_async_tmem_store()
        self.done.sync()
        ctx.out[EPOCHS, me.tid] = _ld(_addr(tmem, "z", me) + 1)
        cute.arch.fence_view_async_tmem_load()
        return fwd, back

    @role(warps=range(4, 8))
    @cute.jit
    def b(self, ctx, *, fwd: Consumer, back: Producer, tmem: Tmem, me: Local):
        for _ in cutlass.range(EPOCHS, unroll=1):
            token = fwd.try_wait()
            skip = self.skip
            reason = None if skip is None else f"test: the {skip} fence is needed"
            with fwd.consume(token=token, skip_fence=skip, reason=reason) as h:
                value = _ld(_addr(tmem, "x", me) + h.index * 32)
            with back.produce():
                _st(_addr(tmem, "y", me), value + 7)
        _st(_addr(tmem, "z", me) + 1, Int32(-2) - me.tid)
        cute.arch.fence_view_async_tmem_store()
        self.done.sync()
        ctx.out[EPOCHS + 1, me.tid] = _ld(_addr(tmem, "z", me))
        cute.arch.fence_view_async_tmem_load()
        return fwd, back

    @cute.kernel
    def kernel(self, out):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_sync_toy")
        self.kernel(out).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(Int32, (EPOCHS + 2, THREADS), assumed_align=4)
    return compile_tvm_ffi(op, out, name="sync_toy")


def _ptx_findings(build, op, tmp_path):
    """Checker findings in the PTX of an uncached compile of ``op``."""
    with capture_ptx(tmp_path) as out:
        build.__wrapped__(op)
    (path,) = out.glob("*.ptx")
    text = read_ptx(path)
    return text, check_tcgen05_sync(text)


def test_toy_hands_tmem_back_and_forth(requires_blackwell):
    fn = compile_toy(SyncToy())
    out = torch.zeros(EPOCHS + 2, THREADS, dtype=torch.int32, device="cuda")
    fn(out)
    tid = torch.arange(THREADS, dtype=torch.int32)
    expected = torch.stack([1000 * e + tid + 7 for e in range(EPOCHS)] + [-2 - tid, -1 - tid])
    assert torch.equal(out.cpu(), expected)


def test_toy_ptx_is_fenced(requires_blackwell, tmp_path):
    """Every path of the toy's PTX is fenced, and the fences are the layer's: the toy itself
    emits none."""
    text, found = _ptx_findings(compile_toy, SyncToy(), tmp_path)
    assert not found, "\n".join(f.describe() for f in found)
    assert text.count("tcgen05.fence::after_thread_sync") >= 6
    assert text.count("tcgen05.fence::before_thread_sync") >= 6


def test_skips_name_a_fence_the_call_emits_and_a_reason():
    with pytest.raises(ValueError, match="one of"):
        fence_skip("all", "r", "kv.wait")
    with pytest.raises(ValueError, match="one of"):
        fence_skip("after", "r", "kv.release", ("before",))
    with pytest.raises(ValueError, match="non-empty reason"):
        fence_skip("after", " ", "kv.wait")
    assert fence_skip(None, "a reason alone skips nothing", "kv.wait") is None


@pytest.mark.parametrize("skip", ["after", "before"])
def test_skip_on_a_needed_fence_is_caught(requires_blackwell, tmp_path, skip):
    """``b`` loads TMEM right after its fwd wait and releases fwd right after that load
    completes, so both fences of that handle are needed: skipping either leaves a PTX path the
    checker reports, and ``describe()`` names the skip, its reason and what it dropped."""
    op = SyncToy(skip)
    _, found = _ptx_findings(compile_toy, op, tmp_path)
    rule = "after_thread_sync" if skip == "after" else "before_thread_sync"
    assert found and all(rule in f.describe() for f in found), [f.describe() for f in found]
    text = op.describe()
    assert re.search(rf"skip {skip} @ test_warpdrive_thread_sync\.py:\d+ \(fwd\.wait\)", text)
    assert f"b: skipped {skip}\n    'test: the {skip} fence is needed'" in text


def test_tcgen05_roles_alone_take_the_fences(requires_blackwell, tmp_path):
    """A role that issues no tcgen05 operation (the GEMM's TMA producer) waits and signals
    without the fences; the MMA and epilogue roles fence theirs."""
    from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm

    op = Gemm()
    _, found = _ptx_findings(compile_gemm, op, tmp_path)
    assert not found, "\n".join(f.describe() for f in found)
    assert [op._issues_tcgen05(r) for r in op._ws_roles] == [
        r.role in ("mma", "epilogue") for r in op._ws_roles
    ]


@pytest.mark.parametrize("pair", [False, True], ids=["cta", "pair"])
def test_persistent_gemm_ptx_is_fenced(requires_blackwell, tmp_path, pair):
    """Warps signals, an Umma-signalled accumulator ring, the work ring and, in a pair, the
    allocator warps' dealloc handshake."""
    from attn_gym.warpdrive.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    op = PersistentGemm(cluster=CtaPair() if pair else None)
    _, found = _ptx_findings(compile_persistent_gemm, op, tmp_path)
    assert not found, "\n".join(f.describe() for f in found)


@pytest.mark.parametrize("pair", [False, True], ids=["cta", "pair"])
def test_fa4_bwd_ptx_is_fenced(requires_blackwell, tmp_path, pair):
    """The pair's raw dS arrival (``signal_ds``) follows the compute role's tcgen05.ld of S,
    whose tail the MMA role's dQ overwrites."""
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_bwd import (
        FlashAttentionBackward,
        FlashAttentionBackward2Cta,
        compile_fa4_bwd,
    )

    op = FlashAttentionBackward2Cta() if pair else FlashAttentionBackward()
    _, found = _ptx_findings(compile_fa4_bwd, op, tmp_path)
    assert not found, "\n".join(f.describe() for f in found)
