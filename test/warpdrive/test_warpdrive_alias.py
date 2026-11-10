"""Test trace-time alias diagnostics and ordered overlap correctness.

Unmodified examples report no potential hazard. Unsafe mutants must identify the
conflicting pair and missing edge; they compile but never launch. The ordered overlap
variant runs for correctness.
"""

import math
import re
import warnings
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute
from test_warpdrive_tokens import PreviewOverlap, compile_toy

from attn_gym.testing.warpdrive import alias_section, raises_protocol, status
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Producer,
    Raster,
    Ring,
    StageView,
    StaticPersistent,
    Threads,
    Tmem,
    Umma,
    WorkCursor,
    WorkStream,
    role,
    unchecked,
)


def compile_quietly(op, compile_fn):
    """Compile, returning the alias diagnostic's warnings (other warnings pass through)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = compile_fn(op)
    alias = [str(w.message) for w in caught if "alias diagnostic" in str(w.message)]
    return fn, alias


class AliasRing(PreviewOverlap):
    def __init__(self, opt_out: bool = False):
        super().__init__(Umma)
        self.opt_out = opt_out

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.opt_out):
                with unchecked("alias", reason="test: UMMA release does not publish thread reads"):
                    c.peek().access("P", read=True)
            else:
                c.peek().access("P", read=True)
            c.wait().release()
        return c


def compile_alias_ring(op):
    return compile_toy(op, "alias_ring")


def test_alias_warning_restores_from_disk(tmp_path, monkeypatch):
    """A cached diagnostic repeats its warnings and retains the traced channel checks."""
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    compile_toy.cache_clear()
    cold = AliasRing()
    _, cold_alias = compile_quietly(cold, compile_alias_ring)
    assert len(cold_alias) == 1 and "Q~P" in cold_alias[0]
    assert status(cold, "Q ~ P") == "potential hazard"
    assert "cons read P by threads" in alias_section(cold)

    compile_toy.cache_clear()
    warm = AliasRing()
    _, warm_alias = compile_quietly(warm, compile_alias_ring)
    info = compile_toy.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 0, 1)
    assert warm_alias == cold_alias
    assert warm.describe() == cold.describe() and warm.checks() == cold.checks()


def test_alias_strict_off_and_unchecked():
    compile_quietly(AliasRing(), compile_alias_ring)
    strict = AliasRing()
    strict.alias_check = "strict"
    raises_protocol(r"alias diagnostic \(strict\)", lambda: compile_alias_ring(strict))

    opted = AliasRing(opt_out=True)
    _, alias = compile_quietly(opted, compile_alias_ring)
    assert alias == []
    assert "unchecked alias @ test_warpdrive_alias.py:" in opted.describe()

    off = AliasRing()
    off.alias_check = "off"
    _, alias = compile_quietly(off, compile_alias_ring)
    assert alias == [] and "Alias diagnostic" not in off.describe()
    assert off.checks().aliases == {}


@pytest.fixture(scope="module")
def fa4():
    pytest.importorskip("flash_attn.cute")
    pytest.importorskip("quack")

    from attn_gym.warpdrive.examples.fa4_fwd import (
        LOG2_E,
        Q_STAGE,
        Fa4Params,
        FlashAttentionForward,
        compile_fa4_fwd,
        umma,
    )

    class QkBeforePv(FlashAttentionForward):
        def __init__(self, opt_out: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.opt_out = opt_out

        def get_name(self) -> str:
            return f"{super().get_name()}_qk_before_pv{int(self.opt_out)}"

        @role(warps=12, regs=48, cta="leader")
        @cute.jit
        def mma(
            self,
            params: Fa4Params,
            *,
            q: Consumer,
            kv: Consumer,
            s: Producer,
            p_last: Consumer,
            work: WorkCursor,
            tmem: Tmem,
        ):
            thr_qk, thr_pv = params.mma_qk.get_slice(0), params.mma_pv.get_slice(0)
            q = q.views(Q=StageView(thr_qk.make_fragment_A(params.sQ), axis=3))
            kv = kv.views(
                K=StageView(thr_qk.make_fragment_B(params.sK), axis=3),
                V=StageView(thr_pv.make_fragment_B(params.sV), axis=3),
            )
            tS = [cute.make_tensor(tmem.ptr("S", j), params.s_layout) for j in range(Q_STAGE)]
            tO = [cute.make_tensor(tmem.ptr("O", j), params.o_layout) for j in range(Q_STAGE)]
            tOrP0 = thr_pv.make_fragment_A(cute.make_tensor(tmem.base, params.p_layout.outer))
            tOrP0 = tOrP0[None, None, None, 0]
            tOrP = [
                cute.make_tensor(tOrP0.iterator + tmem.cols("P", j) * 2, tOrP0.layout)
                for j in range(Q_STAGE)
            ]
            while work.valid:
                k0 = kv.wait()
                qh = []
                for j in cutlass.range_constexpr(Q_STAGE):
                    qh.append(q.wait(j))
                    sp = s.acquire(j)
                    tmem.access("S", j, write=True, by="umma")
                    umma(
                        params.mma_qk,
                        tS[j],
                        qh[j].view(),
                        k0.view("K", read=True, by="umma"),
                        False,
                    )
                    sp.commit()
                k0.release()
                for i in cutlass.range(params.n_blocks - 1, unroll=1):
                    v = kv.wait()
                    k = kv.wait()
                    for j in cutlass.range_constexpr(Q_STAGE):
                        sl = s.acquire(j)
                        # MUTANT: S_j(i) = Q_j K_i^T overwrites P_j before PV_j reads it.
                        if cutlass.const_expr(self.opt_out):
                            with unchecked("alias", reason="mutant: deliberately unordered"):
                                tmem.access("S", j, write=True, by="umma")
                        else:
                            tmem.access("S", j, write=True, by="umma")
                        umma(
                            params.mma_qk,
                            tS[j],
                            qh[j].view(),
                            k.view("K", read=True, by="umma"),
                            False,
                        )
                        tOrV = v.view("V", read=True, by="umma")
                        p_last = self.pv(p_last, j, tO[j], tOrP[j], tOrV, i > 0, params, tmem)
                        sl.commit()
                    v.release()
                    k.release()
                for h in qh:
                    h.release()
                v_last = kv.wait()
                for j in cutlass.range_constexpr(Q_STAGE):
                    with s.produce(j):
                        tOrV = v_last.view("V", read=True, by="umma")
                        p_last = self.pv(p_last, j, tO[j], tOrP[j], tOrV, True, params, tmem)
                v_last.release()
                work.advance()
            return q, kv, s, p_last, work

    class PersistentOverlapOQ(FlashAttentionForward):
        def __init__(self, ordered: bool):
            super().__init__(overlap_o_q=True)
            self.ordered = ordered
            self.q_free = Ring(1, Threads, Threads)  # epilogue: "sO read", load: next Q may land
            self.work = WorkStream(mapping=Raster(along="m"), schedule=StaticPersistent())

        def get_name(self) -> str:
            return f"{super().get_name()}_persistent_oq{int(self.ordered)}"

        def get_key(self) -> str:
            return self.get_name()

        @role(warps=14, regs=48)
        @cute.jit
        def load(
            self,
            params: Fa4Params,
            *,
            q: Producer,
            kv: Producer,
            q_free: Consumer,
            work: WorkCursor,
        ):
            rank = self.cluster.rank()
            thr = params.mma_qk.get_slice(rank), params.mma_pv.get_slice(rank)
            while work.valid:
                q, kv = self.load_item(params, q, kv, thr, work.item)
                if cutlass.const_expr(self.ordered):
                    with q_free.consume():
                        pass
                work.advance()
            return q, kv, q_free, work

        @role(warps=13, regs=48)
        @cute.jit
        def epilogue(
            self, params: Fa4Params, *, epi: Consumer, q_free: Producer, work: WorkCursor
        ):
            rank = self.cluster.rank()
            while work.valid:
                epi = self.epilogue_item(params, epi, rank, work.item)
                if cutlass.const_expr(self.ordered):
                    with q_free.produce():
                        pass
                work.advance()
            return epi, q_free, work

    class RawSoftmaxRead(FlashAttentionForward):
        def get_name(self) -> str:
            return f"{super().get_name()}_raw_s"

        def softmax_step(self, stage, s, p_last, stats, mufu, sm, t, is_first):
            tmem = t.tmem

            class Silent:  # forward everything but the S read annotation
                def access(self, name, *args, **kwargs):
                    if name != "S":
                        tmem.access(name, *args, **kwargs)

            return super().softmax_step(
                stage,
                s,
                p_last,
                stats,
                mufu,
                sm,
                SimpleNamespace(**{**vars(t), "tmem": Silent()}),
                is_first,
            )

    return SimpleNamespace(
        FlashAttentionForward=FlashAttentionForward,
        QkBeforePv=QkBeforePv,
        PersistentOverlapOQ=PersistentOverlapOQ,
        RawSoftmaxRead=RawSoftmaxRead,
        compile_fa4_fwd=compile_fa4_fwd,
        compile_quietly=lambda op: compile_quietly(op, compile_fa4_fwd),
        LOG2_E=LOG2_E,
    )


# Unmodified examples


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"split_p": True}, {"overlap_o_q": True}],
    ids=["single", "split_p", "overlap_o_q"],
)
def test_fa4_examples_report_no_hazard(fa4, kwargs):
    op = fa4.FlashAttentionForward(**kwargs)
    _, alias = fa4.compile_quietly(op)
    assert alias == []
    pairs = ["P ~ S", "V ~ K"] + (["O ~ Q"] if kwargs.get("overlap_o_q") else [])
    for pair in pairs:
        assert status(op, pair) == "no hazard found", alias_section(op)
    assert "potential hazard" not in op.checks().aliases.values(), alias_section(op)


def test_cta_pairs_are_unverified_not_safe(fa4):
    op = fa4.FlashAttentionForward(cluster=CtaPair())
    _, alias = fa4.compile_quietly(op)
    assert alias == []  # out of scope is not a finding
    assert status(op, "P ~ S") == "unverified"
    assert "CtaPair kernels are not modeled" in alias_section(op)


# Mutant 1: the S-overwriting QK moved before the PV that reads P (an alias of S)


def test_qk_overwriting_p_before_pv_is_a_potential_hazard(fa4, tmp_path, monkeypatch):
    """Findings name the conflicting pair and missing edge, including after a disk load.

    Cached kernels restore findings, repeat warnings, and describe like cold builds,
    including the TMA report.
    """
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    fa4.compile_fa4_fwd.cache_clear()
    cold = fa4.QkBeforePv()
    _, cold_alias = fa4.compile_quietly(cold)
    assert len(cold_alias) == 1 and "P~S" in cold_alias[0]
    assert status(cold, "P ~ S") == "potential hazard"
    text = alias_section(cold)
    finding = re.search(
        r"potential hazard: mma write S\[0\] by umma @ test_warpdrive_alias\.py:\d+ then "
        r"mma read P\[0\] by umma @ fa4_fwd\.py:\d+: not a pipelined pair.*\[pipelined\]",
        text,
    )
    assert finding, text
    assert status(cold, "V ~ K") == "no hazard found"

    fa4.compile_fa4_fwd.cache_clear()  # a warm process: the build is on disk
    warm = fa4.QkBeforePv()
    _, warm_alias = fa4.compile_quietly(warm)
    info = fa4.compile_fa4_fwd.cache_info()
    assert (info.hits, info.misses) == (1, 0)
    assert warm_alias == cold_alias
    assert warm.describe() == cold.describe() and warm.checks() == cold.checks()
    assert {c.tma for c in warm.checks().channels.values()} == {"verified", None}


def test_strict_mode_raises_and_off_or_unchecked_opts_out(fa4):
    fa4.compile_quietly(fa4.QkBeforePv())  # a cached "warn" build must not satisfy a "strict" one
    strict = fa4.QkBeforePv()
    strict.alias_check = "strict"
    raises_protocol(r"alias diagnostic \(strict\)", lambda: fa4.compile_fa4_fwd(strict))

    opted = fa4.QkBeforePv(opt_out=True)
    _, alias = fa4.compile_quietly(opted)
    assert alias == []
    assert "potential hazard" not in opted.checks().aliases.values(), alias_section(opted)
    assert "unchecked alias @ test_warpdrive_alias.py:" in opted.describe()

    off = fa4.QkBeforePv()
    off.alias_check = "off"
    _, alias = fa4.compile_quietly(off)
    assert alias == [] and "Alias diagnostic" not in off.describe()
    assert off.checks().aliases == {}


# Mutant 2: sO overlapping sQ under a persistent schedule, with and without the epilogue ->
# load edge (FA4's load_epi pipeline) that keeps the next item's Q off the O being stored


def test_overlapping_o_and_q_needs_the_epilogue_to_load_edge(fa4):
    ordered = fa4.PersistentOverlapOQ(ordered=True)
    _, alias = fa4.compile_quietly(ordered)
    assert alias == []
    assert status(ordered, "O ~ Q") == "no hazard found"

    dropped = fa4.PersistentOverlapOQ(ordered=False)
    _, alias = fa4.compile_quietly(dropped)
    assert len(alias) == 1 and "O~Q" in alias[0]
    assert status(dropped, "O ~ Q") == "potential hazard"
    text = alias_section(dropped)
    assert re.search(
        r"epilogue read O\[\d\] by tma @ fa4_fwd\.py:\d+ \(work item 0\) and load write "
        r"Q\[\d\] by tma @ fa4_fwd\.py:\d+ \(work item 1\): no channel path orders them",
        text,
    ), text


def test_ordered_overlap_runs_correctly(fa4, requires_blackwell):
    b, h, sq, sk = 2, 3, 640, 512
    q, k, v = (
        torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16) for s in (sq, sk, sk)
    )
    op = fa4.PersistentOverlapOQ(ordered=True)
    fn, _ = fa4.compile_quietly(op)
    out, lse = torch.empty_like(q), torch.empty(b, h, sq, device="cuda", dtype=torch.float32)
    workspace, _ = op.work.launch_args(q.device)
    fn(q, k, v, out, lse, 128**-0.5 * fa4.LOG2_E, workspace, 2)  # 2 CTAs: several items each
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() <= 2**-7 * ref.abs().max().item() + 4e-3
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


# Mutant 3: a raw TMEM access (no annotation) to an aliased region


def test_raw_tmem_access_is_unverified(fa4):
    op = fa4.RawSoftmaxRead()
    _, alias = fa4.compile_quietly(op)
    text = alias_section(op)
    assert status(op, "P ~ S") == "unverified", text
    assert re.search(
        r"softmax\[0\] acquires S without annotating its accesses @ fa4_fwd\.py", text
    )
    assert alias == []  # nothing positively conflicting was seen
