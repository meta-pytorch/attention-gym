"""Test round-robin rings, named channels, and the FA4 swap-AB decode port.

Three role instances take a three-stage mbarrier ring's epochs in turn. Each forwards
through a three-stage named ring (one stage per instance) and a one-stage named mailbox
serialized by a turnstile. Decode tests compare with an fp32 reference and, when
``flash_attn.cute`` is installed, with FA4's split partials bit for bit.
"""

import math
import pickle
import warnings
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Instance,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Sharing,
    Slots,
    Threads,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    named_barrier,
    role,
)
from attn_gym.warpdrive.cache import (
    canonical_key,
    compile_tvm_ffi,
    jit_cache,
    make_fake_compact_tensor,
)
from attn_gym.warpdrive.channels import ConsumerEnd

K_MID = 3  # round-robin instances of the toy's middle role


class RoundRobinToy(WarpSpecialized):
    """src (warp 0) -> rr -> mid[0..2] (warps 1-3) -> nr and mail -> sink (warp 4).

    Epoch e's value is ``e * 32 + lane``. mid[j] takes rr epochs j, j + 3, ...; records which
    instance saw each epoch; forwards it through ``nr`` (named, 3 stages: one per instance) and
    ``mail`` (named, 1 stage, taken in turn). ``mode`` injects declaration/trace errors."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        payload = (cutlass.Int32, (32,))
        rr_stages = 2 if mode == "short_ring" else K_MID
        rr = Sharing.ROUND_ROBIN
        self.rr = Ring(rr_stages, Threads, Threads, consumer_sharing=rr, payload=payload)
        self.nr = Ring(K_MID, Threads, Threads, sync="named", producer_sharing=rr, payload=payload)
        turnstile = None if mode == "no_turnstile" else "turn"
        self.mail = Ring(
            1,
            Threads,
            Threads,
            sync="named",
            producer_sharing=rr,
            turnstile=turnstile,
            payload=payload,
        )
        self.turn = Turnstile()

    @role(warps=0)
    @cute.jit
    def src(self, ctx, *, rr: Producer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(ctx.n, unroll=1):
            h = rr.produce()
            h.data[lane] = e * 32 + lane
            h.commit(wait=self.mode == "wait_mbarrier")
        return rr

    @role(warps=[range(1, 2), range(2, 3), range(3, 4)])
    @cute.jit
    def mid(self, ctx, *, stage: Instance, rr: Consumer, nr: Producer, mail: Producer, turn: Turn):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(stage, ctx.n, K_MID, unroll=1):
            value = cutlass.Int32(0)
            with rr.consume() as h:
                value = h.data[lane]
            ctx.out[e * 32 + lane, 0] = value
            if lane == 0:
                ctx.who[e] = stage
            with nr.produce() as h:
                h.data[lane] = value
            if cutlass.const_expr(self.mode == "outside_turn"):
                with mail.produce() as h:
                    h.data[lane] = value
            else:
                with turn.turn(), mail.produce() as h:
                    h.data[lane] = value
        return rr, nr, mail, turn

    @role(warps=4)
    @cute.jit
    def sink(self, ctx, *, nr: Consumer, mail: Consumer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(ctx.n, unroll=1):
            with nr.consume() as h:
                ctx.out[e * 32 + lane, 1] = h.data[lane]
            with mail.consume() as h:
                ctx.out[e * 32 + lane, 2] = h.data[lane]
        if cutlass.const_expr(self.mode == "id_collision"):
            named_barrier(self.mail.full_barriers[0].id, 32)
        return nr, mail

    @cute.kernel
    def kernel(self, out, who, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, who=who, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, who: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix(f"warpdrive_round_robin_toy_{self.mode}")
        self.kernel(out, who, n).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 3), assumed_align=4)
    who = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, who, cutlass.Int32(1), name=f"round_robin_toy_{op.mode}")


def test_round_robin_and_named_channels_deliver_every_epoch(requires_blackwell):
    """Instance j gets epochs j, j + 3, ... of the round-robin ring, across ring wraps; the
    named ring and the turn-taking mailbox deliver them to the sink in epoch order."""
    fn = compile_toy(RoundRobinToy())
    for n in (1, 2, 13):  # runtime epoch counts of one compiled kernel
        out = torch.full((n * 32, 3), -1, device="cuda", dtype=torch.int32)
        who = torch.full((n,), -1, device="cuda", dtype=torch.int32)
        fn(out, who, n)
        expected = torch.arange(n * 32, device="cuda", dtype=torch.int32)
        for column in range(3):
            torch.testing.assert_close(out[:, column], expected)
        expected_who = torch.arange(n, device="cuda", dtype=torch.int32) % K_MID
        torch.testing.assert_close(who, expected_who)


def test_round_robin_and_named_describe():
    text = RoundRobinToy().describe()
    # One instance's arrivals per epoch: 32, not the 96 of a three-instance join.
    assert "rr    Ring(3)  src -> mid[0], mid[1], mid[2]" in text
    assert "empty     32 x3" in text
    assert "consumers round-robin: instance j of 3 takes epochs j, j+3, ..." in text
    # Named barriers count both sides; IDs come from the top, below the layer's 1 and 2.
    assert "named     full ids [10, 11, 12], empty ids [13, 14, 15], 64 x3 threads" in text
    assert "named     full ids [8], empty ids [9], 64 threads, taken in turn (turn)" in text
    assert RoundRobinToy().named_barrier_ids == {
        "nr": ([10, 11, 12], [13, 14, 15]),
        "mail": ([8], [9]),
    }


@pytest.mark.parametrize(
    "mode, message",
    [
        ("no_turnstile", r"3 round-robin producers share the barriers of 1 named stage"),
        ("outside_turn", r"mail\.acquire\(\) at .* outside `with turn\.turn\(\):`"),
        ("id_collision", r"named_barrier\(8\) at .* uses a barrier ID owned by channel mail"),
        ("short_ring", r"3 round-robin consumers need a multiple of 3 stages, got 2"),
        ("wait_mbarrier", r"commit\(wait=True\) on rr\[i\] acquire: only a named-barrier"),
    ],
)
def test_named_channel_errors(mode, message):
    _raises_protocol(message, lambda: compile_toy(RoundRobinToy(mode)))


def _kernel(channels, roles):
    """A kernel class with the given channel attributes and role functions."""
    cls = type("Probe", (WarpSpecialized,), dict(roles))

    def init(self):
        for name, make in channels.items():
            setattr(self, name, make())

    cls.__init__ = init
    return cls()


def _consumer_pair(width2=1):
    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        return ch

    @role(warps=1)
    @cute.jit
    def cons_a(self, ctx, *, ch: Consumer):
        return ch

    @role(warps=range(2, 2 + width2))
    @cute.jit
    def cons_b(self, ctx, *, ch: Consumer):
        return ch

    return {"prod": prod, "cons_a": cons_a, "cons_b": cons_b}


def test_round_robin_declaration_errors():
    with pytest.raises(ValueError, match="consumer_sharing=Sharing.ROUND_ROBIN needs a Ring"):
        Slots(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)
    with pytest.raises(ValueError, match="producer_sharing must be a Sharing"):
        Ring(2, Threads, Threads, producer_sharing="round_robin")
    with pytest.raises(ValueError, match="both sides must be Threads"):
        Ring(2, Umma, Threads, sync="named")
    with pytest.raises(ValueError, match="takes no override"):
        Ring(2, Threads, Threads, sync="named", override={"empty_counts": 4})
    with pytest.raises(ValueError, match="sync must be one of"):
        Ring(2, Threads, Threads, sync="nvbar")
    with pytest.raises(ValueError, match="turnstile= serializes"):
        Ring(2, Threads, Threads, turnstile="t")

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        return ch

    # A round-robin side with one instance: nothing to take turns.
    rr = Sharing.ROUND_ROBIN
    op = _kernel(
        {"ch": lambda: Ring(2, Threads, Threads, consumer_sharing=rr)},
        {"prod": prod, "cons": cons},
    )
    _raises_protocol(
        r"consumer_sharing=Sharing.ROUND_ROBIN needs at least two instances of one role on the "
        r"consumer side, got \['cons'\]",
        op.describe,
    )
    # Two different roles, or instances of different widths, cannot alternate one count.
    op = _kernel({"ch": lambda: Ring(2, Threads, Threads, consumer_sharing=rr)}, _consumer_pair())
    _raises_protocol("must be instances of one role with the same width", op.describe)


def test_named_channels_need_free_ids_and_one_cta():
    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, a: Producer, b: Producer):
        return a, b

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, a: Consumer, b: Consumer):
        return a, b

    roles = {"prod": prod, "cons": cons}
    # 13 IDs (3-15) are free: 4 + 3 stages need 14.
    channels = {
        "a": lambda: Ring(4, Threads, Threads, sync="named"),
        "b": lambda: Ring(3, Threads, Threads, sync="named"),
    }
    _raises_protocol("need 14 barrier IDs; 13 are free", _kernel(channels, roles).describe)

    @role(warps=0, cta="leader")
    @cute.jit
    def lead(self, ctx, *, a: Producer):
        return a

    @role(warps=1)
    @cute.jit
    def both(self, ctx, *, a: Consumer):
        return a

    op = _kernel(
        {"a": lambda: Ring(1, Threads, Threads, sync="named")}, {"lead": lead, "both": both}
    )
    op.cluster = CtaPair()
    _raises_protocol("named barriers are per CTA", op.describe)

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def pair_cons(self, ctx, *, a: Consumer):
        return a

    op = _kernel(
        {"a": lambda: Ring(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)},
        {"lead": lead, "pair_cons": pair_cons},
    )
    op.cluster = CtaPair()
    _raises_protocol("round-robin sides are CTA-local", op.describe)


def test_named_turnstile_must_seat_the_round_robin_instances():
    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, ctx, *, a: Producer):
        return a

    @role(warps=[range(2, 3), range(3, 4)])
    @cute.jit
    def other(self, ctx, *, t: Turn):
        return t

    @role(warps=4)
    @cute.jit
    def cons(self, ctx, *, a: Consumer):
        return a

    channels = {
        "a": lambda: Ring(
            1, Threads, Threads, sync="named", producer_sharing=Sharing.ROUND_ROBIN, turnstile="t"
        ),
        "t": Turnstile,
    }
    op = _kernel(channels, {"prod": prod, "other": other, "cons": cons})
    _raises_protocol(
        r"declare an enabled Turnstile over \['prod\[0\]', 'prod\[1\]'\]", op.describe
    )


def test_named_channel_has_no_mbarrier_accessors():
    chan = Ring(1, Threads, Threads, sync="named")
    chan.name = "a"
    cons = ConsumerEnd(chan, None, {}, SimpleNamespace(role_name="r"), None, None)
    _raises_protocol("named-barrier channel and has no mbarriers", lambda: cons.full_barrier(0))
    _raises_protocol("named-barrier channel and has no mbarriers", lambda: cons.empty_barrier(0))


# The FA4 swap-AB decode port


@pytest.fixture
def decode():
    """The example imports flash_attn.cute and quack; only its tests need them."""
    return pytest.importorskip("attn_gym.warpdrive.examples.fa4_decode_swapab")


def reference_partials(q, k, v, splits, used, counts, tile_n=256):
    """fp32 split partials of FA4's split rule over ``tile_n``-token blocks; (O, LSE, owned)."""
    b, _, h, d = q.shape
    qpk = h // k.shape[2]
    o = torch.zeros(splits, b, h, v.shape[-1], device=q.device)
    lse = torch.full((splits, b, h), -math.inf, device=q.device)
    owned = torch.zeros(splits, b, dtype=torch.bool, device=q.device)
    for i in range(b):
        n_blocks = -(-used[i] // tile_n)
        per_split = -(-n_blocks // counts[i])
        for s in range(counts[i]):
            owned[s, i] = True
            lo, hi = s * per_split * tile_n, min((s + 1) * per_split * tile_n, used[i])
            if hi <= lo:
                continue
            kk = k[i, lo:hi].float().repeat_interleave(qpk, 1)
            vv = v[i, lo:hi].float().repeat_interleave(qpk, 1)
            scores = torch.einsum("hd,thd->ht", q[i, 0].float(), kk) / math.sqrt(d)
            lse[s, i] = torch.logsumexp(scores, -1)
            o[s, i] = torch.einsum("ht,thd->hd", torch.softmax(scores, -1), vv)
    return o, lse, owned


# One split per sequence, for both rings, is test_decode_stats_handoff's case.
DECODE_CASES = [
    # (batch, kv len, heads, kv heads, splits, seqused, per-sequence split counts)
    (2, 1000, 10, 2, 3, None, None),  # GQA 5 padded to an 8-head tile; ragged last block
    (3, 2048, 8, 1, 4, [1, 700, 2048], [1, 2, 4]),  # empty splits and per-sequence counts
    (2, 1500, 32, 1, 8, [300, 1500], [8, 3]),  # 32-head tile with register budgets
    (1, 768, 1, 1, 2, None, None),  # MQA-free single head: tile 1, MMA N 8
]
# Constructor choices other than FA4's: (case, SwapABDecode keyword options). Each exercises one
# knob against the fp32 reference, on one of the two rings; the FA4-bitwise test below stays at
# the defaults. The split ring's head_dim_v=256 is test_decode_combines_wider_v's.
DECODE_CONFIGS = [
    ((2, 1000, 10, 2, 3, None, None), {"tile_n": 128}),
    ((2, 1024, 8, 1, 2, None, None), {"tile_k": 16}),
    ((2, 1000, 10, 2, 3, None, None), {"head_dim": 256}),
    ((3, 2048, 8, 1, 4, [1, 700, 2048], [1, 2, 4]), {"s_stages": 2, "p_stages": 6}),
    ((2, 1000, 10, 2, 3, None, None), {"s_stages": 6, "p_stages": 2, "kv_stages": (4, 4)}),
    # The shared K/V ring: a 4-stage ring is one K block, so every V wait needs the rendezvous.
    ((3, 2048, 8, 1, 4, [1, 700, 2048], [1, 2, 4]), {"shared_kv": True, "kv_stages": (2, 2)}),
    ((2, 1000, 10, 2, 3, None, None), {"shared_kv": True, "kv_stages": (1, 2)}),
    ((2, 1000, 10, 2, 3, None, None), {"shared_kv": True, "tile_k": 32}),
    ((3, 2048, 8, 1, 4, [1, 700, 2048], [1, 2, 4]), {"shared_kv": True, "tile_n": 512}),
    ((2, 1500, 32, 1, 8, [300, 1500], [8, 3]), {"shared_kv": True, "head_dim_v": 256}),
]


def decode_inputs(case, head_dim=128, head_dim_v=128):
    b, s, h, hkv, splits, used, counts = case
    torch.manual_seed(0)
    q = torch.randn(b, 1, h, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, hkv, head_dim_v, device="cuda", dtype=torch.bfloat16)
    used_t = None if used is None else torch.tensor(used, device="cuda", dtype=torch.int32)
    counts_t = None if counts is None else torch.tensor(counts, device="cuda", dtype=torch.int32)
    return q, k, v, splits, used_t, counts_t


def check_against_reference(decode, case, **config):
    b, s, *_ = case
    dims = {k: config.pop(k) for k in ("head_dim", "head_dim_v") if k in config}
    q, k, v, splits, used_t, counts_t = decode_inputs(case, **dims)
    o, lse = decode.decode_partials(
        q, k, v, splits, seqused_k=used_t, splits_per_seq=counts_t, **config
    )
    used = case[5] or [s] * b
    counts = case[6] or [splits] * b
    ro, rl, owned = reference_partials(
        q, k, v, splits, used, counts, tile_n=config.get("tile_n", 256)
    )
    o, lse = o[:, :, 0], lse[..., 0]
    torch.testing.assert_close(o[owned], ro[owned], atol=3e-3, rtol=0)
    assert torch.equal(torch.isneginf(lse[owned]), torch.isneginf(rl[owned]))
    finite = owned[..., None] & torch.isfinite(rl)
    torch.testing.assert_close(lse[finite], rl[finite], atol=1e-4, rtol=0)
    empty = owned[..., None] & torch.isneginf(rl)
    assert torch.count_nonzero(o[empty]) == 0, "an empty split must write O = 0"


# The shared K/V ring runs these cases with a knob in DECODE_CONFIGS.
@pytest.mark.parametrize("case", DECODE_CASES, ids=["gqa_ragged", "empty_splits", "h32", "h1"])
def test_decode_matches_fp32_reference(requires_blackwell, decode, case):
    check_against_reference(decode, case)


@pytest.mark.parametrize("shared_kv", [False, True], ids=["split", "shared"])
def test_decode_stats_handoff(requires_blackwell, decode, shared_kv):
    """Test the one-shot stats handoff on one small split, with a CPU fp32 reference.

    Use this case for compute-sanitizer; see "Decode racecheck" in docs/warpdrive/examples.md.
    """
    q, k, v, splits, *_ = decode_inputs((1, 1024, 8, 1, 1, None, None))
    partial, partial_lse = decode.decode_partials(q, k, v, splits, shared_kv=shared_kv)
    out, lse = partial[0], partial_lse[0]
    ref_out, ref_lse, _ = reference_partials(
        q.cpu(), k.cpu(), v.cpu(), splits, [k.shape[1]], [splits]
    )
    torch.testing.assert_close(out.cpu().float()[:, 0], ref_out[0], atol=3e-3, rtol=0)
    torch.testing.assert_close(lse.cpu()[..., 0], ref_lse[0], atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "case, config",
    DECODE_CONFIGS,
    ids=["_".join(f"{k}{v}" for k, v in c.items()) for _, c in DECODE_CONFIGS],
)
def test_decode_configs_match_fp32_reference(requires_blackwell, decode, case, config):
    check_against_reference(decode, case, **dict(config))


def test_decode_role_overrides(requires_blackwell, decode):
    """A 32-head tile with another register split: describe() shows it, the key changes, and
    the partials still match the reference."""
    from attn_gym.warpdrive import RoleConfig

    roles = {
        "softmax": RoleConfig((range(4, 8), range(8, 12)), regs=112),
        "correction": RoleConfig(range(12, 16), regs=224),
    }
    op = decode.SwapABDecode(32, roles=roles)
    text = op.describe()
    assert "softmax[0]   112" in text and "correction   224" in text and "mma_qk        64" in text
    assert op.__cute_cache_key__() != decode.SwapABDecode(32).__cute_cache_key__()
    with pytest.raises(ProtocolError, match="not roles"):
        decode.SwapABDecode(32, roles={"softmx": RoleConfig(0)}).describe()
    with pytest.raises(ValueError, match="register budgets need"):
        decode.SwapABDecode(
            32, roles={"correction": RoleConfig(range(12, 16), regs=232)}
        ).describe()
    check_against_reference(decode, (2, 1500, 32, 1, 8, [300, 1500], [8, 3]), roles=roles)


def test_decode_unaligned_roles(requires_blackwell, decode):
    """Softmax on warps 2-5 and 6-9 and correction on 10-13, no budgets: their TMEM rows
    follow the hardware warp quadrant, so the partials still match the reference."""
    from attn_gym.warpdrive import RoleConfig

    roles = {
        "mma_qk": RoleConfig(0),
        "mma_pv": RoleConfig(1),
        "softmax": RoleConfig((range(2, 6), range(6, 10))),
        "correction": RoleConfig(range(10, 14)),
        "load": RoleConfig(14),
        "q_store": RoleConfig(15),
    }
    check_against_reference(decode, (2, 1500, 32, 1, 8, [300, 1500], [8, 3]), roles=roles)


def test_decode_combines_wider_v(requires_blackwell, decode):
    """head_dim_v=256 through fa4_decode: the wider partials go through FA4's combine."""
    pytest.importorskip("flash_attn.cute.interface")
    q, k, v, splits, used_t, counts_t = decode_inputs((2, 1000, 10, 2, 3, None, None), 128, 256)
    out, lse = decode.fa4_decode(q, k, v, splits, seqused_k=used_t, splits_per_seq=counts_t)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))  # (b, h, s, d)
    kf, vf = (x.repeat_interleave(q.shape[2] // k.shape[2], 1) for x in (kf, vf))
    scores = qf @ kf.transpose(-1, -2) / math.sqrt(q.shape[-1])
    ref = (torch.softmax(scores, -1) @ vf).transpose(1, 2)
    assert out.shape == (2, 1, 10, 256)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=0)
    torch.testing.assert_close(
        lse[..., 0], torch.logsumexp(scores, -1)[:, :, 0], atol=1e-4, rtol=0
    )


def test_decode_shared_kv_describe(decode, requires_blackwell):
    """SharedKVDecode shares one exclusive ring between both MMA warps.

    FA4's two rendezvous barriers are declared first, shifting the stats-flag IDs.
    The default build stays unchanged.
    """
    op = decode.SharedKVDecode(8)
    assert op.get_name().count("_shared") == 1
    text = op.describe()
    assert "kv       Ring(12) load -> mma_pv, mma_qk" in text, text
    assert "full      1 x12, tx 16384 B" in text and "empty     1 x12" in text
    assert "owners    exclusive: one of mma_pv, mma_qk owns each epoch" in text
    assert "qk_issued    id 3  64 threads" in text and "pv_issued    id 4  64 threads" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decode.compile_decode(op)
    text = op.describe()
    # The MMAs' K/V reads and the loads are annotated: the stage handoff orders them.
    assert "V ~ K (smem)  no hazard found" in text, text
    assert "epochs    unverified (load's acquire runs inside a dynamic loop or branch)" in text
    split = decode.SwapABDecode(8)
    assert "_shared" not in split.get_name()
    assert "max_done     id 3  160 threads" in split.describe()
    assert canonical_key(op) != canonical_key(split)  # the class tells the two kernels apart


def test_decode_shared_kv_selection(decode):
    """The wrappers' ``shared_kv`` picks the class; pickling keeps it."""
    op = decode.SharedKVDecode(8)
    assert op.shared_kv
    again = pickle.loads(pickle.dumps(op))
    assert type(again) is decode.SharedKVDecode
    assert canonical_key(again) == canonical_key(op)
    q = torch.empty(1, 1, 8, 128, dtype=torch.bfloat16)
    kv = torch.empty(1, 256, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shared_kv must be a bool"):
        decode.decode_partials(q, kv, kv, 1, shared_kv=1)
    with pytest.raises(TypeError, match="shared_kv"):
        decode.SwapABDecode(8, shared_kv=True)


def test_decode_rejects_unfit_configurations(decode):
    with pytest.raises(ValueError, match="TMEM plan needs"):
        decode.SwapABDecode(32, tile_n=512)
    with pytest.raises(ValueError, match="power-of-two multiple"):
        decode.SwapABDecode(8, tile_n=384)
    with pytest.raises(ValueError, match="tile_k must be one of"):
        decode.SwapABDecode(8, tile_k=128)
    with pytest.raises(ValueError, match="head_dim must be"):
        decode.SwapABDecode(8, head_dim=96)
    with pytest.raises(ValueError, match="head_dim_v must be"):
        decode.SwapABDecode(8, head_dim_v=64)
    with pytest.raises(ValueError, match="even number"):
        decode.SwapABDecode(8, s_stages=3)
    with pytest.raises(ValueError, match="room for"):
        decode.SwapABDecode(32, tile_n=256, p_stages=12)
    # The one-shot stats flags count one load warp and one correction warpgroup.
    with pytest.raises(ValueError, match=r"roles\['load'\] must be one instance of 1 warp"):
        decode.SwapABDecode(8, roles={"load": RoleConfig(range(16, 18))})


@pytest.mark.parametrize(
    "b, s, used, shared_kv",
    [
        # FA4 picks per-sequence splits from a 32K capacity and the lengths.
        (4, 32768, [300, 32768, 9000, 1], False),
        # No seqused_k: one split count, every sequence at the capacity (no length read).
        (1, 8192, None, True),
    ],
    ids=["seqused_split", "capacity_shared"],
)
def test_decode_matches_fa4_partials_bitwise(
    requires_blackwell, decode, monkeypatch, b, s, used, shared_kv
):
    """Match FA4's split partials and combined output bit for bit.

    Per-sequence lengths use split counts from FA4's prepare kernel, then FA4's combine.
    """
    interface = pytest.importorskip("flash_attn.cute.interface")
    monkeypatch.setenv("FLASH_ATTENTION_SM100_BACKEND", "swapab")
    torch.manual_seed(0)
    h, hkv = 8, 1
    q = torch.randn(b, 1, h, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, hkv, 128, device="cuda", dtype=torch.bfloat16)
    if used is not None:
        used = torch.tensor(used, device="cuda", dtype=torch.int32)
    captured = {}
    combine = interface._flash_attn_fwd_combine

    def capture(out_partial, lse_partial, *args, **kwargs):
        captured.update(
            o=out_partial.clone(), lse=lse_partial.clone(), table=kwargs["num_splits_dynamic_ptr"]
        )
        return combine(out_partial, lse_partial, *args, **kwargs)

    capture.compile_cache = combine.compile_cache  # the combine reads it by global name
    monkeypatch.setattr(interface, "_flash_attn_fwd_combine", capture)
    out_fa4, lse_fa4, *_ = interface._flash_attn_fwd(
        q, k, v, seqused_k=used, max_seqlen_k=s, num_splits=0, return_lse=True
    )
    monkeypatch.undo()
    table = captured["table"]
    assert (table is None) == (used is None), "FA4 took unexpected per-sequence split counts"
    splits = captured["o"].shape[0]
    kw = {"seqused_k": used, "splits_per_seq": table, "shared_kv": shared_kv}
    o, lse = decode.decode_partials(q, k, v, splits, **kw)
    counts = table if table is not None else torch.full((b,), splits, device="cuda")
    owned = torch.arange(splits, device="cuda")[:, None] < counts[None, :]
    assert torch.equal(o[owned], captured["o"][owned])
    assert torch.equal(lse[..., 0][owned], captured["lse"][:, :, 0][owned])
    out, lse_out = decode.fa4_decode(q, k, v, splits, **kw)
    assert torch.equal(out, out_fa4)
    assert torch.equal(lse_out, lse_fa4)
