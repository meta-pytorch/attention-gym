"""Test round-robin ownership and external channel sides without launching kernels.

Instance j of k owns epochs j, j + k, ... on either side. The exclusive-ring ownership
check treats this as skipping other instances' epochs. GPU coverage is in the decode
toy and example. External sides have declaration, collection, and describe() checks;
examples/raw_interop.py runs one.
"""

import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Instance,
    OneShot,
    Producer,
    Ring,
    Sharing,
    StaticPersistent,
    Threads,
    Tma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

RR = Sharing.ROUND_ROBIN


class Probe(WarpSpecialized):
    """Per work item, the producer instances acquire ``fills[j]`` epochs of ``ch`` and the
    consumer instances wait ``waits[j]``, in static loops (``dynamic``: the consumers' in a
    dynamic one). ``side`` names the round-robin side: its role has two instances."""

    def __init__(self, fills, waits, schedule=None, dynamic=False):
        self.fills, self.waits, self.dynamic = fills, waits, dynamic
        self.ch = Ring(2, Threads, Threads, **{f"{self.side}_sharing": RR})
        self.work = WorkStream(schedule=schedule or OneShot())

    @cute.jit
    def fill(self, ch, work, count: int):
        while work.valid:
            for _ in cutlass.range_constexpr(count):
                with ch.produce():
                    pass
            work.advance()
        return ch, work

    @cute.jit
    def drain(self, ch, work, count: int):
        while work.valid:
            if cutlass.const_expr(self.dynamic):
                for _ in cutlass.range(count, unroll=1):
                    ch.wait().release()
            else:
                for _ in cutlass.range_constexpr(count):
                    ch.wait().release()
            work.advance()
        return ch, work

    @cute.kernel
    def kernel(self, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace())

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        bound = self.work.bind((n, 1, 1), max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix("warpdrive_sharing_probe")
        self.kernel(bound).launch(**self.launch_config(bound), stream=stream)


class Consumers(Probe):
    side = "consumer"

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer, work: WorkCursor):
        return self.fill(ch, work, self.fills[0])

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, stage: Instance, ch: Consumer, work: WorkCursor):
        return self.drain(ch, work, self.waits[stage])


class Producers(Probe):
    side = "producer"

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, ctx, *, stage: Instance, ch: Producer, work: WorkCursor):
        return self.fill(ch, work, self.fills[stage])

    @role(warps=2)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer, work: WorkCursor):
        return self.drain(ch, work, self.waits[0])


class BothSides(Producers):
    """Round-robin producers and consumers: the check needs a joining side to count against."""

    def __init__(self):
        super().__init__((2, 2), (2, 2))
        self.ch = Ring(2, Threads, Threads, producer_sharing=RR, consumer_sharing=RR)

    @role(warps=[range(2, 3), range(3, 4)])
    @cute.jit
    def cons(self, ctx, *, stage: Instance, ch: Consumer, work: WorkCursor):
        return self.drain(ch, work, self.waits[stage])


@jit_cache
def compile_probe(op):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name="warpdrive_sharing_probe")


def epochs(op) -> str:
    compile_probe(op)
    return next(line.strip() for line in op.describe().splitlines() if "epochs  " in line)


@pytest.mark.parametrize(
    "op, verdict",
    [
        (Consumers((4,), (2, 2)), "verified: 4 epochs per work item (cons[0] 2, cons[1] 2)"),
        # One item per CTA: instance 0 may take the odd epoch out.
        (Consumers((3,), (2, 1)), "verified: 3 epochs per work item (cons[0] 2, cons[1] 1)"),
        (Consumers((4,), (2, 2), StaticPersistent()), "verified: 4 epochs per work item (cons"),
        (Producers((2, 2), (4,)), "verified: 4 epochs per work item (prod[0] 2, prod[1] 2)"),
        (Producers((2, 1), (3,)), "verified: 3 epochs per work item (prod[0] 2, prod[1] 1)"),
        (
            Consumers((4,), (2, 2), dynamic=True),
            "unverified (cons[0]'s wait runs inside a dynamic loop or branch)",
        ),
        (BothSides(), "unverified (both sides take turns)"),
    ],
)
def test_round_robin_ownership_verdicts(op, verdict):
    assert epochs(op).startswith(f"epochs    {verdict}")


@pytest.mark.parametrize(
    "op, error",
    [
        (
            Consumers((4,), (2, 1)),
            (
                r"epoch 3 of 4 per work item is waited by no consumer; round-robin consumers take "
                r"epochs j, j \+ k, \.\.\. in turn"
            ),
        ),
        (Consumers((4,), (3, 2)), r"cons\[0\] waits on epoch 4, but prod fills only 4 per"),
        # A persistent instance continues into the next item, where the odd epoch out would
        # shift every instance's turn.
        (
            Consumers((3,), (2, 1), StaticPersistent()),
            r"cons\[0\] advances 4 epochs per work item \(2 per wait\), but prod acquires 3",
        ),
        (
            Producers((2, 0), (2,)),
            r"epoch 1 of 2 per work item is acquired by no producer; round-robin producers",
        ),
        (
            Producers((1, 1), (3,)),
            r"cons waits on epoch 2, but prod\[0\] and prod\[1\] fill only 2 per work item",
        ),
        (
            Producers((2, 1), (3,), StaticPersistent()),
            (
                r"prod\[0\] acquires 2 of the 3 epochs per work item; a round-robin producer "
                r"continues into the next item"
            ),
        ),
    ],
)
def test_round_robin_trip_counts_that_do_not_add_up_raise(op, error):
    _raises_protocol(error, lambda: compile_probe(op))


# External sides


def test_external_side_declaration_errors():
    with pytest.raises(ValueError, match="external must be 'producer' or 'consumer'"):
        Ring(2, external="both", override={"empty_counts": 1})
    with pytest.raises(ValueError, match=r"needs override=\{'empty_counts': \.\.\.\}"):
        Ring(2, external="consumer")
    with pytest.raises(ValueError, match=r"needs override=\{'full_counts': \.\.\.\}"):
        Ring(2, external="producer", override={"empty_counts": 1})
    with pytest.raises(ValueError, match="needs an mbarrier channel"):
        Ring(2, sync="named", external="consumer")
    with pytest.raises(ValueError, match="takes Sharing.JOIN on both sides"):
        Ring(2, external="consumer", override={"empty_counts": 1}, producer_sharing=RR)
    # Settings of the external side would never take effect.
    with pytest.raises(ValueError, match="takes no consumer_roles or release_fence"):
        Ring(
            2,
            external="consumer",
            override={"empty_counts": 1},
            consumer_roles=("cons",),
            release_fence="tmem_store",
        )
    with pytest.raises(ValueError, match="takes no commit_fence"):
        Ring(2, external="producer", override={"full_counts": 1}, commit_fence="smem_async")


def _kernel(channels, roles):
    cls = type("Probe", (WarpSpecialized,), dict(roles))

    def init(self):
        for name, make in channels.items():
            setattr(self, name, make())

    cls.__init__ = init
    return cls()


@role(warps=0)
@cute.jit
def _prod(self, ctx, *, ch: Producer):
    return ch


@role(warps=1)
@cute.jit
def _cons(self, ctx, *, ch: Consumer):
    return ch


def test_external_side_binds_no_role():
    external = {"ch": lambda: Ring(2, external="consumer", override={"empty_counts": 1})}
    text = _kernel(external, {"prod": _prod}).describe()
    assert "ch  Ring(2)  prod -> raw code\n" in text, text
    assert "empty     1 x2" in text, text
    assert (
        "external  consumer side driven by raw code; empty counts are the override's (unverified)"
    ) in text
    _raises_protocol(
        r"channel ch: its consumer side is external \(driven by raw code\), but roles "
        r"\['cons'\] take a Consumer end",
        _kernel(external, {"prod": _prod, "cons": _cons}).describe,
    )
    # The managed side still needs its role.
    _raises_protocol(
        "channel ch: slot 0 has no producer role",
        _kernel(external, {"other": _cons}).describe,
    )


class ExternalTmaProducer(WarpSpecialized):
    """Raw code fills ``ch`` by TMA (here: nobody; compiled only); ``cons`` waits once."""

    def __init__(self):
        self.ch = Ring(1, Tma(128), external="producer", override={"full_counts": 1})

    @role(warps=0)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        ch.wait().release()
        return ch

    @cute.kernel
    def kernel(self):
        self.begin_roles()
        self.run_roles(SimpleNamespace())

    @cute.jit
    def __call__(self, unused: cute.Tensor, stream):
        self.kernel.set_name_prefix("warpdrive_sharing_external_tma")
        self.kernel().launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_external(op):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, name="warpdrive_sharing_external_tma")


def test_an_external_tma_producer_is_not_tma_checked():
    op = ExternalTmaProducer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_external(op)
    assert not [w for w in caught if "TMA check" in str(w.message)]
    text = op.describe()
    assert "ch  Ring(1)  raw code -> cons\n" in text, text
    assert not [line for line in text.splitlines() if line.split()[:1] == ["tma"]], text


def test_external_channels_stay_cta_local():
    with pytest.raises(ValueError, match="its TMA copies take no multicast or fills"):
        Ring(2, Tma(128, multicast="m"), external="producer", override={"full_counts": 1})

    @role(warps=2, cta="leader")
    @cute.jit
    def lead(self, ctx, *, ch: Consumer):
        return ch

    external = {"ch": lambda: Ring(2, external="producer", override={"full_counts": 1})}
    op = _kernel(external, {"cons": _cons, "lead": lead})
    op.cluster = CtaPair()
    _raises_protocol(
        "channel ch: its consumer roles run in different CTAs, but its producer side is "
        "external, driven by raw code in each CTA",
        op.describe,
    )
