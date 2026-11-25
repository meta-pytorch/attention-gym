"""Bulk copies that read shared memory: the PTX checker's rules and the paths it follows.

A role that writes shared memory and stores it with TMA needs ``fence.proxy.async`` between the two
and, before it returns, a commit and a wait for the copy. ``check_bulk_copies`` is exercised on
hand-written PTX, then on compiled kernels with one of those instructions removed.
"""

import importlib
import re

import pytest

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.ptx_check import capture_ptx, check_bulk_copies, read_ptx

STORE = "\tst.shared.b32 [%r4], %r5;"
FENCE = "\tfence.proxy.async.shared::cta;"
COPY = "\tcp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd1, {%r6, %r7}], [%r4];"
LOAD = (
    "\tcp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
    " [%r4], [%rd1, {%r6, %r7}], [%r8];"
)
REDUCE = (
    "\tcp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32"
    " [%r6], [%r4], %r7, [%r8];"
)
STMATRIX = "\tstmatrix.sync.aligned.m8n8.x4.shared.b16 [%r4], {%r5, %r6, %r7, %r8};"
COMMIT = "\tcp.async.bulk.commit_group;"
WAIT = "\tcp.async.bulk.wait_group.read 0;"
# The warp index as CuTeDSL computes it, and each lane's index.
WARP = "\tmov.u32 %r1, %tid.x;\n\tshr.u32 %r2, %r1, 5;\n\tshfl.sync.idx.b32 %r3, %r2, 0, 31, -1;"
LANE = "\tmov.u32 %r3, %laneid;"


def _rules(*lines: str, block: str = "", end: str = "\tret;") -> list[str]:
    """The rules a kernel of ``lines`` breaks, with the directive ``block``, leaving by ``end``."""
    registers = "\t.reg .b32 %r<9>;\n\t.reg .b64 %rd<9>;\n\t.reg .pred %p<9>;"
    body = "\n".join((registers, *lines, end))
    ptx = f".visible .entry k(\n\t.param .u64 p\n)\n{block}\n{{\n{body}\n}}\n"
    return [f.rule for f in check_bulk_copies(ptx)]


def _when(index: int, n: int, *lines: str) -> str:
    """``lines`` run by the threads whose %r3 is ``index``, as the kernel's ``n``-th such block."""
    test = f"\tsetp.eq.b32 %p{n}, %r3, {index};\n\t@!%p{n} bra $L_{n};"
    return "\n".join((test, *lines, f"$L_{n}:"))


@pytest.mark.parametrize(
    "lines, rules",
    [
        ((STORE, FENCE, COPY, COMMIT, WAIT), []),
        ((STORE, COPY, COMMIT, WAIT), ["proxy"]),
        ((STORE, "\t@%p1 " + FENCE.strip(), COPY, COMMIT, WAIT), ["proxy"]),  # may not run
        ((STORE, "\tfence.proxy.async.global;", COPY, COMMIT, WAIT), ["proxy"]),
        ((COPY, COMMIT), ["exit"]),
        ((COPY, WAIT), ["exit"]),  # a wait covers committed groups only
        ((COPY, COMMIT, "\tcp.async.bulk.wait_group.read 1;"), ["exit"]),  # one group may remain
        ((COPY, COMMIT, "\tcp.async.bulk.wait_group 0;"), []),
        ((STORE, LOAD), []),  # a TMA load writes shared memory
        ((STMATRIX, COPY, COMMIT, WAIT), ["proxy"]),
        (("\tst.volatile.shared.b32 [%r4], %r5;", COPY, COMMIT, WAIT), ["proxy"]),
        (("\tatom.relaxed.cta.shared.add.u32 %r6, [%r4], %r5;", COPY, COMMIT, WAIT), ["proxy"]),
        (("\tst.global.b32 [%rd1], %r5;", COPY, COMMIT, WAIT), []),
        ((STORE, REDUCE), ["proxy"]),  # it completes on an mbarrier: no group to wait for
        ((COPY, COMMIT, "\tcp.async.bulk.wait_group.read 0x0;"), []),
        ((COPY, COMMIT, "\t@ %p1 " + WAIT.strip()), ["exit"]),
    ],
    ids=[
        "fenced-store",
        "unfenced-store",
        "guarded-fence",
        "global-fence",
        "no-wait",
        "no-commit",
        "wait-leaves-a-group",
        "full-wait",
        "load",
        "stmatrix",
        "volatile-store",
        "scoped-atom",
        "global-store",
        "reduce-to-cluster",
        "hex-wait",
        "spaced-guard",
    ],
)
def test_rules(lines, rules):
    assert _rules(*lines) == rules


@pytest.mark.parametrize("who", [WARP, LANE], ids=["warp", "lane"])
def test_roles_are_checked_apart(who):
    """One role stores to shared memory and another issues a TMA store: no thread runs both, but
    a path through both branches does, which is what a kernel gets whose block is unknown or
    not one-dimensional."""
    lines = (
        who,
        _when(0, 1, STORE),
        _when(1, 2, COPY, COMMIT, WAIT),
    )
    assert _rules(*lines, block=".reqntid 64, 1, 1") == []
    assert _rules(*lines) == ["proxy"]
    for block in (".reqntid 64, 2, 1", ".maxntid 64, 1, 1", ".reqntid 48"):
        # %tid.x does not give the lane, the launch may be (32, 2, 1), a warp is partial
        assert _rules(*lines, block=block) == ["proxy"]
    assert _rules(*lines, block=".reqntid 0x40") == []


@pytest.mark.parametrize("who", [WARP, LANE], ids=["warp", "lane"])
def test_a_repeated_condition_pairs_a_copy_with_its_wait(who):
    """The store and its wait sit under two tests of one condition; under different ones the
    storing threads return without waiting."""
    store = _when(0, 1, COPY, COMMIT)
    assert _rules(who, store, _when(0, 2, WAIT), block=".reqntid 64, 1, 1") == []
    assert _rules(who, store, _when(1, 2, WAIT), block=".reqntid 64, 1, 1") == ["exit"]


def test_a_guard_on_the_thread_id_is_decided():
    lines = (
        WARP,
        "\tsetp.eq.b32 %p2, %r3, 0;",
        "\t@%p2 " + COPY.strip(),
        "\t@%p2 " + COMMIT.strip(),
    )
    assert _rules(*lines, "\t@%p2 " + WAIT.strip(), block=".reqntid 64, 1, 1") == []
    assert _rules(*lines, "\t@!%p2 " + WAIT.strip(), block=".reqntid 64, 1, 1") == ["exit"]


def test_only_a_single_result_takes_the_compared_value():
    """``setp``'s second result is the complement: the wait under it skips the copying lane."""
    lines = (LANE, "\tsetp.eq.b32 %p1|%p2, %r3, 0;", "\t@%p1 " + COPY.strip())
    waits = ("\t@%p1 " + COMMIT.strip(), "\t@%p2 " + WAIT.strip())
    assert _rules(*lines, *waits, block=".reqntid 64, 1, 1") == ["exit"]


def _copy_where(*test: str, block: str = ".reqntid 32") -> list[str]:
    """The rules broken when the lanes whose %p1 ``test`` sets issue an unwaited copy: ["exit"]
    unless %p1 is known false in every thread."""
    copy = ("\t@%p1 " + COPY.strip(), "\t@%p1 " + COMMIT.strip())
    return _rules(LANE, *(f"\t{line}" for line in test), *copy, block=block)


NONZERO = "setp.ne.b32 %p1, %r5, 0;"


@pytest.mark.parametrize(
    "write",
    [
        "tex.2d.v4.u32.s32 {%r2, %r6, %r7, %r5}|%p3, [%rd2, {%r6, %r7}];",  # vector|predicate
        "ld.shared.b32 %r5, [%r4];",
        "@%p3 mov.b32 %r5, 1;",  # on a predicate that is not known: the writes disagree
        "setp.eq.b32 %p2, %r3, 1;\n\t@%p2 ld.shared.b32 %r5, [%r4];",  # only lane 1 reaches it
    ],
    ids=["vector-and-predicate", "load", "disagreeing-write", "one-lane"],
)
def test_every_written_register_loses_its_value(write):
    """%r5 is zero in every lane until something else may write it."""
    assert _copy_where("mov.b32 %r5, 0;", NONZERO) == []
    assert _copy_where("mov.b32 %r5, 0;", write, NONZERO) == ["exit"]


@pytest.mark.parametrize(
    "write",
    [
        "ld.global.v4.u32 %v, [%rd1];",
        "mov.v4.u32 %v, {%r2, %r2, %r2, %r2};",
        "ld.global.u32 %v.r, [%rd1];",
    ],
    ids=["whole-vector", "vector-move", "aliased-component"],
)
def test_a_vector_and_its_components_are_one_register(write):
    declare = ".reg .v4 .u32 %v;"
    test = "setp.ne.u32 %p1, %v.x, 0;"
    assert _copy_where(declare, "mov.u32 %v.x, 0;", write, test) == ["exit"]


def test_each_warp_has_its_own_threads():
    """Warp 1 issues the copy: it exists in a block of 64 threads, not in one of 32."""
    lines = (WARP, _when(1, 1, COPY, COMMIT))
    assert _rules(*lines, block=".reqntid 64") == ["exit"]
    assert _rules(*lines, block=".reqntid 32") == []


@pytest.mark.parametrize(
    "test, taken",
    [
        # %r3 is the lane, 0 to 31. A test that no lane passes drops the copy.
        (("setp.lt.u32 %p1, %r3, 0;",), False),
        (("setp.le.u32 %p1, %r3, 0;",), True),
        (("setp.gt.u32 %p1, %r3, 31;",), False),
        (("setp.ge.u32 %p1, %r3, 31;",), True),
        (("setp.eq.b32 %p1, %r3, 035;",), True),  # octal 29
        (("setp.eq.b32 %p1, %r3, 0x20;",), False),
        (("sub.s32 %r5, %r3, 32;", "setp.gt.s32 %p1, %r5, 100;"), False),  # negative when signed
        (("sub.s32 %r5, %r3, 32;", "setp.gt.u32 %p1, %r5, 100;"), True),
        (("add.s32 %r5, %r3, 1;", NONZERO.replace("ne", "eq")), False),
        (("mul.lo.s32 %r5, %r3, 2;", "setp.eq.b32 %p1, %r5, 62;"), True),
        (("mul.lo.s32 %r5, %r3, 2;", "setp.eq.b32 %p1, %r5, 33;"), False),
        (("and.b32 %r5, %r3, 32;", NONZERO), False),
        (("or.b32 %r5, %r3, 32;", "setp.lt.u32 %p1, %r5, 32;"), False),
        (("xor.b32 %r5, %r3, %r3;", NONZERO), False),
        (("not.b32 %r5, %r3;", "setp.lt.u32 %p1, %r5, 0xFFFFFFE0;"), False),
        (("min.u32 %r5, %r3, 4;", "setp.gt.u32 %p1, %r5, 4;"), False),
        (("max.u32 %r5, %r3, 4;", "setp.lt.u32 %p1, %r5, 4;"), False),
        (("shl.b32 %r5, %r3, 1;", "setp.gt.u32 %p1, %r5, 62;"), False),
        (("shr.u32 %r5, %r3, 4;", "setp.gt.u32 %p1, %r5, 1;"), False),
        (("sub.s32 %r5, %r3, 32;", "shr.s32 %r6, %r5, 40;", "setp.ne.s32 %p1, %r6, -1;"), False),
        (
            ("setp.eq.b32 %p2, %r3, 5;", "selp.b32 %r5, 7, 9, %p2;", "setp.eq.b32 %p1, %r5, 8;"),
            False,
        ),
        (
            ("setp.eq.b32 %p2, %r3, 5;", "selp.b32 %r5, 0, 9, %p2;", NONZERO.replace("ne", "eq")),
            True,
        ),
        (("setp.eq.b32 %p2, %r3, 5;", "not.pred %p3, %p2;", "and.pred %p1, %p2, %p3;"), False),
        # A shuffle is one lane's value only with the full clamp and mask.
        (("shfl.sync.idx.b32 %r5, %r3, 0, 31, -1;", NONZERO), False),
        (("shfl.sync.idx.b32 %r5, %r3, 0, 15, -1;", NONZERO), True),
        (("shfl.sync.idx.b32 %r5, %r3, 0, 31, 0xFFFF;", NONZERO), True),
    ],
    ids=lambda case: (
        None if isinstance(case, bool) else " ".join(line.split()[0] for line in case)
    ),
)
def test_values_that_follow_from_the_thread_id(test, taken):
    assert _copy_where(*test) == (["exit"] if taken else [])


@pytest.mark.parametrize(
    "end", ["\tret;", "\tret.uni;", "\texit;", "", "\tbra $L_end;\n$L_end:"], ids=repr
)
def test_every_way_out_of_the_entry_is_a_return(end):
    assert _rules(COPY, COMMIT, end=end) == ["exit"]
    assert _rules(COPY, COMMIT, WAIT, end=end) == []


def test_an_unknown_condition_takes_both_branches():
    """A condition on data (here a loaded word) decides nothing."""
    load = "\tld.shared.b32 %r3, [%r4];"
    assert _rules(COPY, COMMIT, load, _when(0, 1, WAIT), block=".reqntid 64, 1, 1") == ["exit"]


def _without(ptx: str, pattern: str) -> list[str]:
    """``ptx`` with each instruction matching ``pattern`` blanked in turn."""
    lines = ptx.split("\n")
    sites = [n for n, line in enumerate(lines) if re.match(pattern, line)]
    return ["\n".join(lines[:n] + [""] + lines[n + 1 :]) for n in sites]


@pytest.mark.parametrize(
    "module, kernel, build",
    [
        ("persistent_gemm", "PersistentGemm", "compile_persistent_gemm"),
        ("grouped_gemm", "GroupedGemm", "compile_grouped_gemm"),
    ],
    ids=["persistent_gemm", "grouped_gemm"],
)
def test_epilogue_store_needs_its_fence_and_final_wait(
    requires_blackwell, tmp_path, module, kernel, build
):
    """The epilogue writes each subtile to shared memory, fences, and warp 0 TMA-stores it and
    waits for the last stores after the work loop. The compiled kernel is clean, and removing
    any one fence or the final wait is reported."""
    mod = importlib.import_module(f"attn_gym.warpdrive.examples.{module}")
    with capture_ptx(tmp_path) as out:
        getattr(mod, build).__wrapped__(getattr(mod, kernel)())  # a cached kernel has no PTX
    (path,) = out.glob("*.ptx")
    ptx = read_ptx(path)
    assert not check_bulk_copies(ptx)
    unfenced = _without(ptx, r"\s*fence\.proxy\.async")
    assert unfenced and all({f.rule for f in check_bulk_copies(m)} == {"proxy"} for m in unfenced)
    (unwaited,) = _without(ptx, r"\s*cp\.async\.bulk\.wait_group\.read\s+0;")
    assert {f.rule for f in check_bulk_copies(unwaited)} == {"exit"}
