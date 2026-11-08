"""Small regressions from declaration and compile-only mutation fuzzing."""

import pytest

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive import (
    AtomicPersistent,
    Barriers,
    ClusterLaunchControl,
    Consumer,
    Producer,
    ProtocolError,
    Raster,
    Ring,
    Threads,
    Tma,
    TmemPlan,
    Umma,
    WorkCursor,
    WorkStream,
)
from warpdrive_fuzz.mutate import compile_mutation, mutations, run_mutation_fuzz
from warpdrive_fuzz.static import (
    collect_pair,
    collect_roles,
    declaration_role,
    run_static_fuzz,
)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_tma_requires_positive_integer_bytes(value):
    with pytest.raises(ValueError, match="positive int"):
        Tma(value)


@pytest.mark.parametrize("side", ["producer", "consumer"])
@pytest.mark.parametrize("mechanism", [Threads(), Umma(), object(), Tma])
def test_channel_rejects_unknown_mechanisms(side, mechanism):
    with pytest.raises(ValueError, match="mechanism"):
        Ring(1, **{side: mechanism})


def test_local_channel_rejects_tma_consumer():
    with pytest.raises(ValueError, match="producer mechanism"):
        Ring(1, Threads, Tma(16))


@pytest.mark.parametrize("value", [1.5, True])
@pytest.mark.parametrize("declaration", [Ring, Barriers])
def test_stage_count_requires_integer(declaration, value):
    with pytest.raises(ValueError, match="positive int"):
        declaration(value)


@pytest.mark.parametrize("counts", [[1.5], [True], 1.5, "x"])
def test_raw_barrier_counts_require_positive_integers(counts):
    with pytest.raises(ValueError, match="positive ints"):
        Barriers(1, counts=counts)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_tmem_rejects_nonpositive_or_noninteger_widths(value):
    with pytest.raises(ValueError, match="positive"):
        TmemPlan().region("a", value)
    with pytest.raises(ValueError, match="positive int"):
        TmemPlan().region("a", 32).alias("b", of="a", offset=0, cols=value)


def test_tmem_rejects_duplicate_names_before_plan_corruption():
    plan = TmemPlan().region("a", 32, 2).alias("b", of="a", offset=8, cols=16)
    with pytest.raises(ValueError, match="already declared"):
        plan.region("a", 16)
    with pytest.raises(ValueError, match="already declared"):
        plan.alias("a", of="a", offset=0, cols=16)
    assert plan.used_cols == 64
    assert plan.col("b", 1) == 40


def test_tmem_unknown_alias_parent_is_protocol_error():
    with pytest.raises(ProtocolError, match="no TMEM region"):
        TmemPlan().alias("b", of="missing", offset=0, cols=16)


@pytest.mark.parametrize("warps", [[], -1, 32, range(-1, 1), range(31, 33), True, 1.5, None])
def test_role_rejects_warps_outside_cuda_block_domain(warps):
    with pytest.raises(ValueError, match="role warps"):
        declaration_role(warps)


@pytest.mark.parametrize("regs", [0, 23, 25, 257, 24.0, True, "40"])
def test_role_rejects_illegal_setmaxnreg_budget(regs):
    with pytest.raises(ValueError, match="role regs"):
        declaration_role(0, regs=regs)


def test_kernel_without_roles_has_clear_declaration_error():
    with pytest.raises(ValueError, match="at least one role"):
        collect_roles({})


@pytest.mark.parametrize("warp", [-1, 32, True, 1.5])
def test_stream_scheduler_warp_has_cuda_block_domain(warp):
    with pytest.raises(ValueError, match="scheduler_warp"):
        WorkStream(scheduler_warp=warp)


@pytest.mark.parametrize("record", [-1, True, 1.5])
def test_stream_record_is_nonnegative_integer(record):
    with pytest.raises(ValueError, match="record"):
        WorkStream(record=record)


def test_raster_swizzle_rejects_bool():
    with pytest.raises(ValueError, match="positive int"):
        Raster(swizzle=True)


@pytest.mark.parametrize(
    "override",
    [
        {"full_counts": 1.5},
        {"full_counts": [True]},
        {"full_mask": True},
        {"full_counts": {False: [1]}},
    ],
)
def test_overrides_reject_noninteger_counts_and_masks(override):
    with pytest.raises(ProtocolError, match="override"):
        collect_pair(Ring(1, override=override))


def test_work_cursor_annotation_requires_stream_declaration():
    with pytest.raises(ProtocolError, match="no WorkStream named"):
        collect_roles({"worker": declaration_role(params=(("work", WorkCursor),))})


@pytest.mark.parametrize("schedule", [AtomicPersistent(), ClusterLaunchControl()])
def test_stream_does_not_overwrite_user_channel(schedule):
    with pytest.raises(ProtocolError, match="reserved"):
        collect_roles(
            {"worker": declaration_role(params=(("work", WorkCursor),))},
            {"work": WorkStream(schedule=schedule), "work_ring": Ring(1)},
        )


def test_shared_channel_attributes_are_rejected_before_binding():
    ch = Ring(1)
    with pytest.raises(ProtocolError, match="same declaration"):
        collect_roles(
            {"worker": declaration_role(params=(("a", Producer), ("b", Consumer)))},
            {"a": ch, "b": ch},
        )


@pytest.mark.parametrize(
    "name,message", [("compound_condition", "condition"), ("advance_before_work", "after advance")]
)
@pytest.mark.parametrize("seed", range(4), ids=["oneshot", "static", "atomic", "clc"])
def test_tile_cursor_rejects_undrained_condition_and_next_item_reads(name, message, seed):
    mutation = next(m for m in mutations() if m.name == name)
    with pytest.raises(Exception, match=message) as info:
        compile_mutation(mutation, seed=seed)
    assert "ProtocolError" in repr(info.value)


@pytest.mark.fuzz
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_declaration_fuzz_quick(seed):
    report = run_static_fuzz(seed, budget=128)
    assert not report.findings, report.findings
    assert report.cases == report.accepted + report.rejected == 128
    assert report.accepted > 0 and report.rejected > 0


@pytest.mark.fuzz
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_mutation_fuzz_quick(seed):
    report = run_mutation_fuzz(seed, budget=12)
    assert not report.findings, report.findings
    assert report.cases == report.accepted + report.rejected + report.obligations == 12
