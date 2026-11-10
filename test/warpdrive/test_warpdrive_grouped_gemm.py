import math
import random

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import Int32, cute

from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    OneShot,
    RoleConfig,
    StaticPersistent,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples import grouped_gemm as gg
from attn_gym.warpdrive.examples.grouped_gemm import (
    GroupedGemm,
    compile_grouped_gemm,
    grouped_gemm,
    grouped_gemm_reference,
)


def _offsets(sizes: list[int]) -> torch.Tensor:
    return torch.tensor(sizes, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)


def _inputs(sizes: list[int], n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(sum(sizes), k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(len(sizes), n, k, device="cuda", dtype=torch.bfloat16)
    return a, b, _offsets(sizes)


def _bf16_baseline_error(a, b, sizes, ref) -> float:
    """The error of per-group bf16 ``torch.mm`` against the fp32 reference."""
    worst, start = 0.0, 0
    for g, size in enumerate(sizes):
        if size:
            out = (a[start : start + size] @ b[g].T).float()
            worst = max(worst, (out - ref[start : start + size]).abs().max().item())
        start += size
    return worst


def _check(sizes, n, k, rows_after: int = 0, **config):
    """Compare against per-group fp32 matmuls; rows past the last offset must stay untouched."""
    a, b, offsets = _inputs(sizes, n, k)
    if rows_after:
        a = torch.cat([a, torch.randn(rows_after, k, device="cuda", dtype=torch.bfloat16)])
    out = torch.full((a.shape[0], n), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out=out, **config) is out
    torch.cuda.synchronize()
    ref = grouped_gemm_reference(a, b, offsets)
    rows = sum(sizes)
    err = (out[:rows].float() - ref[:rows]).abs().max().item() if rows else 0.0
    torch_err = _bf16_baseline_error(a, b, sizes, ref)
    assert err <= 2 * torch_err + 1e-3, (sizes, n, k, config, err, torch_err)
    assert torch.isnan(out[rows:]).all(), "rows past the last offset were written"


# (sizes, N, K, rows of A past the last offset): one compiled kernel covers them all.
SIZE_CASES = [
    ([128], 256, 64, 0),  # single group
    ([333], 520, 136, 0),  # single ragged group
    ([256] * 8, 512, 256, 0),
    ([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136, 0),
    ([0, 0, 200, 0, 50, 0, 0, 7], 264, 72, 0),  # empty leading and trailing groups
    ([0] * 7 + [3], 64, 64, 0),  # all but one empty
    ([37] * 64, 1032, 200, 0),
    ([0, 1, 127, 128, 129, 255, 256, 257] * 8, 2056, 264, 0),
    ([300, 0, 128, 77], 512, 256, 131),  # A holds more rows (padded routing buffers)
    ([5, 0, 1000], 2056, 1032, 0),
    ([0, 0, 0, 5], 8, 8, 0),
]


def test_grouped_gemm_shapes_and_routings_reuse_one_compile(requires_blackwell):
    """Shapes, G and the routing are runtime values of one compiled kernel."""
    sizes, n, k, rows_after = SIZE_CASES[0]
    _check(sizes, n, k, rows_after)  # the default configuration is compiled (or loaded) once
    before = compile_grouped_gemm.cache_info()
    for sizes, n, k, rows_after in SIZE_CASES[1:]:
        _check(sizes, n, k, rows_after)
    after = compile_grouped_gemm.cache_info()
    assert after.currsize == before.currsize, "a shape or offset change compiled another kernel"
    assert after.misses == before.misses
    assert after.hits == before.hits + len(SIZE_CASES) - 1  # one launch each
    # The wrapper's launch cache also keys on the configuration, not the shapes.
    ops = len(gg._OPS)
    _check([64, 64], 64, 64)
    assert len(gg._OPS) == ops


def _skewed_sizes(groups: int, total: int, seed: int = 0) -> list[int]:
    """Heavy-tailed routing as in the benchmark: a few large groups, an eighth of them empty."""
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(groups, generator=gen) ** 3
    weights[torch.randperm(groups, generator=gen)[: groups // 8]] = 0
    sizes = (weights / weights.sum() * total).floor().to(torch.int64)
    sizes[sizes.argmax()] += total - int(sizes.sum())
    return sizes.tolist()


SCHEDULE_SIZES = {
    "skewed_96": _skewed_sizes(96, 6000),
    "empty_runs": [0] * 30 + [5] + [0] * 40 + [300] + [0] * 9,
}


@pytest.mark.parametrize(
    "schedule",
    [OneShot(), StaticPersistent(), AtomicPersistent(), ClusterLaunchControl()],
    ids=lambda s: s.get_key(),
)
def test_grouped_gemm_schedules(requires_blackwell, schedule):
    """Every schedule runs every tile once, whether a unit's indices cross several groups per
    item (a full grid) or walk through all of them (7 CTAs)."""
    for sizes in SCHEDULE_SIZES.values():
        for max_ctas in (None, 7):
            _check(sizes, 264, 72, schedule=schedule, max_ctas=max_ctas)


@pytest.mark.parametrize(
    "schedule", [AtomicPersistent(), StaticPersistent()], ids=lambda s: s.get_key()
)
def test_grouped_gemm_cta_pair(requires_blackwell, schedule):
    """256-row MMA tiles: the peer CTA's half of a ragged tile can hold no valid rows at all."""
    _check([1000, 24, 0, 129, 128, 1, 256, 300, 5], 520, 136, schedule=schedule, cluster=CtaPair())


CONFIGS = [
    {"tile_n": 128},
    {"tile_n": 224, "epi_tile_n": 16, "c_stages": 3},
    {"ab_stages": 2, "acc_stages": 1},
    {"roles": {"epilogue": RoleConfig(range(4), regs=152)}},
]


@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=["_".join(f"{k}{v}" for k, v in c.items() if k != "roles") or "roles" for c in CONFIGS],
)
def test_grouped_gemm_configs(requires_blackwell, config):
    _check([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136, **config)


def test_grouped_gemm_unaligned_epilogue(requires_blackwell):
    """Epilogue on warps 2-5 (no budgets), straddling two warpgroups: its TMEM rows follow the
    hardware warp quadrant, its TMA-store warp the role."""
    roles = {
        "load": RoleConfig(0),
        "mma": RoleConfig(1, cta="leader"),
        "epilogue": RoleConfig(range(2, 6)),
    }
    _check([1000, 24, 0, 129, 128, 1, 256, 300], 520, 136, roles=roles)


def _reference_tiles(sizes: list[int], tile_m: int) -> set[tuple[int, int, int]]:
    tiles, start = set(), 0
    for g, size in enumerate(sizes):
        for row0 in range(start, start + size, tile_m):
            tiles.add((g, row0, min(start + size - row0, tile_m)))
        start += size
    return tiles


def _python_locate_tile(mt: int, offsets: list[int], tile_m: int, total_m: int | None = None):
    """Mirror of ``GroupedTiles``' decode: ``(found, group, row0, rows)`` of m-tile ``mt``, the
    groups' tiles numbered back to back, group ends clamped to ``[previous end, total_m]``."""
    total_m = offsets[-1] if total_m is None else total_m
    tile = row = 0
    for group, end in enumerate(offsets):
        end = min(max(end, row), total_m)
        next_tile = tile + -(-(end - row) // tile_m)
        if tile <= mt < next_tile:
            row0 = row + (mt - tile) * tile_m
            return True, group, row0, min(end - row0, tile_m)
        tile, row = next_tile, end
    return False, len(offsets) - 1, row, 0


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize(
    "sizes",
    [
        [128],
        [1000, 24, 0, 129, 128, 1, 256, 300],
        [0, 0, 200, 0, 50, 0, 0, 7],
        [0] * 7 + [3],
        [37] * 64,
        [0, 1, 127, 128, 129, 255, 256, 257] * 16,
    ],
    ids=["one", "ragged", "empty_runs", "last_only", "small_64", "mixed_128"],
)
def test_grouped_tiles_decode_is_a_bijection(sizes, tile_m):
    """Every (group, row0, rows) tile of the work appears exactly once among the
    ``ceil(total_M / tile_m) + G`` m-tile slots the host launches, in order; the slots past
    them are dropped."""
    offsets = list(torch.tensor(sizes).cumsum(0).tolist())
    slots = math.ceil(sum(sizes) / tile_m) + len(sizes)
    kept = [_python_locate_tile(mt, offsets, tile_m) for mt in range(slots)]
    found = [f for f, *_ in kept]
    assert found == sorted(found, reverse=True), "a dropped slot before a kept one"
    kept = [(g, row0, rows) for f, g, row0, rows in kept if f]
    assert kept == sorted(_reference_tiles(sizes, tile_m))


class DecodeProbe:
    """Decodes ``indices`` in order on one warp with ``GroupedTiles``, threading the state, and
    decodes each again from a fresh ``start`` state: ``out[i] = (keep, words..., fresh keep,
    fresh words...)``."""

    def __init__(self, tile_m: int):
        self.mapping = gg.GroupedTiles(tile_m=tile_m)

    @cute.kernel
    def kernel(self, space, indices: cute.Tensor, out: cute.Tensor):
        mapping = self.mapping
        state = mapping.start(space)
        for i in cutlass.range(indices.shape[0], unroll=1):
            keep, words, state = mapping.decode(indices[i], space, state)
            fresh_keep, fresh, _ = mapping.decode(indices[i], space, mapping.start(space))
            row = (cutlass.select_(keep, 1, 0), *words, cutlass.select_(fresh_keep, 1, 0), *fresh)
            if cute.arch.lane_idx() == 0:
                for j in cutlass.range_constexpr(len(row)):
                    out[i, j] = Int32(row[j])

    @cute.jit
    def __call__(self, offsets, total_m: Int32, n_tiles: Int32, indices, out, stream):
        space = self.mapping.bind((offsets, total_m, n_tiles))
        self.kernel(space, indices, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    def __cute_cache_key__(self):
        return ("grouped_decode_probe", self.mapping)


@jit_cache
def compile_decode_probe(op: DecodeProbe):
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        Int32(1),
        make_fake_compact_tensor(i32, (cute.sym_int(),)),
        make_fake_compact_tensor(i32, (cute.sym_int(), 10)),
        name=f"grouped_decode_probe{op.mapping.tile_m}",
    )


def _index_orders(count: int, units: int, seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    return {
        "strided": [i for b in range(min(units, count)) for i in range(b, count, units)],
        "ascending": list(range(count)),
        "descending": list(range(count - 1, -1, -1)),
        "random": rng.sample(range(count), count),
        "jumps": [rng.choice([0, count - 1, rng.randrange(count)]) for _ in range(count)],
    }


@pytest.mark.parametrize("tile_m", [128, 256])
@pytest.mark.parametrize(
    "sizes, n_tiles, extra_rows",
    [
        ([1000, 24, 0, 129, 128, 1, 256, 300], 3, 0),
        ([0, 0, 200, 0, 50, 0, 0, 7], 1, 131),  # rows past the last offset: trailing slots
        ([37] * 64, 5, -400),  # offsets past A: clamped
        ([300, -100, 50, 400, -500, 90, 0, 7], 2, 0),  # out of order: clamped to the last end
    ],
    ids=["ragged", "empty_runs", "clamped", "unordered"],
)
def test_grouped_tiles_state_never_changes_the_item(
    requires_blackwell, sizes, n_tiles, extra_rows, tile_m
):
    """Stateful decoding must match a fresh decode and the host reference at every index.

    Cover static-persistent strides, ascending, descending, random, and large jumps.
    """
    offsets = list(torch.tensor(sizes).cumsum(0).tolist())
    total_m = offsets[-1] + extra_rows
    count = (math.ceil(total_m / tile_m) + len(sizes)) * n_tiles
    fn = compile_decode_probe(DecodeProbe(tile_m))
    offsets_t = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    want = []
    for index in range(count):
        found, g, row0, rows = _python_locate_tile(index // n_tiles, offsets, tile_m, total_m)
        want.append((int(found), g, row0, rows, index % n_tiles))
    for name, order in _index_orders(count, 7, seed=len(sizes)).items():
        indices = torch.tensor(order, dtype=torch.int32, device="cuda")
        out = torch.zeros(len(order), 10, dtype=torch.int32, device="cuda")
        fn(offsets_t, total_m, n_tiles, indices, out)
        torch.cuda.synchronize()
        for index, row in zip(order, out.cpu().tolist(), strict=True):
            stateful, fresh = tuple(row[:5]), tuple(row[5:])
            assert stateful[0] == fresh[0] == want[index][0], (name, index, row)
            if want[index][0]:
                assert stateful == fresh == want[index], (name, index, row)


@pytest.mark.parametrize("tile_m", [128, 256])
def test_grouped_tiles_extreme_offsets_stay_in_int32(requires_blackwell, tile_m):
    """Offsets at the Int32 extremes with ``total_M`` at the host's bound: every decoded row
    stays inside ``A`` and the stateful decode still matches the reference."""
    G, n_tiles = 6, 3
    total_m = 2**31 - 1 - (G + 1) * gg.MAX_TILE_M
    offsets = [2**31 - 1, -(2**31), 5, 2**30, 2**30 + 7, 2**31 - 1]
    count = (math.ceil(total_m / tile_m) + G) * n_tiles
    rng = random.Random(tile_m)
    order = [0, 1, 2, 3, count - 1, count - 2, 3 * (2**30 // tile_m), count // 2, 0]
    order += [rng.randrange(count) for _ in range(40)]
    fn = compile_decode_probe(DecodeProbe(tile_m))
    out = torch.zeros(len(order), 10, dtype=torch.int32, device="cuda")
    fn(
        torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        total_m,
        n_tiles,
        torch.tensor(order, dtype=torch.int32, device="cuda"),
        out,
    )
    torch.cuda.synchronize()
    for index, row in zip(order, out.cpu().tolist(), strict=True):
        found, g, row0, rows = _python_locate_tile(index // n_tiles, offsets, tile_m, total_m)
        assert row[0] == row[5] == int(found), (index, row)
        if found:
            assert tuple(row[1:5]) == tuple(row[6:]) == (g, row0, rows, index % n_tiles)
            assert 0 <= row0 and row0 + rows <= total_m


def test_grouped_gemm_zero_work(requires_blackwell):
    """Empty K writes zeros over the covered rows only; an empty output or G=0 launches
    nothing and leaves ``out`` alone."""
    a, b, offsets = _inputs([64, 0, 24], 24, 0)
    a = torch.cat([a, a[:10]])
    out = torch.full((98, 24), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out) is out
    torch.cuda.synchronize()
    assert torch.equal(out[:88], torch.zeros_like(out[:88]))
    assert torch.isnan(out[88:]).all()
    a, b, offsets = _inputs([0, 0], 8, 64)
    assert grouped_gemm(a, b, offsets).shape == (0, 8)
    a, b, offsets = _inputs([], 8, 64)
    a = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.full((16, 8), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_gemm(a, b, offsets, out) is out
    assert torch.isnan(out).all()


def test_grouped_gemm_clamps_offsets_to_a(requires_blackwell):
    """Out-of-bounds offsets must not read or write outside A.

    Compute existing rows with the group named by the offsets; ignore nonexistent rows.
    """
    sizes = [64, 300]
    a, b, offsets = _inputs(sizes, 72, 64)
    a = a[:200].clone()  # offsets say [0, 64) and [64, 364) but A has 200 rows
    out = torch.full((200, 72), float("nan"), device="cuda", dtype=torch.bfloat16)
    grouped_gemm(a, b, offsets, out=out)
    torch.cuda.synchronize()
    ref = torch.cat([a[:64].float() @ b[0].float().T, a[64:].float() @ b[1].float().T])
    assert (out.float() - ref).abs().max().item() <= 0.1


@pytest.mark.parametrize(
    "schedule", [StaticPersistent(), AtomicPersistent()], ids=["static", "atomic"]
)
def test_grouped_gemm_clamps_out_of_order_offsets(requires_blackwell, schedule):
    """Clamp negative or decreasing offsets to the previous group's end.

    All groups stay inside A. Here group 0 owns [0, 300), groups 1-3 are empty, and group 4
    owns the remaining rows of the 400-row input.
    """
    a, b, _ = _inputs([80] * 5, 72, 64)
    offsets = torch.tensor([300, 100, -5, 250, 2000], device="cuda", dtype=torch.int32)
    out = torch.full((400, 72), float("nan"), device="cuda", dtype=torch.bfloat16)
    grouped_gemm(a, b, offsets, out=out, schedule=schedule)
    torch.cuda.synchronize()
    ref = torch.cat([a[:300].float() @ b[0].float().T, a[300:].float() @ b[4].float().T])
    assert (out.float() - ref).abs().max().item() <= 0.1


@pytest.mark.parametrize(
    "mutate, error, match",
    [
        (lambda a, b, o: (a.float(), b, o), TypeError, "bfloat16"),
        (lambda a, b, o: (a, b, o.long()), TypeError, "int32"),
        (lambda a, b, o: (a, b[:, :, :32], o), ValueError, "incompatible shapes"),
        (lambda a, b, o: (a, b, o[:3]), ValueError, "offsets must have shape"),
        (lambda a, b, o: (a[:, ::2], b[:, :, ::2], o), ValueError, "contiguous"),
        (
            lambda a, b, o: (a[:, :60].contiguous(), b[:, :, :60].contiguous(), o),
            ValueError,
            "multiples of 8",
        ),
    ],
    ids=[
        "a_dtype",
        "offsets_dtype",
        "k_mismatch",
        "offsets_shape",
        "strided",
        "k_mod_8",
    ],
)
def test_grouped_gemm_rejects_bad_operands(requires_blackwell, mutate, error, match):
    a, b, offsets = _inputs([64, 64, 64, 64], 64, 64)
    with pytest.raises(error, match=match):
        grouped_gemm(*mutate(a, b, offsets))


def test_grouped_gemm_rejects_too_many_rows(requires_blackwell):
    """The decode's Int32 arithmetic covers ``total_M + (G + 1) * 256`` rows."""
    _, b, offsets = _inputs([64, 64], 64, 64)
    a = torch.empty(1, 64, device="cuda", dtype=torch.bfloat16).expand(2**31 - 600, 64)
    with pytest.raises(ValueError, match="fit in int32"):
        grouped_gemm(a, b, offsets)


def test_grouped_gemm_rejects_bad_out(requires_blackwell):
    a, b, offsets = _inputs([64, 64], 64, 64)
    with pytest.raises(ValueError, match="must not overlap"):
        grouped_gemm(a, b, offsets, out=a)
    with pytest.raises(ValueError, match="must not overlap"):
        grouped_gemm(a, b, offsets, out=b.view(128, 64))
    with pytest.raises(TypeError, match="out must be bfloat16"):
        grouped_gemm(a, b, offsets, out=torch.empty(128, 64, device="cuda"))
    with pytest.raises(ValueError, match="out must have shape"):
        grouped_gemm(a, b, offsets, out=torch.empty(64, 64, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        grouped_gemm(
            a, b, offsets, out=torch.empty(64, 128, device="cuda", dtype=torch.bfloat16).T
        )


def test_grouped_gemm_rejects_unfit_configurations():
    with pytest.raises(ValueError, match="epi_tile_n must be one of"):
        GroupedGemm(epi_tile_n=128)
    with pytest.raises(ValueError, match="multiple of epi_tile_n=64"):
        GroupedGemm(tile_n=224, epi_tile_n=64)
    with pytest.raises(ValueError, match="c_stages must be in"):
        GroupedGemm(c_stages=0)
    assert GroupedGemm(tile_n=128).get_name() != GroupedGemm().get_name()
    assert GroupedGemm(cluster=CtaPair()).work.mapping.tile_m == 256
    assert "GroupedTiles(tile_m=128, state=Window: 5 x Int32)" in GroupedGemm().describe()
