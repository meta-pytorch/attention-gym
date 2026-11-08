"""Static schedules drawn per CTA: ``WorkStream.assignment`` into ``schedule_html``.

* ``persistent_gemm_view``: the output tiles each CTA of a ``StaticPersistent`` GEMM computes,
  in order. The CTA x slot grid shows the wave quantization (a last slot only some CTAs
  fill); the tile map shows where each wave lands under the raster's L2 swizzle.
* ``grouped_gemm_view``: the same for a grouped GEMM, whose items the kernel decodes on the
  device from the group offsets. ``grouped_tiles_decode`` walks the same offsets on the host,
  so the padded index space's dropped items are known.

    python -m attn_gym.warpdrive.examples.schedule_views OUTDIR [--ctas 148]
"""

import argparse
import math
from itertools import accumulate
from pathlib import Path

import torch

from attn_gym.warpdrive.examples.grouped_gemm import GroupedGemm
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm
from attn_gym.warpdrive.schedule_html import Item, schedule_html


def _rows(assignment, items: list[Item], make) -> list[list[int]]:
    """Item positions per CTA; appends ``make(index, item)`` to ``items`` per assigned item."""
    rows = []
    for row in assignment:
        rows.append([])
        for index, item in row:
            rows[-1].append(len(items))
            items.append(make(index, item))
    return rows


def persistent_gemm_view(m: int, n: int, ctas: int) -> str:
    op = PersistentGemm()  # one CTA per work unit, so the rows are the units
    tile_m, tile_n = op.mma_tile[:2]
    space = (math.ceil(m / tile_m), math.ceil(n / tile_n), 1)
    items: list[Item] = []

    def tile(index, item):
        mt, nt, _ = item
        return Item(f"tile m{mt} n{nt}\nindex {index}", "tile", coord=(mt, nt))

    rows = _rows(op.work.assignment(space, max_ctas=ctas), items, tile)
    tiles = len(items)
    return schedule_html(
        f"PersistentGemm {m} x {n}: {tiles} tiles on {len(rows)} CTAs",
        f"{op.work.describe()} · {tile_m} x {tile_n} tiles · {tiles / len(rows):.2f} waves",
        items,
        rows,
        color_by="slot",
        notes=[
            (
                "WorkStream.assignment: StaticPersistent gives CTA b the indices b, b + G, ...; "
                "Raster decodes each index to its output tile."
            ),
            f"max_ctas = {ctas} (the launch caps the grid at it).",
        ],
    )


def grouped_tiles_decode(offsets: list[int], total_m: int, n_tiles: int, tile_m: int):
    """Host counterpart of ``GroupedTiles.decode``: ``index -> (keep, (group, row0, rows,
    n_tile))`` from the groups' cumulative row offsets, clamped the same way."""
    windows = []  # (group, first m-tile, end m-tile, first row, end row)
    tile = row = 0
    for group, end in enumerate(offsets):
        end = min(max(end, row), total_m)
        tiles = math.ceil((end - row) / tile_m)
        windows.append((group, tile, tile + tiles, row, end))
        tile, row = tile + tiles, end

    def decode(index: int):
        mt = index // n_tiles
        for group, lo, hi, first, end in windows:
            if lo <= mt < hi:
                row0 = first + (mt - lo) * tile_m
                return True, (group, row0, min(end - row0, tile_m), index % n_tiles)
        return False, None

    return decode


def grouped_gemm_view(sizes: list[int], n: int, ctas: int) -> str:
    op = GroupedGemm()
    tile_m, tile_n = op.mma_tile[:2]
    offsets = list(accumulate(sizes))
    total_m, n_tiles = offsets[-1], math.ceil(n / tile_n)
    space = (torch.tensor(offsets, dtype=torch.int32), total_m, n_tiles)
    decode = grouped_tiles_decode(offsets, total_m, n_tiles, tile_m)
    items: list[Item] = []

    def tile(index, item):
        group, row0, rows, nt = item
        mt = index // n_tiles
        label = f"group {group}: rows {row0}..{row0 + rows - 1}, n tile {nt}\nindex {index}"
        return Item(label, f"group {group}", coord=(mt, nt))

    rows = _rows(op.work.assignment(space, max_ctas=ctas, decode=decode), items, tile)
    return schedule_html(
        f"GroupedGemm: {len(sizes)} groups, {len(items)} tiles on {len(rows)} CTAs",
        f"{op.work.describe()} · {tile_m} x {tile_n} tiles · group rows {sizes}",
        items,
        rows,
        kinds=[f"group {g}" for g in range(len(sizes))],
        notes=[
            (
                "The index space pads every group by one m-tile (the host cannot count tiles "
                "before the launch); the padding indices are dropped, so their CTAs run fewer "
                "items."
            ),
            (
                "Items decoded on the host by grouped_tiles_decode, which walks the offsets like "
                "the device decode."
            ),
            f"max_ctas = {ctas}.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ctas", type=int, default=148, help="max_ctas (the SM count)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pages = {
        "persistent_gemm_schedule": persistent_gemm_view(4096, 4096, args.ctas),
        "grouped_gemm_schedule": grouped_gemm_view(
            [700, 2048, 64, 1500, 300, 4096, 900, 17], 2048, args.ctas
        ),
    }
    for name, text in pages.items():
        path = args.out / f"{name}.html"
        path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()
