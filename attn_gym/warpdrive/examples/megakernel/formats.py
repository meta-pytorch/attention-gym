"""The expert weight formats of the quantized GEMVs (``quant_gemv``) and their geometry on the
weight tiles (``constants``), which a model's loader (``inkling_megakernel.quant``) packs to.
Imports only ``constants``: loaders and oracles need no CuTeDSL, and kernel modules import no
host module."""

from attn_gym.warpdrive.examples.megakernel.constants import TILE_K

BF16, NVFP4, MXFP8 = "bf16", "nvfp4", "mxfp8"
FORMATS = (BF16, NVFP4, MXFP8)
ACTIVATION_FORMATS = (NVFP4, MXFP8)  # expert input quantizations (``act_quant``)
BLOCK_COLUMNS = {NVFP4: 16, MXFP8: 32}  # columns per scale byte
LANE_WORDS = {NVFP4: 1, MXFP8: 2}  # weight words per lane, row, subtile
SUBTILE_WORDS = {f: 32 * w for f, w in LANE_WORDS.items()}
WEIGHT_BITS = {BF16: 16, NVFP4: 4, MXFP8: 8}  # bits per stored weight
MIN_MX_SCALE = 10  # smallest MXFP8 scale byte whose products with e4m3 values are normal
# Tile-major scale words per 256-column subtile: 8 warps x blocks (each word: a block's 4 warp
# rows).
BLOCKS = {fmt: TILE_K // columns for fmt, columns in BLOCK_COLUMNS.items()}
