"""The tile geometry and shared-memory budget the megakernels share. Plain Python with no
imports, so host code (loaders, schedulers) can read it without CuTeDSL."""

# A weight stage is ROWS rows x TILE_K bf16 columns (16 KB), and a GEMV work item's tile is
# ROWS rows; a KV stage is BLOCK positions of one HEAD_DIM head, also 16 KB. The attention
# helpers (``attention``) and the KV ring are written for this head dimension only.
ROWS, TILE_K, BLOCK, HEAD_DIM = 32, 256, 64, 128
STAGE_BYTES = ROWS * TILE_K * 2
MAX_STAGES = 12  # cap on the default ring depth (``ring.ring_stages``)
SMEM_BYTES = 232448  # SM100 per-block shared memory (227 KB)
SMEM_HEADROOM = 1024  # for barriers and alignment
# The compute role: COMPUTE_WARPS warps, warp w owning rows ROWS_PER_WARP * w .. of a tile.
COMPUTE_WARPS = 8
ROWS_PER_WARP = ROWS // COMPUTE_WARPS
COMPUTE_THREADS = 32 * COMPUTE_WARPS
RED_VALUES = 16  # values a ``regmath.cta_sum`` slot reduces at once
# Sleep between failed polls of a wait that spans other CTAs' work (``tagged.fetch``'s backoff).
BACKOFF_NS = 100
EVICT_FIRST = 0x12F0000000000000  # TMA L2 cache hint (CUTLASS CacheHintSm90::EVICT_FIRST)
LOG2E = 1.4426950408889634
