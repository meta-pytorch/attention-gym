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
EVICT_LAST = 0x14F0000000000000  # TMA L2 cache hint (CUTLASS CacheHintSm90::EVICT_LAST)
LOG2E = 1.4426950408889634
# The multi-token GEMVs' geometry (``multi_gemv``), which TMA boxes and shared-memory plans take:
# defined here so that a launch whose module does not import ``multi_gemv`` (Inkling's one-token
# visit) keeps it out of its compile key.
MMA_N = 8  # tokens per MMA
BOX = 64  # bf16 columns per swizzled box (128 bytes)
QBOX = 32  # quantized words per swizzled box
X_PAD = 8  # bf16 elements after each GEMV input row: rows 4 mod 32 words apart
DBUF_FLOATS = 4 * 32 * MMA_N  # ``multi_gemv.finish``'s partial tiles: quarter, row, token
