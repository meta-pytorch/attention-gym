# CuTeDSL Top-K Indexer Kernel — Plan

## Problem

Given `A (M, K)` and `B (K, N)`, produce `out (M, topk)` containing the column
indices of the largest `topk` values in each row of `A @ B`, without
materializing the full M×N product.

## Core Idea

Each thread owns **one row** of A and maintains a **min-heap of size `topk`** in
SMEM (one heap per row, `TILE_M` heaps per block).  We stream B in tiles along N
(columns), and for each N-tile we pipeline the K-reduction (dot product) through
shared memory.  After each N-tile's dots are fully reduced, the tile's scores are
**sorted cooperatively in SMEM** (bitonic sort), and each thread walks the sorted
list from the top, inserting into its heap only until the sorted scores drop
below its heap minimum.  At the end, each thread writes its heap's indices to
global memory.

## Heap Entry Struct

Rather than maintaining two parallel arrays (scores and indices), each heap entry
is a single struct:

```
struct HeapEntry {
    float score;
    int32 col_idx;
};
```

The heap is an array of `topk` HeapEntry values stored in SMEM (one array per
row).  Comparisons use the `.score` field; swaps move the full 8-byte struct.

Why SMEM instead of registers: `TILE_M` rows × `topk` entries × 8 bytes per
entry.  For TILE_M=128, topk=32 this is 32 KB — too many registers per thread
(64 regs just for the heap), and the heap is only touched between N-tiles, not in
the hot K-reduction inner loop.  SMEM keeps it off the register file.

## Work Decomposition

| Dimension | Mapping |
|-----------|---------|
| **Grid** | `ceil(M / TILE_M)` blocks — one block handles `TILE_M` rows |
| **Block** | `BLOCK_THREADS` threads (e.g. 128–256) |
| **Thread → row** | Thread `t` owns row `block_start + t` (for `t < TILE_M`) |
| **Outer loop (N)** | `ceil(N / TILE_N)` iterations — each produces `TILE_N` scores per row |
| **Inner loop (K, pipelined)** | `ceil(K / TILE_K)` iterations with multi-stage `cp.async` |

## Pipelined K-Reduction (within one N-tile)

For each N-tile we compute `A_block (TILE_M, K) @ B_tile (K, TILE_N)`.  We tile
along K in chunks of `TILE_K` with a 2- or 3-stage async-copy pipeline:

```
Prologue:  issue loads for stages 0 .. (NUM_STAGES - 1)

Main loop (k = 0 .. K/TILE_K - 1):
   wait(stage[k % NUM_STAGES])           — SMEM tile pair is ready
   each thread reads its A-row slice from SMEM (TILE_K values)
   each thread reads TILE_N B-columns from SMEM (TILE_K × TILE_N)
   accumulate:  acc[n] += dot(a_slice, b_col[n])   for n in 0..TILE_N-1
   arrive_done(stage[k % NUM_STAGES])    — release SMEM slot
   issue load for stage[(k + NUM_STAGES) % NUM_STAGES]

Epilogue:  drain remaining stages
```

This follows the repo's existing `cp.async` + mbarrier pattern (see
`tile_dsl/barrier.py`, `tile_dsl/tma.py`).

After the K-loop finishes, each thread holds `TILE_N` fully-reduced f32 scores
in its register accumulator.

## Score Sorting + Heap Insertion (replacing the naive per-element loop)

### The problem with naive per-element insertion

The original plan had each thread independently iterating over `TILE_N` scores
and conditionally inserting into its heap:

```
for n in 0..TILE_N-1:
    if score > heap_min: replace root, sift down
    else: continue
```

This causes severe **thread divergence** within a warp: threads whose scores
mostly miss the heap skip through the loop quickly, while threads that insert
many values execute 5-level sift-downs.  All 32 warp threads must wait for the
slowest one.

### Sort-then-scan approach

Instead, after the K-loop produces `TILE_N` scores per row:

**Step 1 — Cooperative bitonic sort in SMEM.**  All threads in the block
cooperate to sort the `TILE_M × TILE_N` score-index pairs.  Each row's `TILE_N`
scores are sorted independently (descending).  This is uniform work — every
thread executes the same compare-and-swap network — so there is no divergence.

The sorted pairs live in SMEM as `(score, col_index)` structs:

```
sort_buf[TILE_M][TILE_N]   // in SMEM, each entry is (f32 score, i32 col_idx)
```

**Step 2 — Truncated scan into heap.**  Each thread (owning one row) reads from
its sorted row in descending order and stops as soon as the sorted score falls
below its heap minimum:

```
heap_min = heap[row][0].score           // read from SMEM

for i in 0 .. TILE_N - 1:
    entry = sort_buf[row][i]            // descending order
    if entry.score <= heap_min:
        break                           // everything after is even smaller

    heap[row][0] = entry                // replace root
    sift_down(heap[row], topk)          // restore heap in SMEM
    heap_min = heap[row][0].score       // updated minimum
```

The early-break is now **convergent across threads**: after the heap has filled
up (after the first few N-tiles), most rows will have high heap minimums, and the
loop terminates after 0–2 iterations for almost all threads in a warp.  The worst
case is `min(TILE_N, topk)` insertions per row, but the common case is much
better.

### Why this is better

| Property | Naive per-element | Sort-then-scan |
|----------|------------------|----------------|
| Divergence in insertion loop | High (variable sift-down depth per thread) | Low (early-break is convergent once heaps fill) |
| Work per N-tile | O(TILE_N × log(topk)) worst case per thread | O(TILE_N × log(TILE_N)) sort (uniform) + O(few × log(topk)) inserts |
| Warp efficiency | Poor (threads wait for slowest inserter) | Good (sort is uniform; scan converges quickly) |

## Min-Heap in SMEM

### Layout

```
heap_buf[TILE_M][topk]   // in SMEM, each entry is HeapEntry {f32 score, i32 col_idx}
```

Size: TILE_M × topk × 8 bytes.  For TILE_M=128, topk=32: 32 KB.

### Initialization

Each thread initializes its own row's heap entries:

```
for i in 0..topk-1:
    heap[row][i] = { score: -inf, col_idx: -1 }
```

### Sift-Down (in SMEM)

Standard min-heap sift-down operating on HeapEntry structs in SMEM.  Since topk
is a `Constexpr`, the loop is fully unrolled:

```
pos = 0
for _ in range(ceil(log2(topk))):
    left  = 2*pos + 1
    right = 2*pos + 2
    smallest = pos
    if left  < topk and heap[row][left].score  < heap[row][smallest].score:
        smallest = left
    if right < topk and heap[row][right].score < heap[row][smallest].score:
        smallest = right
    if smallest == pos: break
    swap(heap[row][pos], heap[row][smallest])
    pos = smallest
```

Each thread operates exclusively on its own row — no cross-thread conflicts, no
atomics needed.

## Shared Memory Layout

```
Buffer                                 Size (bytes)         Stages/Notes
A tile  (TILE_M × TILE_K, bf16)       TILE_M×TILE_K×2      NUM_STAGES
B tile  (TILE_K × TILE_N, bf16)       TILE_K×TILE_N×2      NUM_STAGES
sort_buf (TILE_M × TILE_N, 8B each)   TILE_M×TILE_N×8      1 (reuse across N-tiles)
heap_buf (TILE_M × topk, 8B each)     TILE_M×topk×8        1 (persistent across N-tiles)
```

Example with TILE_M=128, TILE_K=64, TILE_N=64, topk=32, NUM_STAGES=2:

- A pipeline: 128 × 64 × 2 × 2 = 32 KB
- B pipeline: 64 × 64 × 2 × 2 = 16 KB
- sort_buf: 128 × 64 × 8 = 64 KB
- heap_buf: 128 × 32 × 8 = 32 KB
- **Total: ~144 KB** (within the ~228 KB budget)

Note: sort_buf and the A/B pipeline buffers are not needed simultaneously.
sort_buf is written after the K-loop finishes (when the pipeline buffers are
drained).  We can **alias** sort_buf with the pipeline buffers to save SMEM:

- Pipeline buffers: 48 KB
- sort_buf: 64 KB → overlaps with pipeline (64 > 48, so sort_buf dominates)
- heap_buf: 32 KB
- **Aliased total: ~96 KB**

## Register Budget Per Thread

| Item | Count | Bytes |
|------|-------|-------|
| Accumulator (TILE_N f32s) | 64 | 256 |
| A-row fragment (TILE_K bf16) | 64 | 128 |
| Misc (loop counters, pointers, heap_min) | ~16 | 64 |
| **Total** | | **~448 B = 112 regs** |

The heap is now in SMEM, so register pressure is significantly reduced compared
to the original plan.

## Outer N-Loop Structure

```
initialize heap_buf to (-inf, -1) × topk per row

for n_tile in 0 .. ceil(N / TILE_N) - 1:
    zero accumulator[0..TILE_N-1]

    # --- pipelined K-loop ---
    for k_tile in 0 .. ceil(K / TILE_K) - 1:
        (pipeline: async-load A[:,k], B[k,n] tiles → SMEM)
        (compute:  acc[n] += A_row[k] · B[k,n]  for each n)

    # --- write scores + indices to sort_buf in SMEM ---
    for n in 0 .. TILE_N - 1:
        sort_buf[thread_row][n] = { acc[n], n_tile * TILE_N + n }
    __syncthreads()

    # --- cooperative bitonic sort (per-row, descending) ---
    bitonic_sort(sort_buf, TILE_M, TILE_N)
    __syncthreads()

    # --- truncated scan into heap ---
    heap_min = heap_buf[thread_row][0].score
    for i in 0 .. TILE_N - 1:
        if sort_buf[thread_row][i].score <= heap_min: break
        heap_buf[thread_row][0] = sort_buf[thread_row][i]
        sift_down(heap_buf[thread_row], topk)
        heap_min = heap_buf[thread_row][0].score

# --- write output ---
for i in 0 .. topk - 1:
    out[row, i] = heap_buf[thread_row][i].col_idx
```

## Bitonic Sort Details

Each row's `TILE_N` elements are sorted independently.  With TILE_M rows and
TILE_M threads, each thread sorts its own row.  For TILE_N = 64, a bitonic sort
network has `log2(64) × (log2(64)+1) / 2 = 21` passes of compare-and-swap, each
pass touching `TILE_N/2 = 32` pairs.  A single thread can execute this
sequentially over its row's SMEM entries, or multiple threads can cooperate on
one row if TILE_N is large.

Since each thread owns exactly one row and the rows are independent, the simplest
approach is: each thread runs the full bitonic network on its own row's SMEM
entries.  This is fully uniform work (same comparisons for every thread), so no
warp divergence.

## Alternative to Heap: Sorted Buffer with Merge

An alternative to the min-heap is maintaining a **sorted buffer** of size `topk`
per row and merging each N-tile's sorted scores into it:

**Approach:** After sorting the N-tile's scores (Step 1 above), perform a
**sorted merge** of the tile's top-topk scores with the existing sorted buffer,
keeping only the top-topk results.

**Merge:** Two sorted arrays of size `topk` each → merged into one sorted array
of size `topk` (keep top-topk from the merged 2×topk).  This is a single linear
pass: walk both arrays with two pointers, emit the larger element, stop after
`topk` outputs.

| Property | Min-heap | Sorted buffer + merge |
|----------|----------|-----------------------|
| Insertion cost | O(log topk) per element | O(topk) per merge (but only one merge per N-tile) |
| Total per N-tile | O(inserted × log topk) | O(topk) (one merge regardless of how many qualify) |
| Code complexity | Sift-down in SMEM | Two-pointer merge in SMEM |
| Early termination | Natural (stop when score < min) | Natural (merge stops after topk outputs) |
| Divergence | Low with sort-then-scan | None (merge is fixed-length work) |

**Analysis:** The sorted-merge approach has **zero divergence** — every thread
does exactly the same amount of work per N-tile (one `O(topk)` merge) regardless
of the data distribution.  The heap approach can be faster when few elements
qualify (the scan terminates early), but the merge approach is more predictable.

**Recommendation:** Implement the **heap with sort-then-scan** as the primary
approach (it wins when the heap is well-populated and few inserts happen), but
structure the code so the merge alternative can be swapped in as a tuning
variant.  Both share the same sort step; they differ only in the
post-sort selection logic.

## A-Matrix Reuse

Within each N-tile's K-loop, every thread re-reads its A row from SMEM.  Since A
is the same across all N-tiles, we have two options:

1. **Re-load A per N-tile** — simplest; A is already in SMEM from the pipelined
   load.  Each thread reads only one row (TILE_K values), so the SMEM traffic is
   cheap.
2. **Cache A in registers across N-tiles** — load A once into registers and keep
   it resident.  This saves SMEM bandwidth but increases register pressure
   (K values × bf16 per thread).  Only viable for small K.

Start with option 1 for simplicity.

## CuTeDSL Implementation Outline

Following the repo's patterns (`copy_reads_example.py`, `tile_dsl/*`):

```python
class IndexerConfig(NamedTuple):
    tile_m: int        # rows per block (= threads per block)
    tile_n: int        # columns per N-tile
    tile_k: int        # K-reduction tile size
    num_stages: int    # async-copy pipeline depth

class TopkIndexerOp:
    @cute.kernel
    def kernel(self, A, B, Out, M, N, K, topk, ...):
        # 1. Thread/block identity
        # 2. Allocate SMEM for pipeline bufs, sort_buf, heap_buf
        # 3. Initialize heap in SMEM
        # 4. N-tile outer loop:
        #    a. K-tile pipelined inner loop (cp.async load + dot accumulate)
        #    b. Write scores to sort_buf, bitonic sort
        #    c. Truncated scan from sorted scores into heap
        # 5. Write heap indices to Out

    @cute.jit
    def execute(self, A, B, Out, M, N, K, topk, config, stream):
        blocks = cute.ceil_div(M, config.tile_m)
        self.kernel(...).launch(
            grid=(blocks, 1, 1),
            block=(config.tile_m, 1, 1),
            stream=stream,
        )

    @staticmethod
    @jit_cache
    def compile(config: IndexerConfig, topk: int):
        # Symbolic dims, fake tensors, compile_tvm_ffi
        ...
```

## Output Semantics

- Output indices are **not sorted** within each row (matches
  `torch.topk(sorted=False)`).
- Ties are broken arbitrarily (whichever column was encountered first stays in
  the heap).

## Test (if __name__ == "__main__")

```python
A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

result = index(A, B, topk=32)

# Reference
scores = (A.float() @ B.float())
expected = scores.topk(32, dim=1).indices

# Compare as sets per row (order may differ)
for i in range(M):
    assert set(result[i].tolist()) == set(expected[i].tolist())
```

## Open Questions / Tuning Knobs

1. **TILE_N vs topk:** If `TILE_N >> topk`, most insertions hit the early-exit
   after the sort.  Larger TILE_N amortizes pipeline overhead but increases
   sort_buf SMEM and sort cost.
2. **MMA vs scalar dots:** For the K-reduction, we could use `mma.sync`
   (m16n8k16) for higher throughput instead of scalar dot products.  This would
   require restructuring the thread mapping (warp-level MMA tiles instead of
   one-thread-per-row).  Start scalar, upgrade to MMA if the K-loop is the
   bottleneck.
3. **bf16 vs f32 accumulation:** Accumulate in f32 for correctness (matching
   PyTorch reference), store heap scores as f32.
4. **Predication for partial tiles:** Handle M, N, K not divisible by tile sizes
   with bounds checks in the final CTA / final N-tile / final K-tile.
5. **Heap vs sorted-merge selection:** Both share the bitonic sort step.  The
   heap wins when few elements qualify per N-tile (common after the first few
   tiles); the merge wins when divergence is the bottleneck.  Expose as a config
   knob or auto-select based on topk/TILE_N ratio.
6. **SMEM aliasing:** sort_buf can alias the pipeline A/B buffers since they are
   never live simultaneously.  Implement this to reduce peak SMEM from ~144 KB
   to ~96 KB, allowing larger tiles or more pipeline stages.
