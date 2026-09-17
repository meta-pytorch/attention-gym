"""Row addressing shared by the bounded-slab Triton indexer scorers."""

import triton
import triton.language as tl


@triton.jit
def slab_row(row, FirstBatch, FirstPair, T, USE_INT64_OFFSETS: tl.constexpr):
    pair = FirstPair + row // 2
    pairs_per_batch = tl.cdiv(T, 2)
    # Most slabs stay in the first batch or cross only its end. A compare then
    # suffices to locate the row; only slabs spanning many short batches need a divide.
    if pair >= 2 * pairs_per_batch:
        delta = pair // pairs_per_batch
    else:
        delta = (pair >= pairs_per_batch).to(tl.int64 if USE_INT64_OFFSETS else tl.int32)
    batch = FirstBatch + delta
    query = 2 * (pair - delta * pairs_per_batch) + row % 2
    return batch, query
