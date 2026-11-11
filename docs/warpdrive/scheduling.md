# Scheduling

[Back to warpdrive](index.md)

Use a `WorkStream` to give cooperating roles the same work items. By default, each CTA
gets one tile from `Raster()` using `OneShot()`:

```python
self.work = WorkStream()
```

A **work unit** is one CTA, or one whole cluster with `CtaPair()` or `Cluster(...)`.
Declare at most one stream per kernel. The schedule assigns indices; the mapping turns
each index into the item that roles read:

```text
schedule                    mapping                   role
choose linear index 5  ---> decode 5 as (m, n, l) ---> work.item
                                ^
                         optional ready wait
                         before decoding starts
```

## Choose a schedule

Start with `OneShot()`. Use `StaticPersistent()` to reuse each unit for several items.

| Schedule | How a unit gets its next index | Distribution |
|---|---|---|
| `OneShot()` | None: return terminal index `total`; dropped items are not reassigned | Local |
| `StaticPersistent()` | Start at unit index b, then add the number of launched units | Local |
| `TilePersistent(axis)` | Step one tile coordinate by `gridDim[axis]` | Local |
| `AtomicPersistent(stages=2)` | Claim from a global atomic counter after its first index b | Scheduler warp |
| `ClusterLaunchControl(stages=2)` | Cancel an unlaunched unit and take its work | Scheduler warp |

With eight items and three static-persistent units:

```text
unit 0:  0 -> 3 -> 6 -> terminal
unit 1:  1 -> 4 -> 7 -> terminal
unit 2:  2 -> 5      -> terminal
```

For local schedules, `work.assignment(space, max_ctas)` lists each CTA's `(index, item)`
sequence before launch ([drawn per CTA](tracing.md#see-which-cta-runs-what)).

A correctly launched and drained stream assigns each kept index to exactly one unit.
Within that unit, roles see the same sequence wherever they run: every CTA for `all`,
pair leaders for `leader`, rank 0 for `rank0`.

### The role loop

This role fragment works with either local or broadcast distribution:

```python
self.work = WorkStream(schedule=StaticPersistent())


@role(warps=4, regs=40)
@cute.jit
def load(self, params: PersistentGemmParams, *, ab: Producer, work: WorkCursor):
    while work.valid:
        m, n, _ = work.item
        ...  # load this item's operands
        work.advance()
    return ab, work
```

Every managed role must drain a persistent stream, even if it has no per-item work.
Use `while work.valid:`, read the item before advancing, call `work.advance()` once directly
at the end of the body, and return the cursor. `StaticPersistent` and `TilePersistent` also
support the [counted form](#a-counted-loop).

The checker rejects item reads after advance, including lookahead reads of the next item
inside the same iteration. It does not prove advance is lexically last, detect every cursor
rebind/stale restore, or prove useful and matching work happened before it.

## Bind and launch

Keep scheduling choices static and problem dimensions runtime. The
[persistent GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py)
binds and launches work at three boundaries:

```python
# Python, outside the compiled entry point:
workspace, max_ctas = op.work.launch_args(device)

# In __call__ (@cute.jit):
bound = self.work.bind((m_tiles, n_tiles, 1), workspace, max_ctas)
self.kernel(..., bound).launch(**self.launch_config(bound), stream=stream)

# In the kernel:
smem = self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, "C": c_layout})
```

`bind()` returns the kernel's runtime `BoundWork`. Shapes, counts and the persistent grid
cap stay out of the compile key. `work=bound` matches the declaration `self.work`. Supply
layouts for every `SmemPlan` buffer; omit `layouts` if you allocate storage yourself.

Keep `workspace` and `max_ctas` in the compiled entry point's signature even when a schedule
ignores them. This lets you switch schedules without changing the launch ABI.

### Grid cap

Static and atomic persistence require `max_ctas`. It caps **CTAs**, not clusters:
with cluster size `cs`, they launch `max(1, min(total, max_ctas // cs))` units.
`TilePersistent` without `units` instead uses it to cap units along its axis.

`launch_args()` estimates how many CTAs fit in resident clusters: the SM count for single
CTAs or pairs, fewer for four-CTA clusters (36 clusters on a 152-SM GB300). This is a capacity
estimate, not a co-residency guarantee.

### Workspace lifetime

Atomic persistence uses two zeroed Int32 words, `[claims, finished]`:

```text
first item:       unit b gets index b
later claim:      launched_units + atomicAdd(claims, 1)
last report:      fence, then reset both words to zero
```

The last scheduler report resets claim state for the next ordered launch, including CUDA
graph replay. It is not a statement that all earlier item computations have finished.

`launch_args()` caches workspace by device and **current stream**. Ordered launches on one
stream can share it; different streams get separate workspaces. Never share one between
concurrent launches: their claims and resets would interfere.

Without `record`, schedules that use no workspace (`OneShot`, `StaticPersistent`,
`TilePersistent`, CLC) receive one placeholder per device. The kernel never accesses it; graph
capture on a new stream therefore records no fill.

## Built-in schedule details

### Raster order

For `space=(3, 2, 1)`, plain Raster maps indices 0–5 to
`(0,0,0), (1,0,0), (2,0,0), (0,1,0), (1,1,0), (2,1,0)`. The schedule decides which unit gets
each index; changing the schedule need not change these items.

Raster options:

- `along="m"` chooses the raster axis; `"n"` chooses the other one.
- `swizzle=8` groups the other axis into eight-tile bands, visiting the band's tiles
  innermost. The final band may be narrower.
- `reverse="m"` visits m high-to-low. l is always the slowest axis.

For positive extents Raster is a bijection. Its decode intermediates stay within
`max(index, item_count)`, avoiding overflow for admitted indices.

### Tile persistence

`TilePersistent("n")` over `(pairs, chunks, heads)` launches `(pairs, workers, heads)`,
with `workers` capped by `max_ctas`. Each CTA keeps its pair and head from `blockIdx` and
steps through chunks by `gridDim.y`, without division.

The axis can be `"m"`, `"n"` or `"l"`. `max_ctas` caps that axis per slice of the other two,
not the total CTA count. The stream index is the axis coordinate and `total` is its extent,
so a [limit](#a-device-read-limit) bounds that coordinate. This schedule needs plain
`Raster()`, a single-CTA cluster and no `record`.

`TilePersistent(axis, units=n)` fixes the grid's axis extent and stride at compile time.
CTAs past the work extent run nothing; `max_ctas` is ignored. Bucket `n` to bound the number
of compiled variants.

### A counted loop

To unroll work across items, use a counted loop. `while work.valid` lowers to `scf.while`
and cannot use `cutlass.range(..., unroll=n)`. The counted form supports local
`StaticPersistent` or `TilePersistent` streams over `Raster()` without `record`: the mapping
must keep every item, decode any index and carry no state.

```python
for k in cutlass.range(work.trip_count(), unroll=2):  # two items' loads overlap
    m, n, l = work.item_at(k)
    ...
```

Item `k` is index `first + k * stride`, and `trip_count()` is
`ceil((end - first) / stride)` (0 when the first index is already terminal), with `end` the total
or the [limit](#a-device-read-limit). A cursor runs in one mode: calling `advance()` on a counted
cursor, or `trip_count()` on an advanced one, raises at trace time. Return the cursor from the
role as usual.

### What the scheduler warp buffers

Atomic claims are unique but have no fixed inter-unit order. CLC gets the canceled work
index from its CTA x coordinate divided by cluster size. It waits for every cancellation
response and never queries again after failure.

Local schedules compute and decode indices in each role. Atomic and CLC schedules add a
scheduler warp that claims, decodes and broadcasts items through a work ring. In a cluster,
rank 0 sends to every CTA. The warp is the lowest free warp unless selected with
`WorkStream(scheduler_warp=...)`.

```text
scheduler: acquire free slot -> claim/decode -> publish item
role:      receive item      -> release slot -> compute item
```

`stages` is **work-ring depth**, not A/B buffering. A role releases its work slot on receipt,
not when computation finishes. All ring consumers also receive a terminal marker, which is
not work.

For cluster broadcasts, rank 0 uses `st.async` to fill every CTA's slot and complete its
FULL barrier. Receivers return the slot with `remote_scope="cta"`: they publish no data.
Cluster-scope releases would add a GPU-scope membar per item in every receiving thread.

### The grid and empty work

Plain `Raster()` with `OneShot()` normally launches
`(m_tiles * cluster_size, n_tiles, l)` and reads `blockIdx` coordinates. If y or z exceeds
65535, it switches to a decoded 1-D grid. A hand-built `(total * cluster_size, 1, 1)` grid is
also supported for this path. Apart from `TilePersistent`, other built-in combinations
use 1-D grids.

`OneShot(grid="auto")` handles both grids with a runtime branch. The merged item and its
derived addresses and predicates leave the uniform datapath. To avoid that branch, choose
a static grid. The table shows single-CTA grids; clusters multiply x by `cluster_size`:

| `OneShot(grid=...)` | Launch grid | Item |
|---|---|---|
| `"auto"` (default) | Tile space, or 1-D past 65535 `n_tiles` or `l` | `blockIdx`, or decoded behind a runtime branch |
| `"3d"` | `(m_tiles, n_tiles, l)`; needs `n_tiles`, `l` <= 65535 | `blockIdx`, no branch and no division |
| `"2d"` | `(m_tiles * l, n_tiles, 1)`; needs `n_tiles` <= 65535 | `n` is `blockIdx.y`; `l` and `m` from `blockIdx.x` with one division |
| `"1d"` | 1-D only | Always decoded: two divisions for `Raster()` |

`"3d"` and `"2d"` need plain `Raster()` without `record` and a launch on
`bound.grid()`. `OneShot.grid_for((m_tiles, n_tiles, l))` returns the cheapest grid that
launches a space given as Python ints, so a wrapper picks the variant before compiling:
`"3d"` when `n_tiles` and `l` fit, `"2d"` when only `n_tiles` does, else `"1d"`.

```python
op = MyKernel(grid=OneShot.grid_for((chunks, heads, batch)))
# in MyKernel.__init__:
self.work = WorkStream(schedule=OneShot(grid=grid))
```

This needs at most three compiled variants. If heads on n always fit the grid limit,
only `"3d"` and `"2d"` are selected as batch on l grows.

Past a `"3d"` or `"2d"` limit, `bound.grid()` raises `ValueError` for static extents.
For runtime extents, CUDA rejects the grid before any CTA runs:
`CUDA launch failed: cudaErrorInvalidValue (1)`, `CUDA rejected the launch configuration before
running the kernel.`

Pass `m_tiles` and `n_tiles` as Python ints when they are compile-time constants (heads, a
channel-block count): binding keeps them static beside runtime extents, so `Raster.decode` and
the `"2d"` unfold divide by constants. `l` is never a divisor; with any runtime extent it
binds as a runtime value.

If any Raster extent is `<= 0`:

- Host-side binding normalizes space to `(1, 1, 0)` (a runtime-empty space with static tile
  extents keeps them and binds `l = 0`).
- Every built-in schedule launches one unit, with `work.valid == False` in every role.
- No guarded decode runs; any-index decoders may evaluate and discard a result.
- No divisor is zero and the atomic workspace remains zeroed.

Direct `Raster.decode` calls must also use **bound** space. Raw `(0, 1, 1)` is not safe.

The entry point calls `mapping.count64()` once and passes narrowed Int32 `BoundWork.total`
to the kernel. Before launch, the count must satisfy:

| Schedule | Required bound |
|---|---|
| One-shot or CLC | `total * cluster_size <= 2**31 - 1` |
| Static or atomic persistent | `total + launched_units - 1 <= 2**31 - 1`, including terminal indices |

Use `launch_config(bound)` or `bound.grid()` for this check. Failure gives a zero-width grid,
which CUDA rejects instead of running wrapped work. A hand-built grid is your responsibility.
The failed launch leaves no sticky CUDA error, but invalidates active CUDA-graph capture.

Raster runtime extents become Int32 before binding; passing wider values cannot enlarge the
supported space. Custom mapping arithmetic, `record` offsets and other launch resources must
independently fit Int32 and CUDA limits.

## Custom mappings

Subclass `WorkMap` when `Raster` does not describe your items. A map decodes an index;
it does not choose the next one. For example, it can return table row `index` and drop
unused rows. [scheduling.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/scheduling.py)
defines the full protocol.

| Method or field | Contract |
|---|---|
| `words` | Positive number of Int32 words in each item |
| `bind(space)` | Once in the host entry point; normalize runtime inputs into a DSL-compatible tuple, tensor or other value |
| `count64(bound_space)` | Host-visible nonnegative count, exact through `2**31` or saturated there; never wrap |
| `decode(index, space, state)` | On device, return `(keep, words, next_state)` |
| `unpack(words)` | Public `work.item`; defaults to a tuple |
| `get_key()` | Every static choice, not runtime extents; lowercase letters, digits and underscores |

Host `bind` and `count64` cannot read a device-built length. Bind a capacity instead
([device-built list](#a-device-built-list)). A stateless decoder passes `state=None` through;
see [Decode state](#decode-state) for a carried cache.
Set `skips=True` if `keep` can be false; otherwise it must be the Python constant `True`,
checked while tracing. Dropping an index permanently removes it from every role's sequence.

Correctness obligations:

- All roles and CTAs must decode the same answer. Tables must be ready before decoding and
  unchanged while the stream uses them.
- Kept indices must represent distinct items, and `[0, count)` must reach every needed item.
  The layer does not prove either property.
- Each skipping decoder runs once per visited index at each decoding site.

### Guarded versus speculative decode

Normally decode runs only for `0 <= index < total`, after readiness. A table map may safely
read row `index` without handling terminal indices.

`decodes_any_index=True` removes that guard. Opt in only for a **pure launch-data decoder**
defined for every nonnegative Int32 index, including empty spaces and indices beyond the
end. It may run before readiness in the tile-grid prologue. Speculative results are discarded,
but any updated state must remain valid. A skipping map's `keep` is still used only for
valid indices.

Raster opts in; an item-table reader must not. A subclass replacing
`decode` must explicitly opt in again.

### Decode state

State avoids repeating expensive lookup work. In grouped GEMM, group m-tiles are numbered
back-to-back. A small cached window says which group contains the previous tile:

```text
cached group window:  [tile_lo ........ tile_hi)
next tile inside?     reuse the window, no offset load
next tile ahead?      walk forward through group offsets
next tile behind?     restart from group 0
```

A shortened grouped-map excerpt (with `_seek` supplied by the example):

```python
class Window(NamedTuple):
    group: Int32
    tile_lo: Int32  # group's m-tiles [tile_lo, tile_hi), rows [row_lo, row_hi)
    tile_hi: Int32
    row_lo: Int32
    row_hi: Int32


class GroupedTiles(WorkMap):
    def start(self, space):
        return Window(Int32(-1), Int32(0), Int32(0), Int32(0), Int32(0))  # before group 0

    def decode(self, index, space, state):
        offsets, total_m, n_tiles = space
        mt = index // n_tiles
        w = _seek(state, mt, offsets, total_m, self.tile_m)
        row0 = w.row_lo + (mt - w.tile_lo) * self.tile_m
        rows = cutlass.min(w.row_hi - row0, self.tile_m)
        return row0 < w.row_hi, (w.group, row0, rows, index % n_tiles), w
```

Each decoding site owns its state: every role in a local schedule, the scheduler warp in a
broadcast schedule, or `single_item`. Ring receivers carry no decode state.

- `start(space)` runs once after readiness, **even for an empty stream**. It must be safe
  for every bound space. It may read readiness-published data, but put item-table reads in
  guarded `decode`.
- Each decode receives the previous returned state, including after dropped items. The
  first uses `start`'s state. Exception: with `decodes_any_index` and no skip/readiness hooks,
  every role decodes its first item from a fresh `start` state.
- Use DSL values (`Int32(0)`, not `0`) in tuples or `NamedTuple`s. Plain dataclass fields do
  not carry. Return the same structure and types; tracing checks this.

**State is a cache, never an input to the answer.** From any reachable state, decode must
return the same `keep` and kept words as from a fresh start. Static and atomic indices
increase within a unit; CLC and custom schedules may go backwards. Optimize forward walks,
but make every order correct.

State lives in each decoding warp's registers; keep it small and name it in `describe()`.
The grouped example reports `GroupedTiles(tile_m=128, state=Window: 5 x Int32)`.

If decoding remains expensive, a broadcast scheduler decodes once per item ahead of the
roles. If state makes decoding cheap, a local schedule avoids the scheduler warp/workspace.
Measure both on your workload.

### A device-built list

[PackedWork](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/packed_work.py) binds a host-visible capacity and reads the actual
length only after readiness:

```python
from attn_gym.warpdrive import AtomicPersistent, PdlWait, WorkStream
from attn_gym.warpdrive.examples.packed_work import PackedWork

self.work = WorkStream(mapping=PackedWork(words=3), schedule=AtomicPersistent(), ready=PdlWait())
# In the entry point: table is int32 (capacity, 3), length is an int32 device tensor.
bound = self.work.bind((table, length), workspace, max_ctas)
```

`count64()` returns capacity. `decode()` reads row `index`, keeps it if `index < length[0]`,
and passes `state=None` through. Even a dropped row read stays within capacity; its words
are ignored. The schedule visits all capacity, so avoid excessive padding.

### A device-read limit

If live items form a prefix, bind a `Limit` rather than decoding and dropping the rest.
Each decoding site reads it once after readiness. The stream ends at `min(total, limit)`
in the schedule's index order.

```python
from attn_gym.warpdrive import LastEntry

# In the entry point: chunk_offsets is the int32 prefix sum whose last entry counts live chunks.
bound = self.work.bind((chunk_capacity, 1, 1), None, max_ctas, limit=LastEntry(chunk_offsets))
```

Indices at or past the limit are terminal: no role decodes them, `single_item()` reports them
invalid, and a limit of 0 runs nothing. Each role reads the limit itself, so `read()` must give
every role and CTA the same value: it may read only memory written before the launch (or
before `ready` returns) and not written during it. `LastEntry(table, scale)` reads
`scale * table[size - 1]`; a custom `Limit` implements `read()` and the DSL-value protocol
(`__extract_mlir_values__`/`__new_from_mlir_values__`). Only local schedules take a limit, and
`assignment()` ignores it.

`PdlWait()` issues `griddepcontrol.wait` once per decoding site: each local role, or the
broadcast scheduler. It supplies only the wait, not the producer or dependent-launch setup.
See the PDL tests in
[test_warpdrive_scheduler_interface.py](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive/test_warpdrive_scheduler_interface.py).
A custom `Readiness` supplies `wait()` and `get_key()`; the wait must terminate in every
calling warp.

Split work also needs distinct indices. A split-K map can return `(tile, split)` and its K
range, dropping empty splits; a later kernel reduces partials. The decode example's
`SplitKV` returns `(split, kv_head, batch, splits)`. With per-sequence counts from a preceding
kernel, it drops excess splits after `PdlWait()`.

## Custom index schedules

An `IndexSchedule` selects indices, not items. This local example visits each static
sequence backwards:

```python
from dataclasses import dataclass
from attn_gym.warpdrive import IndexSchedule, Raster, WorkStream


@cute.jit
def countdown(index, problem):
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    def first_index(self, problem):
        return countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):
        return countdown(index - problem.units(), problem)


self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ReverseStatic())
```

Local schedule rules:

- Return a valid index or terminal index `>= total`, never a negative index.
- `first_index` depends only on launch values and the bound problem. `next_index` is a pure
  function giving every role the same sequence.
- Every local stream user calls `first_index` at initialization and `next_index` on advance,
  including once per dropped valid index in skip loops.
- Terminate and assign each kept index to exactly one unit for once-only coverage. These are
  caller obligations, not trace-time proofs.
- A nonpersistent schedule must return a terminal index from `next_index`.

### Broadcast schedules

Set `broadcast=True`, choose positive `stages`, and implement
`claim_next(fetch, problem, index)`. The previous index permits a stateful claim sequence
without another buffer. Only the rank-0 scheduler calls it, after acquiring a free work slot
and while the previous claim is valid.

The layer broadcasts `[index, *words]` and owns the work ring's payload, counts, fences,
tails and drain. `fetch_channels()` may declare extra scheduler channels using the same
channel derivation. `close_claims(fetch, problem)` runs once after the last claim; earlier
items can still be computing. **Close claim state there, not work:** no teardown, fixups or
reductions.

Normally the scheduler decodes/broadcasts the first item too, so roles never decode.
Exception: if `decodes_any_index=True` and the stream neither skips nor waits for readiness,
each role decodes `first_index` locally from launch-time data; later items arrive via the ring.

Host-side contracts:

| Method/field | Responsibility |
|---|---|
| `launch_units(total, max_units)` | Return at least one launched unit; cap is in work units when `uses_max_ctas`, otherwise `None` |
| `index_limit(total, units, cs)` | Bound computed indices and grid x extent for the Int32 launch check; default `total + units - 1` fits strided terminal indices |
| `uses_max_ctas`, `uses_workspace` | Declare resource use truthfully |
| `get_key()` | Include all static choices |

Override `index_limit` if the default does not bound your sequence. See `IndexSchedule` in
[scheduling.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/scheduling.py) for the full method contract.

## External tile orders: subclass WorkMap

Wrap an external tile order in a `WorkMap`; warpdrive still owns index distribution,
the grid and the work ring. [external_maps.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/external_maps.py)
wraps CUTLASS's static persistent order and FA4's tile schedulers. These are user mappings,
not core APIs; their authors own the library-specific input domains.

When wrapping a library, keep the [WorkMap contract](#custom-mappings):

- `bind` must make empty spaces safe, including any external host-side parameter construction.
- `count64` must be exact up to `2**31` (and may saturate there), never wrapping before the
  launch checks it. The external library's own arithmetic must also support the bound space.
- `decode` must return an item determined only by the index and bound space, independent of
  decode state; it wraps an order, not a library-owned pipeline, advance or tail.
- Set `decodes_any_index=True` only when decoding is safe for every nonnegative Int32 index,
  including empty spaces and indices past the end.

## One item in a raw role

Use `single_item()` instead of reading `blockIdx` yourself:

```python
@role(warps=range(4), regs=160, raw=True)
@cute.jit
def epilogue(self, params, *, tmem: Tmem):
    valid, (m, n, _) = self.work.single_item()
    if valid:
        ...
```

It handles the fallback grid, clusters, mappings, readiness and dropped items like a managed
cursor. Call it after `begin_roles()`, only with a nonpersistent local schedule such as
`OneShot()`. Persistent and broadcast schedules reject it; managed roles instead take a
`WorkCursor` and loop.

## Without roles

A kernel with no warp roles can still take its grid and items from a stream. Declare the
stream on a plain class, bind it in the entry point, pass `bound` to the `@cute.kernel`, and
read the work from `bound`:

```python
self.work = WorkStream(schedule=OneShot(grid=grid))  # in __init__ of a plain class

# In __call__ (@cute.jit):
bound = self.work.bind((chunks, heads, batch))
self.kernel(x, bound).launch(grid=bound.grid(), block=(128, 1, 1), stream=stream)

# In the kernel:
valid, (chunk, head, batch) = bound.single_item()
if valid:
    ...
```

`bound.cursor()` returns a `WorkCursor` for a local persistent schedule; loop on it as a role
does. There is no `begin_roles()`/`run_roles()`, so the kernel pays no barrier-initialization
fence or CTA sync, and gets no placement, channels or storage plans.

Both methods raise `ProtocolError` for:

- a stream declared on a `WarpSpecialized` kernel (its roles take a `WorkCursor` or call
  `self.work.single_item()`);
- `WorkStream(record=...)`, which logs per role;
- a broadcast schedule (`AtomicPersistent`, `ClusterLaunchControl`), which needs a scheduler
  warp and a work ring;
- `single_item()` with a persistent schedule.

One loop runs every item, so the cursor's loop shape is not checked here.
