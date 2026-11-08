# Storage

[Back to warpdrive](index.md)

Channels control **when** a stage is usable. `SmemPlan` declares **where** its data lives.
You can instead allocate tensors yourself and index them with `h.index`; neither approach
chooses your layout or changes the channel's synchronization.

```python
self.smem_plan = SmemPlan()
a = self.smem_plan.buffer("A", cutlass.BFloat16, align=1024)
b = self.smem_plan.buffer("B", cutlass.BFloat16, align=1024)
self.ab = Ring(4, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
```

```text
SmemPlan      declares A/B types, alignment and stage axes
begin_roles   binds layouts and allocates the whole buffers
handle        selects A/B at its own h.index
channel       orders publication and reuse; stores no A/B data in its barriers
```

## Allocate in the prologue

```python
# Inside the kernel; a_layout and b_layout include all four stages.
smem = self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout})
sA = self.buffer("A")
sB = self.buffer("B")
...  # partition sA/sB for TMA and MMA, build params
self.run_roles(params)
```

`SmemPlan.buffer(name, dtype, *, align=16, stage_axis=-1, overlaps=None)` declares a buffer.
Layouts bind later because they often depend on the entry point's tiled MMA.

- Supply a layout for **every** plan buffer, including overlaps. Plain and swizzled layouts
  work; allocation preserves dtype, alignment and swizzle.
- Allocation follows declaration order, after internal barriers, payloads and TMEM
  bookkeeping. You can continue allocating through the returned `smem` allocator.
- `self.buffer(name)` returns the whole tensor after `begin_roles()`. These tensors bypass
  handle stage selection and acquire no channel ownership.
- `stage_axis` defaults to the last layout mode. For a bound buffer, that mode's extent must
  be static and equal the channel's stage count.
- `buffer=` takes one buffer or a nonempty dictionary of named buffers from this kernel's
  plan. It cannot combine with `payload=`.

## The handle selects the stage

With one bound buffer, `h.data` is a tensor; with several, it is a dictionary:

```python
with ab.produce() as h:
    stage_a = h.data["A"]
    stage_b = h.data["B"]
    ...  # A and B at h.index, each with its stage mode removed
```

That is the **canonical view**: the original layout with the stage mode selected. TMA/MMA
usually need a partitioned view instead. Build the partition as usual, then register its
stage axis with `StageView`:

```python
ab = ab.views(A=StageView(tAsA, axis=1), B=StageView(tBsB, axis=1))
with ab.produce() as h:
    cute.copy(tma_a, tAgA[None, m, k], h.view("A"), tma_bar_ptr=h.barrier)
    cute.copy(tma_b, tBgB[None, n, k], h.view("B"), tma_bar_ptr=h.barrier)
```

```text
h.data["A"]  -> canonical A at h.index
h.view("A")  -> registered tAsA at h.index (canonical A if none registered)
```

Rebind the result of `views()` and return the endpoint from the role. It retains the same
state and checks the partition's stage extent, **not** that the tensor addresses the declared
buffer. A view registered in a loop holds that loop's values; use it in that region only.

`h.view()` always uses the handle's snapshot, never the endpoint's next stage. Omit the name
only for a single buffer. Registering partitions does not change `h.data`.

To select other modes too, supply one coordinate per nonstage mode (`None` keeps a mode):

```python
ab = ab.views(A=StageView(tCrA, axis=3))
# Inside an acquired handle:
a_fragment = h.view("A", None, None, kb)  # tCrA[None, None, kb, h.index]
```

These excerpts assume the GEMM examples' TMA/MMA partitions; storage binding does not build
the partitions for you.

## Overlapping buffers

Use `overlaps=` to give the same bytes another layout or dtype:

```python
k = self.smem_plan.buffer("K", cutlass.BFloat16, align=1024)
v = self.smem_plan.buffer("V", cutlass.BFloat16, align=1024, overlaps=k)
self.kv = Ring(kv_stages, Tma(stage_bytes), Umma, buffer={"K": k, "V": v})
```

V starts at the root allocation's first byte, recast to V's dtype/swizzle/layout, and adds no
storage. It must fit and cannot demand stricter alignment. Allocation checks static byte
sizes; dynamic sizes get no static bounds check. Multiple channels can bind one buffer
without allocating again.

**Overlapping bytes does not order accesses.** Channel epochs must keep K readers from
racing V writers. If the layouts overlap across different stage numbers, ordinary ring
reuse is insufficient:

```text
ring orders:       read stage 0 -> release stage 0 -> overwrite stage 0
ring does not add: read stage 0 -> overwrite overlapping bytes in stage 1
```

In a cluster, each CTA allocates its own copy at the same offset. Bound buffers are allowed
on pair-scoped channels, unlike layer-owned `payload`s. Binding adds no DSMEM publication
or fence; `remote_scope` still defaults to `"cta"`. For ordinary memory read by a peer,
follow the [pair memory-scope rules](clusters.md#memory-scope).

## Check declared aliases

The alias diagnostic looks for conflicting accesses to declared SMEM overlaps and
`TmemPlan.alias` regions. **It checks annotations, not all kernel memory accesses.**

```python
op.alias_check = "warn"  # default; set before compiling
compile_kernel(op)  # your kernel's compile helper
print(op.describe())
```

| Mode | Effect |
|---|---|
| `"warn"` | Warn about findings and keep a report |
| `"strict"` | Raise `ProtocolError` for a potential hazard, not an unverified case |
| `"off"` | Skip analysis |

Analysis runs while tracing. `jit_cache` keys `alias_check` through
`op.__cute_cache_key__()`: warn/off builds cannot satisfy strict requests. Cache loads restore
reports and warnings ([compile keys](roles.md#compile-keys)).

Annotate **at the operation**, not merely when obtaining a pointer:

```python
with kv.produce() as h:
    dst = h.view("K", write=True, by="tma")
    cute.copy(tma_k, src, dst, tma_bar_ptr=h.barrier)
```

If a helper already holds the tensor, use `h.access("K", write=True, by="tma")` to record the
access without returning a view. Set `read=True`, `write=True`, or both. SMEM agents are
`"threads"` (ordinary loads/stores), `"tma"`, and `"umma"` (MMA operand reads).

TMEM offers the same two forms:

```python
# At the MMA writing S, inside its producer handle:
s_ptr = tmem.view("S", stage, write=True, by="umma")
...  # build/use the tensor for this MMA

# At an access through a tensor built earlier:
tmem.access("P", stage, read=True, by="umma")
```

`tmem.view()` returns an fp32 pointer, **not a CuTe tensor**; `tmem.access()` only records.
Its slot is a Python int [TMEM plan slot](channels.md#tmem), not a dynamic ring index. Agents
are `"threads"` for `tcgen05.ld/st` and `"umma"` for MMA.

For a narrower TMEM footprint, declare `warps`, `lanes` and `cols`:

- `lanes`/`cols` are ranges or callables of the warp index; columns are relative to the region.
- Default thread accesses cover that warp's 32-lane TMEM subpartition; MMA covers all 128 lanes.
- Both default to the whole region width. These are your declarations, not inferred addresses.

### Make ordering visible

Use recording wrappers when the diagnostic needs to see a raw instruction's ordering:

```python
from attn_gym.warpdrive import (
    bulk_commit_group,
    bulk_wait_group,
    fence_async_shared,
    named_barrier,
)

fence_async_shared()  # cute.arch.fence_view_async_shared()
bulk_commit_group()  # cute.arch.cp_async_bulk_commit_group()
bulk_wait_group(0, read=True)  # cute.arch.cp_async_bulk_wait_group(...)
named_barrier(barrier_id, number_of_threads=128)  # cute.arch.barrier(...)
```

These are separate examples, **not a synchronization recipe**. IDs 1 and 2 are reserved;
avoid allocated channel/barrier IDs too. Prefer a declared
[`NamedBarrier`](channels.md#named-barriers), whose `sync()` records the same event, or
choose an available ID with the correct participant count.

- Wrappers emit the underlying instruction plus a host trace event, not another device op.
- Access annotations emit no device code. Enabling/disabling this host-only analysis does
  not change generated device code.
- Managed signals and declared fences are recorded automatically. Raw instructions remain
  legal but add no modeled ordering edge.
- `signal=False` records the managed close, not your manual signal. An ordering supplied only
  by that signal is `unverified`. So is one that might depend on a handle opened under
  `unchecked("unclosed")` and never closed.

### Read the result

`describe()` reports each declared alias:

| Result | Meaning |
|---|---|
| `potential hazard` | Conflicting annotations with an established missing ordering edge; includes accesses, source lines and rule |
| `unverified` | Not enough modeled information to decide |
| `no hazard found` | No hazard among the annotated conflicting pairs; **not proof of safety** |

Inspect the report even without a warning. Warn mode reports findings, but some unverified
cases appear only in `describe()`.

**Model boundary.** This first-phase diagnostic models single-CTA kernels, whole-region/stage
footprints, same-channel handoffs and cross-channel paths. Its bounded trace runs loops twice
and persistent work items twice. Clusters, raw/unannotated accesses and unmatched epochs can
remain unverified. `h.data`, `tmem.ptr`, `tmem.cols` and `tmem.base` are unannotated acquisitions,
not descriptions of later accesses. Cross-role convergence is not proved.

**Which handoffs order storage?**

- An `n`-stage ring orders epoch `e` before `e+n`, not `e+1`. The stage-reuse edge covers only
  storage indexed by that ring's stage, with static, disjoint stage spans.
- Ring-stage SMEM shared across channels is unverified. Consumers of one fan-out epoch, and
  producers of one fan-in epoch, are unordered with respect to each other.
- Ring reuse inside an inner work-item loop adds no modeled edge. A pair ordered only by that
  reuse can therefore be reported as a hazard.
- Exclusive-ring same-stage handoffs are checked normally: consumers do not share epochs.
  Paths through the exclusive ring and another channel are unverified because owners follow
  runtime skips.
- Round-robin epoch matching is not modeled. An ordering through that channel is unverified;
  a pair no possible epoch pairing could order remains a potential hazard.

**Unequal traced counts mean unknown epoch correspondence.** Prologues, peeled final tiles
or differently shaped loops can give two roles different per-item counts. The model then
keeps only these definite cross-role links:

```text
item's commits -> consumer's last wait for that item
Slots release  -> next item's first acquire
```

Other pairings within the item and the next are guesses. If only a guess orders a pair,
the report says `unverified` with the channel, both counts and "unknown epoch correspondence",
without warning. If no pairing could order it, the result stays `potential hazard`.

Those guesses over-approximate: an unbalanced channel can downgrade a real hazard to
unverified. Treat such entries as unchecked; balance traced counts or inspect the overlap
by hand. The diagnostic does not choose loop counts to balance channels, which could hide a
missing publication/release.

Other assumptions and limits:

- Loops run at least twice; TMEM loads finish before the next signal.
- `tcgen05` thread-sync fences surround recorded barriers and channel signals; the analysis
  assumes rather than verifies them.
- A role's SMEM accesses are treated as one thread set: no cross-warp SMEM check.
- Annotations are trusted to cover the role's uses of each annotated region.

For an intentionally unmodeled access:

```python
with unchecked("alias", reason="access is ordered by the raw protocol below"):
    tmem.access("P", stage, write=True, by="threads")
```

Only annotations inside the block are excluded. It adds no fence and certifies no safety.
See [alias.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/alias.py) for modeled rules/PTX references and
[GUARANTEES.md](guarantees.md) for the broader synchronization contract.
