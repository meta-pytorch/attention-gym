# Storage

[Back to warpdrive](index.md)

Use `SmemPlan` to declare the shared-memory buffers behind a channel, as in the built-in
[GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py).
The plan allocates storage; channel handles select the stage you can use. You still supply
the layouts and TMA/MMA partitions.

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

You can also allocate tensors yourself and index them with `h.index`.

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

This **canonical view** selects the stage in the original layout. For TMA/MMA, build a
partitioned view and register its stage axis with `StageView`:

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

Reassign the result of `views()` and return that endpoint from the role. It keeps the same
state and checks the stage extent, **not** whether the tensor addresses the named buffer.
Use a view registered in a loop only in that region; it holds the loop's values.

`h.view()` selects the handle's stage, never the endpoint's next one. Omit the name only
for a single buffer. Registered partitions do not change `h.data`.

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

V reuses the root allocation from its first byte, with V's dtype, swizzle and layout.
It adds no storage, must fit, and cannot require stricter alignment. Allocation checks
static byte sizes, not dynamic ones. Multiple channels can bind a buffer without reallocating.

**Sharing bytes does not order accesses.** Keep K reads separate from V writes through
channel epochs. Ordinary ring reuse is not enough when different stages overlap:

```text
ring orders:       read stage 0 -> release stage 0 -> overwrite stage 0
ring does not add: read stage 0 -> overwrite overlapping bytes in stage 1
```

Each CTA in a cluster allocates its own copy at the same offset. Pair-scoped channels can
bind buffers, but cannot own `payload`s. Binding derives `commit_fence="smem_async"` for
`Threads`/`Warps` → `Umma` ([Storage and fences](channels.md#storage-and-fences)); it adds
no DSMEM publication. `remote_scope` still defaults to `"cta"`. For ordinary memory read
by a peer, follow the [pair memory-scope rules](clusters.md#memory-scope).

## Check declared aliases

The alias diagnostic checks declared SMEM overlaps and `TmemPlan.alias` regions for
conflicting accesses. **It reads your annotations, not all kernel memory accesses.**

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

The same mode runs the [fence check](channels.md#checking-explicit-fences).

Analysis runs during tracing. `alias_check` is part of `op.__cute_cache_key__()`, so
`jit_cache` cannot reuse warn/off builds for strict requests. Cache hits restore reports
and warnings ([compile keys](roles.md#compile-keys)).

Annotate **at the operation**, not merely when obtaining a pointer:

```python
with kv.produce() as h:
    dst = h.view("K", write=True, by="tma")
    cute.copy(tma_k, src, dst, tma_bar_ptr=h.barrier)
```

If a helper already has the tensor, `h.access("K", write=True, by="tma")` records the access
without returning a view. Set `read=True`, `write=True`, or both. `by=` names the agent:
`"threads"` for ordinary SMEM loads/stores, `"tma"` for TMA, or `"umma"` for MMA operand reads.

TMEM offers the same two forms:

```python
# At the MMA writing S, inside its producer handle:
s_ptr = tmem.view("S", stage, write=True, by="umma")
...  # build/use the tensor for this MMA

# At an access through a tensor built earlier:
tmem.access("P", stage, read=True, by="umma")
```

`tmem.view()` returns an fp32 pointer, **not a CuTe tensor**. `tmem.access()` only records.
The slot is a Python int [TMEM plan slot](channels.md#tmem), not a dynamic ring index.
Use `"threads"` for `tcgen05.ld/st` and `"umma"` for MMA.

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
  by that signal is `unverified`.

### Read the result

`describe()` reports each declared alias:

| Result | Meaning |
|---|---|
| `potential hazard` | Conflicting annotations with an established missing ordering edge; includes accesses, source lines and rule |
| `unverified` | Not enough modeled information to decide |
| `no hazard found` | No hazard among the annotated conflicting pairs; **not proof of safety** |

Inspect `describe()` even without a warning: some unverified cases appear only there.

The model covers single-CTA kernels, whole-region/stage footprints, same-channel handoffs
and cross-channel paths. It runs loops and persistent work items twice. Clusters,
raw/unannotated accesses and unmatched epochs can remain unverified. `h.data`, `tmem.ptr`,
`tmem.cols` and `tmem.base` do not annotate later accesses. Cross-role convergence is not proved.

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

**Unequal traced counts leave epoch matching unknown.** Prologues, peeled tiles or different
loops can give roles different per-item counts. Only these cross-role links remain definite:

```text
item's commits -> consumer's last wait for that item
Slots release  -> next item's first acquire
```

Other pairings within this item and the next are guesses. If only a guess orders a pair,
the report says `unverified`, naming the channel, both counts and "unknown epoch correspondence",
without warning. If no pairing could order it, the result remains `potential hazard`.

An unbalanced channel can therefore downgrade a real hazard to unverified. Treat these
entries as unchecked: balance traced counts or inspect the overlap by hand. The diagnostic
does not choose loop counts to balance channels; that could hide a missing publication/release.

Other assumptions and limits:

- Loops run at least twice; TMEM loads finish before the next signal.
- `tcgen05` thread-sync fences surround recorded barriers and channel signals; the analysis
  assumes rather than verifies them.
- A role's SMEM accesses are treated as one thread set: no cross-warp SMEM check.
- Annotations are trusted to cover the role's uses of each annotated region.

To leave an access unmodeled, do not annotate it; `alias_check="off"` skips the whole
diagnostic. Neither adds a fence or certifies safety.
See [alias.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/alias.py) for modeled rules/PTX references and
the [safety contract](guarantees.md) for the synchronization rules beyond aliasing.
