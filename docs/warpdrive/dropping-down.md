# Dropping down

[Back to warpdrive](index.md)

Keep managed handoffs where they fit. Use raw CuTeDSL for the part they cannot express: one
signal, one channel side, or a whole barrier set. The layer still allocates resources;
**you own the synchronization you replace**.

```text
managed handle + signal=False   you replace one signal and its fences
channel + external=            you drive an entire unbound side
Barriers                       you drive the whole protocol
```

Start with [`raw_interop.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/raw_interop.py):
a managed producer and raw consumer share a ring. A raw watcher joins a `NamedBarrier`
through its native ID.

## Example: managed and raw on one ring

The example keeps the producer managed and replaces the consumer side:

```text
managed producer: acquire EMPTY -> write -> commit FULL
raw consumer:     wait FULL -> 64 threads read -> read.sync() -> one EMPTY arrival
raw completion:   64 consumers publish results and arrive on done
                  32 watchers sync natively on done.id -> read results
```

This excerpt keeps all participants but omits the math and prologue. The full runnable
version is [`raw_interop.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/raw_interop.py).

```python
self.stages = 2
self.ring = Ring(self.stages, external="consumer", override={"empty_counts": 1})
self.read = NamedBarrier(threads=64)  # consumer's two warps
self.done = NamedBarrier(threads=96)  # 64 consumers + 32 watchers


@role(warps=range(1, 3), raw=True)
@cute.jit
def consumer(self, ctx, me: Local):
    t = me.tid  # consumer-local thread 0..63
    full = self.barrier_ptr("ring")
    empty = full + self.stages
    state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
    for e in cutlass.range(ctx.n, unroll=1):
        cute.arch.mbarrier_wait(full + state.index, state.phase)
        ...  # read stage state.index
        self.read.sync()  # all 64 threads finished reading before the one release
        if t == 0:
            cute.arch.mbarrier_arrive(empty + state.index)
        state.advance()
    ...  # store per-thread results in SMEM
    self.done.arrive()


@role(warps=3, raw=True)
@cute.jit
def watcher(self, ctx):
    cute.arch.barrier(barrier_id=self.done.id, number_of_threads=self.done.threads)
    ...  # read the results published before the consumers' arrival
```

The producer uses `ring.produce()` unchanged. `read.sync()` orders all 64 readers before the
single EMPTY arrival. **The count alone would not.**

`describe()` makes both raw boundaries visible:

```text
ring  Ring(2)  producer -> raw code
external  consumer side driven by raw code; empty counts are the override's (unverified)
done: arrive by consumer; native id by watcher;
      unverified: roles calling it have 64 of 96 threads; ...
```

`test/warpdrive/test_warpdrive_hardening.py` exercises several stage counts and epoch counts.

## Escape hatches by feature

### Replace a handle's signal

A handle exposes `h.index`, `h.phase`, `h.full_barrier` (also `h.barrier`),
`h.empty_barrier`, `h.data` and `h.view(...)`.

`h.commit(signal=False)` or `h.release(signal=False)` closes the handle but emits
**neither its signal nor its fences**. Supply both, with the effective count. This includes
the derived `smem_async` fences: every ordinary reader of a TMA-filled stage, or writer of
an MMA-consumed SMEM stage, runs `fence_async_shared()` before the raw arrival. See
[Storage and fences](channels.md#storage-and-fences) for `release_fence` and `commit_fence`.
The alias diagnostic marks ordering supplied only by a raw signal as `unverified`.

Changing a count does not make a one-lane release safe. Conceptually:

```text
all reader threads finish their accesses and required fences
  -> synchronize those readers
  -> selected thread(s) arrive
  -> producer may reuse the stage
```

For one arrival per warp, use `Warps`, not a custom count. It runs the side's fence in all
lanes, then synchronizes the warp and elects one lane to arrive. The raw local arrival is:

```python
cute.arch.sync_warp()
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(bar)
```

This supplies **only** the local arrival, not fences or remote targets. `Threads` arrives
from every bound thread. Other sparse counts need an override and your own participation
and ordering argument.

### Override counts, bytes or masks

```python
self.ring = Ring(2, override={"empty_counts": 1})
```

The managed consumer must now supply one raw arrival and close with `release(signal=False)`.
The override does not select the arriving lane.

| Override key | Replaces | Accepted values |
|---|---|---|
| `full_counts`, `empty_counts` | Expected arrivals | Positive int, per-slot list, or rank map of either form |
| `tx` | Bytes armed on FULL | Nonnegative int; requires a TMA channel |
| `full_mask`, `empty_mask` | UMMA completion masks | Nonnegative int |

Count lists need one entry per slot; rank maps must cover every CTA rank. Inspect:

- `end.scope`: effective counts, masks and transaction bytes, after overrides.
- `end.channel.derived_scope`: values before overrides.
- `end.targets`: target CTAs for the end's signals.

`describe()` shows overrides beside derived values; it does not validate the raw protocol.
For example, `tx=0` and a well-typed mask naming a nonexistent rank are accepted.
Named-barrier channels do not accept overrides.

### Drive a channel end by state and pointer

Managed ends expose:

| Accessor | Meaning |
|---|---|
| `end.full_barrier(i)`, `end.empty_barrier(i)` | This CTA's mbarrier pointers |
| `end.state` | Ring `PipelineState`, or Slots `Int32` phase mask |
| `end.index` | Ring's next stage |
| `end.phase(slot)` | Next wait parity; omit `slot` for a Ring |
| `self.barrier_ptr(name)` | Base pointer after `begin_roles()`, also available in roles |

For `n` stages, FULL is `[0, n)` and EMPTY `[n, 2*n)` from the base. Raw code can wait with
`cute.arch.mbarrier_wait(end.full_barrier(st.index), st.phase)` and signal with
`cute.arch.mbarrier_arrive(end.empty_barrier(st.index))`. Advance the state you read and
restore it with `end.state = st` before the next managed call.

Raw waits bypass handle-lifetime and alias-ordering checks. Reading **or** assigning
`end.state` invalidates outstanding try tokens and unadopted previews. Even a read exposes
live state that raw code may advance.

Sharing changes the state contract:

- **Exclusive consumers:** `skip(n)` is `advance_state(end.state, n)` plus trace bookkeeping
  (`advance_state` is CuTeDSL 4.8's `state.advance(n)`, on every supported version). Use a
  positive Python int or dynamic `Int32` with `n >= 0`. Raw state or dynamic skips make
  ownership unverified. `override={"empty_counts": ...}` lifts equal consumer contributions;
  each owner must supply the replacement count.
- **Round-robin sides:** instance `j` of `k` starts at epoch `j`; each managed acquire/wait
  steps `k`, exposed by `end.channel.turns(side)`. Raw advancement must also step `k`.
  Raw state makes trip-count ownership unverified; alias orderings through the ring are
  already unverified.

### Leave one side external

```python
self.ring = Ring(2, external="consumer", override={"empty_counts": 1})
```

**No role binds the external side.** Raw code drives its pointers and keeps its own state.
The managed producer still uses `ring.produce()`.

- `external="consumer"` requires `empty_counts`; `external="producer"` requires `full_counts`.
  Without bound roles, the layer cannot derive these counts. `describe()` marks them unverified.
- Managed end bindings, role names and fences on the external side raise. Supply every wait,
  completion, fence and signal yourself. No release fence is derived for external consumers
  of TMA-filled stages: ordinary readers must run `fence_async_shared()` before release.
  No commit fence is derived for external producers of SMEM stages read by MMA, either.
- Only CTA-local mbarrier channels with `Sharing.JOIN` on both sides are supported. There is
  no TMA multicast; raw code drives the side in each CTA.

Alternatively, bind a managed role and use raw code through its endpoint state/accessors.
If neither side is managed, use `Barriers` instead.

### Tokens and previews

- Loop-carried `Token`s are checked normally. If the checker cannot follow your protocol,
  pass a raw Boolean (`cute.arch.mbarrier_try_wait(bar, phase)` or `token.ready`) inside
  `with unchecked("token", reason=...)`. This skips the end, epoch and dynamic-region checks.
- A `peek()` preview exposes `preview.index` and `preview.phase` for raw tensor indexing.
  The next wait on the end owns the epoch; raw accesses through those indices are outside
  the alias diagnostic.

### Named barriers and named channels

For a declared `NamedBarrier`, `.id` and `.threads` let native code join the same generation
as managed `sync()`/`arrive()`:

```python
cute.arch.barrier(barrier_id=nb.id, number_of_threads=nb.threads)
```

Reading `.id` records `native id by ...`; the checker cannot see the native calls. Missing
arrivals in a CTA with a native caller become `unverified`, not an error. Counts beyond that
CTA's available role threads still raise. An ID read outside roles allows the whole block.

A `sync="named"` channel exposes one `NamedBarrier` per slot through
`end.channel.full_barriers[s]` and `end.channel.empty_barriers[s]`.
`op.named_barrier_ids` lists the allocated consecutive IDs. Use the objects' `.sync()` and
`.arrive()`, or native `cute.arch.barrier`/`barrier_arrive` with their ID/count. Named channels
have no mbarrier pointers and take no count overrides. Per-generation participation is
never verified.

`alias.named_barrier(id, n)` is for an ID no declaration owns. It checks collisions with
layer, channel and `NamedBarrier` IDs; native `cute.arch` calls do not. See
[named barriers](channels.md#named-barriers) for allocation and reuse rules.

### Own the whole protocol

`Barriers(n, counts=...)` reserves mbarriers initialized and fenced with the layer's barriers.
Access them through a `name: Barriers` role parameter or `self.barrier_ptr(name)`.
There are no protocol checks.

With `counts=None`, initialize them yourself **between `begin_roles()` and `run_roles()`**.
For example, a CUTLASS pipeline can take the storage with
`cutlass.pipeline.PipelineUmmaAsync.create(barrier_storage=self.barrier_ptr(name), defer_sync=True)`.

A `@role(raw=True)` role receives context and may take `Instance`, `Local`, `Tmem` and
`Barriers`, or use `self.<NamedBarrier>`. It cannot take managed ends, work cursors or turnstile
seats. Returns are ignored; handle and stream-drain checks do not run. For a nonpersistent
local schedule, use `self.work.single_item()`. For persistence, use a managed role with a raw
body instead.

### Suppress one check

`unchecked(check, reason=...)` supports `unclosed`, `cross_region`, `tma`, `alias` and `token`.
It suppresses only that check inside the block and records the opt-out in `describe()`.
It changes diagnostics, not device synchronization.

## Compiler pitfall: saved indices across loops

Optimized ptxas (reproduced with 12.8–13.5, SM90+ targets) can miscompile this valid
PTX dataflow:

```text
read tid -> role dispatch branch -> saved = tid + nonzero_constant
                                -> loop advances tid's PTX register
                                -> address after loop uses saved
```

It may fold `base + (tid + C) * scale` into `base + tid * scale + C * scale`, but use
the **loop-advanced register** instead of the original tid. The result can silently
corrupt an in-bounds location; a clean memcheck alone does not rule it out. The bug
also occurs with non-thread-index sources. Role dispatch makes the block placement
natural, but neither managed channels nor raw synchronization causes the wrong code.

Inspect kept PTX and SASS when an index survives a loop, and confirm a suspect with a
numerical or sentinel check against a ptxas `-O0` build. Minimal repro and NVIDIA report:
[gist](https://gist.github.com/drisspg/8ec9afe991ae22c1d6e6edaed8e046a6).

Verified PTX workarounds include computing the address before the loop, keeping a
separate induction register, or defining the saved index in the same basic block as
its source read. A post-loop fresh `%tid.x` read also works at PTX level, but another
`cute.arch.thread_idx()` call can be removed by NVVM's common-subexpression elimination.
Check the emitted PTX rather than trusting the Python spelling. `cute.PtxasOptions("-O0")`
is a diagnostic/workaround for the whole kernel, not a cheap performance-neutral fix.
There is no universal layer-level thread-index guard for arbitrary role bodies.

## Rules that still apply

- **Counts are not identities.** An mbarrier count does not elect a lane or order other
  threads' accesses before its arrival.
- **Pointers are local.** Peer CTAs have barriers at the same offsets, but a pointer alone
  is not a remote arrival. Raw signals must honor effective counts, target CTAs
  (`end.targets`) and masks.
- **Raw state is not checked ownership.** Accessors do not validate bounds or ownership:
  `Slots.phase(slot)` checks for a Python int, not range or instance binding. Restoring old
  `end.state` can silently break the protocol.
- **Opt-outs add no ordering.** Document raw counts, completion waits and memory ordering at
  the call site. Opt-out bookkeeping is tracing-global, not thread-safe; compile in isolated
  processes rather than concurrent threads.
- **Alias opt-outs certify nothing.** `unchecked("alias")` excludes only annotations inside
  its block from the [alias diagnostic](storage.md#check-declared-aliases). A TMA opt-out is
  separate from a copy the checker could not recognize.
