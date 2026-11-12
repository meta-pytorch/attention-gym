# Channels

[Back to warpdrive](index.md)

A channel controls **when a producer may fill a stage, when consumers may read it, and when
it can be reused**. Start with `Ring` for an ordered stream of tiles, as in the built-in
[GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py).
The channel manages synchronization; the data can live in separate SMEM or TMEM.

```python
self.ab = Ring(3, Tma(stage_bytes), Umma)
self.acc = Ring(2, Umma, Warps)
```

`Ring(2, Umma, Warps)` has two reusable stages. MMA completion publishes a stage; consumer
warps release it. With one MMA warp and four consumer warps:

```text
producer acquires EMPTY stage
  -> issues MMA -> elected tcgen05.commit -> MMA completes -> FULL
  -> consumers wait, then read
  -> all lanes finish the handoff; each warp syncs and elects one EMPTY arrival
  -> 4 EMPTY arrivals -> producer may reuse the stage
```

The stage count is capacity, not a consumer count. FULL/EMPTY barriers hold protocol state,
not data. A channel without `payload=` or `buffer=` allocates no payload storage.

Roles receive [producer and consumer ends](roles.md#arguments) by parameter name.
Use a handle to hold one stage while working on it:

```python
with ab.produce() as h:
    ...  # fill stage h.index; commit on exit

with ab.consume() as h:
    ...  # read stage h.index; release on exit
```

These are separate producer and consumer excerpts. Return the updated end from the role.
After collection, `channel.bound(side)` gives `(role instance, slot)` pairs for either side.

## Stages and phases

A **stage** is one unit of reusable channel capacity. `Ring` selects stages in order;
a **slot** is an explicitly selected stage of `Slots`. An **epoch** is one exchange,
numbered in ring order or per slot; for a standalone named barrier, one use of that barrier.
A **phase** is the mbarrier wait's parity bit, not an epoch counter. It repeats every two
uses of the same barrier. Named barriers have no phase bit.

An **end** is a role's advancing channel state; a **handle** holds one taken stage while the
end moves on. **Acquire** takes an empty stage; **wait** takes a full one. A producer
**commits** to publish it; a consumer **releases** it for reuse. A **signal** is the mechanism
that completes either handoff; an **arrival** contributes to a barrier's count. A commit or
release need not complete synchronously, and a TMA commit emits no arrival.

For zero-based operation `k` on an `n`-stage ring end without round-robin rotation:

| State | Value |
|---|---|
| Stage | `k % n` |
| Consumer wait phase | `(k // n) % 2` |
| Producer acquire phase | `1 XOR ((k // n) % 2)` |

For a two-stage ring:

```text
epoch                  0  1  2  3  4  5
stage                  0  1  0  1  0  1
consumer phase         0  0  1  1  0  0
producer phase         1  1  0  0  1  1
```

The opposite initial phases let producers acquire empty stages while consumers wait for the
first publication. Matching epochs preserve publication order and prevent reuse until all
declared consumers release.

**Acquire/wait advances the end; commit/release does not.** A handle keeps its index
and phase after the end moves on. In each CTA, full mbarriers occupy `[0, n)` and empty
mbarriers `[n, 2*n)` from the channel's base.

The mechanisms choose what completes the handoff:

| Mechanism | Signal/completion |
|---|---|
| `Tma(tx_bytes)` | Acquire arms the full barrier's expected bytes; TMA copies complete it. Commit adds no arrival. |
| `Umma` | An elected `tcgen05.commit` signals completion of the issuing thread's prior MMAs. |
| `Threads` | Every thread of every bound role arrives. |
| `Warps` | All bound threads participate; each warp syncs, then one elected lane arrives. |

`Tma` is producer-only. `tx_bytes` is the bytes per stage landing in each destination CTA's
SMEM, whichever CTA issues the copies. See [multicast](clusters.md#multicast) for
`Tma(tx_bytes, multicast="n")`.

A single-mechanism TMA or UMMA side binds to one single-warp role: one active instance per
epoch for round-robin, or one owner per epoch for [exclusive consumers](#exclusive-consumers).

## Wait mode

Blocking acquires and waits default to `wait="suspend"`: CuTeDSL's `mbarrier.try_wait.parity`
with a suspend-time hint, which ptxas lowers to a `NANOSLEEP.SYNCS` fallback. `wait="spin"`
retries a hint-less `try_wait` instead, the loop CUTLASS's C++ `ClusterBarrier::wait` and the
cuDNN-frontend GDN/KDA kernels use:

```python
self.trip = Ring(1, Umma, Threads, wait="spin")  # a serial MMA <-> CUDA-core handoff
```

Spinning can shorten a serial chain of handoffs that are usually satisfied quickly (GDN's
backward summary reached parity with cuDNN only this way), but the spinning warps compete for
issue slots with co-resident warps (spinning every wait made the KDA prefill ~1% slower).
Choose it per channel and measure. It applies to `acquire`/`wait`, span `begin` and
`peek`; probes, producer tails and named channels are unaffected. `describe()` lists
`wait spin` under the channel.

## CUTLASS crosswalk

CUTLASS's state-taking calls leave advancement to the caller. Warpdrive's `acquire()`/`wait()`
advance the end and return a handle for the taken stage; the APIs are not interchangeable.

| CUTLASS Python API | warpdrive counterpart |
|---|---|
| `PipelineAsync` with async-thread sides | `Ring(n, Threads, Threads)` |
| `PipelineTmaUmma` | `Ring(n, Tma(bytes_per_cta), Umma)`; acquire arms bytes, TMA completes FULL, UMMA releases EMPTY |
| `producer_acquire(state)` / `consumer_wait(state)` | `end.acquire()` / `end.wait()` |
| `producer_commit(state)` / `consumer_release(state)` | `h.commit()` / `h.release()` |
| `producer_try_acquire(state)` / `consumer_try_wait(state)` | `end.try_acquire()` / `end.try_wait()`; a checked `Token`, not a raw Boolean |
| `NamedBarrier.arrive_unaligned()` | `NamedBarrier.arrive()`; warpdrive allocates the ID |
| `NamedBarrier.arrive_and_wait()` | `NamedBarrier.sync(aligned=True)`; warpdrive defaults to unaligned sync |

The definitions are in CUTLASS's `cutlass/pipeline/sm90.py`, `sm100.py` and `helpers.py`,
and warpdrive's
[`channels.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/channels.py)
and [`named.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/named.py).
Cluster byte counts and targets follow [placement](clusters.md#placement-rules), not a direct
copy of CUTLASS constructor arguments.

## Counts come from the roles

For CTA-local mbarriers, each stage's count sums its bound roles' contributions:

| Mechanism | Contribution per role instance |
|---|---|
| `Threads` | `32 * role.num_warps` |
| `Warps` | `role.num_warps` |
| `Tma` / `Umma` | 1 |

Thus the four-warp consumer above supplies 4 EMPTY arrivals with `Warps`, or 128 with
`Threads`. Multiple joined consumers must all release before reuse. Cross-CTA counts and
targets follow the [placement rules](clusters.md#placement-rules); `Warps` counts one arrival
per warp of each signalling CTA.

For both `Threads` and `Warps`, predicate data accesses, not the handoff. A missing lane
can strand its warp. See [participation](guarantees.md#participation) and the
[liveness check](guarantees.md#liveness) for what tracing establishes.

`Warps` requires mbarriers; named channels count threads and reject it. Prefer it for
multi-warp releases: FA4 and CUTLASS signal per warp.

Use [count overrides or raw barriers](dropping-down.md) for async-copy relay warps or sparse
protocols. Changing a count neither elects lanes nor synchronizes their work.

## Overlapping handles

Use explicit handles when lifetimes overlap without nesting. In FA4, K and V share a ring.
MMA holds V(i-1) and K(i), then releases V first:

```python
for i in cutlass.range(n_blocks - 1, unroll=1):
    v = kv.wait()  # V(i-1)
    k = kv.wait()  # K(i)
    pv(o, p, v.index)
    v.release()
    qk(s, q, k.index)
    k.release()
```

`pv`/`qk` stand for the kernel's MMA code. Nested `with` blocks would release K first.

Follow the [lifetime rule](guarantees.md#lifetimes) and
[held-stage check](guarantees.md#participation). A [span](#spans) holds a stage from a loop's
first iteration to its last. `h.retire()` closes a final unused producer acquire without
publication; automatically tailed pair channels reject it because that stage cannot drain.

## Storage and fences

Keep buffers separate and index them with `h.index`, or attach simple CTA-local storage:

```python
self.stats = Ring(2, Threads, Threads, payload=(cutlass.Float32, (32,)))
```

This allocates uninitialized, 16-byte-aligned SMEM of shape `(32, 2)`—generally `(*shape, n)`.
`h.data` selects the handle's stage. User pair-scoped channels cannot own these payloads.
For swizzled tiles, multiple buffers or overlaps, use [`SmemPlan` and `buffer=`](storage.md).
Binding a buffer leaves remote memory scope unchanged. It derives a commit fence for
`Threads`/`Warps` -> `Umma` channels (see below).

Declare fences on the side that needs them, before its managed signal:

| Fence kind | Instruction |
|---|---|
| `"tmem_store"` | `tcgen05.wait::st` |
| `"tmem_load"` | `tcgen05.wait::ld` |
| `"smem_async"` | `fence.proxy.async` |

[`fences.RULES`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/fences.py)
defines the access pairs used by both default derivation and the
[explicit fence check](#checking-explicit-fences).

`tmem_store` completes TMEM stores for a reader, usually MMA. `tmem_load` completes
`tcgen05.ld` before a later MMA overwrites those columns. Declare
`release_fence="tmem_load"` when every release needs it (FA4 backward's `dq` and `dkv`).
If only one path needs it, call `fence_view_async_tmem_load()` there instead (persistent
GEMM's `early_release`). Channel fences run on every managed signal.
`smem_async` orders generic SMEM accesses against the async proxy (TMA, `tcgen05.mma`).

`commit_fence`/`release_fence` accept a string or ordered tuple of distinct kinds. For
example, FA4 backward's dS uses `commit_fence=("tmem_store", "smem_async")` to hand over both
TMEM and SMEM stores. A string is a one-element tuple; the layer emits exactly that order.
Unknown/repeated kinds and commit fences on TMA producers are rejected.

Two SMEM handoffs get a default proxy fence:

- **TMA producer → `Threads`/`Warps` consumer:** `release_fence="smem_async"` orders ordinary
  reads (`ld.shared`, `autovec_copy`, `ldmatrix`) before the next TMA fill. Without it, a
  pending read may observe that next fill. TMA completion covers the other direction:
  consumers can read after the FULL wait without a fence. `Umma` consumers read through the
  async proxy and get no default release fence.
- **`Threads`/`Warps` producer → `Umma` consumer, with `payload=` or `buffer=`:**
  `commit_fence="smem_async"` orders ordinary stores before MMA's async-proxy reads.
  FA4 decode's P ring uses this default. Without SMEM storage, no commit fence is derived;
  declare `"tmem_store"` for TMEM stores.

A proxy fence acts only in its issuing thread. Every reading/writing thread must run it
before arrival, including every lane before a `Warps` arrival. See PTX's Memory Consistency
Model ("Proxies" and the causality axiom) and tcgen05 "Shared Memory Accesses".
The fence is still required across CTAs and does not change `remote_scope`
([Memory scope](clusters.md#memory-scope)). TMA stage releases still default to `"cta"`,
as in CUTLASS.

Use `release_fence=()` only when consumers make no generic access to the filled storage
(for example, they only issue TMA stores or MMAs), or every reader calls
`fence_async_shared()` after its last read and before release. CTA-pair FA4 backward's dPsum
uses the latter.

Use `commit_fence=()` only when producers make no generic writes or every writer calls
`fence_async_shared()` after its last write and before arrival. CTA-pair FA4 backward's `ds`
opts out: `make_ds` fences its writers before raw signaling; the managed relay writes no data.

Explicit values **replace**, not extend, defaults. Include `"smem_async"` when adding
`"tmem_store"`, as single-CTA FA4 backward's `ds` does with `("tmem_store", "smem_async")`.
`describe()` shows replacements as `override  commit_fence smem_async -> ...` or
`override  release_fence smem_async -> ...`.

A joining consumer that does not read an epoch (another instance or role processes it) can
pass over it with `end.consume_unread()`: it waits for the epoch and releases it at once,
without the release fences. Nothing runs between that wait and release, so the end's reads
of the stage all precede an earlier release, whose fence ordered them; a previewed epoch
keeps its fences. Accesses to the same storage through another channel (aliased buffers)
are not covered: order those yourself. The fence check and alias diagnostic treat the pass
as an unfenced release. The KDA prefill's two `cg0` instances join every `raw` epoch but
read alternate ones; passing over the other instance's epochs this way dropped one
`fence.proxy.async` per chunk and made the kernel 0.9-1.7% faster (GB300, CuTeDSL 4.8.1,
8x4096 H32: 451.3 → 447.1 us with per-instance bodies, 453.5 → 445.9 us with a shared body).

`commit(signal=False)` and `release(signal=False)` omit **the signal and all its fences**,
including defaults. Other async work, including TMA stores and TMEM aliases, still needs
its own completion waits and proxy fences.

### Checking explicit fences

Annotate accesses with [`view` or `access`](storage.md#check-declared-aliases), then inspect
`describe()` or `checks().fences` after compilation. See [fence checks](guarantees.md#fences)
for the verdicts and limits of manual fences and cross-channel ordering.

## TMEM

A `TmemPlan` assigns named column ranges; aliases share columns, not lifetimes:

```python
self.tmem_plan = (
    TmemPlan().region("S", 128, 2).region("O", 128, 2).alias("P", of="S", offset=64, cols=64)
)
```

S uses 256 columns, O another 256, and P uses no additional allocation.
`print(self.tmem_plan.describe())` shows:

```text
  tmem S   [0, 128), [128, 256)
  tmem O   [256, 384), [384, 512)
  tmem P   [64, 128), [192, 256) (inside S)
```

- `tmem.ptr("S", stage)` returns an fp32 TMEM pointer; `tmem.cols("S", stage)` its column
  offset. Here `stage` is a **Python int plan index**, not a dynamic ring index.
- To select a ring stage dynamically, build a tensor spanning the region's entries and index
  it with `h.index`, as GEMM does. Reinterpret packed BF16 yourself.
- Nonaliases occupy declaration-order intervals of `cols * slots`; each alias stays inside
  its parent entry. Allocation rounds up to a power of two, 32–512 columns. Capacity,
  duplicate names and alias bounds are checked.

**Every TMEM-using role, including raw roles, must take `tmem: Tmem`.** All declared users
join an unaligned teardown barrier before free. With no annotated users, nothing allocates.

In one CTA, `begin_roles()` allocates from the lowest TMEM-user warp other than warp 0,
or warp 0 if it is the only user. This overlaps warp 0's barrier initialization. The same
warp frees. Clusters allocate after the cluster sync in `run_roles()`
([work and exit](clusters.md#work-and-exit)).

See [alias annotations](storage.md#check-declared-aliases) and
[alias checks](guarantees.md#aliases) before overlapping S/P accesses.

## Slots and instances

Use `Slots` when buffers are reused independently, not in ring order:

```python
self.s = Slots(2, Umma, Threads)


@role(warps=[range(4), range(4, 8)], regs=192)
@cute.jit
def softmax(self, params, *, s: Consumer[Instance], stage: Instance):
    with s.consume(stage) as h:
        ...
    return s
```

```text
softmax[0] -> slot 0 -> phase 0, 1, 0, ...
softmax[1] -> slot 1 -> phase 0, 1, 0, ...   (independent)
```

- Pass a Python int in `[0, n)` on every acquire/wait; there are at most 31 slots.
- Producers start each slot at phase 1; consumers at phase 0. Taking a slot flips only its bit.
- Visit slots in any order. An instance-bound end contributes only to its own slot and must
  still pass that slot explicitly; another instance's slot is an error.
- An unbound end can visit any slot. The [FA4 example](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_fwd.py) uses one MMA
  warp to produce two slots consumed independently by two softmax instances.

## Sharing epochs

`producer_sharing=` and `consumer_sharing=` independently select who takes each epoch:

| `Sharing` | Who takes an epoch? | Supported sides |
|---|---|---|
| `JOIN` (default) | Every bound instance; counts sum | Either side of `Ring` or `Slots` |
| [`ROUND_ROBIN`](#round-robin-instances) | Instances take turns; one instance's count | Either side of `Ring` |
| [`EXCLUSIVE`](#exclusive-consumers) | One declared consumer owns it; others skip it | Consumers of an mbarrier `Ring` |

For `Slots`, “bound” means bound to that slot. Non-join sides give each epoch one owner.
When their acquire/wait/skip sequences are static, collection checks
[ownership](#static-ownership-check).

## Per-role consumer mechanisms

When an MMA warp and CUDA threads both read every stage, give each consumer role its own
mechanism:

```python
self.h = Ring(2, Tma(stage_bytes), {"mma": Umma, "cuda": Threads}, buffer=h_buf)
```

The dict names every consumer role and its mechanism; it implies `consumer_roles`.
Every key must bind a consumer end, and no consumer role may be omitted. Both roles
take every epoch. Each instance contributes its role's [count](#counts-come-from-the-roles).
With a one-warp `mma` and an eight-warp `cuda`:

| Role | Mechanism | Release | Fence before it | EMPTY arrivals |
|---|---|---|---|---|
| `mma` | `Umma` | elected `tcgen05.commit` | none (MMAs read through the async proxy) | 1 |
| `cuda` | `Threads` | every thread arrives | `smem_async` (derived on a `Tma` ring) | 256 |
| `cuda` | `Warps` | warp sync, one elected arrival per warp | `smem_async` | 8 |

With `Threads`, the empty count is 257. `describe()` shows
`Tma(… B) -> {mma: Umma, cuda: Threads}`, `empty 257` and
`release_fence smem_async (cuda)`. For these bindings, only an `empty_counts`
[override](dropping-down.md) changes the count.

The [derived release fence](#storage-and-fences) applies to the `Threads`/`Warps` roles only,
as does an explicit `release_fence=`. Declare `release_fence=()` only if the thread roles make no
generic access to what the producer fills; the alias diagnostic checks each handle against its
own role's mechanism and fences.

This requires a CTA-local mbarrier `JOIN` side, with no multicast.
Round-robin, exclusive and named channels reject a dict. Each `Umma` role must have one
single-warp instance. Producers take one mechanism.

## Round-robin instances

```python
self.s = Ring(4, Umma, Threads, consumer_sharing=Sharing.ROUND_ROBIN)


@role(warps=[range(4, 8), range(8, 12)], regs=120)
@cute.jit
def softmax(self, params, *, stage: Instance, s: Consumer):
    for j in cutlass.range(stage, n_blocks, 2, unroll=1):
        with s.consume() as h:  # block j's S
            ...
    return s
```

```text
epoch         0  1  2  3  4  5
consumer      0  1  0  1  0  1
ring stage    0  1  2  3  0  1
```

Instance `j` of `k`, in binding order, takes epochs `j, j+k, j+2k, ...`. Its operation `m`
uses stage `(j + m*k) % n` and that epoch's [phase](#stages-and-phases). The example counts
128 arrivals per epoch, not the 256 of a join. `describe()` shows the rotation and count.

Constraints:

- At least two instances of one role, with equal width. Either or both sides may rotate.
- Loops must visit the instance's own epochs. Static trip-count mismatches raise; dynamic
  counts remain `unverified` by the [ownership check](#static-ownership-check).
- Rotation continues across work items; a new item does not reset the end.
- CTA-local only, with no multicast. On mbarrier rings, `n` must be a multiple of `k`.
  The layer adds no turn-taking: a phase wait is exact only if the waiter saw that stage's previous
  epoch. With three instances on two stages, instance 2's first wait could pass before any fill.

The [decode example](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py) alternates KV blocks this way.

## Exclusive consumers

Use an exclusive ring when different roles consume different epochs of one ordered stream.
The producer still fills every epoch; the owner waits/releases, and all other consumers skip.

```python
self.kv = Ring(
    12,
    Tma(stage_bytes),
    Umma,
    consumer_roles=("mma_qk", "mma_pv"),
    consumer_sharing=Sharing.EXCLUSIVE,
    buffer={"K": k_buf, "V": v_buf},
)

# In mma_qk, for each KV block j >= 3:
kv.skip(v_stages)  # V(j - 3), owned by mma_pv, was filled before K(j)
for _ in cutlass.range_constexpr(k_stages):
    with kv.consume() as h:
        ...  # wait for, read and release this warp's own epoch
```

`skip(n)` emits only index/phase/count advancement, not a wait or arrival. Counts are one
owner's contribution (1 for `Umma`), not a [join](#counts-come-from-the-roles). Other channels
are unaffected.

The layer requires:

- An mbarrier `Ring`, one joining producer instance, and CTA-local placement.
- Explicit `consumer_roles`, each bound to a consumer end; at least two consumer instances.
- Equal consumer contributions, with single-warp roles for `Umma`, unless `empty_counts`
  is [overridden](dropping-down.md).

Overrides for `tx`, `full_counts` and `full_mask` do not change ownership. An `empty_counts`
override lifts the equal-contribution check and marks counts unverified: every owner must
supply that count, typically with `release(signal=False)` and raw arrivals.

`skip()` is consumer-only on exclusive rings. It accepts a positive Python int or dynamic
`Int32` with runtime value `n >= 0`, as `PipelineState.advance` does. Dynamic skips, or reading
or assigning `end.state`, make ownership `unverified`. Skips stale outstanding
[try tokens](#try-tokens); skipping past an unconsumed [preview](#previews) raises.

### Static ownership check

Use static acquires, waits and skips consistently per item or per launch, with one joining
instance opposite the shared side. For round-robin instance `j` to resume at `j` next item,
`k` must divide the item's epoch count. `describe()` reports `epochs  verified: ...` or
`unverified (...)`; see [ownership coverage](guarantees.md#participation).

**Ownership does not make a skipped-to phase wait exact.** Epoch `e` waits for
`(e // n) % 2` at stage `e % n`. That wait is exact once epoch `e-n` was filled. Before
then, an older epoch can make it pass immediately.

A consumer has enough history if its first wait is below `n` and each later wait follows
a wait for epoch `e-n` or later. Join/valid round-robin consumers observe this themselves.
An exclusive consumer may skip it and need a rendezvous:

```text
3-stage ring; consumer's first wait is epoch 4 (stage 1)
  needs evidence that epoch 1 was filled
  skipping epochs 0..3 supplies no such evidence
  another consumer must rendezvous before this wait
```

The ownership report names the first wait needing this rendezvous, but inserts no handshake.
Dynamic schedules leave the check to you; a consumer that skips the rendezvous can hang.

[Alias checks](guarantees.md#aliases) distinguish same-stage reuse from paths through
another channel.

### Rendezvous recipe: FA4's QKIssued/PVIssued

[`SharedKVDecode`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py) shares one ring between QK and PV MMA
warps. Its loader fills:

```text
K(0), K(1), K(2), V(0), K(3), V(1), ... , last two V blocks
```

With 12 ring stages and four stages per K or V block, PV's first wait, V(0), is epoch 12—a
full lap after launch. Two [named barriers](#named-barriers) bound the skew. This is a protocol
excerpt; `...` stands for the example's MMA work:

```python
self.qk_issued = NamedBarrier(threads=64)  # mma_qk arrives, mma_pv syncs
self.pv_issued = NamedBarrier(threads=64)  # mma_pv arrives, mma_qk syncs

# mma_qk, per block j (after skipping V(j - 3) for j >= 3):
self.pv_issued.sync()
...  # wait for, issue and release K(j)'s stages
self.qk_issued.arrive()
# After the last block: two more pv_issued.sync(); qk_issued.arrive()

# mma_pv: pv_issued.arrive(), then for each of K(0) and K(1):
# skip it, qk_issued.sync(), pv_issued.arrive().
# Per block j (after skipping K(j + 2) if it exists):
self.qk_issued.sync()
...  # wait for, issue and release V(j)'s stages; arrive on pv_issued after the last MMA
```

- Before PV waits for V(j), QK has issued K(j+2), the group filled just before it.
- Before QK waits for K(j), `j >= 3`, PV has issued V(j-3), the group just before it.

Thus all earlier epochs were filled and both waits are exact at any ring depth. Each side
syncs on one barrier between arrivals on the other; neither can arrive twice in an epoch
([reusable rendezvous](#reusable-rendezvous)). PV's final `pv_issued` arrival is unmatched,
as in FA4. That is sound for this `OneShot` decode, one item per CTA; a persistent version
must balance it. These barriers also pace the two MMA warps.

## Named-barrier channels

Choose `sync="named"` for a CTA-local thread exchange. It replaces each stage's mbarriers
with two CTA named barriers; **there is no phase bit**.

```text
                 producer                 consumer
FULL barrier     commit: arrive           wait: arrive + wait
EMPTY barrier    acquire: arrive + wait   release: arrive
count at either barrier = producer threads + consumer threads
```

Both mechanisms must be `Threads`; overrides are not supported. Operations use unaligned
`barrier.cta.sync`/`barrier.cta.arrive` (the unaligned counterparts of `bar.sync`/`bar.arrive`).

For a running maximum that producers also read:

```python
rr = Sharing.ROUND_ROBIN  # two softmax warpgroups produce in turn
self.row_sum = Ring(2, Threads, Threads, sync="named", producer_sharing=rr)
self.row_max = Ring(1, Threads, Threads, sync="named", producer_sharing=rr, turnstile="mufu")
self.mufu = Turnstile()

with mufu.turn():
    mh = row_max.acquire()
    ...  # atomically max this warpgroup's block max into sRowMax
    mh.commit(wait=True)  # every producer and consumer thread has arrived
    ...  # read the shared max
```

`commit(wait=True)` also waits on FULL, making the producers' prior writes visible to each
other. It runs the ordinary `commit_fence`. It cannot combine with `signal=False` or run on
an mbarrier channel. FA4's decode uses it for the shared running max.

Initialization and reuse:

- At role start, every consumer instance arrives once on its stages' EMPTY barriers: its
  bound slot, all stages for an unbound nonrotating end, or stages `s % k == j` for round-robin
  instance `j` of `k`. This supplies the initial empty capacity.
- Consumer work before its first explicit wait cannot publish data that the producer's
  first acquire depends on: the startup arrival already happened.
- Final release is unmatched at exit. Keep participation balanced each epoch; a fast side
  must not arrive twice in one epoch.
- If round-robin instances share a stage's barriers (`n % k != 0`), they could complete the
  barrier between themselves. Name an enabled turnstile over exactly those instances and
  acquire/wait inside its turn; tracing checks this.

IDs are allocated per channel, in declaration order, in consecutive blocks from 15 downward:
`n` full IDs followed by `n` empty IDs. `channel.full_barriers[s]` and
`channel.empty_barriers[s]` are [`NamedBarrier`s](#named-barriers), named `<channel>.full[s]`
and `<channel>.empty[s]`, with `.id` and `.threads`. `op.named_barrier_ids` and `describe()`
list them. Named channels have no mbarrier pointer accessors.

ID 0 is for CTA synchronization; 1 and 2 belong to the layer. The remaining 13 IDs are shared
by named channels and declared named barriers. Hand-picked IDs must avoid allocated IDs,
even small ones. `alias.named_barrier` checks collisions; raw `cute.arch.barrier` and
`barrier_arrive` do not.

## Named barriers

Use `NamedBarrier` for a CTA rendezvous without a channel's full/empty protocol. It allocates
one barrier ID with a thread count:

```python
self.compute_sync = NamedBarrier(threads=256)  # both compute warpgroups
self.max_done = NamedBarrier(threads=160)  # correction warpgroup + load warp

self.compute_sync.sync()  # compute: arrive and wait
self.max_done.arrive()  # correction: arrive and continue
self.max_done.sync()  # load: arrive and wait
```

Declare it in `__init__` and use `self.<name>` in managed or raw roles. An epoch
completes after `threads` arrivals through `sync()` or `arrive()`.

- Default calls are unaligned, so participants can reach them from different code sites.
- `aligned=True` emits `bar.sync`/`bar.arrive`: every thread of each participating warp must
  execute that same instruction convergently, as GEMM epilogues do.
- IDs allocate from 3 upward in declaration order, independently of channels allocating
  from 15 downward. Collection rejects declarations exceeding the shared 13-ID budget.
- `.id` and `.threads` support native calls, e.g.
  `cute.arch.barrier(barrier_id=nb.id, number_of_threads=nb.threads)`. These may mix with
  managed methods on the same barrier. Native calls are invisible, but reading `.id` while
  tracing records `native id by ...` in `describe()`.
- `describe()` lists IDs, counts and traced callers. Trace wait ranges wrap `sync()` in
  `<name>.sync`; [pipeline plans](tracing.md) model channels, not standalone named barriers.

The count is a multiple of 32, at most 1024 and at most the kernel's thread count. A role
may use only some warps. See the [named-barrier contract](guarantees.md#named-barriers) for
participation, liveness and alias checks.

`NamedBarriers(n, threads=N)` declares `n` such barriers with consecutive IDs. `bars[i]` with
a Python int is the `NamedBarrier` named `<name>[i]`; with a runtime index it returns a view
whose `sync()`/`arrive()` compute the ID at run time. Each barrier is allocated and checked
like a separate declaration. A [shared-body](roles.md#shared-bodies) role indexing with its
`Instance` value is checked as instance `i` using barrier `i`; any other runtime index
counts as native use of every barrier, so short participation is unverified.

### One-shot done flag

Use one barrier, without an empty/acknowledgement barrier, when publishers will not reuse it:

```python
self.done = NamedBarrier(threads=128 + 32)  # publishers + waiters

# Publishers (128 threads):
...  # write the result to SMEM
self.done.arrive()

# Waiters (32 threads):
self.done.sync()
...  # read it
```

```text
publishers: ordinary SMEM stores -> arrive -> continue
waiters:                           sync   -> read published stores
```

The [PTX `bar`/`barrier` contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
makes prior accesses performed relative to participants when **sync, red or arrive** completes
the barrier. Performed writes are visible to participants; sync also holds new accesses until
completion. Thus `st.shared; barrier.arrive id, N` publishes to
`barrier.sync id, N; ld.shared` without an extra `membar`. Mixed arrive/sync is allowed;
`bar.arrive`/`bar.sync` are the aligned spellings.

Every writer must arrive and every waiter must sync with the same ID/count. This argument
covers ordinary SMEM, not asynchronous proxies: TMA/TMEM still need their completion waits
and fences.

Use this once per launch, or once per item with at most one item per CTA (the `OneShot`
[decode](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py)).
Otherwise a fast publisher can add its next arrival to the current epoch. Add an
acknowledgement edge, or use a named channel's EMPTY barrier.

**Racecheck can flag this pattern.** compute-sanitizer racecheck reports stores published
by `arrive()` against reads after `sync()` on the same barrier (swap-AB decode's row sums are
a known false positive). That is not a blanket suppression: for every new finding, recheck
the accesses, participants, epochs and emitted instructions.

### Reusable rendezvous

```python
self.compute_sync = NamedBarrier(threads=256)

for t in cutlass.range(n_tiles, unroll=1):
    ...  # each warpgroup writes its half
    self.compute_sync.sync()
    ...  # read the other half
```

Exactly `threads` participants must sync each epoch. None can leave epoch `e`
until all arrive, so none arrives early for `e+1`. This does not protect data overwritten
after the sync from readers still using it.

With arrive-only participants, your protocol must prevent a second arrival before the current
epoch completes. A wait on another edge can do that; the named channel's EMPTY wait is
one example. `NamedBarrier` itself supplies no such edge.

## Spans

Use a span to hold a stage from a counted loop's first iteration to its last. An ordinary
handle must close in the region that opened it. A span can wait in the first iteration and
signal in the last. KDA's WY backward uses this to accumulate dq over its V tiles:

```python
dq_span = dq.span()  # takes the stage and advances the end; no wait yet
for v in cutlass.range(self.num_v_tiles):  # a Python int: 0, 1, ..., n - 1
    ...
    dq_span.begin(v)  # if v == 0: wait until the stage is empty
    mma(dq_span.index, accumulate=v != 0)
    dq_span.commit(v)  # if v == n - 1: tcgen05.commit
```

```text
              v = 0                 v = 1 ... n - 2        v = n - 1
span()   begin: wait EMPTY          begin: nothing         begin: nothing
         commit: nothing            commit: nothing        commit: signal FULL
```

`mma` stands for the kernel's MMA code. To publish after the loop, call `commit()` without
an index in the region that opened the span. WY's dA does this after one more MMA.
A consumer span waits for FULL in the first iteration, then closes with `release(v)` or
`release()`. Both closes accept `signal=False`. A span also exposes `index`, `phase`,
`view()`, `data` and barrier pointers.

Prefer acquiring before the loop when the role has no work to do first. An ordinary handle
also works with `cutlass.range_constexpr`; a span keeps the loop dynamic and delays the wait.

Tracing checks, and raises `ProtocolError` otherwise:

- `begin(i)`, `commit(i)` and `release(i)` run directly in the body of one
  `for i in cutlass.range(n)` loop with a Python int `n >= 1`, start 0 and step 1; `i` is
  that loop's index; and the loop sits in the region that opened the span. A branch, a nested
  or constexpr loop, dynamic bounds or another loop's index are rejected. The trip count comes
  from the loop, so the two guards cannot disagree with it.
- `begin` is traced once, before the close. `commit()`/`release()` without an index is in
  the opening region.
- The span is closed exactly once; `retire()` is rejected.
- Named-barrier channels and `Tma` producers cannot hold a span: a named wait is an arrival,
  and a `Tma` producer arms its stage when it acquires. Use an ordinary handle there.

`span()` counts as the end's acquire/wait for the [ownership check](#static-ownership-check),
[try tokens](#try-tokens) (outstanding ones go stale) and [previews](#previews) (the span owns
a previewed epoch). The [alias diagnostic](storage.md#check-declared-aliases) places a span's
wait in its loop's first repetition and an in-loop signal in the last.

Use the stage only between the wait and signal; see [lifetime coverage](guarantees.md#lifetimes).

## Try tokens

A token probes the next epoch without taking it, letting independent work run before the
ordinary acquire/wait:

```python
token = kv.try_wait()  # probe; no handle, no advance
...  # independent work
h = kv.wait(token=token)  # blocking wait only if the probe failed; advances once

token = out.try_acquire()
h = out.acquire(token=token)  # a Tma producer arms the stage here, once, either way
```

`try_wait()` probes FULL; `try_acquire()` probes EMPTY. Each emits one
`mbarrier.try_wait.parity` and returns a `Token` with dynamic Boolean `ready`. For `Slots`,
pass the same slot to probe and take. Without `token=`, codegen is the ordinary blocking wait.

A probe may suspend for a hardware-dependent time. **It is not a nonblocking poll and has
no latency bound.** A token records readiness, not a stage index. It cannot arm, commit or
release a stage; leaving it unused is harmless.

A token belongs to its probing end's next epoch (the same slot for `Slots`). Acquiring,
waiting or reading/assigning `end.state` makes it stale. Probe inside a dynamic region or
carry a fresh token into each iteration; see [token checks](guarantees.md#lifetimes).

Probe in the consuming region, as FA4 does inside an unrolled K-step loop, or carry the DSL
value. CUTLASS mainloops use this pattern:

```python
token = ab.try_wait()
for k in cutlass.range(k_tiles, unroll=1):
    h = ab.wait(token=token)
    ...
    h.release()
    token = ab.try_wait()  # next iteration's epoch
```

For a checked carry:

1. Between the entering probe and loop entry, do not acquire/wait or access `end.state`.
2. Consume the carried token at the body's top level before another acquire/wait on that end.
3. Between the outgoing probe and body end, do not acquire/wait or access `end.state`.

The back-edge check runs at role exit and names the probe on failure. After the loop, the
last probe remains usable (the entering probe for zero trips). A token merged by an `if`
reassignment does not; probe after the branch.

`token=` takes only a `Token`. If the checker cannot follow your protocol, wait on the
end's barriers with [raw code](dropping-down.md#drive-a-channel-end-by-state-and-pointer).

Named channels reject probes and `token=`: their wait is itself an arrival. Probing is not
automatically faster; measure before keeping it.

## Previews

Use a preview to read the next epoch before opening the handle that will release it:

```python
preview = q.peek()  # wait for the next epoch, without advancing
issue_s(preview.view("Q", read=True, by="umma"))
...
h = q.wait()  # owns that epoch; its release must cover the preview's reads
...
h.release()
```

`peek()` waits for FULL and returns a read-only `Preview`, without arrival, handle or state
advance. It is consumer-only and mbarrier-only. `preview.index`/`phase` identify the stage and phase;
`view(name, *coords, by=...)` and `access(name, by=...)` record reads. `write=True`, `commit`,
`release` and `retire` raise. FA4 backward previews Q(t+1) for S(t+1) while Q(t)'s handle
remains open for dK(t).

The next `wait()` on the end (same slot for `Slots`) **in trace order** owns the preview.
Its release mechanism/fences cover the preview's annotated accesses in the alias diagnostic.
Tracing rejects:

- Use after that handle closes, or outside the dynamic region of `peek()`.
- Reading/assigning `end.state` before the owning wait.
- A preview left without a later wait in the role.

Trace order is an approximation: a preview in a loop is attributed to the wait after the loop,
although at runtime the next iteration owns all but the final preview. Peel the last
iteration, as backward does, so a peeled wait owns that final preview. CuTeDSL also refuses
to carry a preview through a dynamic loop/branch that calls its methods.

See [preview lifetime limits](guarantees.md#lifetimes).

## Turnstiles

A turnstile gives instances of one role method a round-robin critical section:

```python
self.mufu = Turnstile()


@role(warps=[range(4), range(4, 8)], regs=192)
@cute.jit
def softmax(self, params, *, mufu: Turn):
    with mufu.turn():
        ...  # exp2 work
    return mufu
```

It needs at least two instances with matching CTA placement. With `p` seats, it orders a
finite prefix of `0, 1, ..., p-1, 0, ...`. Every seat's threads participate; its barrier counts
the previous seat's threads. FA4 alternates two softmax instances around exp2 this way.

`Turnstile(enabled=False)` emits no synchronization and provides no exclusion. The
declaration must still be valid.
