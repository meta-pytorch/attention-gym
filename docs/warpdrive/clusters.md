# Clusters and CTA pairs

[Back to warpdrive](index.md)

Use `CtaPair()` to let two CTAs cooperate on one work item. In the built-in GEMM, both
CTAs load operands and store results; rank 0 issues the group-TWO MMA:

```text
one work item
  rank 0 (leader): load its A/B half -> issue MMA -> epilogue its result half
  rank 1 (peer):   load its A/B half -------------> epilogue its result half
                                      MMA reads/writes both CTAs
```

```python
self.cluster = CtaPair()  # cluster (2, 1, 1), tcgen05 CtaGroup.TWO


@role(warps=5, regs=40, cta="leader")
@cute.jit
def mma(self, params, *, ab: Consumer, acc: Producer, tmem: Tmem):
    ...
    return ab, acc
```

The pair shares **one work item**, not one per CTA. The declaration does not convert
single-CTA atoms or layouts. To write the two-CTA path:

- Double the MMA tile's M dimension.
- Build atoms with `self.cluster.cta_group` and `self.cluster.layout_vmnk()`;
  partition with `self.cluster.rank()`.
- Call `rank()` from converged warp code, never a lane-0 branch.
- Call `begin_roles()` and `run_roles()` from converged CTA/cluster code; use
  `launch_config()` for the declared block and cluster shape.

The [persistent GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py)
and [FA4](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_fwd.py)
implement both single-CTA and pair paths. GEMM also supports [larger clusters](#larger-clusters).

## Which channels cross CTAs?

Placement decides where roles run; channel scope decides which barriers their signals reach.

| Participants and copies | Channel scope |
|---|---|
| Same placement, no multicast | Independent CTA-local channels |
| Mixed placement | Cross-CTA; pair-scoped in a `CtaPair` |
| [Multicast](#multicast) reaches another CTA | Cross-CTA even with the same placement |

With no multicast, all/all gives one local channel per CTA; leader/leader gives
one per pair leader. Pair GEMM's two main channels cross CTAs:

```text
ab:   both CTAs issue TMA -> leader's FULL barrier -> leader issues MMA
      both CTAs reuse A/B <- leader's UMMA release

acc:  leader's UMMA completion -> FULL in both CTAs -> both epilogues read
      leader reuses TMEM      <- both epilogues release
```

For `ab`, use group-TWO copies. The leader arms **twice** the declared per-CTA TMA bytes.
For `acc`, the layer counts both epilogues and derives their masks; do not double the thread
counts yourself. Every UMMA signaler in a `CtaPair` must be leader-only, even on a local channel.

`print(op.describe())` shows the counts, targets, masks and bytes. The
[placement rules](#placement-rules) give the full derivation.

## Work and exit

Local schedules compute the same sequence in every CTA of the cluster. `AtomicPersistent`
and `ClusterLaunchControl` claim on rank 0 and broadcast items to every CTA. CLC cancels
whole clusters.

If the kernel uses TMEM, at least one TMEM-using role must run in all CTAs. The first such
role allocates after cluster synchronization in `run_roles()`. All declared users, including
raw roles, join teardown before free. Pair allocation/free handshakes with the peer, `rank XOR 1`.

A producer automatically tails if a remote UMMA consumer can still read its storage,
including a multicast partner outside its pair. Return the final end state:

- Ring tails drain from the returned stage and phase.
- Slots tails wait only on that end's acquired slots, using their final phase bits.
- Retired acquires are rejected on these channels: they cannot be tailed.

All threads reach the exit cluster barrier after dispatch and TMEM teardown. See the
[exit obligations](guarantees.md#what-a-successful-check-means) for work outside that protocol.

## Larger clusters

```python
self.cluster = Cluster(2, 2, pair=True)  # two MMA pairs side by side along N
self.ab = Ring(stages, Tma(stage_bytes, multicast="n"), Umma, buffer={"A": a, "B": b})
```

`Cluster(m, n, pair=...)` groups up to 8 CTAs around one item. With `pair=True`, M must be
even: consecutive CTAs along M form MMA pairs. `CtaPair()` is `Cluster(2, 1, pair=True)`;
`SingleCta()` is `Cluster(1, 1)`.

```text
Cluster(2, 2, pair=True)       N ->
                         pair 0     pair 1
  M / pair member 0      rank 0     rank 2     <- leaders issue MMA
  M / pair member 1      rank 1     rank 3
```

`cta="leader"` selects even ranks (every CTA without pairs). `coords(rank)` returns `(v, mi, ni)`, with the pair member fastest, matching the
CUTLASS `(V, M, N, K)` `layout_vmnk()`. The launch cluster is `(m * n, 1, 1)`, so a work
unit remains consecutive `blockIdx.x` values.

### Multicast

Pairs along N need the same A rows; pairs along M need the same B rows. Multicast splits the
load among CTAs sharing those rows, then delivers each part to its partners.

In the 2x2 pair cluster above, only A is multicast: rank 0 shares with 2, rank 1 with 3.
B is already split across the two CTAs of each pair, and there is only one pair along M.

```python
share_a = self.cluster.multicast("n")  # coord, layout, mask, ctas
tAsA, tAgA = cpasync.tma_partition(tma_a, share_a.coord, share_a.layout, sA, gA)
...
cute.copy(tma_a, tAgA[None, k], s.view("A"), tma_bar_ptr=s.barrier, mcast_mask=share_a.mask)
```

Build the atom with `CopyBulkTensorTileG2SMulticastOp(self.cluster.cta_group)` and pass
`layout_vmnk().shape` to `make_tiled_tma_atom_A`/`_B`. Declare `Tma(bytes, multicast="n")`;
`bytes` remains the per-CTA SMEM bytes per stage, the same as without sharing.

Multicast also changes when storage can be reused. Each CTA's A/B stage waits for both
pair leaders: the next multicast from either pair could overwrite it.

```text
leader 0 finishes reading --+--> EMPTY in ranks 0, 1, 2, 3
leader 2 finishes reading --+
                            each EMPTY waits for both releases
```

Releasing only your own pair can race with the other pair's MMA reads.

Get partitions and masks from `Cluster.multicast` after `begin_roles()`, in converged code;
see [TMA checks](guarantees.md#tma). Persistent GEMM requires power-of-two counts of MMA
tiles along both modes because a 3-way descriptor partition can compile and fault.

### Work items and the scheduler

[Persistent GEMM](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py) runs one MMA tile per pair. At `tile_n=224`,
the 2x2 work item is a **256 x 448 cluster tile**. Each pair offsets the item's tile indices
by its position: `m * pairs_m + mi`, `n * pairs_n + ni`. On a ragged edge, entire pairs can
be out of bounds; TMA zero-fills loads and clips stores.

The rank-0 scheduler publishes each item to all four CTAs. Its ring's rank-0 EMPTY barrier
counts every stream user in every CTA. `launch_args()` caps a persistent grid at the estimated
resident cluster capacity ([grid cap](scheduling.md#grid-cap)).

### What the 2x2 GEMM derives

```python
op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=Cluster(2, 2, pair=True), tile_n=224)
compile_persistent_gemm(op)
print(op.describe())
```

Representative excerpt; print your build for current placement, stages and source lines:

```text
PersistentGemm: 8 warps, 256 threads
  cluster  cluster 2 x 2 (M x N), tcgen05 CtaGroup.TWO, leaders = even ranks
  params   PersistentGemmParams

Roles
  warps  role            regs  cta     raw
  0-3    epilogue         160  all
  4      load              40  all
  5      mma               40  leader
  6      work.scheduler    40  rank0
  7      (padding)         40

Channels
  ab         Ring(6)  load -> mma
             signal    Tma(30720 B) -> Umma
             full      rank 0 1 x6, rank 2 1 x6, tx 61440 B (leader-armed)
             empty     rank 0 2 x6, rank 1 2 x6, rank 2 2 x6, rank 3 2 x6
             scope     cross-CTA, multicast n, release_mask 0b1111, tail
             storage   buffer A, B
             tma       verified CtaGroup.TWO, 30720 B/stage per CTA @ persistent_gemm.py:194,195
  acc        Ring(2)  mma -> epilogue
             signal    Umma -> Threads
             full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2
             empty     rank 0 256 x2, rank 2 256 x2
             scope     cross-CTA, commit_mask rank 0 0b0011, rank 2 0b1100, remote_scope cta
  work_ring  Ring(2)  work.scheduler -> epilogue, load, mma, work.scheduler
             signal    Tma(16 B) -> Threads
             full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2, tx 16 B (leader-armed)
             empty     rank 0 736 x2
             scope     cross-CTA, remote_scope cta
             storage   payload Int32(4,)
  ...
```

- `ab`: each leader arms 60 KiB for its pair; every CTA waits for two leaders' releases.
- `acc`: ownership stays inside each pair—rank 0 commits to 0–1, rank 2 to 2–3.
- `work_ring`: `736 = 4 * (128 + 32) + 2 * 32 + 32` arriving threads: epilogue/load in
  all CTAs, MMA in two leaders, and one scheduler.

## Memory scope

`remote_scope` controls what thread arrivals publish, not where payload bytes live.

| Channel data | Derived scope | Why |
|---|---|---|
| No `payload=` | `"cta"` | Normal TMA/MMA/TMEM ordering comes from those mechanisms; a release only returns a stage |
| Thread producer writes and commits a payload | `"cluster"` | The commit must publish those words to the remote reader |
| `Tma` producer fills a payload | `"cta"` | Bytes complete on the reader's FULL barrier; consumer releases publish no data |

For ordinary SMEM/global writes **outside a payload** that a peer will read, explicitly set
`remote_scope="cluster"` and use the correct peer address. warpdrive cannot infer either.
Cluster scope emits a GPU-scope membar per arriving thread, which is expensive on
per-item handoffs; use it only when the arrival must publish data.

Storage is separate from that barrier ordering:

- User cross-CTA channels cannot have `payload=`: layer-owned payloads are CTA-local.
  The internal work ring is the exception; its scheduler fills every CTA's copy with `st.async`.
- `SmemPlan` buffers may bind to cross-CTA channels. They allocate separately in each CTA at
  matching offsets. `buffer=` leaves scope unchanged and derives a commit fence for
  `Threads`/`Warps` -> `Umma`, as `payload=` does
  ([Storage and fences](channels.md#storage-and-fences)).

See [alias checks](guarantees.md#aliases) before relying on cluster storage overlaps.

## Placement rules

<details>
<summary>Exact derivation: participants → targets/counts → TMA completion, with PTX basis</summary>

### 1. Find the participants and written storage

Let `C(role)` be the ranks where a role runs:

| Placement | Ranks |
|---|---|
| `all` | Every CTA |
| `leader` | Even ranks with MMA pairs; every rank without pairs |
| `rank0` | Rank 0 only: the broadcast scheduler in a cluster larger than a pair, not declarable |

For producer rank `p`:

- `reach(p)` is `p` plus `Cluster.line(p, mode)` for each multicast mode. With both modes,
  this is a **union of lines**, not a Cartesian product.
- `W(p) = reach(p)`: the CTAs whose SMEM `p`'s copies write.

Same `C` for every participant, with no multicast, means CTA-local: ordinary
per-stage counts, unscaled bytes, self-targets and no producer tail. Masks are 1, except a
UMMA commit to its own CTA uses `1 << rank`.

### 2. Find the synchronization unit and counts

For a cross-CTA channel, rank `r`'s unit is:

1. `r` alone if all participants have the same `C` (only copies cross CTAs).
2. `r`'s MMA pair if each participant runs somewhere in every pair.
3. The whole cluster otherwise, when a participant is restricted to rank 0.

A producer targets the consumer CTAs in its unit with FULL signals; a consumer targets the
producer CTAs in its unit with EMPTY signals. **User TMA channels use the next subsection
instead.** In a `CtaPair`, mixed placement therefore targets the consumer/producer union.

For each target and stage, count the signaling instances and CTAs:

| Mechanism | Arrival contribution |
|---|---|
| `Threads` | `32 * role.num_warps` per signaling instance and CTA |
| `Warps` | `role.num_warps` per signaling instance and CTA; all lanes participate, one elected arrival per warp |
| TMA / UMMA | One per signaling CTA |

Nontarget ranks have placeholder count 1 and no promised waiter. A mask has one bit per
target: one integer if all signalers agree, otherwise a per-rank mapping.

With pairs, UMMA signalers must be pair leaders (`cta="leader"`) using group TWO: each CTA's
operands and accumulator belong to its own leader's MMAs.

### 3. Route TMA completion and reuse

Let `W` be all written CTAs (the union of `W(p)`), and `F` the consumer CTAs.

| Written CTAs and consumers | Where bytes complete | Copy group |
|---|---|---|
| `W` is a subset of `F` | Each destination's own FULL barrier | ONE |
| Otherwise, pairs exist and `F` contains only pair leaders | Each destination's pair leader, which must be in `F` | TWO |
| Any other shape, or a consumer with no completing copy | Rejected | — |

Exactly one CTA arms each FULL barrier: its own CTA. It arms `k * Tma.tx_bytes`, where `k` is the number of CTAs completing there
(2 for group TWO, otherwise 1). Each FULL barrier expects **one arrival**.

Consumer `c` releases every producer `p` whose `W(p)` intersects either:

- the destinations completing on `c`'s barrier, or
- `c`'s synchronization unit.

Examples and limits:

- GEMM's all-CTA producers feeding pair leaders: each leader arms `2 * Tma.tx_bytes`.
- Same-placement TMA: each CTA arms its own barrier with unscaled bytes.
- All/all requires group ONE. With pairs, leader/leader accepts either group in the channel
  check; descriptor legality remains yours.
- `Tma.tx_bytes` always means bytes landing in **each CTA's SMEM per stage**. A copy delivers
  its box to every CTA in its mask. The declaration derives releases; it does not issue copies.
- The internal scheduler is separate: rank 0 explicitly fills each target's payload and
  arms unscaled payload bytes, not ordinary multicast TMA.

### Why remote arming is safe

These rules use PTX ISA 9.5:

- `cp.async.bulk.tensor` (§9.7.10.28.5.3) with `.shared::cluster` may write any cluster CTA.
  A `.multicast::cluster` `ctaMask` selects destinations, not necessarily the issuer; data
  lands at the same offset in each.
- With `.cta_group::1`, each destination's mbarrier at `mbar`'s offset is signaled. With
  `.cta_group::2`, the signal targets the destination or its peer according to the parity
  of the CTA holding `mbar`. A leader's barrier therefore collects its pair's bytes.
  CUTLASS `SM100_TMA_2SM_LOAD*` clears the barrier address's peer bit for the same effect.
- `mbarrier.arrive.expect_tx.shared::cluster` (§9.7.15.16.16) arms a remote barrier. The
  tx-count spans `±(2^20 - 1)` (§9.7.15.16.3), so copies may complete before the arm. A barrier
  completes only when pending arrivals and tx-count are both zero (§9.7.15.16.8).
- A wait must observe completion before the next epoch's arrival (§9.7.15.16.5.1).
  The producer rearms only after EMPTY reports the consumers' release, which follows their
  FULL wait. This is why a filled stage cannot have no waiter.

At exit, consumers have waited on every stage filled for them: copies and remote arms have
landed before the exit cluster barrier. Thread releases precede those threads' arrival at
that barrier. Remote UMMA releases need the producer tails described next.

</details>

## Limits

- Copies may land only in CTAs with a consumer, or whose pair leader has one.
- Multicast needs a producer in each destination to arm there.
- `external=` channels are CTA-local: no multicast, and one placement on the managed
  side. Raw code drives the external side independently in each participating CTA.

See the [safety contract](guarantees.md) for target and analysis limits.
