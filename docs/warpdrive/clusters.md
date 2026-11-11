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
| Same placement, no multicast or fills | Independent CTA-local channels |
| Mixed placement | Cross-CTA; pair-scoped in a `CtaPair` |
| [Multicast](#multicast) or [fills](#one-cta-loads-for-others) reaches another CTA | Cross-CTA even with the same placement |

With no multicast or fills, all/all gives one local channel per CTA; leader/leader gives
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
including a multicast partner outside its pair. Return the final endpoint state:

- Ring tails drain from the returned stage and phase.
- Slots tails wait only on that end's acquired slots, using their final phase bits.
- Retired acquires are rejected on these channels: they cannot be tailed.

All threads reach the exit cluster barrier after dispatch and TMEM teardown. This covers
the managed protocol, **not arbitrary async work** started by a role.

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

`cta="leader"` selects even ranks (every CTA without pairs); `cta="rank0"` selects rank 0
once. `coords(rank)` returns `(v, mi, ni)`, with the pair member fastest, matching the
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

Get partitions and masks from `Cluster.multicast` after `begin_roles()`, in converged code.
The TMA check verifies the declared mode and byte count. It rejects undeclared multicast;
a mask from elsewhere leaves the channel `unverified`. It does not prove descriptor legality.
A 3-way partition can compile and fault, so persistent GEMM requires power-of-two counts
of MMA tiles along both modes.

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
- The alias diagnostic reports cluster aliases as `unverified`; it does not analyze cross-CTA
  ordering.

## One CTA loads for others

Use `Tma(fills=...)` when some CTAs cannot issue their own loads. This is experimental
and used only by tests. It did not speed up GEMM over loading in every CTA.

```python
self.ld = Ring(2, Tma(stage_bytes, fills="pair"), Threads)  # load on cta="leader"


@role(warps=0, cta="leader")
@cute.jit
def load(self, ld: Producer, ctx):
    rank = self.cluster.rank()
    own, peer = self.cluster.fill(rank), self.cluster.fill(rank + 1)
    for i in cutlass.range(ctx.n, unroll=1):
        with ld.produce() as s:
            cute.copy(
                ctx.tma,
                ctx.g_own[None, i],
                ctx.s[None, s.index],
                tma_bar_ptr=s.barrier,
                mcast_mask=own.mask,
            )
            cute.copy(
                ctx.tma,
                ctx.g_peer[None, i],
                ctx.s[None, s.index],
                tma_bar_ptr=s.barrier,
                mcast_mask=peer.mask,
            )
    return ld
```

| Declaration | Required producer placement | Copies it issues |
|---|---|---|
| `Tma(bytes, fills="pair")` | `leader`, in every pair | Both CTAs' copies |
| `Tma(bytes, fills="cluster")` | `rank0` | Every CTA's copies |

`bytes` is still per destination CTA per stage. `Cluster.fill(rank)` targets that CTA alone,
at the issuer's stage offset. Use a multicast atom even for the issuer's own copy: the mask
selects its destination. `Cluster.fill(rank, "n")` additionally multicasts as that rank's
own `multicast("n")` would; fills and multicast compose.

Consumer placement determines completion:

- Consumers in every written CTA: each CTA's own FULL barrier, group ONE.
- Pair-leader consumers: both halves complete on their leader, group TWO.

The filler arms those barriers, using remote `mbarrier.arrive.expect_tx` where needed;
each expects one arrival. Consumers release the fillers whose copies they depend on.

For a 2x2 cluster with pair fills and N multicast:

```text
  ld  Ring(2)  load -> use
      signal    Tma(4096 B) -> Threads
      full      rank 0 1 x2, rank 1 1 x2, rank 2 1 x2, rank 3 1 x2, tx 4096 B, rank 0 arms 0+1, rank 2 arms 2+3
      empty     rank 0 128 x2, rank 2 128 x2
      scope     cross-CTA, fills pair, multicast n, remote_scope cta
      fences    release_fence smem_async
```

Rank 1 receives half a tile from each leader, so it releases both, but only rank 0 arms its
barrier. Each leader's EMPTY expects `4 * 32` consumer threads. The TMA check counts 8192
bytes per filling CTA per stage and reports 4096 per destination CTA. It cannot prove where
each byte lands: fewer distinct `Cluster.fill` masks than filled CTAs (e.g. two copies into
one CTA) leaves the stage `unverified`.

`LeaderLoadGemm` in
[test_warpdrive_fills.py](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive/test_warpdrive_fills.py) is a
complete pair-fill GEMM.

### A rank0 consumer on a multicast channel

A rank-0 observer can read rank 0's stages alongside other consumers. Every written CTA
still needs a waiter, directly or through its pair leader, to complete the wait/rearm cycle.
The `rank0` participant widens the unit to the whole cluster, so each consumer releases
every producer.

## Placement rules

<details>
<summary>Exact derivation: participants → targets/counts → TMA completion, with PTX basis</summary>

### 1. Find the participants and written storage

Let `C(role)` be the ranks where a role runs:

| Placement | Ranks |
|---|---|
| `all` | Every CTA |
| `leader` | Even ranks with MMA pairs; every rank without pairs |
| `rank0` | Rank 0 only |

For producer rank `p`:

- `reach(p)` is `p` plus `Cluster.line(p, mode)` for each multicast mode. With both modes,
  this is a **union of lines**, not a Cartesian product.
- `filled(p)` is `p` alone, its pair for `fills="pair"`, or the cluster for `fills="cluster"`.
- `W(p)` is the union of `reach(d)` for every `d` in `filled(p)`: the CTAs whose SMEM those
  copies write. Without fills, `W(p) = reach(p)`.

Same `C` for every participant, with no multicast or fills, means CTA-local: ordinary
per-slot counts, unscaled bytes, self-targets and no producer tail. Masks are 1, except a
UMMA commit to its own CTA uses `1 << rank`.

### 2. Find the synchronization unit and counts

For a cross-CTA channel, rank `r`'s unit is:

1. `r` alone if all participants have the same `C` (only copies cross CTAs).
2. `r`'s MMA pair if each participant runs somewhere in every pair.
3. The whole cluster otherwise, when a participant is restricted to rank 0.

A producer targets the consumer CTAs in its unit with FULL signals; a consumer targets the
producer CTAs in its unit with EMPTY signals. **User TMA channels use the next subsection
instead.** In a `CtaPair`, mixed placement therefore targets the consumer/producer union.

For each target and slot, count the signaling instances and CTAs:

| Mechanism | Arrival contribution |
|---|---|
| `Threads` | `32 * role.num_warps` per signaling instance and CTA |
| `Warps` | `role.num_warps` per signaling instance and CTA; all lanes participate, one elected arrival per warp |
| TMA / UMMA | One per signaling CTA |

Nontarget ranks have placeholder count 1 and no promised waiter. A mask has one bit per
target: one integer if all signalers agree, otherwise a per-rank mapping.

With pairs, UMMA signalers must be pair leaders using group TWO. The UMMA side must include
the leader of **every pair** used by the other side: each CTA's operands and accumulator
belong to its own leader's MMAs. Without pairs, a UMMA producer must include every consumer
CTA; a UMMA consumer may release storage in other CTAs.

### 3. Route TMA completion and reuse

Let `W` be all written CTAs (the union of `W(p)`), and `F` the consumer CTAs.

| Written CTAs and consumers | Where bytes complete | Copy group |
|---|---|---|
| `W` is a subset of `F` | Each destination's own FULL barrier | ONE |
| Otherwise, pairs exist and `F` contains only pair leaders | Each destination's pair leader, which must be in `F` | TWO |
| Any other shape, or a consumer with no completing copy | Rejected | — |

Exactly one CTA arms each FULL barrier: its own CTA, which must run a producer, or the CTA
that fills it. It arms `k * Tma.tx_bytes`, where `k` is the number of CTAs completing there
(2 for group TWO, otherwise 1). Each FULL barrier expects **one arrival**.

Consumer `c` releases every producer `p` whose `W(p)` intersects either:

- the destinations completing on `c`'s barrier, or
- `c`'s synchronization unit (this preserves releases for placements accepted before fills).

Examples and limits:

- GEMM's all-CTA producers feeding pair leaders: each leader arms `2 * Tma.tx_bytes`.
- Same-placement TMA: each CTA arms its own barrier with unscaled bytes.
- All/all requires group ONE. With pairs, leader/leader **without fills** accepts either
  group in the channel check; descriptor legality remains yours.
- `Tma.tx_bytes` always means bytes landing in **each CTA's SMEM per stage**. A copy delivers
  its box to every CTA in its mask. The declaration derives releases; it does not issue copies.
- A `rank0` consumer may join multicast only if every written CTA still has a waiting
  consumer, directly or through its pair leader.
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
  tx-count spans `±(2^20 - 1)` (§9.7.15.16.3), so copies may complete before the arm. A phase
  completes only when pending arrivals and tx-count are both zero (§9.7.15.16.8).
- A wait must observe phase completion before the next phase's arrive (§9.7.15.16.5.1).
  The filler rearms only after EMPTY reports the consumers' release, which follows their
  FULL wait. This is why a filled stage cannot have no waiter.

At exit, consumers have waited on every stage filled for them: copies and remote arms have
landed before the exit cluster barrier. Thread releases precede those threads' arrival at
that barrier. Remote UMMA releases need the producer tails described next.

</details>

## Limits

- Copies may land only in CTAs with a consumer, or whose pair leader has one.
- Multicast needs a producer in each destination to arm there, unless `fills` names its filler.
- `fills="pair"` needs MMA pairs and `leader` producers; `fills="cluster"` needs `rank0`.
- `external=` channels are CTA-local: no multicast or fills, and one placement on the managed
  side. Raw code drives the external side independently in each participating CTA.
- Clusters have at most 8 CTAs, the portable limit. The alias diagnostic does not model them.
