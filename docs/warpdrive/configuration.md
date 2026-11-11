# Configuration: what is tunable and what is fixed

[Back to warpdrive](index.md)

Start with an example's defaults. Pass the choices you want to change to its constructor:

```python
small = Gemm(tile_n=128, ab_stages=2)
wide = Gemm(tile_n=256, ab_stages=3)
```

Constructor choices identify the [compiled variant](roles.md#compile-keys). Keep runtime
shapes out of them. A legal configuration is not necessarily faster; changing a fixed
algorithm choice may need a different role graph.

## The examples

These five examples expose different choices. Parenthesized numbers are defaults,
not recommendations for every workload.

### gemm.py

Vary tile width and buffering within one layout family.

| Constructor choice | Range / purpose |
|---|---|
| `tile_n` (256) | Positive multiple of 32, at most 256; output width |
| `ab_stages` (3), `acc_stages` (1) | Input and accumulator ring depths |
| `roles=` | Overrides epilogue warps 0–3, load 4, MMA 5; budgets 160 / 48 / 48 |

Fixed choices:

- `TILE_M=128`: epilogue maps one `tcgen05.ld 32x32b` lane to each output row.
- `TILE_K=64`: 128-byte K-major BF16 rows use SW128. Smaller steps are possible, as in
  decode's `tile_k`, but GEMM keeps one layout.
- `EPI_COLS=32`: the epilogue uses the TMEM load's x32 repetition.

### persistent_gemm.py

Persistence adds work scheduling and an SMEM-staged TMA-store epilogue.

| Constructor choice | Range / purpose |
|---|---|
| `tile_n` (224 for a pair, 256 otherwise) | Positive width ≤256, divisible by `epi_tile_n`; per-MMA-CTA width must be divisible by 8 |
| `ab_stages` (fills ~192 KB), `acc_stages` (2) | A/B buffering and overlapping accumulator lifetimes |
| `c_stages` (2) | 1–8 epilogue staging buffers |
| `epi_tile_n` (32) | 16, 32 or 64 columns, dividing `tile_n`: 32–128-byte BF16 rows for the TMA-store box |
| `early_release` (True) | Release the accumulator after its last subtile reaches registers, before SMEM staging/TMA store. False releases after staging and issuing that store |
| `schedule`, `mapping`, `cluster` | Work assignment and CTA grouping |
| `roles=` | Overrides epilogue 0–3, load 4, MMA 5; budgets 160 / 40 / 40 |

Fixed choices:

- `CTA_TILE_M=128`, `TILE_K=64`: as in GEMM; CUTLASS's epilogue partition helpers assume
  128 rows per CTA.
- `epi_sync=NamedBarrier(threads=128)`: intra-epilogue synchronization. It receives ID 3
  and syncs aligned.
- MMA K=16: BF16 `tcgen05.mma` instruction shape.

### fa4_fwd.py

Two softmax warpgroups each own a Q tile. Buffer and math options preserve that role graph.

| Constructor choice | Range / purpose |
|---|---|
| `tile_n` (128) | 64 or 128 keys per K/V tile; TMEM needs `Q_STAGE * (tile_n + head_dim)` columns, so 256 does not fit |
| `head_dim` (128, from tensors) | 64 or 128 |
| `kv_stages` (3 per CTA) | K/V ring depth, default `3 * cluster_size` |
| `split_p`, `turnstile`, `overlap_o_q`, `use_ldred_rowmax` | Early P release, ordered softmax turns, storage overlap and row-max instruction choice |
| `rescale_threshold` (8.0) | Nonnegative FA4 BF16 lazy-rescale threshold in log2 units; 0 rescales on every row-max growth |
| `schedule`, `mapping`, `cluster` | Work assignment; this example supports one CTA or a pair |
| `roles=` | Overrides the placement/budgets below |

Default roles: softmax 0–3 and 4–7 at 192; correction 8–11 at 80; MMA 12, epilogue 13 and
load 14 at 48.

`overlap_o_q` requires a nonpersistent schedule: the next item's Q would overwrite the
previous item's O staging bytes. `use_ldred_rowmax` requires an SM103-family target.

Fixed choices:

- `M_BLOCK=128`: one softmax thread per query row.
- `Q_STAGE=2`: one Q tile, S region and O region per softmax warpgroup. A different count
  needs a different role graph, not just a depth change.
- `SPLIT_P=3` out of 4: release after `floor(P_store_count * 3 / 4)` stores. Both sides use
  that count: at `tile_n=64`, it is 1 of 2 stores. Exposing it would require a
  `split_p_stores` argument.
- 16-column TMEM loads in rescale and O-store loops: fixed to FA4's register footprint.

### fa4_decode_swapab.py

The decode exposes tile sizes and ring depths, but keeps FA4's two-softmax-group issue order.

| Constructor choice | Range / purpose |
|---|---|
| `tile_hq` (from tensors) | Power of two up to 32 query heads per KV head |
| `tile_n` (256) | Power-of-two multiple of 128, with `tile_hq * tile_n / 128 <= 128` for one S-load repetition |
| `tile_k` (64) | 16, 32 or 64: 32–128-byte K-major rows. K and Vᵀ tiles have equal bytes |
| `s_stages`, `p_stages` (4, 4) | Even, at least 2; half the stages per softmax warpgroup |
| `head_dim` (128, from tensors) | Positive multiple of `tile_k`; sizes Q and QK K steps |
| `head_dim_v` (128, from tensors) | Positive multiple of 128; adds O tiles subject to `S + L + O <= 512` TMEM columns. 256 fits every `tile_hq` with the default S ring |
| `kv_stages` | FA4's total capacity split about 7:5 between K and V; with `shared_kv`, one ring uses their sum |
| `roles=` | Computed defaults: 64 / 120 / 208 for 32-head tiles, no adjustment otherwise |

`fa4_decode` / `decode_partials` default to split K/V rings. `shared_kv=True` selects
`SharedKVDecode`: one ring, two exclusive MMA consumers, and FA4's QKIssued/PVIssued
named-barrier rendezvous ([shared K/V ring](examples.md#shared-kv-ring)). Results vary on
ragged shapes; benchmark yours before switching.

Fixed choices:

- `O_STAGES=2`: one O slot per softmax warpgroup (even/odd blocks), merged in correction's tail.
- `TILE_M=128`: one lane per KV-token row of an S subtile and per head dimension of an O tile.
- `MMA_N_MIN=8`: smallest hardware `tcgen05.mma` N.
- `lookahead=2`: K issues two blocks ahead of V. The K ring bounds the lead; the shared
  ring's skips/rendezvous are written for this order.

Increasing lookahead with a shallow P ring can deadlock. The dependency cycle is:

```text
load waits for free K -> QK waits for free P -> PV waits for V
        ^                                      |
        +---------- load has not issued V -----+
```

Recheck dependencies when changing issue order; the [liveness report](guarantees.md#liveness)
only covers its stated inputs and assumptions.

### fa4_bwd.py

This port fixes FA4's single-CTA, head-dimension-128 tiling (`FlashAttentionBackward`; the
2-CTA `FlashAttentionBackward2Cta` uses FA4's pair split of 136 / 104 registers).

| Constructor choice | Range / purpose |
|---|---|
| `causal` | Also selects FA4's causal register split |
| `dq_stages` (2) | 1 or 2 dQ buffers for TMA reduce-add; more does not fit SMEM |
| `roles=` | Overrides reduce 0–3, compute 4–11, MMA 12, load 13. Budgets: 152 / 136 / 88, or 136 / 144 / 88 for causal |

Fixed choices:

- `TILE_M=TILE_N=HEAD_DIM=128`: the 512-column TMEM map (S, dV, dP, dK) is full.
- `Q_STAGE=2`: `S[t+1]` reads `Q[t+1]` before `dK[t]` releases `Q[t]`. One stage deadlocks;
  three do not fit SMEM.
- dO/K/V stages `(1,1,1)`, LSE/dPsum `(2,1)`: FA4's counts. With two dQ buffers, usage is
  231 of 232 KB SMEM.
- `DQ_COLS=32`, `EPI_COLS=64`: FA4's dQ reduce width (postprocess depends on tile order)
  and the 128-byte TMA-store box.
- `OneShot()`, `Raster()`: retiring the last S acquire and staging epilogue tiles over Q/dO
  require one item per CTA.

A non-fitting configuration fails at the relevant boundary: construction for the TMEM plan
or default K/V stage budget (`ValueError`), first collection for role checks
(`ProtocolError` / `ValueError`), or compilation for excessive explicit SMEM staging.
SMEM is not fully budgeted at construction; explicit `kv_stages`, `ab_stages` or `c_stages`
can still exceed capacity at compile time.

## The layer

The layer derives synchronization and launch resources from your roles, channels, storage,
work stream and cluster. It does not autotune them.

Static algorithm choices belong in constructor arguments. A fixed module constant can
express an instruction or algorithm constraint: editing tracked source invalidates the
cache, but mutating a Python global does not select a specialization. Create a new kernel
instance instead of changing its declarations. The [compile-key contract](roles.md#compile-keys)
lists the few settings that may change after construction and when they become frozen.

### Placement and registers

| Choice | Where / constraint |
|---|---|
| Warps and budgets | `@role(warps=, regs=)`; constructor overrides in `self.roles` ([roles](roles.md#choose-placement-in-the-constructor)) |
| Numeric budget | Multiple of 8 in `[24, 256]`; one budget per warpgroup; all budgets must fit the launch's starting register pool |
| Idle warps | Any numeric budget pads to whole warpgroups. Holes take their group's budget; a wholly idle group takes the hardware floor, 24 |
| Scheduler warp | Lowest unclaimed warp by default; choose explicitly with `WorkStream(scheduler_warp=...)`. It uses its group's budget |

Use [`Local`](roles.md#index-the-role-not-the-hardware) indices so `roles=` can move a role
without changing its math. Raw `thread_idx()`/`warp_idx()` depend on placement.
Some role sizes are fixed and checked at construction or trace time: GEMM epilogues,
FA4 and decode softmax/correction, and backward reduce need 4 warps; backward compute
needs 8; Qwen megakernel compute needs one instance of 8.

Register dispatch uses one warpgroup-uniform branch per run of adjacent warpgroups with
the same budget, then branches by role. `setmaxnreg` is `.sync.aligned` over a warpgroup;
adjusting before the role branches lets `ptxas` allocate each role under its budget.
Non-adjacent groups get separate branches even if their budgets match: merging them lets
one role's pressure cause spills in another. See
[register accounting](roles.md#registers-belong-to-warpgroups).

### Channels and storage

| Choice | Where / constraint |
|---|---|
| Mechanisms and ordering | `Tma(bytes, multicast=)`, `Umma`, `Threads`, `Warps`, `commit_fence=`, `release_fence=`: declared per channel |
| Round-robin depth | An mbarrier round-robin ring needs a stage count divisible by its instance count, so each instance sees the previous epoch of stages it waits on |
| Exclusive consumers | `consumer_sharing=Sharing.EXCLUSIVE`: CTA-local mbarrier `Ring`, one producer, equal consumer contributions. The author supplies any rendezvous needed after more than a ring lap of skips ([exclusive consumers](channels.md#exclusive-consumers)) |
| TMEM allocation | Plan columns round up to a power of two, minimum 32: hardware `tcgen05.alloc` granularity |

Named barrier IDs are allocation details, not tuning knobs:

```text
0          CTA bar.sync 0
1, 2       TMEM base exchange, TMEM teardown
3 upward   declared NamedBarrier objects
15 downward  named-channel FULL/EMPTY barriers
```

Hardware provides 16 IDs. Declare `NamedBarrier` rather than picking one. Tracing rejects
`alias.named_barrier` calls that use allocated IDs. The choice of free ID does not affect speed.

### Fixed lifecycle and waits

Setup and teardown run in this order:

1. Warp 0 initializes mbarriers.
2. For a single CTA, the lowest nonzero warp belonging to a TMEM role allocates TMEM before
   CTA sync (warp 0 if no other candidate). For a cluster, allocation follows cluster sync,
   from the first warp of the first eligible TMEM role.
3. `SmemPlan` buffers are allocated last in `begin_roles`; `run_roles` performs the common
   fences/sync, register adjustments and dispatch. Cluster TMEM allocation occurs there
   after its sync.
4. TMEM users arrive on barrier 2 after their roles. The allocator waits and frees TMEM;
   a cluster then arrives and waits before exit.

Setup order is an implementation choice. Teardown has hardware requirements: all TMEM
readers finish before `tcgen05.dealloc`, and remote arrivals land before a CTA exits.

Mbarrier waits use retrying `mbarrier.try_wait.parity` with CuTeDSL's 10 ms suspend hint.
Named channels use unaligned `bar.sync` / `bar.arrive`. A role can choose a per-wait initial
probe with [try tokens](channels.md#try-tokens).
