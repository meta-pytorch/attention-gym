# Examples

[Back to warpdrive](index.md)

Start with GEMM for the basic handoff; use the other examples for the feature you need.
These are teaching kernels with the input limits below, not drop-in replacements for every
PyTorch or FA4 feature.

| Start here to… | Source | Guide |
|---|---|---|
| Write a basic GEMM | [gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py) | [GEMM](#gemm) |
| Add persistent scheduling and CTA clusters | [persistent_gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py) | [GEMM](#gemm), [tile boundaries](#tile-boundaries) |
| Handle grouped GEMM and ragged groups | [grouped_gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/grouped_gemm.py) | [Grouped GEMM](#grouped-gemm) |
| Build attention with multiple consumers | [fa4_fwd.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_fwd.py) | [Forward](#attention-forward) |
| Study attention backward and overlapping lifetimes | [fa4_bwd.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_bwd.py) | [Backward](#attention-backward) |
| Consume a device-built work list | [packed_work.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/packed_work.py) | [Scheduling](scheduling.md) |
| Decode one token with split KV | [fa4_decode_swapab.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py) | [Decode](#single-token-decode), [shared KV](#shared-kv-ring) |
| Run a whole Qwen3 decode step in one launch | [qwen_megakernel.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/qwen_megakernel.py) | [Qwen3 megakernel](#qwen3-decode-megakernel) |
| Mix managed and raw synchronization | [raw_interop.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/raw_interop.py) | [Dropping down](dropping-down.md) |
| Reuse CUTLASS or FA4 tile orders | [external_maps.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/external_maps.py) | [External tile orders](scheduling.md#external-tile-orders-subclass-workmap) |
| Visualize which CTA runs each item | [schedule_views.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/schedule_views.py) | [Tracing](tracing.md#see-which-cta-runs-what) |

The larger examples have a `bench_*.py` script beside them. Run benchmarks under a GPU
reservation with `--output` to save versions, source hashes and every replay sample; compare
only columns of one run.

## GEMM

```python
import torch
from attn_gym.warpdrive.examples.gemm import gemm

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
d = gemm(a, b)  # A @ B.T
```

To use a persistent two-CTA kernel:

```python
from attn_gym.warpdrive import ClusterLaunchControl, CtaPair
from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

d = persistent_gemm(a, b, schedule=ClusterLaunchControl(), cluster=CtaPair())
```

```text
load A/B -- ab ring --> MMA -- acc ring --> epilogue writes A @ B.T
```

Both wrappers require:

- BF16 `a: (M, K)`, `b: (N, K)` on one CUDA device; contiguous and 16-byte aligned.
- K and N multiples of 8.
- Optional `out`: BF16 `(M, N)`, same device, contiguous, 16-byte aligned, disjoint from
  both inputs. Empty outputs return early; K=0 writes zeros.

`PersistentGemm` defaults to two accumulator stages: the next mainloop can use one while
the epilogue drains the other. All stage choices must fit SMEM and TMEM.

<details markdown="1">
<summary>Tile choices, benchmarks and buffer reuse</summary>

For two pairs sharing A along N, pass `cluster=Cluster(2, 2, pair=True)`; see
[clusters](clusters.md#larger-clusters).

`tile_n` defaults to 224 for `CtaPair()` and 256 otherwise. Tile width trades work per tile
against how evenly tiles fill resident CTAs. There is no shape-based selector; benchmark the
shapes you run.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_persistent_gemm \
  --variants mm single pair pair_n256 pair_clc --output agent_space/persistent_gemm.json
```

Variants are `mm` (`torch.mm`), `single`/`pair` with optional `_late` (`early_release=False`),
`_clc`/`_atomic` schedules, `_acc1` (`acc_stages=1`) and `_n<tile_n>`.

### Tile boundaries

Two accumulator buffers let MMA and epilogue overlap. `Umma` publishes asynchronously;
the epilogue can return TMEM as soon as its final read completes:

```text
early_release=True:   last tcgen05.ld -> wait::ld -> release TMEM -> SMEM/TMA store
 early_release=False: last tcgen05.ld -> wait::ld -> SMEM/TMA store -> release TMEM
```

`early_release=True` is the default. It changes **when TMEM is reusable**, not when the
output store finishes. It matters where the epilogue is long relative to the mainloop
(short-K pairs); the second accumulator stage matters more, most where the epilogue is as
long as the mainloop.

Work-ring scope is a tile-boundary cost too: cluster-scope thread arrivals emit a
`MEMBAR.ALL.GPU` per receiving thread. Work payloads become ready from completed bytes and
their releases publish nothing, so the layer derives `remote_scope="cta"` for them
([scheduling](scheduling.md)); avoid forcing cluster scope on a ring whose release publishes
no data.

</details>

## Grouped GEMM

```python
from attn_gym.warpdrive.examples.grouped_gemm import grouped_gemm

a = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)  # the experts' rows, concatenated
b = torch.randn(8, 2048, 7168, device="cuda", dtype=torch.bfloat16)  # one (N, K) weight per expert
offsets = torch.tensor([700, 700, 1500, 2100, 2100, 3000, 4000, 4096], device="cuda", dtype=torch.int32)
out = grouped_gemm(a, b, offsets)  # out[s_g:e_g] = a[s_g:e_g] @ b[g].T
```

`offsets` contains **end rows**, not sizes (`torch._grouped_mm` calls it `offs`):

```text
offsets = [700, 700, 1500, ...]
group 0: rows [0, 700)      group 1: empty      group 2: rows [700, 1500)
```

**Inputs and limits:**

- `a: (total_M, K)`, `b: (G, N, K)`, BF16; GEMM's device, contiguity, alignment and
  K/N-multiple-of-8 limits apply.
- `offsets: (G,)`, contiguous int32 on the same device; nonnegative, nondecreasing,
  at most `total_M`, and unchanged during execution. Values are read only on-device.
- `total_M + (G + 1) * 256` must fit int32. `out` must not overlap `a`, `b` or `offsets`.
- Empty/arbitrary-size groups are allowed; rows past `offsets[-1]` stay untouched. Invalid
  offset values are clamped to the previous end and A's end: wrong rows, but in bounds.

One compiled kernel handles dynamic `G`, `total_M`, `N`, `K` and routing. Options are
`schedule` (default `StaticPersistent()`), `cluster` (`SingleCta()` or `CtaPair()`),
`max_ctas`, `tile_n`, `ab_stages`, `acc_stages`, `c_stages`, `epi_tile_n` and `roles`.

<details markdown="1">
<summary>How grouped work and ragged stores are handled</summary>

**Finding work.** The host reserves an upper bound of
`ceil(total_M / tile_m) + G` m-tiles, since each ragged group adds at most one. `GroupedTiles`
numbers real tiles back to back, dropping only indices after the last tile. Its
[decode state](scheduling.md#decode-state) carries the prefix-sum walk between items,
testing eight groups per round trip instead of restarting. Consecutive real-tile indices keep
dropped entries from unbalancing static units' strided sequences.

**Stopping stores at a group boundary.** A/out each use one descriptor plus
`cute.domain_offset`; B is `(N, K, G)`, with group as a TMA coordinate. The output's ragged
view has shape `(2^30, N, 2^31 - 1)`: its row mode ends at TMA's clipping boundary, and its
third coordinate selects the group's end row. A partial group tile is clipped like a tensor edge.

</details>

**Choosing a cluster.** Small skewed groups waste rows in 128-row tiles; a pair's 256-row
tiles waste about twice as many, so `CtaPair()` pays only with few groups and large N.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_grouped_gemm \
  --output agent_space/grouped_gemm.json
```

The bench uses skewed routing with 1/8 of the experts idle and compares `torch._grouped_mm`,
a `torch.mm` loop, Inductor's Triton template and FlexGEMM's QUACK backend (`--help` lists
variants).

## Attention forward

```python
from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

out, lse = fa4_fwd(q, k, v)
```

This teaching kernel computes **unmasked attention** using `flash_attn.cute` and `quack`
helpers. It does not implement the full FA4 API.

| Input/output | Contract |
|---|---|
| Q/K/V | BF16 `(batch, seqlen, heads, head_dim)`, contiguous, 16-byte aligned, one CUDA device |
| Dimensions | Head dimension 64 or 128; matching batch/head counts; K and V shapes match |
| Sequence lengths | Positive key length divisible by `tile_n` (128 or 64); ragged Q allowed |
| O / LSE | O has Q's shape; LSE is fp32 `(batch, heads, seqlen_q)`, natural-log units |
| Not supported | Varlen or GQA |

The tiling needs `2 * (tile_n + head_dim) <= 512` TMEM columns. Wrapper checks are not a
complete validator of rank, head dimension or device requirements; enforce the full contract
in another wrapper.

Use `schedule=` to choose a schedule and `mapping=` to choose the tile order: `Raster` or a
custom `WorkMap`, including [external tile orders](scheduling.md#external-tile-orders-subclass-workmap).
Override warp placement and register budgets with
[`roles={...}`](roles.md#choose-placement-in-the-constructor).

`tile_n=64` halves the K/V tile; `rescale_threshold` controls FA4's lazy rescale. Head dimension
comes from the tensors. See [configuration](configuration.md) for all options and fixed choices.

Compare `StaticPersistent()` (FA4's default) with `ClusterLaunchControl()` on your workload;
they can launch different numbers of CTAs.

K/V share channel-bound SMEM. The constructor-only `overlap_o_q=True` also reuses Q's bytes
for epilogue SMEM; `fa4_fwd` does not expose it. This requires a non-persistent schedule
(`OneShot()` among built-ins). Persistence would need an epilogue-to-load ordering edge:
**sharing an address does not create a handoff**.

### Hardware row max on SM103

```python
from attn_gym.warpdrive import ClusterLaunchControl, CtaPair

out, lse = fa4_fwd(
    q,
    k,
    v,
    cluster=CtaPair(),
    schedule=ClusterLaunchControl(),
    split_p=True,
    use_ldred_rowmax=True,
)
```

`use_ldred_rowmax=True` combines score loading and maximum reduction with `tcgen05.ld.red`:

```text
128 scores -> four 32-column hardware maxima -> existing software row-max updater
```

- SM103 only; an SM100 target raises `ValueError`. Default is false, retaining the software path.
- Rescale threshold, exponentials, BF16 P conversion and row sums are unchanged, including
  fast-exp2's flush-to-zero rescale behavior.
- Tests compare O/LSE bitwise: ragged Q, nonuniform scales, repeated launches, persistent
  wraps, and subnormal rescale amplified by large finite V. SASS and wrapper kernel-name
  checks verify that the hardware path runs.
- `LDTM.STAT.x32.MAX.F32` replaces the software path's maximum reduction without reducing
  the score bytes loaded.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 60s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_fwd \
  --shape 0 --output agent_space/fa4_fwd.json
```

Shapes 0-2 are `(batch, heads, length)` `(4,16,4096)`, `(2,16,8192)` and `(8,16,1024)`, head
dimension 128, non-causal; variants are CLC split-P with software and hardware max, static
with hardware max, and FA4's default. `--profile` warms every variant and profiles one forward
each between CUDA profiler start/stop (`ncu --profile-from-start off`).

## Attention backward

```python
from attn_gym.warpdrive import CtaPair
from attn_gym.warpdrive.examples.fa4_bwd import fa4_bwd

dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, causal=False)  # FA4's 1-CTA kernel
dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, causal=False, cluster=CtaPair())  # FA4's default
```

This example ports both head-dimension-128 kernels from FA4's `flash_bwd_sm100.py`. The
1-CTA path is `FlashAttentionBackward` (FA4's `disable_2cta` option); `cluster=CtaPair()`
selects `FlashAttentionBackward2Cta`, FA4's default.

```text
FA4 preprocess -> warpdrive main kernel -> FA4 dQ postprocess
  dPsum = rowsum(dO * O)
  LSE_log2 = LSE * log2(e)
  zero fp32 dQ accumulator
```

**Inputs and limits:**

- BF16 `(batch, seqlen, heads, 128)` inputs, contiguous, 16-byte aligned, one CUDA device;
  matching batch/head counts, no GQA. Forward LSE: fp32 `(batch, heads, seqlen_q)`.
- Arbitrary sequence lengths: TMA zero-fills partial loads, keys past `seqlen_k` are masked,
  and dK/dV stores are clipped. `causal=True` is bottom-right aligned.
- No varlen, paging, block sparsity, score mods, softcap or deterministic mode.

<details markdown="1">
<summary>Role layout, cross-CTA exchange and design notes</summary>

**1-CTA.** Each `OneShot()` item is a 128-key tile. `Raster()` over `(n_blocks, heads, batch)`
matches FA4's `SingleTileScheduler` order. Four roles use 14 warps: reduce 0–3, two compute
warpgroups 4–11 sharing each tile, MMA 12, load 13. Eleven channels (seven with `Warps`
sides) pipeline five MMAs per tile through overlapping handles, three TMEM aliases and two
SMEM epilogue overlaps.

**2-CTA.** A pair owns a 256-key item and the leader issues `CtaGroup.TWO` MMAs; a fifth role,
the relay warp 14, joins them. S^T, dP^T, dV and dK keep their rows in their CTA, so the compute
warps and epilogue are the 1-CTA code (shared helpers). The other operands split differently
across the pair: Qt, dOt and Kt are separate TMA loads, and dQ = dS K needs each CTA's dS for
the other's query rows:

```text
compute (CTA r): dS^T for its 128 keys x 128 queries
   own query half  -> own sdS, key block r
   peer's half     -> sdSx --cp.async.bulk.shared::cluster--> peer's sdS, key block r
                                     completes on peer's xchg barrier (armed by thread 0)
relay (CTA r):   wait xchg ----------------------------------------\
compute (both):  sdS written and S_{t+1} read --------------------- ds full (leader, 18 arrivals)
                                                                    -> leader issues dQ_t
```

Thirteen channels (those with an MMA side pair-scoped), one raw barrier, and TMEM S / dV / dP /
dK with P over S, dS over dP and this CTA's 64 dQ columns over S's tail. Both
[class docstrings](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_bwd.py) map channels, TMEM and MMA order.

Design points worth reusing:

- **Read the next epoch before owning it.** S_{t+1} needs Q_{t+1} while dK_t still holds Q_t:
  the port uses `q.peek()` and `preview.view("Q", read=True, by="umma")`, and the next
  `q.wait()` owns that epoch ([previews](channels.md#previews)). The 2-CTA kernel previews dO
  the same way.
- **Defer a commit past the loop's back edge** with `commit(signal=False)` and a raw arrival
  after S_{t+1}'s load (FA4's "dS_t is published once S_{t+1} is read"). The handle still closes
  in its opening region; only the raw signal is deferred.
- **Arrive once per warp.** `Warps` sides synchronize readers before electing an arrival.
- **Two Q stages are required:** with one, S_{t+1} waits for a load that needs Q_t's release.
- **Write the peer's dS straight to SMEM** once the ds acquire guards it, and acquire ds only
  after both dS stages are in TMEM.
- dQ reduce and dK/dV epilogues order SMEM staging with declared `NamedBarrier`s and
  `bulk_commit_group`/`bulk_wait_group`, as in the persistent GEMM epilogue.
- The pair's dS exchange is raw: compute thread 0 arms the peer's `xchg` mbarrier
  (`mbarrier.arrive.expect_tx` on the peer) and issues `cp.async.bulk.shared::cluster`; the relay
  waits with a hand-kept phase. The ds channel orders sdS/sdSx and barrier reuse.
- See [alias checks](guarantees.md#aliases) when reading reports for peeled iterations and
  pair storage.

</details>

### Correctness

dK and dV are bit for bit FA4's results from the same kernel configuration (1-CTA against
`disable_2cta`, the pair against FA4's default), non-causal and causal, including ragged
lengths. dQ differs only by the order in which CTAs' TMA reduce-adds land in the fp32
accumulator (FA4 is not bitwise reproducible there either). All three match an fp32 autograd
reference within FA4's own error.
dK/dV are deterministic across launches; dQ is not, as in FA4.

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=10s 1500s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_bwd \
  --output agent_space/bwd.json  # add --causal for the causal rows
```

The bench graph-replays one whole backward call (preprocess, main kernel, postprocess) per
variant and times the main kernel with CUPTI.

## Single-token decode

```python
from attn_gym.warpdrive.examples.fa4_decode_swapab import fa4_decode

out, lse = fa4_decode(q, k, v, num_splits=8, seqused_k=lengths, splits_per_seq=counts)
```

This ports FA4's `flash_fwd_decode_sm100.py`. Swapping the MMA operands puts 128 KV tokens
along M and one KV head's query heads along N:

```text
S^T = K Q^T -> softmax -> P^T -> O^T = V^T P^T
```

| Input | Contract |
|---|---|
| Q | BF16, contiguous `(batch, 1, heads, d)` |
| K / V | BF16, contiguous `(batch, kv_len, kv_heads, d or d_v)` |
| Heads | Query heads divisible by KV heads, at most 32 per KV head |
| Dimensions | `d` divisible by K step (default 64); `d_v` divisible by 128. Defaults 128/128; 256 works for either |
| `seqused_k` | int32 per-sequence lengths; rows past the length must be finite—FA4's bounded V descriptor is not ported |
| `splits_per_seq` | int32 per-sequence counts, at most `num_splits`; caps splits as FA4's prepare kernel does |
| Not supported | Paging, varlen Q, masks, score mods |

`decode_partials` returns fp32 O `(splits, batch, 1, heads, d_v)` and LSE
`(splits, batch, heads, 1)` in FA4's combine layout. Splits past a sequence's count are
unwritten; kept splits with no KV blocks write O=0, LSE=-inf. `fa4_decode` uses FA4's combine
kernel and returns BF16 O `(batch, 1, heads, d_v)` and fp32 LSE `(batch, heads, 1)`.

<details markdown="1">
<summary>Decode roles and handoffs</summary>

The roles keep FA4's placement: QK and PV MMA warps, K/V-load plus split-LSE warp,
Q-load/O-store warp, two alternating softmax warpgroups, and correction warpgroup.

- `SplitKV` supplies `(split, kv_head, batch, splits)` items; the fourth word is the sequence's
  split count. Per-sequence counts are read after `PdlWait()`; indices past the count are
  dropped.
- `Sharing.ROUND_ROBIN` on S, P and row-max/row-sum mailboxes gives softmax[i] blocks
  i, i+2, … with one warpgroup's counts. **OneShot only:** after an odd block count, another
  item would start on softmax[1]'s epochs.
- Thread mailboxes use NamedBarrier-backed channels and a turnstile for the shared row-max
  exchange. Three one-shot `NamedBarrier` flags finish split statistics; one-shot flags
  instead of ring channels free named-barrier IDs (12 to 9 of 13 here).
- P occupies a 4-stage SMEM ring as PV's B operand, with no TMEM alias.

</details>

`SwapABDecode` configures `tile_n` (default 256), `tile_k` (64), S/P depths (4/4), head
dimensions and role registers; wrappers forward these keywords. See
[configuration](configuration.md) for valid ranges.

Default K/V storage is **split rings**, 7/5 stages for 8-head tiles. `shared_kv=True` selects
[FA4's shared ring](#shared-kv-ring), with two MMA cursors and two-block K lookahead. Other
port differences: the first PV overwrites its O slot instead of zero-filling TMEM first;
empty splits use the normal tail with zero sums.

Tests match FA4 partials and combined outputs bitwise, using FA4's actual split counts and
the table captured at its combine call.

Tips for short decodes, where setup and tail dominate:

- **Precompute divisors on the host.** `SplitKV.bind()` builds `FastDivmodDivisor` values once,
  so roles avoid general integer division.
- **Order the tail by dependency.** Correction runs the row-sum/1/L stats round trip before
  waiting for the final O slots.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_decode \
  --output agent_space/decode.json  # add --shared-kv for the shared ring
```

The bench runs FA4's prepare/decode/combine against the port's decode/combine on seven dense
and ragged shapes, timing the decode kernel with CUPTI, and asserts bitwise partials.
For NCU source attribution add
`--profile --shapes 0` under
`CUTE_DSL_LINEINFO=1 ncu --profile-from-start off --cache-control none --import-source yes`,
filtering kernels to `attn_kernel|warpdrive_fa4_decode`.

### Decode racecheck

compute-sanitizer racecheck reports RAW hazards between correction's lane-0 stores to
`sRowSums` and `split_stats`'s vectorized read. They are **false positives**: all correction
threads arrive on the one-shot `sum_done` named barrier after the stores and the load warp
syncs before reading, which publishes ordinary SMEM writes
([one-shot done flag](channels.md#one-shot-done-flag)); FA4's own decode reports the same
handoff. `test/warpdrive/test_warpdrive_decode.py::test_decode_stats_handoff` isolates the
handoff for a sanitizer run. New access pairs need a fresh audit.

### Shared KV ring

```python
out, lse = fa4_decode(q, k, v, num_splits, seqused_k=lengths, splits_per_seq=counts, shared_kv=True)
```

`shared_kv=True` chooses `SharedKVDecode`: one ring of `sum(kv_stages)` stages containing
**either K or V^T** (their buffers overlap), not two readers joining on every stage.

```text
one TMA producer -> mixed K / V^T epochs
                   QK owns K epochs; skips V
                   PV owns V epochs; skips K
```

The declaration uses `consumer_sharing=Sharing.EXCLUSIVE` with QK/PV consumer roles.
FA4's QKIssued/PVIssued `NamedBarrier`s bound cursor skew; follow the
[rendezvous recipe](channels.md#rendezvous-recipe-fa4s-qkissuedpvissued).

Both MMA warps probe the next stage after releasing, using try tokens.
The loader probes after each commit within a block's K or V group; its token
cannot cross those dynamic regions, unlike FA4's loader token.

With split rings, V-stage reuse waits on the previous PV before the loader reaches the next K.
The shared ring lets V reuse K stages that QK already released. Compare with `--shared-kv`;
the shape determines whether that reuse helps.

## Qwen3 decode megakernel

```python
from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    load_qwen3_weights,
    make_decode_state,
)

weights = load_qwen3_weights()  # Qwen/Qwen3-0.6B from the HF hub, or a local checkpoint path
caches, token, pos = make_decode_state(weights, ctx_len=1024, max_len=1088)  # random prompt
runner = QwenMegakernel(weights, max_len=1088)  # one compile serves every max_len
logits = runner(weights, token, pos, *caches)  # decode_step_ref's contract: writes K/V at pos
```

One persistent launch runs a whole batch-one decode step. The host lowers the step into a
table of 20.3k instructions and assigns them to CTAs ahead of time:

```text
for layer in 28 layers:
    QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
    ATTN    8 x 16      kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
    OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
    GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
    DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials;
                        the last split also sums them into the residual
LMHEAD  594         8 x 32 rows of the tied embedding x final RMSNorm    -> logits
```

| Input | Contract |
|---|---|
| Model | Qwen3 0.6B, 1.7B, 8B, 14B and 32B, and Qwen3-MoE 30B-A3B (`load_qwen3_weights` reads the shape from `config.json`; `check_model` names what a shape breaks), bf16 weights, batch 1, one token per launch |
| Caches | Contiguous bf16 `(layers, kv_heads, max_len, 128)`, `max_len` a multiple of 64, `0 < pos < max_len`; only row `pos` is written |
| GPU | SM100 family; one CTA per SM, cooperative launch |
| Not supported | Batch > 1, prefill, paging, Qwen3-4B (hidden 2560 is not a multiple of 1024), MoE models across GPUs (235B-A22B) |

The table above shows 0.6B's counts. Larger models change only the counts: an attention row
serves two q heads of one kv head (one for 14B's odd group of 5), an o_proj split covers a
quarter of the q heads, the residual polls run 8 elements per thread at a time, the ring keeps
as many 16 KB stages as fit (11 for 32B), and the default KV split count targets about 128
attention rows per layer (16 for 0.6B and 1.7B, 8 above).

Median us per step on a GB200 (CUDA-graph replay, 2026-11-11; `bench_qwen_megakernel`;
"roofline" is the logical bytes per step at the measured 6.2 TB/s read bandwidth, so above
100% means L2 hits):

| model | ctx | `torch.compile` + graph | megakernel | speedup | roofline / megakernel |
|---|---:|---:|---:|---:|---:|
| 0.6B | 1024 / 8192 | 1327 / 1451 | 465 / 645 | 2.85x / 2.25x | 45% / 53% |
| 1.7B | 1024 / 8192 | 1693 / 1819 | 887 / 1061 | 1.91x / 1.71x | 64% / 66% |
| 8B | 1024 / 8192 | 3858 / 4011 | 2949 / 3368 | 1.31x / 1.19x | 90% / 84% |
| 14B | 1024 / 8192 | 5918 / 6106 | 4760 / 5634 | 1.24x / 1.08x | 101% / 88% |
| 32B | 1024 / 8192 | 12849 / 13101 | 10650 / 12035 | 1.21x / 1.09x | 99% / 91% |
| 30B-A3B (MoE) | 1024 / 8192 | 4650 / 4818 | 2202 / 2768 | 2.11x / 1.74x | 50% / 44% |

A MoE model adds a router phase after o_proj (128 logits from 4 tiles) and routes the MLP:
the gate/up and down GEMVs keep the dense layout over `experts_per_token x moe_intermediate_size`
activation columns, with slot s's weight rows taken from its routed expert. Each consumer computes
the top-k itself from the logits (k rounds of a warp max over packed (logit, index) keys; ties go
to the lower expert), the load warp for expert addresses and one compute warp for the routing
weights, which gate/up folds into its output. The torch.compile baseline gathers the top-k
experts' weights for batched matmuls. Its floor counts only the routed experts' bytes, and it
stays latency-bound like the small dense models: one more dependent phase per layer, routing on
the critical path, and expert weights that cannot stream before the router finishes.

**Tensor-core engine.** ``QwenMegakernel(..., engine="tcgen05")`` swaps the GEMV engine
(``TcGen05Gemv``, ``QwenDecodeMegakernelTc``): an MMA warp accumulates each 128-row weight tile
against the input with tcgen05 swap-AB MMAs (M = 128 weight rows, N = 16 token columns) into
TMEM, and the compute warps run the same epilogues. Each 16 KB stage is one 128 x 64 SW128
operand, so a stage is four MMAs; the input stays the contiguous vector the CUDA-core engine
reads, viewed through a B layout whose core matrices overlap, so batch one needs no zero
padding. At batch one it is not faster (GB200, ctx 1024, us): 0.6B 465 vs 637, 8B 2959 vs 3048,
32B 10629 vs 11294, 30B-A3B 2199 vs 2496. The 128-row items give the latency-bound phases fewer
CTAs (30B-A3B's router is one item). It is the base for batched decoding, where the CUDA-core
GEMV's work grows with every token.

**Batched decoding.** ``QwenMegakernel(..., engine="tcgen05", batch=B)`` decodes B sequences (dense
models, B <= 64) per launch through ``decode(tokens, positions, k_cache, v_cache)``, each sequence
with its own position and cache. The weights stream once for all B: the MMA's N columns are the
tokens, and every activation carries a token axis. Attention rows are per (sequence, kv head,
split), with the split count divided by B. Each warp builds whole tokens' GEMV inputs, and
o_proj's last split publishes the post-attention residual so gate/up rows read one word per
element. A sequence's logits do not depend on the rest of the batch (bit-identical in batches
of 8 and 3). The inputs stay in shared memory (B x hidden x 2 bytes), which bounds Qwen3-8B
at B = 16. GB200, all sequences at one context, tokens per second
(``bench_qwen_megakernel_batch``):

| model, ctx | CUDA cores, B=1 | B=1 | B=4 | B=16 | B=32 | B=64 |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B, 256 | 1993 | 1422 | 5312 | 12251 | 15010 | 13568 |
| 0.6B, 1024 | 2049 | 1600 | 4860 | 10146 | 11963 | 10991 |
| 8B, 256 | 323 | 332 | 1095 | 2721 | | |
| 8B, 1024 | 339 | 330 | 1041 | 2438 | | |

Beyond B = 32 the input buffer leaves 0.6B five ring stages, and at ctx 1024 the batch's KV
reads (about 270 MB per layer at B = 64) bound attention.

The megakernel removes launch and dependency latency, so its lead shrinks as weight bytes
dominate the step. Its one-step error against fp32 stays within the spread of eager bf16's,
which varies about 5x across prompts on the deep models.

<details markdown="1">
<summary>How the megakernel runs the model</summary>

- **Schedule.** `InstructionTable` is a `WorkMap` over the int32 table. `list_schedule`
  assigns rows to CTAs with a row-latency model and `assign_static` lays the table out
  `[slot][cta]` for `StaticPersistent`; `schedule="atomic"` claims rows in dependency order with
  `AtomicPersistent` instead. Each CTA's rows stay in dependency order, so with every CTA
  resident (the reason for `cooperative=True`) the earliest unfinished row can always run.
- **Roles.** Eight compute warps and one TMA load warp. The load warp streams 16 KB weight
  and KV stages through a 12-stage `Ring(Tma, Warps)` and never waits on activations, so it
  prefetches the next rows' weights while compute polls for inputs. The GEMV is packed FFMA2
  on CUDA cores (or tcgen05 MMAs from a tenth warp, above); weights and KV carry an L2 evict-first hint.
- **Dependencies between CTAs** are raw code: every activation is a 64-bit word
  `(fp32 value, launch epoch)` published with one relaxed store, and consumers spin on relaxed
  (mostly 16- and 32-byte vector) loads until each word carries the current epoch (NCCL's LL
  protocol). The last CTA to finish advances the epoch. Consumers sum their Split-K partials,
  except down's: its last split (last because its GATEUP rows run in a second wave) fetches the other
  partials and publishes the finished residual, so QKV and the LM head read one word per
  element instead of five.
- **Numerics** follow HF's bf16 rounding boundaries with fp32 accumulation; every per-step
  intermediate has its own buffer, so a launch has no write-after-read hazards.

Dependent model steps exchange flags between SMs; already-published words still need to be
fetched. Use the trace to distinguish those waits from weight loading.

`QwenDecodeMegakernel(splits=19)` (passed through `QwenMegakernel(..., splits=19)`) gives one
attention item per GB300 SM. The split count is static; benchmark it for the contexts the
runner decodes.

</details>

Checkpoint tests skip unless Qwen3-0.6B is already in the local HF cache; they never download it
(`hf download Qwen/Qwen3-0.6B` fetches it):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 600s \
  .venv/bin/python -m pytest test/warpdrive/test_warpdrive_qwen_megakernel.py \
  test/warpdrive/test_warpdrive_qwen_megakernel_ref.py
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 2400s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_qwen_megakernel \
  --output agent_space/qwen_megakernel.json
```

The bench compares eager and `torch.compile` CUDA graphs and a logical-bytes read roofline at
ctx 1024/4096/8192 (`--contexts`).

`QwenMegakernel(..., trace="globaltimer")` compiles the layer's
[globaltimer ranges](tracing.md#globaltimer): every CTA records its compute and load work items
(payload: the table row) and the compute role's `trace_region`s, also with the row as payload:
`poll` (waiting for the row's inputs; it ends when they are ready), `stream` (a GEMV tile's
weight stages), `attn.blocks` (an attention row's KV blocks) and `reduce.poll` (the last down
split fetching the other partials). `qwen_megakernel_trace` decodes one launch:

```python
from attn_gym.warpdrive.examples import qwen_megakernel_trace as mkt

runner = QwenMegakernel(weights, max_len=1088, trace="globaltimer")
runner(weights, token, pos, *caches)  # compile and warm up
with mkt.trace_session(runner, trace_path="step.pftrace") as session:
    runner.launch(*caches)
mkt.phase_spans(session.ranges, runner.table)  # {"QKV": us, "ATTN": us, ...} per layer
mkt.input_waits(session.ranges, runner.table)  # median row start -> inputs ready, us
```

The trace has one process per SM and one track per role; work items are named after their
instruction (`QKV`, `load QKV`, ...) with `layer`, `a` and `b` in their args, and an arrow runs
from each input group's last-finishing producer row to the row waiting for it, across SMs
(`write_step_trace`). `bench_qwen_megakernel --trace-dir DIR` writes one trace per context and
reports both views. `trace="iket"` emits the same ranges for `run-iket` (as in
[Tracing](tracing.md#iket)), and `waits=True` adds the layer's wait ranges with either backend.
Disabled, both emit no instructions.

<details markdown="1">
<summary>Implementation pitfalls and current limits</summary>

- **Co-residency.** A static schedule over spinning CTAs needs every CTA resident;
  `launch_args()`'s SM-count cap is only a capacity estimate. Launch with `cooperative=True`
  (CUDA-graph capture keeps the attribute). When another job shares the GPU, steps can slow
  down several times or stall; `schedule="atomic"` does not need co-residency.
- **Materialize tensors before `run_roles`.** Argument tensors materialized lazily inside a
  role, dicts of tensors in params and Python loops in the kernel prologue fail with "operand
  does not dominate this use" at an unrelated line. Keep params flat.
- **SMEM over-subscription fails only at launch** (`cudaErrorInvalidValue`, e.g. `stages=13`);
  `describe()` cannot report bytes before compiling.
- **Register budgets don't help a 9-warp CTA.** Three warps share one SM sub-partition, so
  ptxas caps every thread at 168 registers whatever `RoleConfig` asks for; pushing the compute
  role past 168 spills in hot loops. Check `cuobjdump --dump-resource-usage` before timing.
- **Unroll GEMV k-loops fully;** the rolled loop kept a stack frame.
- A one-entry buffer dict makes `h.data` the bare tensor (use `h.view("W")`), and a rank-1
  SMEM layout needs a tuple shape (`cute.make_layout((3072,))`).
- Bit-level helpers (fp32/int32 bitcasts, i64 packing, vector relaxed loads) need private MLIR
  imports such as `cutlass._mlir_helpers.arith.bitcast`. Each strong load is one L2 request per
  lane, so use vector loads when polling.

The tagged-word protocol crosses clusters and uses raw code, not the layer's
[cluster channels](clusters.md). `plan_from_warpdrive` sees only this kernel's CTA-local ring,
not its cross-CTA critical path; see [raw code and opt-outs](guarantees.md#opt-outs).

</details>

## Known limits

Both decode variants are OneShot-only. See the [safety contract](guarantees.md) for layer
mechanisms and analysis boundaries.

For the normal and quick-fuzz commands, see [Tests](index.md#tests). Long campaigns and
what the harness covers are in the [fuzz harness guide](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md).
