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
| Run a whole Qwen3 decode step as one persistent launch per layer | [qwen_megakernel](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/qwen_megakernel/runner.py) | [Qwen3 megakernel](#qwen3-decode-megakernel) |
| Decode a looped MoE (Inkling 2 Mini) with one launch per layer visit | [inkling_megakernel](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/inkling_megakernel/megakernel.py) | [Inkling 2 Mini megakernel](#inkling-2-mini-decode-megakernel) |
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

- SM103-family targets only (SM103 and SM107); an SM100 target raises `ValueError`. Default is false, retaining the software path.
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
from attn_gym.warpdrive.examples.qwen_megakernel.ref import (
    load_qwen3_weights,
    make_decode_state,
)

weights = load_qwen3_weights()  # Qwen/Qwen3-0.6B from the HF hub, or a local checkpoint path
caches, token, pos = make_decode_state(weights, ctx_len=1024, max_len=1088)  # random prompt
runner = QwenMegakernel(weights, max_len=1088)  # one compile serves every max_len
logits = runner(weights, token, pos, *caches)  # decode_step_ref's contract: writes K/V at pos
```

A batch-one decode step runs as one persistent launch per layer, then one for the LM head:
a megakernel whose launch boundaries are the layer boundaries ([why](#why-one-launch-per-layer)).
The host lowers the step into a table of 24.5k instructions, assigns them to CTAs ahead of time
and splits the table by layer:

```text
for layer in 28 layers:
    QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
    ATTN    8 x 16      kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
    OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
    GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
    DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials;
                        the last split also sums them into the residual
LMHEAD  4748        32 rows of the tied embedding x final RMSNorm         -> logits
```

| Input | Contract |
|---|---|
| Model | Qwen3 0.6B, 1.7B, 8B, 14B and 32B, and Qwen3-MoE 30B-A3B (`load_qwen3_weights` reads the shape from `config.json`; `check_model` names what a shape breaks), bf16 weights, batch 1, one token per launch |
| Caches | Contiguous bf16 `(layers, kv_heads, max_len, 128)`, `max_len` a multiple of 64, `0 < pos < max_len`; only row `pos` is written |
| GPU | SM100 family; also DGX Spark (SM121) with `engine="cuda_core"` at batch one. One CTA per SM, a programmatic-dependent launch per layer. |
| Not supported | Batch > 1, prefill, paging, Qwen3-4B (hidden 2560 is not a multiple of 1024), MoE models across GPUs (235B-A22B) |

The table above shows 0.6B's counts. Larger models change only the counts: an attention row
serves two q heads of one kv head (one for 14B's odd group of 5), an o_proj split covers a
quarter of the q heads, the residual polls run 8 elements per thread at a time, the ring keeps
as many 16 KB stages as fit beside the fixed scratch space (up to 12 by default), and the
default KV split count targets about 128 attention rows per layer (16 for 0.6B and 1.7B, 8 above).
On DGX Spark, load 30B-A3B with interleaved gate/up weights so prefill and decode share
one allocation:

```python
weights = load_qwen3_weights("Qwen/Qwen3-30B-A3B", interleave_gate_up=True)
runner = QwenMegakernel(weights, max_len, engine="cuda_core")
```

The runner sizes the ring for the weights' device; no manual stage count is needed. With
0.6B at batch one, `QwenDecodeMegakernel(smem_bytes=99 * 1024).stages` is 5, versus
12 with the default SM100 budget.

The GB200 and GB300 tables in this section were measured with the earlier single cooperative
launch of the whole step.

Median us per step on a GB200 (CUDA-graph replay, 2026-11-11; `qwen_megakernel.bench`;
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
(``qwen_megakernel.bench_batch``):

| model, ctx | CUDA cores, B=1 | B=1 | B=4 | B=16 | B=32 | B=64 |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B, 256 | 1993 | 1422 | 5312 | 12251 | 15010 | 13568 |
| 0.6B, 1024 | 2049 | 1600 | 4860 | 10146 | 11963 | 10991 |
| 8B, 256 | 323 | 332 | 1095 | 2721 | | |
| 8B, 1024 | 339 | 330 | 1041 | 2438 | | |

Beyond B = 32 the input buffer leaves 0.6B five ring stages, and at ctx 1024 the batch's KV
reads (about 270 MB per layer at B = 64) bound attention.

**Batched MoE.** Qwen3-30B-A3B batches too (B <= 32, its inputs' shared-memory bound). The router
row computes every token's logits, takes each token's top 8 and publishes per-expert token lists
(count, token and rank, routing weight) plus the normalized MLP input. Each expert has four
gate/up and four down rows (a quarter of its tiles each) whose MMA columns are its tokens:
gate/up gathers their inputs and writes each token's activation slot times its routing weight,
down writes per-(token, rank) outputs, and REDUCE rows sum a token's eight into the next
residual. Rows of experts no token chose skip after one poll. A sequence's results do not
depend on which tokens share its experts (bit-identical in batches of 8 and 3). GB200, tokens
per second at ctx 256 / 1024: B=1 410 / 407 (the batch-one kernel), B=2 442 / 446, B=4 834 /
817, B=8 1401 / 1326, B=16 1925 / 1771, B=32 2279 / 2052. At small batches most of the 1024
expert rows per layer only skip; a device-built list of active experts would remove them.

**Streamed inputs.** `QwenMegakernel(..., engine="tcgen05", batch=B, stream_inputs=True)`
(`QwenDecodeMegakernelTcStream`) keeps GEMV inputs in global memory and stages chunks in
shared memory, leaving room for the weight ring. With resident inputs, batches above 16
on 8B and 14B, or 8 on 32B, no longer fit. NORM rows, one per token, normalize QKV and
LM-head inputs into a shared bf16 buffer. For dense models, they also normalize gate/up
inputs, and gate/up epilogues write their activations into that buffer. NORM rows publish
an epoch flag per token; gate/up publishes one per 64-column chunk. A dedicated warp polls
these flags and bulk-copies chunks into the activation ring read by the MMAs. For dense
models with equal KV split counts, each sequence's results are bit-identical to the
tcgen05 kernel at batch one. o_proj, the MoE router and expert rows still build inputs in
per-CTA scratch. The table compares megakernel throughput (the faster of resident and
streamed inputs) with `torch.compile` + CUDA graph, in tokens per second on GB300 at
ctx 1024. Each cell is megakernel / compiled:

| model | B=16 | B=32 | B=64 |
|---|---:|---:|---:|
| 1.7B | | 10598 / 5020 | 11282 / 5875 |
| 8B | 2946 / 2342 | 4325 / 3301 | 5256 / 4166 |
| 14B | 1767 / 1764 | 2502 / 2581 | 2999 / 3464 |
| 32B | 845 / 934 | 1244 / 1439 | 1452 / 2055 |

In these measurements, resident inputs are faster at B = 8 and below; NORM rows add a
dependency hop per phase. Attention on CUDA cores is the largest remaining cost on 14B
and 32B.

The megakernel removes launch and dependency latency, so its lead shrinks as weight bytes
dominate the step. Its one-step error against fp32 stays within the spread of eager bf16's,
which varies about 5x across prompts on the deep models.

### Why one launch per layer

A decode step is a chain of short dependent phases. Two considerations determine where to put
launch boundaries:

- **A kernel boundary requires materializing any live on-chip state.** Registers, shared
  memory and tensor memory belong to a CTA and die with it, so a value that one op leaves on
  chip for the next must be written out and read back if a launch boundary falls between
  them. Activations that already cross CTAs through global memory pay nothing extra.
- **A programmatic-dependent launch hands off like an in-kernel dependency.** Each launch
  calls `griddepcontrol.launch_dependents` at its start, so the next launch's CTAs take SMs as
  this launch's CTAs exit, and its consumers poll the same epoch-tagged words they would poll
  inside one kernel. Only a step's first launch calls `griddepcontrol.wait`, to read the
  epoch the previous step's last launch wrote; later launches do not wait for their
  predecessors to complete. The added cost per boundary is a launch hop of under a
  microsecond, against several to tens of microseconds per layer.

The layer boundary is where both hold: the residual sum after the MLP is an all-to-all (every
CTA's next QKV row needs the whole hidden vector, built from every down partial), so nothing
local is live there, and every layer is long enough to hide the launch hop. Inside a layer the
phases are short and point to point (QKV to attention to o_proj to the MLP), and the rows and
weights a CTA keeps for the next item need one persistent CTA per SM; those stay inside a
launch.

What a layer boundary still costs: each launch's CTAs start fresh, so an SM's prologue and its
prefetch of the next layer's weights wait until its previous CTA exits. Against one cooperative
launch of the whole step, most configurations we measured ran as fast or faster; the worst lost
several percent of a step, where a layer's first phase waited on that fresh start. The host
enqueues a step's launches faster than the GPU runs them, so eager steps time within a few
percent of CUDA-graph replay. In return each launch has its own table, which keeps tables small
and traces per layer, needs no cooperative launch, and leaves room for launches that differ per
layer (hybrid models whose attention layers want another kernel configuration).

<details markdown="1">
<summary>How the megakernel runs the model</summary>

- **Code.** The `qwen_megakernel` package over the shared [megakernel package](#shared-megakernel-package):
  `core.py` holds the base kernel, `tc.py` the tcgen05 engine and streamed inputs, `schedule.py` the host's instruction
  table and CTA assignment, and `runner.py` the `QwenMegakernel` runner. One prologue builds every GEMV input: a group of
  `8 // min(batch, 8)` warps per token, so batch one uses all eight warps per token and a batch
  of eight or more one warp each.
- **Schedule.** `InstructionTable` is a `WorkMap` over the int32 table. `list_schedule`
  assigns rows to CTAs with an engine-specific row-latency model (`row_cost`, fitted to
  traced work items across batch sizes), placing each row on the CTA that finishes it first (at
  batch one a row costs less on a CTA whose previous GEMV row wrote the same input: it skips the
  prologue), and `layout_static` lays each launch's table out
  `[slot][cta]` for `StaticPersistent`; `schedule="atomic"` claims rows in dependency order with
  `AtomicPersistent` instead. Each CTA's rows stay in dependency order, so with every CTA of a
  launch resident the earliest unfinished row can always run. `build_tables` schedules the
  whole step once and splits each CTA's rows by layer into that layer's launch, ordering a
  launch's CTAs by their first row's predicted start; the atomic schedule gives each launch its
  own claim counter.
- **Roles.** Eight compute warps and one TMA load warp. The load warp streams 16 KB weight
  and KV stages through a device-sized `Ring(Tma, Warps)` and never waits on activations, so it
  prefetches the next rows' weights while compute polls for inputs. The GEMV is packed FFMA2
  on CUDA cores (or tcgen05 MMAs from a tenth warp, above); weights and KV carry an L2 evict-first hint.
- **Dependencies between CTAs** are raw code: every activation is a 64-bit word
  `(fp32 value, step epoch)` published with one relaxed store, and consumers spin on relaxed
  (mostly 16- and 32-byte vector) loads until each word carries the current epoch (NCCL's LL
  protocol), within a launch or across one. The last CTA of the step's last launch advances
  the epoch. Consumers sum their Split-K partials,
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
  .venv/bin/python -m attn_gym.warpdrive.examples.qwen_megakernel.bench \
  --output agent_space/qwen_megakernel.json
```

The bench compares eager and `torch.compile` CUDA graphs and a logical-bytes read roofline at
ctx 1024/4096/8192 (`--contexts`).

`QwenMegakernel(..., trace="globaltimer")` compiles the layer's
[globaltimer ranges](tracing.md#globaltimer): every CTA records its compute and load work items
(payload: the table row) and the compute role's `trace_region`s, also with the row as payload:
`poll` (waiting for the row's inputs; it ends when they are ready), `stream` (a GEMV tile's
weight stages), `attn.blocks` (an attention row's KV blocks) and `reduce.poll` (the last down
split fetching the other partials). `qwen_megakernel.trace` decodes all launches of one step.
Each launch records into its own buffer; ranges use launch-local CTA indices, and work-item
payloads index `runner.table`:

```python
from attn_gym.warpdrive.examples.qwen_megakernel import trace as mkt

runner = QwenMegakernel(weights, max_len=1088, trace="globaltimer")
runner(weights, token, pos, *caches)  # compile and warm up
with mkt.trace_session(runner, trace_path="step.pftrace") as session:
    runner.launch(*caches)
mkt.phase_spans(session.ranges, runner.table)  # {"QKV": us, "ATTN": us, ...} per layer
mkt.input_waits(session.ranges, runner.table)  # median row start -> inputs ready, us
```

The trace has one process per schedule CTA and one track per role; work items are named after their
instruction (`QKV`, `load QKV`, ...) with `layer`, `a` and `b` in their args, and an arrow runs
from each input group's last-finishing producer row to the row waiting for it, across CTAs
(`write_step_trace`). `qwen_megakernel.bench --trace-dir DIR` writes one trace per context and
reports both views. `trace="iket"` emits the same ranges for `run-iket` (as in
[Tracing](tracing.md#iket)), and `waits=True` adds the layer's wait ranges with either backend.
Disabled, both emit no instructions.

<details markdown="1">
<summary>Implementation pitfalls and current limits</summary>

- **Co-residency.** A static schedule over spinning CTAs needs every CTA of a launch resident
  eventually. A launch starts only after every CTA of the previous one has started, and those
  never wait on later launches, so the step cannot deadlock on its own. A concurrent kernel that
  holds SMs while it spins (another static runner on another stream, an MPS peer) can deadlock
  it, which the cooperative launch used to reject at launch; `schedule="atomic"` does not need
  co-residency.
- **Materialize tensors before `run_roles`.** Argument tensors materialized lazily inside a
  role, dicts of tensors in params and Python loops in the kernel prologue fail with "operand
  does not dominate this use" at an unrelated line. Keep params flat.
- **Ring depth is checked at construction.** An explicit `stages` outside `[2, fit]` raises
  `ValueError`; `fit` reserves room for fixed buffers, barriers and alignment. This is a budget
  check, not a compiled resource report: CUDA still checks the actual allocation at launch.
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

## Inkling 2 Mini decode megakernel

``inkling_megakernel`` decodes Thinking Machines' Inkling 2 Mini (``thinkingmachines/Inkling-2-Mini``)
at batch one. The model runs its 24 layers as 48 *visits*: layers 0-5 once, layers 6-17 three
times with shared weights (every pass restarts from layer 5's output and reads gated messages
from the previous pass), layers 18-23 once. Each visit has its own KV cache and short-convolution
rings.

```bash
python -m attn_gym.warpdrive.examples.inkling_megakernel.chat --prompt "Hi" --reasoning-effort low
python -m attn_gym.warpdrive.examples.inkling_megakernel.bench --context 1100
```

- **Oracle.** ``ref.py`` is a PyTorch implementation with HF's dtype boundaries; loaded with
  ``activation_quant=False`` and given the same weights it equals HF's eager
  ``Inkling2LoopForCausalLM`` bit for bit on the tiny test model and short real prompts (its
  docstring states the scope). By default it computes the experts' declared numerics (below).
  ``fast.py`` runs the same dataflow as a CUDA-graph capturable torch step.
- **Launches.** A step is 51 launches: one per visit (its attention half, then its MLP half in
  the same persistent grid, the halves' rows ordered so the static schedule cannot deadlock), one
  per loop pass after the first (all 48 messages and the message half of every gate's down
  projection, which depend only on the previous pass), and one head launch (final block residual,
  norm, LM head, softcap). Cross-CTA dependencies are the [megakernel package](#shared-megakernel-package)'s
  epoch-tagged words.
- **Experts.** The checkpoint stores experts as NVFP4 (layers 0-19) and MXFP8 (layers 20-23);
  the kernel streams them as stored and decodes their block values in registers exactly, so its
  outputs equal those of the same kernel on the bf16 block values bit for bit
  (``--backend megakernel-bf16`` keeps that path). Weights take 19 GB on the device.
- **Expert numerics.** The checkpoint's ``hf_quant_config.json`` declares the quantization of
  the experts' inputs (``required_activation_quantization``: NVFP4 with a per-token amax, 16-column
  blocks and ``fouroversix_mae`` e4m3 scales for layers 0-19; MXFP8 with 32-column ue8m0 blocks
  for 20-23). The oracle, the torch step and the kernels compute it as vLLM's SM100 MoE kernels
  do (``act_quant.py``, bit for bit against vLLM's Triton quantizers): each expert input
  quantized once (a token's MLP input, each routed row's SwiGLU), the weights' block values
  without the NVFP4 per-expert factor, which joins the token's amax in a row scale on the fp32
  dot product, and the routing weight applied before the down input's quantization. HF's
  modeling code (and vLLM on Hopper) instead dequantizes the weights to bf16 with the factor
  folded in, keeps activations bf16 and applies the routing weight after the down projection;
  the oracle keeps that path behind ``activation_quant=False`` for its HF parity tests.
- **Numerics.** Per visit within a few bf16 ulps of the oracle (fp32 summation order and the
  flash-decoding merge) on the launch's own inputs; quantization amplifies rounding
  differences, so a whole step's logits stay within a few units of the oracle's.

The declared numerics against HF's, teacher-forced on the 6 DSpark chat prompts (low effort,
each with HF's 256-token greedy continuation) and a 1328-token code prompt (64-token
continuation); KL(hf || variant) in nats at the 1600 continuation positions. "Structure only"
applies the declared structure (factor outside the dot, routing weight before the down
projection) with an identity quantizer; the noise floor is the HF oracle prefilled in 64-token
chunks:

| | KL mean / median / p99 / max | top-1 agreement |
|---|---|---:|
| declared numerics | 0.015 / 0.0024 / 0.136 / 0.32 | 95.9% |
| structure only | 0.0032 / 0.0004 / 0.036 / 0.08 | 97.8% |
| noise floor | 0.0029 / 0.0004 / 0.038 / 0.13 | 98.2% |

Decode step at context 1100, GB300, CUDA-graph replay (``bench --context 1100``, 2026-11-24;
one run, all rows in the same process; declared numerics):

| Step | ms per token |
|---|---:|
| torch step (``fast.decode_step``) | 60.2 |
| megakernel, bf16 experts | 3.31 |
| megakernel, quantized experts | 3.01 |

The declared numerics cost the quantized-expert megakernel nothing measurable (3.02 ms per
token with HF's numerics, same session), the bf16-expert one 2.4%, and the torch step its
quantizers' small elementwise kernels (32.3 ms with HF's numerics).

The chat (``--prompt "What is the capital of France? One sentence." --reasoning-effort low
--max-len 2048``, a 57-token reply, megakernel default backend) decoded at 305 tok/s (GB300,
2026-11-24). Greedy tokens equal the oracle's on the tested checkpoint decodes except where the two
best logits nearly tie.

**Speculative decoding.** ``--backend megakernel-spec`` drafts with the checkpoint's DSpark
drafter (``draft.py``: three launches) and verifies a block of ``k`` = 4 tokens per step on the
multi-token decoder (``InklingDecoder.decode_block``), accepting on the device, so a CUDA graph
holds whole steps (``megakernel.spec``); ``--verify-tokens 2`` or ``8`` picks another ``k``.
On DGX Spark, keep the default ``k = 4``: at ``k = 8`` the full-attention input tiles alone
exceed GB10's 99 KiB per-block shared-memory limit. To try the quantized checkpoint on Spark:

```bash
python -m attn_gym.warpdrive.examples.megakernel.chat --model thinkingmachines/Inkling-2-Mini --backend megakernel-spec
```

To inspect a single `megakernel-spec` round offline, capture one warm CUDA graph replay with
`torch.profiler` (CPU and CUDA activities) and `export_chrome_trace`. Add model tracks to the
original GPU timeline, then open `annotated.json` in [Perfetto](https://ui.perfetto.dev/):

```bash
python -m attn_gym.warpdrive.examples.inkling_megakernel.schedule_trace round.json annotated.json
```

The replay must contain exactly one complete step on one stream; a multi-step chat graph is not
an input to this exporter. It matches kernel names against `InklingConfig.visits()` (default:
Inkling 2 Mini; `--config /path/to/config.json` selects a checkpoint config). The tracks show
drafting, target verification, each core pass and layer visit, and acceptance from target argmax
through the final device updates. Timestamps do not reveal how many drafts matched. Supply
`--emitted-tokens N` only if you recorded that replay's output count (accepted drafts plus one).

A step is the block's 51 model launches and the drafter's three, plus about 15 small torch kernels that accept
on the device (argmax, prefix match) and update the step's state (count, tokens, anchor,
position). It decodes greedily only, and its replies are the greedy decode in the block's
rounding, which can resolve a near tie differently from the one-token step. Chat replies on a
GB300 (5 prompts x low/high reasoning effort, up to 256 tokens, 2026-11-24). tok/s pools a
backend's replies (their tokens over their decode time); the speedup is the geometric mean of
each reply's tok/s ratio to megakernel's on the same prompt:

| Backend | tokens per step | tok/s, context ~40 / ~1100 | speedup, context ~40 / ~1100 |
|---|---:|---:|---:|
| megakernel | 1 | 342 / 330 | 1 |
| megakernel-spec, k = 2 | 1.65-1.66 | 397 / 400 | 1.16x / 1.21x |
| megakernel-spec, k = 4 | 2.35-2.36 | 428 / 438 | 1.28x / 1.34x |
| megakernel-spec, k = 8 | 2.59-2.78 | 326 / 362 | 0.98x / 1.12x |

A ``k``-token verify costs 3.59 / 4.72 / 6.45 ms at ``k`` = 2 / 4 / 8 against 3.01 ms for the
one-token step (context 1100), drafting 0.35-0.47 ms: at ``k`` = 4 a step pays off above
about 1.7 tokens per step ((4.72 + 0.35 .. 0.47) / 3.01 = 1.68 .. 1.72). Code and
step-by-step reasoning replies gain most (1.54-1.74x at ``k`` = 4), prose (story, explanation,
email) least (0.96-1.35x).

The bf16 bandwidth floor is about 1.4 ms per token (about 0.6 ms with quantized experts). The
step is latency-bound: the serial chain from a visit's attention output to its first expert row
(residual, norm, router, top-8) costs about 9 us per visit, and NVFP4 decoding is compute-bound,
so the quantized experts save less than their 3.5x fewer bytes suggest. A visit grid needs all its
CTAs resident at once, so visit launches are cooperative: one CTA per SM, admitted only when the
whole grid fits (otherwise the launch fails, e.g. under MPS or green-context SM limits). Callers
keep ``0 <= pos < max_len``; positions are read on the device.

``--backend spec-torch`` runs the same greedy speculative step on the torch decode step
(``fast.py``'s kernels) and the torch drafter, ``--verify-tokens`` 2 to 8 (default 4). It is a
reference for the megakernel backends, not a fast path: the eager activation quantizers make
the torch step launch-bound, so at ``k`` = 4 it decodes at 26-30 tok/s against 17 tok/s for
``--backend torch`` (GB300, low effort, 128 tokens, two prompts, 2026-11-24).

## Shared megakernel package

`examples/megakernel/` holds what the decode megakernels share: the tile geometry, the epoch-tagged
words that carry activations between CTAs, register helpers, the CUDA-core GEMV engine and its
NVFP4/MXFP8 expert decode, the weight ring's TMA loads, the instruction table, the tensor-core
multi-token GEMV, the speculative step and the chat driver. A model directory keeps its oracle,
kernels, runners, bench and chat family module. Its
[`AGENTS.md`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/megakernel/AGENTS.md)
states what belongs in the package and the rules its kernels follow (residency, K/V store ordering,
compile-key boundaries); the `adding-megakernel-model` skill
(`.agents/skills/adding-megakernel-model`) walks through adding a model.

The chat picks the model family from the checkpoint's `config.json`:

```bash
python -m attn_gym.warpdrive.examples.megakernel.chat --model Qwen/Qwen3-8B
python -m attn_gym.warpdrive.examples.qwen_megakernel.chat --backend compile --prompt "Hi"
python -m attn_gym.warpdrive.examples.inkling_megakernel.chat --prompt "Hi" --reasoning-effort low
```

Before the first prompt, the chat compiles the backend's kernels that are missing from the
[compile cache](cache.md) (all of them after a code change) with a progress bar, several in
parallel worker processes and a single one in the chat's own process, then captures its decode
step; with a warm cache it starts without a word. Cold, Inkling's `megakernel-spec` (11 kernels)
took 71 s to its first reply against 307 s compiled one at a time on the first turn, and
`megakernel` (8 kernels) 24 s against 71 s (GB300, 2026-11-24).

## Known limits

Both decode variants are OneShot-only. See the [safety contract](guarantees.md) for layer
mechanisms and analysis boundaries.

For the normal and quick-fuzz commands, see [Tests](index.md#tests). Long campaigns and
what the harness covers are in the [fuzz harness guide](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md).
