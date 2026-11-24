# Megakernel package

`examples/megakernel/` holds what the warpdrive decode megakernels share; a model directory
(`qwen_megakernel/`) composes it into that model's launches. Adding a model should mean writing
model code (its oracle, its per-layer kernels, its chat family module) over these pieces. The
[`adding-megakernel-model`](../../../../.agents/skills/adding-megakernel-model/SKILL.md) skill
is the procedure; this file holds the package's contracts.

## Modules

| Module | Contents |
|---|---|
| `constants.py` | Tile geometry (`ROWS` x `TILE_K` weight stages of `STAGE_BYTES`, `BLOCK` KV positions of a `HEAD_DIM` head), the compute role's warps, the SMEM budget and default ring depth cap (`MAX_STAGES`), `EVICT_FIRST`, `LOG2E`. Imports nothing. |
| `tagged.py` | The cross-CTA protocol: `publish`, `fetch`, `load_words`, `pack_tagged`/`unpack_tagged`; the scratch (`scratch_offsets`, `tagged_views`) and the epoch (`advance_epoch`) |
| `regmath.py` | `bf16_round`, `bf16x8_to_f32`, `group_sum`, `warp_transpose_reduce`, `load_vec`/`load_into`/`store_vec` |
| `gemv.py` | `gemv_tile`: one token against a 32-row weight tile on CUDA cores, over the input row and the weight ring; `owns_row`, `owns_pair`, `pair_value` say where its results land. Qwen wraps it in its engine protocol (`qwen_megakernel.core.CudaCoreGemv`), beside its tcgen05 engine. |
| `attention.py` | The flash-decoding rows' helpers: `split_blocks` (a KV split's blocks and the newest split), `head_vec` and `stage_row` (a lane's slices of a head) |
| `ring.py` | The weight ring: `ring_stages` (its depth in the SMEM budget), `weight_stages`/`kv_stages` (stage layouts), `tma_partition`, and `load_tiles` (the load warp's stages) |
| `table.py` | `InstructionTable` (the row-table `WorkMap`) and `layout_static` (its `[slot][cta]` layout) |
| `launch.py` | Host side: `tagged_values` (a tagged buffer's fp32 values). Torch only. |
| `chat.py` | The chat driver; each family provides a `chat` module (`FAMILIES`) |

## Shape preconditions

The geometry fixes what a model must have to use these pieces; check them in the model's
constructor (Qwen's `check_model`):

- **Head dimension 128** (`HEAD_DIM`): a KV stage is `BLOCK` = 64 positions of one head in
  16 KB, and `attention`'s lane layout pairs each dim d < 64 with its rotate-half partner
  64 + d.
- **Hidden size a multiple of 256**: whole `TILE_K` stages, and the compute role's 256 threads
  each own whole elements of a hidden vector. GEMV row counts (vocabulary, projections) are
  whole 32-row tiles (`ROWS`).
- **SM100**: the SMEM budget (`SMEM_BYTES`), TMA loads and the residency rule below.

## What belongs here

- **Here:** code defined by the tile geometry, the tagged-word protocol or a protocol every
  family implements, with no model semantics: helpers, GEMV engines, ring loads, the row table,
  host drivers.
- **In the model directory:** the config, oracle and weight loader; op codes, table words and
  row schedule; the scratch layout; the per-layer kernels and runners; rounding points that follow
  the model's reference; and the check of its shapes against the preconditions above.
- Move a piece when a second model needs it, or when it has no model semantics and a current
  user. Do not add frameworks for a model nobody is writing. Moving a piece updates every
  importer; leave no re-export under the old name.
- Package modules import `attn_gym.warpdrive` and each other, never a model; only the chat
  driver names the models, loading each family's `chat` module by name (`FAMILIES`).

## Compile keys

`jit_cache` keys hash every module the compile module imports
([compile cache](../../../../docs/warpdrive/cache.md)):

- A model's modules import the package and their own model's modules only (a family's chat
  module reaches the others only through the chat driver), so editing one model recompiles no
  other model's kernels. `test/warpdrive/test_warpdrive_megakernel_package.py` checks every
  `*_megakernel` directory; add to its `REQUIRED` a few modules each new kernel must hash, so a
  computed import cannot drop them from the key unnoticed.
- Pass sizes to kernel constructors and keep oracles, loaders, runners, benches and the chat out
  of kernel modules' imports, so host edits recompile nothing. (Qwen's kernels import
  `QwenConfig` from `qwen_megakernel/ref.py`, so its oracle edits recompile them.)
- Editing a package module recompiles the kernels whose import closure reaches it: every
  model's, for the device modules they share. When a change is meant to be pure
  (a move, a rename), compare the PTX of each model's kernels before and after
  (`attn_gym.warpdrive.ptx_check.capture_ptx`).

## Kernel rules

- **Residency.** CTAs that poll each other's tagged words need the whole grid resident: one CTA
  per SM, each CTA's rows in dependency order (`layout_static`), so the earliest unfinished row
  can always run. A grid that cannot be resident (an MPS or green-context SM limit) hangs. Launch
  such grids cooperatively so the driver refuses them instead (CUDA error 720); Qwen's
  programmatic-dependent launches assume residency.
- **Epochs.** Tagged words carry the epoch of their dependency chain: the launches whose words
  feed each other share one epoch and one scratch (Qwen: a step's layer launches, which overlap
  under programmatic dependent launch; a one-launch chain is the simplest case). Every CTA reads
  the epoch at its start; the chain's final launch advances it in its last CTA
  (`tagged.advance_epoch`), once every CTA of that launch has read it. Retire an epoch only when
  no launch of the chain can still poll for it: the next run of the chain must start after the
  final launch's CTAs finished reading (stream order or a dependent-launch wait), and a launch
  of another chain must not share the scratch while this one runs. The runner owns the chain's
  scratch and `state` (`[epoch, finished-CTA counter per launch]`).
- **K/V stores after TMA reads.** The new K/V row lands in a cache block that attention rows
  read through TMA (the async proxy). Order its generic store after every TMA read that uses
  the row: within a CTA, after its loads completed (Qwen's writing GQA group; the other groups
  read the block but mask the row, so the race cannot reach a result); across CTAs, with
  `fence.proxy.async` and a release/acquire handoff from the readers.
- **Masked rows.** Cache rows past the position or outside the window hold stale values, NaN
  included. A masked score does not stop `0 * NaN = NaN` in a product, so such rows must never
  reach a GEMV or MMA. Tests fill unused rows with NaN.
- **Alignment.** A fake tensor's `assumed_align` is what the kernel's vector loads may rely on.
  Declare what the widest access needs: TVM-FFI checks it at call time, so a misaligned buffer
  raises `ValueError` instead of faulting.
- **32-bit offsets.** Index within one launch's slice (a layer's cache, not every layer's) so
  offsets fit int32, and state the context limit that implies; offset in 64 bits where a slice
  cannot fit (Qwen's batched caches).
- **Capture-safe host paths.** A step reads its token and position on the device and syncs
  nothing, and the captured path allocates nothing per step (outputs the runner owns or the
  caller passes), so CUDA graphs hold whole steps (a
  chat replay feeds up to `SYNC_EVERY` positions). Validate shapes once, when a runner is built.
- **Exposed sizes.** Test every static specialization a runner or chat flag can select (a
  constructor or `const_expr` choice: engine, batch, format, token count) against the oracle,
  not a representative one: each is separate code. Runtime sizes (positions, cache capacity,
  `--max-len`) are one code path: test representative values at their boundaries (position 0
  and 1, block and split edges, a full cache) and that changing them reuses the compiled kernel
  (`compile_fn.cache_info()`).
