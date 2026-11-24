# Megakernel package

`examples/megakernel/` holds what the warpdrive decode megakernels share; a model directory
(`qwen_megakernel/`, `inkling_megakernel/`) composes it into that model's launches. Adding a
model should mean writing model code (its oracle, its per-layer kernels, its chat family module)
over these pieces. The
[`adding-megakernel-model`](../../../../.agents/skills/adding-megakernel-model/SKILL.md) skill
is the procedure; this file holds the package's contracts.

## Modules

| Module | Contents |
|---|---|
| `constants.py` | Tile geometry (`ROWS` x `TILE_K` weight stages of `STAGE_BYTES`, `BLOCK` KV positions of a `HEAD_DIM` head), the compute role's warps, the SMEM budget and default ring depth cap (`MAX_STAGES`), `EVICT_FIRST`/`EVICT_LAST`, `LOG2E`, the polling backoff `BACKOFF_NS`, and `multi_gemv`'s geometry (`MMA_N`, `BOX`, `QBOX`, `X_PAD`, `DBUF_FLOATS`). Imports nothing. |
| `tagged.py` | The cross-CTA protocol: `publish`, `fetch` (warp-uniform polls with `nanosleep` backoff on request), `load_words`, `pack_tagged`/`unpack_tagged`; the scratch (`scratch_offsets`, `tagged_views`) and the epoch (`advance_epoch`) |
| `regmath.py` | `bf16_round`, `bf16x8_to_f32`, `group_sum`, `warp_transpose_reduce`, `cta_sum` (over the compute warps through SMEM), `load_vec`/`load_into`/`store_vec`/`load_words16`; `fp32`/`fp32_reciprocal` (constants as PyTorch rounds them) |
| `gemv.py` | `gemv_tile`: one token against a 32-row weight tile on CUDA cores, over the input row and the weight ring; `owns_row`, `owns_pair`, `pair_value` say where its results land. Qwen wraps it in its engine protocol (`qwen_megakernel.core.CudaCoreGemv`), beside its tcgen05 engine. |
| `formats.py` | Expert weight formats (bf16, NVFP4, MXFP8), their geometry on the weight tiles, and the expert input quantizations (`ACTIVATION_FORMATS`). Imports only `constants`. |
| `quant_gemv.py` | `quant_tile`: `gemv_tile` of a quantized expert tile, its words decoded in registers to the weights' exact block values and accumulated in `gemv_tile`'s order |
| `act_quant.py` | Expert input quantization on the device, vLLM's SM100 NVFP4 (per-row amax, `fouroversix_mae` block scales) and MXFP8 quantizers bit for bit: `quantize_rows` replaces bf16 rows in shared memory by their exact block values; `row_scale` is the NVFP4 GEMM output scale |
| `multi_gemv.py` | The tensor-core GEMV: a 32-row tile against up to 8 tokens per `mma.sync.m16n8k16`, bf16 (`gemv_bf16`) or quantized weights (`gemv_quant`, `quant_tile`'s stages) |
| `attention.py` | The flash-decoding rows' helpers: `split_blocks` (a KV split's blocks, sliding windows included, and the newest split), `head_vec` and `stage_row` (a lane's slices of a head) |
| `ring.py` | The weight ring: `ring_stages` (its depth in the SMEM budget), `weight_stages`/`kv_stages` (stage layouts), `tma_atom`/`tma_partition`, and the load warp's stages, whole (`load_tiles`) or as `multi_gemv`'s swizzled boxes (`load_boxes`, `box_layout`, `swizzled_atom`). Qwen builds its atoms with its own helpers, which take the tcgen05 engine's layouts. |
| `table.py` | `InstructionTable` (the row-table `WorkMap`) and `layout_static` (its `[slot][cta]` layout) |
| `launch.py` | Host side: `Launch` (a one-launch chain's table, tagged scratch and state), `table_of`/`deal` (instruction tables dealt to CTAs), `tagged_values` |
| `spec.py` | `SpecStep`: greedy speculative decoding on the device over a model's `TargetBlock` and `DraftModel`; `prefill_blocks`: a prompt through the same block step |
| `chat.py` | The chat driver; each family provides a `chat` module (`FAMILIES`), whose backends decode one token (`DecodeBackend`) or several (`MultiTokenBackend`) per step |

## Shape preconditions

The geometry fixes what a model must have to use these pieces; check them in the model's
constructor (Qwen's `check_model`):

- **Head dimension 128** (`HEAD_DIM`): a KV stage is `BLOCK` = 64 positions of one head in
  16 KB, and `attention`'s lane layout pairs each dim d < 64 with its rotate-half partner
  64 + d.
- **Hidden size a multiple of 256**: whole `TILE_K` stages, and the compute role's 256 threads
  each own whole elements of a hidden vector. GEMV row counts (vocabulary, projections) are
  whole 32-row tiles (`ROWS`).
- **Device requirements:** TMA loads and the residency rule below. `SMEM_BYTES` is the
  SM100 budget; Qwen's batch-one CUDA-core runner also supports SM121 and sizes its weight
  ring against the device's opt-in per-block limit. Do not assume a fixed ring depth or
  select the tcgen05 engine on SM121.

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
  such grids cooperatively so the driver refuses them instead (CUDA error 720), as Inkling's
  visit launches do; Qwen's programmatic-dependent launches assume residency.
- **Epochs.** Tagged words carry the epoch of their dependency chain: the launches whose words
  feed each other share one epoch and one scratch (Qwen: a step's layer launches, which overlap
  under programmatic dependent launch; Inkling: each launch alone, its next launch ordered by
  the stream). Every CTA reads the epoch at its start; the chain's final launch advances it in
  its last CTA (`tagged.advance_epoch`), once every CTA of that launch has read it. Retire an
  epoch only when no launch of the chain can still poll for it: the next run of the chain must
  start after the final launch's CTAs finished reading (stream order or a dependent-launch
  wait), and a launch of another chain must not share the scratch while this one runs. The
  runner owns the chain's scratch and `state` (`[epoch, finished-CTA counter per launch]`;
  `launch.Launch` for one-launch chains).
- **K/V stores after TMA reads.** The new K/V row lands in a cache block that attention rows
  read through TMA (the async proxy). Order its generic store after every TMA read that uses
  the row: within a CTA, after its loads completed (Qwen's writing GQA group; the other groups
  read the block but mask the row, so the race cannot reach a result); across CTAs, with
  `fence.proxy.async` and a release/acquire handoff from the readers (the protocol in
  `inkling_megakernel.attn_half`'s docstring).
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
- **Exposed sizes.** Each static specialization a runner or chat flag can select (a constructor
  or `const_expr` choice: engine, batch, format, token count) is separate code and a separate
  compile, the tests' main cost: test the shipped defaults against the oracle, keep at least one
  oracle comparison per launch kind, and prefer configurations the checkpoint test compiles
  anyway. Runtime sizes (positions, cache capacity,
  `--max-len`) are one code path: test representative values at their boundaries (position 0
  and 1, block and split edges, a full cache) and that changing them reuses the compiled kernel
  (`compile_fn.cache_info()`).

## Quantized experts

- `formats` fixes the layout: 32-row x 256-column subtiles, warp w owning rows 4w .. 4w + 3 and
  lane l columns 8l .. 8l + 7; NVFP4 codes and e4m3 bytes as the checkpoint stores them;
  tile-major scale bytes, so a lane reads its four rows' scales as one word.
- `quant_gemv` decodes in registers to the weights' exact block values (NVFP4 `e2m1 x e4m3`
  without the per-expert global scale, MXFP8 `e4m3 x 2^(s - 127)`) and accumulates in
  `gemv_tile`'s order, so quantized dot products equal the bf16 GEMV's on the block values bit
  for bit. Test that equality bit for bit (random codes and scale bytes), not a tolerance.
- `quant_tile` reads a tile through the weight ring's word view `"QW"` and a per-tile stage
  of a meta ring holding its scale words (`"S"`); `quant_gemv`'s docstring gives their
  layouts. A launch declares those views and loads them in its load role (Inkling's
  `mlp_half`).
- Packing a checkpoint into the layout is the model's loader (`inkling_megakernel.quant`).
  Move the packing here when a second quantized model reuses it.

## Expert numerics

A checkpoint can declare how its experts' inputs are quantized (Inkling 2 Mini's
`hf_quant_config.json`: `required_activation_quantization`, NVFP4 for layers 0-19 and MXFP8 for
20-23). The kernels then compute those numerics, as vLLM's SM100 MoE does, not HF's
dequantize-to-bf16 numerics, and the model's oracle defines them
(`inkling_megakernel.ref.ExpertQuant`, `declared_experts_out`):

- each expert input row (a token's MLP input; each routed row's SwiGLU) is quantized once
  (`act_quant.quantize_rows`) and replaced by its exact bf16 block values, so the existing bf16
  GEMVs and MMAs run unchanged; the weights decode to their block values;
- the fp32 dot product is multiplied by the NVFP4 row scale `(x_amax * w_amax[e]) * rcp(1536 *
  w_den)` (`act_quant.row_scale`; MXFP8 has none) before its bf16 rounding;
- the routing weight multiplies the SwiGLU before the down input's quantization, and a token's
  down rows add up in fp32, rounded once.

Quantization makes the numerics discontinuous: a flipped bf16 rounding upstream can move a code,
a block scale or a row's amax, so an output can differ from the oracle's on slightly different
inputs by far more than the rounding (Inkling: up to ~100 ulps of a visit output for a few ulps
of residual). Test each stage on the kernel's own inputs (Inkling's `check_visit`: the
activations against the oracle's, then the down projection and the rest against the oracle on
the launch's activations), and bit-match the device quantizers against the oracle's
(`test_warpdrive_megakernel_act_quant`). The NVFP4 per-row amax needs every element of the row:
quantize where the row is complete in one CTA, after a norm (`quantize_rows`, a block a thread)
or as a fetch holds it (`quantize_chunk`, in registers, a block over 2 or 4 lanes), one CTA
maximum (`act_quant.cta_max`) per row set.

## Speculative decoding

- `SpecStep` drafts, verifies and accepts on the device over two protocols a model implements:
  `TargetBlock` (the target over `k` tokens: logits and the drafter's features) and `DraftModel`
  (commit features, propose `k - 1` drafts). It never syncs, so a CUDA graph holds whole steps;
  the chat runs it as a `MultiTokenBackend`. It decodes greedily only.
- Rejected rows need no undo in position-indexed caches. Other state (short-convolution rings,
  recurrent state) bounds `k` by how far it can rewind; the model's target checks that bound
  (Inkling's `spec.max_block_tokens`).
- Its output is the target's greedy decode in the rounding of a `k`-token forward, which can
  resolve a near tie differently from one-token decode. Test a model's step against a host loop
  over the same verify forward and drafter, not against one-token decode;
  `test_warpdrive_megakernel_spec` covers the acceptance on scripted components.

## Multi-token verify

- `multi_gemv` runs a 32-row tile against up to 8 tokens per `mma.sync.m16n8k16` from 128-byte
  swizzled TMA boxes (`ring.load_boxes`); bf16 and quantized weights give bit-identical dot
  products. Its geometry lives in `constants`, so a launch whose module does not import
  `multi_gemv` keeps it out of its compile key: Inkling's one-token visit (its host test checks);
  the message and head modules, which hold both forms, hash it.
- Share code between the one-token and block forms behind `const_expr` token counts, and check
  that the one-token kernels' PTX is unchanged.
- A block's rows are causal, so `spec.prefill_blocks` can feed a prompt through the verify block
  (padding the last one) and the caches come from the same kernels as the reply.
- In a block, the masked-row rule covers future positions within the block and keys before the
  window: Inkling's block kernel clamps V rows to the visible range, its draft kernel zeroes
  masked rows; the block kernel's test fills those rows with NaN.
- Keep the accepted block sizes in one place and test the default and one other against the
  oracle:
  `InklingDecoder` and the chat's `megakernel-spec` take `inkling_megakernel.spec.BLOCK_TOKENS`
  (2, 4 and 8), `spec-torch` 2..8.
- Inkling's block form quantizes a token's MLP input once, in the CTA that shares it, and each
  DOWN item quantizes its slot's activation rows itself as it fetches them. Electing one item
  per slot to quantize for the others (the slot's last GATEUP item, or its first DOWN item)
  made the others wait for it: 1.3-1.4x slower block steps. Per-item quantization is amortized
  over `DOWN_TILES` = 6 row tiles (2 before the quantization: 6.8-8.4% slower launches).

## Not shared yet

Pieces a third model would copy today, each with two or more variants whose shapes differ;
move one when that model fixes the shape it needs:

- the flash-decoding partial loop, merge and empty partial (Qwen, Inkling one-token and block,
  DSpark's even-share splits);
- a launch of row tiles of one matrix (Inkling's `MatrixRowsKernel`, the LM head);
- router top-k and the expert union (different semantics in Qwen and Inkling);
- prefill by one-token decode steps (Inkling's chat) beside `spec.prefill_blocks`;
- the tests' error-budget helpers (Qwen's `compare_logits`, Inkling's ulp bounds);
- chat options declared by their family instead of `ChatOptions`.
