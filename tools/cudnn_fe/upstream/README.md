# DRAFT upstream cudnn-frontend patches (not filed)

> **DRAFT. Nothing here has been filed, sent to NVIDIA, or proposed upstream.** These are
> reviewable drafts of Attention Gym fixes translated back to upstream cudnn-frontend style.
> Filing any of them is a separate, explicitly approved action.

Prepared against **cudnn-frontend v1.30.0** (`42286a6fb8792055f3db80403350e80114ea086c`). Each
patch maps to a row of the modification ledger in
[`MAINTENANCE.md`](../../../attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md); the narrative
(how each bug was found, evidence, why the AG fix looks the way it does) is in
[`WORKLOG.md`](../../../attn_gym/linear/_delta_rule/cudnn_fe/WORKLOG.md), and
[`../fixes.toml`](../fixes.toml) links every ledger ID to its patch, repro and expected result.
[`../verify_fixes.py`](../verify_fixes.py) `--upstream-python` runs all repros and classifies each
bug as still present or fixed upstream.

## Ranked results

Rank is suggested upstream value, not application order. "Fixed" means the repro passed, not that
every optional specialization was tested.

| Rank | Patch | Ledger | Stock v1.30 behavior reproduced? | Patched validation |
|---:|---|---|---|---|
| 1 | `01-gdn-beta-free-dbeta.patch` | B7 | **Yes:** dBeta=0 instead of 1 at beta=0, BF16/FP16, both inverse-factor routes | All five gradients exact vs FP64 in 16 dtype × beta × factor-route cases |
| 2 | `02-kda-fp32-delta-residual.patch` | B6 | **Yes:** output 2 instead of 1.5; dBeta 0 instead of −256; later dQ 128 instead of 96 | 16 forward plan cases and four uncut backward cases |
| 3 | `03-compact-empty-unsplit-work.patch` | B8 | **Yes:** padded empty intervals stay in the work table; 8/20 work-count checks fail | 20/20 checks, incl. exact empty final-state/cotangent preservation |
| 5 | `05-scalar-scan-head-stride.patch` | B13 | **Yes:** raw scan wrong in 256/516 entries, max abs error 4.6166; host rejects stride-2 replay | Raw scan and compact→strided host replay match exactly |
| 6 | `04-omit-zero-chunk-split-work.patch` | B10 | **Yes:** 8 items instead of 4 (not a reproduced TMEM hang) | Opted-in table has 4 items, all-empty 0; stateful default unchanged |
| 7 | `07-single-thread-gdn-mbarrier-init.patch` | B2 | **No isolated failure:** all-thread init pattern exists | Stock and patched 32-iteration multi-wave GDN fwd/bwd stress both pass |
| 8 | `08-derive-scheduler-arrival-counts.patch` | S12 | **No current wrong count:** default literals are correct | Stock and patched 32-iteration multi-wave KDA fwd/bwd smoke both pass |

## Running the repros

| Script | Arguments | Stock v1.30 | Patched |
|---|---|---|---|
| `repro_01_gdn_beta.py` | `--beta B...`; `--checkpoints` selects the compute-factor route (default gmem factors) | fails (dBeta 0) | pass |
| `repro_02_kda_residual.py` | default: public-API uncut fwd+bwd; `--scheme dv\|prep\|chain` forward plan probes via upstream planning hooks | fails | pass |
| `repro_03_empty_unsplit.py` | optional export dir holding `python/cudnn` (defaults to the installed package) | 8/20 fail | 20/20 |
| `repro_04_zero_chunk_walk.py` | `--skip-empty` on patched (opt-in) | fails (8 vs 4) | pass with `--skip-empty` only |
| `repro_05_scalar_head_stride.py` | default: host replay; `--raw`: explicit dynamic signature isolates address arithmetic | fails (both) | pass |
| `repro_07_mbarrier_stress.py` | `--iterations 32` | pass | pass |
| `repro_08_scheduler_counts.py` | `--iterations 32` | pass | pass |
| `repro_09_replay_abi_scope.py` | none; scope check for excluded B14 | pass (`None` fails at build) | n/a |

## Source/history audit

Translated from these AG commits (by subject; see `../fixes.toml`), **without AG restyling or
dependencies**:

- 01 (B7): "Compute the GDN bprop dBeta without dividing by beta".
- 02 (B6): "Add an FP32 KDA delta-residual staging helper", "Keep the KDA forward delta residual
  in FP32 before the MMA pack", "Keep the KDA backward delta residual in FP32 through subtraction
  and beta scaling".
- 03 (B8): PR #603 and its v1.30 replay in "Route unpaged GDN and KDA and native CP summaries to the
  v1.30 kernels"; adapted conservatively to upstream state ownership.
- 04 (B10): "Omit zero-chunk split work items"; adds an upstream-specific safe opt-in.
- 05 (B13): "Respect scalar-gate head strides in split scans"; uses upstream CuTeDSL's
  `make_fake_tensor` instead of the AG fake-tensor helper.
- 07 (B2): "Initialize the GDN prefill mbarriers from one thread", "Initialize the GDN backward
  mbarriers from one thread", summary part of "Initialize GDN summary mbarriers from one thread and
  tidy vendored kernel names".
- 08 (S12): the arrival-count part of the three "Validate … launch contracts …" commits only, not
  AG's launch-validation framework.

## Draft issue 01 — GDN backward loses dBeta at zero and tiny post-activation beta

**Symptom.** For two tokens with q=k=e₀, v=(e₀,2e₀), log-gate=0, beta=(1,b), and loss=o₁, the exact
recurrence gives loss=1+b and dBeta₁=1, including b=0. FROST returns dBeta₁=0 at zero.

**Root cause.** Backward recovers an already beta-scaled quantity by dividing by `beta + 1e-10`.
This is undefined at zero and attenuates tiny-beta gradients. Removing the constant or clamping beta
cannot restore the missing unscaled value.

**Repro.** `repro_01_gdn_beta.py` via `cudnn.pygraph().gdn_bwd()` with sigmoid and QK normalization
disabled: four zero-beta failures on stock across both dtypes and both factor sources; 16 exact
five-gradient cases patched.

**Fix.** Keep beta-free inverse factors and stage Z separately from beta·Z; compute dBeta directly.
Retains the AG fix's coupled GEMM issue order and scratch-lifetime changes. Covers compute and gmem
factor sources, including reconstructing the beta-free inverse from gmem factors.

**Draft limitations.** The patch adds explicit guards for:

1. `d_k != d_v`: sdQ is borrowed for a value-width Z tile; K≠V needs independent Z storage or a
   correctly dimensioned view, MMA descriptors and layout/stride handling.
2. Multiple dQ/dK SMEM stages: the fix uses one-stage sdQ/sdK lifetimes and reduction scratch;
   general support needs stage-indexed scratch plus barriers proving every async consumer finished.
3. Fused QK L2 normalization at `d_v=128`: its scratch overlaps the beta-free path's sDm lifetime.

**Historical performance (AG port, not this export).** GB200 fwd+bwd +0.4% to +3.6% across four
workloads (lower is better).

## Draft issue 02 — KDA rounds away the delta residual before subtraction and beta scaling

**Symptom.** A small delta next to a large state contraction disappears: 4098 rounds to 4096 in BF16
and FP16, so V=4096 minus the packed contraction gives 0 instead of −2. The exact 64-token fixture
expects output=1.5, dBeta₁₆=−256, dQ₃₂=96; stock yields 2, 0, 128.

**Root cause.** The state·K accumulator is packed before subtraction, then residual/beta are packed
again. Backward consumes the rounded residual for its value contribution to dBeta.

**Repro.** `repro_02_kda_residual.py` (default and `--scheme dv|prep|chain`) fails on stock at the
exact probes; patched passes 16 forward cases (4 plans × 2 dtypes × seed absent/present) and four
uncut backward cases.

**Fix.** A packed helper unpacks V, subtracts FP32 state·K, applies FP32 beta and packs once for MMA,
keeping the FP32 residual for dBeta. Covers prefill, prep-prefill, summary, recompute and bprop. Prep
keeps residual-only staging because its factors already contain beta.

**Historical performance (AG port).** T=32768, H=48 chain forward 1360→1378 µs (+1.3%, lower is
better), others within noise; backward ≈+1% at T2048/H8 and packed; REG/STACK unchanged.

## Draft issue 03 — Empty packed intervals consume unsplit scheduling slots

**Symptom.** Padding `cu_seqlens` with repeated boundaries increases main-kernel work despite adding
no tokens.

**Root cause.** `order_body(gen=True)` synthesizes every sequence×head item and the THD builders
construct descriptors even for empty intervals. Empty stateful items own required pass-through
stores, so blindly dropping zero-length intervals introduces a separate correctness bug.

**Repro.** `repro_03_empty_unsplit.py`: stock passes 12 checks and fails the 8 work-count checks;
patched passes all 20, including deterministic order, capacity fallbacks, forward zero/seeded/indexed
state and backward cotangents with/without exit input.

**Fix.** Deterministic ballot/prefix compaction keeping original sequence IDs and destinations; skip
empty descriptors in the three shared THD builders. Internal `skip_empty=False` opt-in, enabled only
by uncut GDN/KDA forward hosts when `state_out is None`.

**Scope.** Stateful outputs, backward, chain, KDA prep, GDN2/GDP and >4096-sequence batches stay
uncompacted. Upstream has no per-sequence has-initial/fresh-state mask, so AG's fresh-empty paged
clearing is not transplanted.

**Historical performance (AG PR #603).** 71 padded empty intervals: GDN forward 29.9→96.3 µs before
compaction, 32.3 µs after (lower is better). This narrower draft was not timed.

## Draft issue 04 — Split walk emits zero-chunk items

**Symptom.** Bounds `[0,32,32,48,48]`, H=2: eight items instead of four; all-empty bounds also emit
work. A scheduler inefficiency, not a reproduced persistent-TMEM hang.

**Root cause.** The no-cut branch emits a whole-sequence item unconditionally.

**Fix / safety.** Compile-time `skip_empty=False` threaded through launch and compile cache; zero
chunks omitted only when opted in (split-forward hosts without a final state). State/cotangent-
writing callers keep empty items. Distinct from 03's unsplit compaction.

## Draft issue 05 — Scalar split scan ignores the head stride

**Symptom.** A gate view in alternating columns of a poisoned `[4096,4]` tensor produces wrong
per-head decay sums (logical heads −0.1 and −0.2; unused columns +37).

**Root cause.** The scalar load adds bare `h` rather than `h * stride[1]`; the standalone host also
marks the last dimension unit-stride, blocking legitimate non-unit-head-stride replay.

**Repro.** `--raw`: stock 256/516 entries differ, max abs error 4.6166239; patched exact. Default:
stock compact→strided replay raises a TVM-FFI stride mismatch; patched passes.

**Fix.** Widen before multiplying by the actual head stride; explicit symbolic-stride fake signature
for scalar gates. `mark_layout_dynamic(leading_dim=None)` is not enough: it infers stride 1 from a
compact first call.

## Draft issue 07 — Initialize each GDN CTA mbarrier from one thread

**Hardening, not a reproduced deadlock.** GDN prefill, recompute, bprop, bprop-summary and forward
summary initialize mbarriers from every CTA thread; each sync object should have one initializer.
`repro_07_mbarrier_stress.py` (32 sequences × 128 tokens, H=32, D=128, seeded state, backward; 1024
items across persistent waves) passes 32 iterations on both stock and patched.

**Fix.** Wrap each init inventory in `if tidx == 0`, keeping the init fence and CTA sync outside. No
counts or steady-state handshakes change.

## Draft issue 08 — Derive scheduler consumer arrivals from the active warp roles

**Maintenance hardening, not a current bug.** Several GDN/KDA barriers use literal 11/15 arrivals
though the consumer count follows the role map. For fully occupied role maps the patch derives CTA
warps minus the TMA publisher; KDA bprop-summary derives two compute groups plus three scalar
consumer roles (warps 8–11 idle), so it stays 11. `repro_08_scheduler_counts.py` passes on stock and
patched. AG's broader commits produced identical default-kernel SASS (8 KDA backward cubins).

## Excluded — split-table replay absent-scheduler ABI (B14)

AG guards a `None` scheduler slot with `r.has_sched` ("Preserve the scheduler ABI when replaying
split tables"). Upstream `build_split_table` converts `scheduler_counter` through `from_dlpack`
unconditionally and every caller passes a tensor. `repro_09_replay_abi_scope.py` shows `None` failing
at build time (`AttributeError: 'NoneType' object has no attribute '__dlpack__'`) while tensor
build+replay succeeds. Supporting it upstream would be a feature request, so nothing is proposed.
