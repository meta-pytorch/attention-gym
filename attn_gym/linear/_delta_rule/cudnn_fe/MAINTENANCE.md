# Maintaining the vendored cudnn-frontend kernels

This package holds NVIDIA cudnn-frontend's scalar GDN and KDA linear-attention kernels
(`kernel/`, `common/`, `tile_dsl/`, vendored at **v1.30.0**) plus the Attention Gym drivers
(`gdn.py`, `kda.py`, `summary.py`, `plan.py`) and AG-authored helpers (`common/launch.py`,
`common/tvm_ffi.py`, `common/paged_state.py`). Tooling lives in
[`tools/cudnn_fe/`](../../../../tools/cudnn_fe/README.md). Read this file first when upgrading.
This file is the procedure and generated ledger index;
[`fixes.toml`](../../../../tools/cudnn_fe/fixes.toml) is the authoritative ledger.
[WORKLOG.md](WORKLOG.md) is the narrative per ledger ID: evidence, decisions, replay notes,
known limitations, lessons and the v1.30 verification baseline.

## Stack layering

An upgrade lands as a stack; each layer is reviewable on its own terms. v1.30 landed as the
tooling base PR → #606 vendor (A) → #607 restyle (B) → #608 integration (C) → #609 cleanup (D) →
#611 gates (E).

| Layer | Content | Acceptance |
|---|---|---|
| 0 tooling base | `tools/cudnn_fe/` the upgrade needs first: vendor, audit, codemod, fix ledger | tool tests |
| A verbatim | `vendor.py` output for the new tag: closure, license files, per-file notice | `vendor.py --verify <A>` passes; nothing routes to it |
| B restyle | mechanical storage/style changes (restyle rules below), one commit per kernel family | SASS gate: identical, or reviewed offset-only noise (see SASS lessons) |
| C behavior | drivers, fake-signature compiles, every ledger row that changes behavior, tests | guarding tests pass; bug-reversal audit; bench gate |
| D cleanup | prune unreached code and upstream-only knobs, ruff, notices | SASS gate for prunes; notices describe the final modification set |
| E gates | SASS, CUTracer and bench gate tooling, strict audit, this file | tool tests; `audit --strict` prints `CLEAN` |

Commit rules: one ledger item (or one kernel family for restyle) per commit; subject says what
changed, body says why and how it was validated (SASS result, fails-without evidence). Never mix
restyle and behavior in one commit: a SASS diff must be attributable. Keep the vendored files'
SPDX headers and append to (never replace) the per-file "Modified by Attention Gym" notice.
Commit with `git commit --only -- <paths>` when several agents share the checkout.

## Upgrade procedure

Run these commands from a clean, dedicated upgrade checkout with its own editable `.venv`;
never switch branches in a checkout where another agent is working. Set `<old-drop-commit>` to
the previous **verbatim vendor commit**, not the previous integration tip. For v1.30 it is the
#606 commit "Vendor the cudnn-frontend v1.30 GDN and KDA kernels verbatim"; hashes change on every
rebase and squash merge, so `--verify auto` finds it by that subject (`vendor.DROP_SUBJECT`), and
`git log --format=%H --fixed-strings --grep='<subject>'` finds any other drop. The merge below has
base = old drop, theirs = new drop, ours = current AG.

1. **Churn and reproducibility.** Before changing branches:
   ```bash
   python -m tools.cudnn_fe.vendor --upstream <clone> --diff-upstream v1.30.0 <new-tag>
   python -m tools.cudnn_fe.vendor --upstream <clone> --rev v1.30.0 --verify auto
   ```
   Review files added, changed and removed from the driver-import closure. The GDP d_v=64
   prologue fork is pruned automatically; unresolved `cudnn.*` imports fail closed.
2. **Three-way import (layer A into the existing AG layers).** Generate the new drop before
   switching to the old commit, where the current tooling may not exist. Replace the placeholder
   values below, and keep `drop` outside the tracked package:
   ```bash
   old_drop=<old-drop-commit>
   ours=$(git branch --show-current)
   package=attn_gym/linear/_delta_rule/cudnn_fe
   drop=$(mktemp -d /tmp/cudnn-new-drop.XXXXXX)
   python -m tools.cudnn_fe.vendor --upstream <clone> --rev <new-tag> --dest "$drop"
   git switch -c cudnn-new-verbatim "$old_drop"
   git rm -r "$package"
   mkdir -p "$package"
   cp -a "$drop/." "$package/"
   git add "$package"
   git commit -m "Vendor cudnn-frontend <new-tag> verbatim"
   git switch "$ours"
   git merge --no-commit --no-ff cudnn-new-verbatim
   git diff --name-only --diff-filter=U
   ```
   Merge conflicts are expected: preserve unchanged AG files and resolve changed ones against
   both versions, rather than replacing the current package with the new drop. For an unresolved
   file, `git show :1:<path>`, `git show :2:<path>`, and `git show :3:<path>` show base, ours and
   theirs. Review delete/modify conflicts and new closure files explicitly. Resolve behavior
   conflicts using step 5, then stage only resolved paths and commit the merge. Do not run the
   codemod on files still containing conflict markers.
3. **Mechanical restyle (layer B).** Assemble the surviving changed/conflicted Python paths
   relative to the package, e.g. `changed=(kernel/gdn_prefill_f16.py common/split_k.py)` in zsh.
   Leave unchanged files alone; helper resolution still needs the full candidate package:
   ```bash
   candidate=$PWD/attn_gym/linear/_delta_rule/cudnn_fe
   python -m tools.cudnn_fe.restyle --root "$candidate" --files "${changed[@]}" --check
   python -m tools.cudnn_fe.restyle --root "$candidate" --files "${changed[@]}" --write
   python -m tools.cudnn_fe.audit --root "$candidate" --files "${changed[@]}" --strict
   ```
   The codemod handles only proven mechanical forms; install the required AG helpers before
   rewriting, and resolve remaining findings manually. Allowlisted findings are skipped unless
   `--no-allowlist` is given. After the focused edits, run the full-package acceptance check:
   `python -m tools.cudnn_fe.audit --root "$candidate" --strict`. It must print `CLEAN`, including
   no stale entries in [`audit_allowlist.txt`](../../../../tools/cudnn_fe/audit_allowlist.txt).
4. **SASS gate.** Before and after each B commit run
   `python -m tools.cudnn_fe.sass snapshot <dir> [--tree <ref_tree>]` (every GDN/KDA cubin through
   the AG drivers), then `python -m tools.cudnn_fe.sass diff <ref> <new>`. Accept identical
   instruction text and resources. Treat `same-histogram` as a review requirement, not proof of
   equivalence; offset-only noise additionally needs unchanged mbarrier offsets, `DSMEM`, `REG`
   and `STACK`. Use the same version of the gate on both trees; see
   [sass/README.md](../../../../tools/cudnn_fe/sass/README.md).
5. **Port the behavior layer (C/D).** For upstream-changed hosts/configs that AG converted to
   frozen dataclasses and `@jit_cache`, port the upstream semantic delta into the AG host; do not
   restore upstream live-tensor compilation. Check each item explicitly:
   - New/changed config fields, defaults, derived SMEM cosizes, TMA bytes and launch geometry.
   - Static cache-key inputs, compile target and explicit O2; do not key on runtime shapes.
   - Fake tensor shapes/strides/dtypes/alignment, independent extents, optional slots and int64 ABI.
   - Runtime argument order, scheduler presence, buffer ownership/lifetime and validation before
     compile; preserve empty/paged routes and all existing AG driver plans.
   - Shape-reuse, forced-int64 and relevant ledger regression tests, then SASS and bench gates.

   First ask stock upstream which bugs it still has:
   `python -m tools.cudnn_fe.verify_fixes --gpu-run --upstream-python <env>/bin/python`
   runs every upstream repro ("bug still present" → keep the fix; "fixed upstream" → candidate
   to drop; "error" → update the repro for the new API before concluding). Walk `fixes.toml`
   top to bottom with the replay notes in [WORKLOG.md](WORKLOG.md). If upstream contains a fix,
   mark its `upstream_status` superseded and keep its test; otherwise reapply it and keep the
   commit subject. Then run
   `python -m tools.cudnn_fe.verify_fixes --gpu-run --tree . --upstream-python <env>/bin/python`.
   Every fix must report `pass` or `no pytest guard`; `MISSING` means a guarding test was renamed,
   so update `fixes.toml`. Compare with the [v1.30 baseline](WORKLOG.md#verification-baseline-v130).
6. **Bug-reversal audit.** For every WORKLOG entry with "Fails without = YES", revert the
   production hunk in a scratch tree and confirm the guarding test fails, then passes restored.
   Update its narrative evidence.
7. **CUTracer.** `gpu-run --timeout 900 auto -- python -m tools.cudnn_fe.cutracer.stress --family all --ref-dir <reference> --out <results>`
   records the bitwise oracle and runs the `random_delay` ladder (or `--mode deadlock`);
   `results.md` must start with `PASS`. Alternatively, omit the outer reservation and pass
   `--gpu-run` to reserve per attempt with no waiting (exit 75 stops the ladder). Never nest
   reservations. Any new hang follows the cute-kernel-hang-debug skill; see
   [cutracer/README.md](../../../../tools/cudnn_fe/cutracer/README.md).
8. **Bench gate.** `python -m tools.cudnn_fe.bench run --suite gdn kda summary paged --new cand=. --base prev=<prev-tip-tree> --out <results>`
   compares forward, backward, summaries and paged paths with the previous stack tip. It fails
   above the 2% default threshold, on output drift or on a changed launch set;
   `--threshold 0.01` selects a stricter run
   (cells within ~1.3% have been noise on GB200). Bisect regressions; see
   [bench/README.md](../../../../tools/cudnn_fe/bench/README.md).
9. Update the tag everywhere (`NOTICE.md`, notices, this file), `fixes.toml`, then run
   `python -m tools.cudnn_fe.verify_fixes --write-ledger` to regenerate the index below
   (`test/test_cudnn_fe_tools_fixes.py` checks it is in sync). Update WORKLOG.md with findings
   and the new verification baseline. Refresh draft upstream patches for bugs still present
   (`git apply --check` against a scratch checkout of the new tag); filing upstream is a
   separate, user-approved action.

## Restyle rules

- Register arrays become `cute.make_rmem_tensor`; SMEM data buffers move into a
  `@cute.struct SharedStorage` allocated with `SmemAllocator` and read through `smem_data_ptr` /
  `SmemTile`; inline swizzle math becomes `swizzle_box_offset_{128b,32b}` only on exact match.
- mbarrier/commit/packed-fp32 primitives use `cute.arch` wrappers, except `fmul2`/`ffma2`, which
  stay inline PTX (the wrappers changed 35 cubins). Keep upstream `try_wait`/`spin` waits.
- Configs are frozen dataclasses; `build_cfg` returns `replace(...)` with derived SMEM cosizes
  and TMA byte counts. Hosts compile through module-level `@jit_cache` functions over fake
  TVM-FFI signatures (`common/tvm_ffi.py`) and launch with live tensors.
- **SASS lesson 1, barrier-block placement.** Keep upstream's SMEM order: tiles first, then the
  mbarrier/scheduler/gate-staging arrays. Moving the arrays to the start of dynamic SMEM kept the
  instruction stream identical but cost gdn_prefill ~2.8% (3.7x shared-load bank conflicts);
  "Restore the v1.30 SMEM order in the GDN prefill, tinv and summary kernels" restored the order
  with `LeadStorage`/`TailStorage` around the arrays. The SASS gate
  alone does not catch this, so the bench gate is mandatory for storage changes.
- **SASS lesson 2, fake signatures must match the `mark_layout_dynamic` ABI.** Upstream compiled
  from live tensors marked `mark_layout_dynamic(leading_dim=rank-1)`: int32 shapes and int64
  outer strides with divisibility 1 (`make_dynamic_signature_tensor`). int64 shapes changed
  register allocation (REG 52->46); promising aligned strides grew a kernel from 200 to 512
  instructions. The wide variant (`use_int64_offsets`) is an explicit, separately selected ABI.
- Compile at **opt level 2** (`OPT_LEVEL = 2`), as upstream does; O3 was mixed (+1.3%/-4.0%).
- Barrier init is elected (thread 0) plus fence and sync; never all-thread init.

## Ledger: Attention Gym modifications on top of v1.30.0

The 40 modification rows below are generated from
[`fixes.toml`](../../../../tools/cudnn_fe/fixes.toml), the single source for titles, commit
subjects, guarding node IDs, gates and upstream status. Regenerate with
`python -m tools.cudnn_fe.verify_fixes --write-ledger`; do not edit this table by hand.
[WORKLOG.md](WORKLOG.md) records the replay notes and fails-without evidence.

<!-- BEGIN GENERATED FIX LEDGER -->
<!-- Generated from tools/cudnn_fe/fixes.toml: 40 rows. -->
| ID | Title | Guarding test / gate |
|---|---|---|
| B7 | GDN bprop dBeta computed beta-free instead of rowsum/(beta+eps) (reverted; tests xfail) | `test/gdn/cudnn/test_gdn_cudnn_backward.py::test_gdn_cudnn_backward_preserves_small_beta_gradient`; `test/gdn/cudnn/test_gdn_cudnn_backward.py::test_gdn_cudnn_backward_mixed_small_beta_matches_reference` |
| B6 | KDA delta residual kept in FP32 through subtraction and beta scaling (fwd + bwd) | `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_forward_plans_keep_delta_residual_in_fp32`; `test/kda/cudnn/test_delta_rule_numerics.py::test_kda_cudnn_delta_residual_keeps_fp32_precision` |
| B8 | Compact empty cu_seqlens intervals out of the unsplit work table | `test/gdn/cudnn/test_gdn_cudnn_training.py::test_gdn_cudnn_padding_is_bitwise_and_emits_no_empty_work`; `test/kda/cudnn/test_kda_cudnn_v130.py::test_cudnn_compaction_keeps_empty_state_cotangents` |
| B10 | Split-table walk omits zero-chunk sequences | `test/test_cudnn_fe_common.py::test_split_table_omits_zero_chunk_sequences` |
| B13 | Scalar-gate split scan uses the head stride instead of assuming 1 | `test/test_cudnn_fe_common.py::test_scalar_split_scan_respects_head_stride` |
| B14 | Split-table replay passes sched_ctr only when compiled with it | `test/test_cudnn_fe_common.py::test_split_table_replay_preserves_absent_scheduler_abi` |
| E12 | get_dtype matches exact dtype names, not substrings | `test/test_cudnn_fe_common.py::test_cudnn_dtype_names_are_exact` |
| B15 | Driver launch caches keyed on static config, not shape | `test/gdn/cudnn/test_gdn_cudnn_training.py::test_gdn_cudnn_changing_batch_shape_in_one_process_matches_default`; `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_changing_shapes_reuses_only_static_configuration` |
| B16 | Copy 4-byte-aligned gate/beta to 16-byte bases | `test/gdn/cudnn/test_gdn_cudnn_layouts.py::test_gdn_cudnn_accepts_four_byte_aligned_gate_and_beta` |
| R15 | Composed KDA stateful bwd gets a compact 128-byte-aligned beta; traceable under Dynamo | `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_stateful_backward_accepts_forward_beta_layouts`; `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_training_fullgraph_and_six_gradients` |
| R3 | Warmup forward state_in/state_out get independent fake extents | none (no dedicated test; covered indirectly by GDN warmup split tests) |
| R4 | Restore kda_recompute int64 selector; stop forcing selectors on GDN modules that no longer launch | `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_plans_match_reference`; `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_forced_int64_forward_backward_matches_int32`; `test/gdn/cudnn/test_gdn_cudnn_layouts.py::test_gdn_cudnn_forced_int64_forward_backward_matches_int32` |
| F1/F2/F4/F6/F9 | GDN/KDA drivers (uncut, d_v split, prep, chain, warmup split, staged stateful bwd, AG plan floors) | `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_plans_match_reference` |
| F7/F8 | Native CP forward summaries [B;A] and reverse maps [C;R] for arbitrary bounds | `test/kda/cudnn/test_kda_cudnn_native_summary.py::test_native_selected_bounds` |
| F5 | Grouped q/k GDN backward through deterministic AG group_sum | `test/gdn/cudnn/test_gdn_cudnn_training.py::test_public_gdn_cudnn_grouped_h4_h12_forward_backward` |
| F3 | Paged recurrent state (null/fresh/resumed/empty routes) in the v1.30 GDN/KDA prefill | `test/test_cudnn_fe_paged_order.py::test_paged_order_keeps_only_nonempty_and_fresh_empty_routes`; `test/gdn/cudnn/test_gdn_cudnn_paged.py::test_cudnn_paged_negative_and_zero_routes_are_null`; `test/kda/cudnn/test_kda_cudnn_paged.py::test_cudnn_paged_negative_and_zero_routes_never_touch_the_pool` |
| R6 | Paged outputs past cu_seqlens[-1] documented as unspecified (zero-fill reverted for cost) | `test/gdn/cudnn/test_gdn_cudnn_paged.py::test_cudnn_paged_writes_every_token_up_to_the_last_interval`; `test/kda/cudnn/test_kda_cudnn_paged.py::test_cudnn_paged_writes_every_token_up_to_the_last_interval` |
| R1 | Restore upstream SMEM order (tiles before barriers) in GDN prefill/tinv/summary | bench (tools/cudnn_fe/bench; gdn_prefill main kernel vs previous tip) |
| R2 | compile_tvm_ffi takes opt_level; hosts keep O2 | `test/test_cute_cache.py::test_compile_tvm_ffi_adds_fake_stream_and_typed_option` |
| S8/S6/S3 | Prune unreached tile_dsl/common helpers; cute.arch wrappers; SmemTile tensor bases | sass (74/74 identical) |
| S1-S4 | Kernel storage restyle: rmem tensors, SharedStorage, smem_data_ptr, swizzle helpers, derived cosizes | `test/kda/cudnn/test_smem_swizzle.py::test_swizzle_box_offsets_match_original_layouts`; sass (identical or offset-only noise) |
| S5/S13 | Frozen cfgs; @jit_cache compiles over fake TVM-FFI signatures; compile key includes target | `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_changing_shapes_reuses_only_static_configuration`; `test/test_cute_cache.py::test_runtime_cache_includes_compile_target`; sass (byte-identical per family) |
| S15 | Remaining exact KDA prefill swizzle forms; common-helper rmem tensors; state-chain SharedStorage | sass (55/55 byte-identical SASS and resources) |
| R5 | Driver simplification: shared plan.py, no device/stream args, shared allocators, _GATE_FLAGS | `test/kda/cudnn/test_kda_cudnn_v130.py::test_v130_plans_match_reference`; sass + existing suites bitwise unchanged |
| S10 | Ruff lint/format the vendored kernels | sass (42/42 identical) |
| R16 | Rename sdq_reduction to v_term_scratch; dedupe residual_f16x2; host docstring | sass |
| S12 | Launch-contract validation before compile; scheduler arrivals derived from warp counts | `test/test_cudnn_fe_launch_contract.py::test_split_table_rejects_invalid_launch_geometry`; `test/test_cudnn_fe_launch_contract.py::test_warp_role_and_named_barrier_tables_reject_conflicts`; `test/test_cudnn_fe_launch_contract.py::test_gdn_backward_cfgs_reject_unsupported_geometry`; `test/test_cudnn_fe_launch_contract.py::test_gdn_bprop_host_rejects_invalid_launch_metadata`; `test/test_cudnn_fe_launch_contract.py::test_gdn_warmup_backward_rejects_invalid_plan_buffers`; `test/test_cudnn_fe_launch_contract.py::test_gdn_chain_backward_rejects_invalid_plan_buffers`; `test/test_cudnn_fe_launch_contract.py::test_kda_forward_cfgs_reject_unsupported_geometry`; `test/test_cudnn_fe_launch_contract.py::test_kda_warmup_forward_rejects_invalid_plan_buffers`; `test/test_cudnn_fe_launch_contract.py::test_kda_chain_forward_rejects_invalid_plan_buffers` |
| R7 | Reject misaligned per-channel gate rows before the vectorized split scan | `test/kda/cudnn/test_kda_cudnn_forward.py::test_cudnn_split_table_rejects_misaligned_vector_gate_rows` |
| B2 | Elect-one (thread 0) mbarrier init in GDN prefill/bprop/recompute/bprop_summary/summary | `test/gdn/cudnn/test_gdn_cudnn_training.py::test_public_gdn_cudnn_repeated_stateful_launches_cross_wave_boundary`; `test/gdn/cudnn/test_gdn_cudnn_training.py::test_public_gdn_cudnn_repeated_backward_crosses_wave_boundary`; cutracer random_delay (hardening; tests pass without the fix) |
| S14 | int64 ABI selectors, wide-extent GDN forward variant, non-vacuous forced-int64 tests | `test/gdn/cudnn/test_gdn_cudnn_layouts.py::test_gdn_cudnn_forced_int64_forward_backward_matches_int32`; `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_forced_int64_forward_backward_matches_int32`; `test/gdn/cudnn/test_gdn_cudnn_layouts.py::test_gdn_cudnn_oversized_singleton_stride_executes_int64_path` |
| R10 | Delete the legacy 085d50b kernel copy, schedule.py, kda_plain_gate_bwd | `test/test_cudnn_optional_import.py::test_linear_import_does_not_load_cudnn_kernel_dependencies` |
| R11 | Drop the GDP d_v=64 bprop fork from the GDN chain prologue | vendor.py applies the GDP fork cut automatically; import test |
| R13 | Delete unreached/test-only upstream code (gate_bwd, head_reduce, l2norm, standalone hosts, ...) | reachability (package import + full suite) |
| R14 | Prune upstream-only constexpr knobs AG never sets | sass (51/51 identical) |
| R12 | Accurate NOTICE.md and per-file modification notices | review (notices list every ledger row touching the file) |
| B3 | Test: native KDA stateful bwd past sort capacity with zero-chunk items (fix is upstream) | `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_backward_past_sort_capacity_runs_empty_work_items` |
| B9 | Test: contracting-gate split test pins the uncut plan so an ignored split fails | `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_split_forward_matches_reference_on_a_contracting_gate` |
| B17 | Test: GDN backward rejects HQ > HV before dispatch | `test/gdn/cudnn/test_gdn_cudnn_backward.py::test_gdn_backward_rejects_more_query_than_value_heads` |
| R8 | Sharded KDA dbeta bounded by the operand-pack budget | `test/test_delta_rule_stages.py::test_simulated_context_parallel_matches_unsharded_op` |
| R9 | Tighten tests: cache_info reuse, match= on raises, stress rename | `test/kda/cudnn/test_kda_cudnn_training.py::test_cudnn_repeated_stateful_backward_with_empty_sequences_stress` |
<!-- END GENERATED FIX LEDGER -->

Superseded by v1.30 (no AG commit; tests kept): B1 seeded-state wait, B5 KDA FP32 factors,
B11 terminal TMA overfetch, B12 checkpoint descriptor layout, S11 V-major state. Inherited from
main: B4 dO dtype check. Their full test pointers are in `fixes.toml`. Decisions: S7 keep upstream
waits; F11 keep upstream's 4 KQ SMEM stages.

Fails-without evidence was collected on the pre-reorder branch (audit HEAD "[C] Validate the KDA
backward warp-role maps and derive the scheduler arrival count") and has not been re-run on the
final stack; re-run it in step 6 of the next upgrade.
