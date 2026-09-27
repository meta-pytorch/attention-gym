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
| 3 | `03-compact-empty-unsplit-work.patch` | B8 | **Yes:** padded empty intervals stay in the work table; 8/20 work-count checks fail | 20/20 checks, incl. exact empty final-state/cotangent preservation |
| 6 | `04-omit-zero-chunk-split-work.patch` | B10 | **Yes:** 8 items instead of 4 (not a reproduced TMEM hang) | Opted-in table has 4 items, all-empty 0; stateful default unchanged |

## Running the repros

| Script | Arguments | Stock v1.30 | Patched |
|---|---|---|---|
| `repro_03_empty_unsplit.py` | optional export dir holding `python/cudnn` (defaults to the installed package) | 8/20 fail | 20/20 |
| `repro_04_zero_chunk_walk.py` | `--skip-empty` on patched (opt-in) | fails (8 vs 4) | pass with `--skip-empty` only |

## Source/history audit

Translated from these AG commits (by subject; see `../fixes.toml`), **without AG restyling or
dependencies**:

- 03 (B8): PR #603 and its v1.30 replay in "Route unpaged GDN and KDA and native CP summaries to the
  v1.30 kernels"; adapted conservatively to upstream state ownership.
- 04 (B10): "Omit zero-chunk split work items"; adds an upstream-specific safe opt-in.

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
