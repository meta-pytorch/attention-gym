# Test Suite Conventions

## Shared test helpers live in `attn_gym/testing/`

Before writing input factories, offset builders, reference oracles, or tolerance
assertions in a test file, check `attn_gym/testing/` (e.g. `attn_gym/testing/kda.py`)
for an existing helper:

- `cumulative_sequence_offsets(lengths)` — packed `cu_seqlens` boundary tensors
- `make_kda_test_inputs(tokens, ...)` / `clone_kda_inputs(...)` — public KDA operands
  with production dtypes and independent autograd leaves
- `assert_matches_low_precision_reference(...)` — data-derived pointwise error budgets
- `assert_relative_rms_within(...)` — aggregate error budgets measured in source-dtype epsilon
- fp64 backward oracles (`bwd_intra_reference`, `bwd_wy_dqkg_reference`, ...)

When a helper you are about to write would have a second caller — or duplicates the
shape/dtype/seed conventions of an existing one — add or extend it in
`attn_gym/testing/` instead of keeping a private copy in the test file. Local `_inputs`
style helpers are fine only for operand sets no shared factory covers. Numerical kernel
contracts should pair a pointwise maximum-error check with an aggregate relative-RMS check;
either metric alone can hide a different class of regression.
