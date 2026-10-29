# Numerical Contract for Linear-Attention Variants

The repository standard comes from #348 (precision and state dtypes), #446 and #450 (numerical
coverage and contracts), #464 (FP32 state contract), #465 (keep residuals FP32 until the MMA
pack), #572, #616, and #619, and from the numerics reviews of the Mamba2 integration that lands
with this skill.

## 1. State the contract

Document, in the variant's `docs/linear.md` section and the numerics test module docstring:

- **FP32:** the recurrent state carry and its cotangent, decays (log2 domain), and every
  intermediate that feeds a parameter-gradient reduction or a nonlinear epilogue.
- **Activation dtype:** chunk-entry states, state-cotangent tapes, and MMA operands. Name every
  such buffer. Do not describe a buffer as FP32 if it is stored in the activation dtype.
- **FP16 consequences:** states beyond about 65504 or cotangents below `2^-14` lose range when a
  chunk reads them.
- **Safe domains:** for example, in-chunk sums of log decays whose small increments vanish below
  half an FP32 ulp of the running sum.
- **Storage choices:** for example, BF16 paged pools rounding once per call (every decode token).
- **Boundary conventions:** for example, zero gradient at clamp bounds, like `Tensor.clamp`.

## 2. Oracles

- Primary: the FP64 token-by-token recurrence on the same rounded inputs. It is independent of
  the chunked decomposition and has no in-chunk cancellation.
- Secondary: the FP64 chunked reference, which checks the chunk algebra only.
- Pointwise envelope: the FP32 eager recurrence, as the low-precision reference of
  `attn_gym.testing.kda.assert_matches_low_precision_reference`.

## 3. Assertions

- Pair the pointwise envelope with `assert_relative_rms_within` in source-dtype epsilon
  (`test/AGENTS.md`). Set limits from measured healthy maxima with about 1.5-2x headroom, and
  record the measurement and its location in a comment.
- Bound cancelling reductions (`dA`, `ddt_bias`, scalar-parameter sums) relative to the summed
  magnitudes of their per-token terms; a max-relative check fails on well-conditioned kernels.
- Require exact zeros where the mathematics gives them (null routes, masked tails).
- Check the output, the final state, and every gradient, including the initial state and all
  parameters.

## 4. Cases

- Model range: dense, packed (empty, length-one, and partial-chunk sequences), forced
  split or segment schedules (monkeypatch the planner), and long weak-decay memory.
- Extreme inputs compared by value, never only `isfinite`: softplus tails, huge cotangents or
  writes on one token, steep decay, clamp bounds and their `nextafter` neighbors.
- FP16 range limits as `xfail(strict=True, raises=AssertionError)` paired with passing BF16
  controls.

## 5. Failure classes and their stable forms

| Unstable form | Symptom | Stable form |
|---|---|---|
| `log(1 + exp(u))` | step sizes below `u = -16` become 0 | `log1p(exp(u))` above a threshold of 20 |
| separable `exp(+g) * exp(-g)` | overflow, `0 * inf` | masked pairwise `exp2(g_i - g_j)` (#259) |
| `rowsum(Q) - colsum(Q)` then reverse cumsum | large diagonal terms swamp small ones | sum the pairs `j < k <= i` that cross each increment |
| `-w_j + [i == L] * sum(w)` then reverse cumsum | sign-reversed decay gradient | exclusive prefix `sum_{j < k} w_j` |
| narrowing before a subtraction | residual rounds away (#465) | stay FP32 until the MMA pack |
| `exp2` of a combined decay, then scale | underflow to 0 of a representable product | split into factors no smaller than `2^-126` |
| storing an LP intermediate, then a nonlinearity | double rounding, FP16 overflow | keep it FP32 or apply the epilogue to the accumulator |
| LP cotangent feeding a parameter reduction | `dD`-style sums lose precision | reduce from the FP32 value |
| inclusive clamp mask in the backward | gradient at the bound disagrees with eager | strict masks, as `Tensor.clamp` |
| cumulative-sum differences inside a chunk | small decays vanish next to a huge one | document the safe domain |

## 6. Process

- Each numerical fix needs a regression shown failing on the previous commit. Say how many
  targeted cases fail; controls and xfails are expected to behave as before.
- Compare against the upstream implementations the variant claims parity with (for example
  `mamba_ssm`, FLA, cuDNN), on the same harness, per tensor. Upstream parity does not override
  this contract: cuDNN v1.32 Mamba2 shipped the unstable softplus and a steep-decay backward bug.
- Run a numerics-focused review by a separate reviewer with this file as its brief, and re-run it
  after fixes; on the Mamba2 kernels the first pass found six defects and the second found one
  more plus an inaccurate precision claim in the docs.
