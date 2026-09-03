# Linear Attention

Attention Gym provides functional linear-attention operators with eager reference implementations.

## Gated Delta Rule

`chunk_gdn` and `recurrent_gdn` use the token-major layout
`[batch, sequence, heads, dimension]` and return an output/state tuple. For each token, the scalar
natural-log gate decays the previous state before the delta update, and the query reads the updated
state:

```text
decayed_state = exp(gate) * state
residual = beta * (value - decayed_state @ key)
state = decayed_state + outer(residual, key)
output = scale * state @ query
```

Public and persistent state uses axis order `[N, H, V, K]`; paged pools replace `N` with the
slot count. `chunk_gdn` uses a chunk-parallel decomposition for training and prefill.
`recurrent_gdn` consumes
tokens in order for decoding, inference prefill, and state-carrying correctness checks. `chunk_gdn`
defaults to eager PyTorch (`impl="reference"`); `chunk_gdn(..., impl="fused")` selects the
repo-local scalar chunk pipeline. Pass `kernel_options={"backend": "mega"}` to select the optional
Mega CuTeDSL backend. `recurrent_gdn(..., impl="fused")` selects the inference-only Triton scan.

```python
from attn_gym.linear import chunk_gdn

output, final_state = chunk_gdn(
    query,
    key,
    value,
    gate,
    beta,
    impl="fused",
    output_final_state=True,
)
```

### Supported capabilities

- Fixed-length inputs in `[batch, sequence, heads, dimension]` layout and packed batch-one inputs
  with `cu_seqlens`.
- Separate recurrent and chunked operations with explicit initial and final state.
- CPU and CUDA execution through eager PyTorch operations.
- A repo-local fused chunk pipeline on CUDA capability 8.0+ with dense and packed inputs, grouped
  Q/K heads, tails, empty sequences, initial/final state, strict `torch.compile`, and CUDA Graph
  support. On CUDA capability 8.0+, `paged_chunk_gdn` advances selected
  `[num_slots, H, V, K]` cache rows in place for inference prefill without caller-side
  gather/scatter copies.
- An opt-in Mega fused chunk implementation with the same public training/state contract and
  direct paged-cache prefill on SM100/SM103.
- An inference-only fused recurrent implementation on CUDA, including mutable paged state caches
  shaped `[num_slots, H, V, K]` selected by `state_indices`.
- Autograd for the reference and fused chunk implementations, including initial-state gradients.
- Q/K/V share one dtype. FP16 and BF16 inputs use FP32 recurrence math and state while returning
  output in the Q dtype.
- Gate and beta may use independent floating dtypes and are converted to the recurrence compute
  dtype. A provided initial state uses FP32 for low-precision QKV.
- FP64 reference inputs retain FP64 recurrence math and state.

!!! note "What the fused paths keep in FP32"

    Every optimized GDN and KDA path (repo-local fused and Mega) carries the recurrent state
    from `initial_state` to `final_state`, and the state cotangent from `d_final_state` to
    `d_initial_state`, in FP32 accumulators with FP32 decay. The chunk-entry state that each
    chunk reads, the per-chunk state checkpoints, and the per-chunk state-cotangent tape are
    staged in the Q/K/V dtype because they are tensor-core operands. BF16 shares the FP32
    exponent range, so this only costs mantissa precision. FP16 also limits the range: state
    values that round beyond the FP16 finite range (about 65504) become non-finite when a chunk
    reads them, and cotangent values in the FP16 subnormal range (below `2^-14`) lose precision
    or round to zero before they reach `dgate`, `dk`, and `dv`. Keep FP16 states and state
    cotangents within the FP16 normal range, for example through loss scaling, or use BF16 when
    state magnitudes can leave it. Use L2-normalized Q/K with FP16, as the training example does:
    unnormalized Q/K or unusually large V can likewise push attention, solve, and value
    intermediates outside the FP16 range between FP32-accumulating GEMMs. The Mega KDA backend
    additionally applies the per-chunk decay to the carried state, and to the state cotangent of
    its no-state local backward, through a diagonal MMA in the Q/K/V dtype rather than in FP32.

The repo-local fused chunk backend requires CUDA capability 8.0+, matching FP16 or BF16 Q/K/V,
and
`K = V = 128`, but has no Mega runtime dependency. Its scalar natural-log gate is not lower-bounded:
kernels contract raw QK/KK before applying masked nonpositive causal decay differences. The public
chunk size is BT64; internal 16x16 blocks are only the hierarchical triangular-solve representation.
Layouts requiring int64 tensor offsets are rejected until every repo-local kernel has a
wide-address path. Backward uses the tuned CuTe kernels on SM100/SM103 and portable Triton
kernels on Ampere, Hopper, and other supported architectures.

The Mega chunk backend requires the optional `mega` dependencies and SM100/SM103,
FP16/BF16 Q/K/V, FP32 state, and `K = V = 128` contract. It also consumes the scalar natural-log
gate without a lower bound and uses exact execution without an approximate forgetting-horizon
split.

The fused recurrent implementation requires CUDA with Triton, Q/K/V in FP16, BF16, or FP32, and
`K <= 256`. Packed offsets must begin at zero, be nondecreasing, and end within the physical token
capacity. Output rows beyond the terminal offset are inactive capacity and unspecified. Unsupported
implementation choices fail rather than falling back to the reference.

### Migration from the prototype API

- `naive_recurrent_gated_delta_rule(...)` and `gated_delta_rule(..., mode="recurrent")` become
  `recurrent_gdn(...)`.
- `naive_chunk_gated_delta_rule(...)` and `gated_delta_rule(..., mode="chunked")` become
  `chunk_gdn(...)`.
- Inputs use `[batch, sequence, heads, dimension]`, matching the KDA operations; keyword callers
  use `q`, `k`, and `v`.
- The public chunk size is fixed at 64, matching KDA's fused decomposition.
- `initial_state` may be positional, and `output_final_state` controls the optional state output.
- Both operations return `(output, final_state)`, matching the KDA operations.

`recurrent_gdn_decode` is the serving-specific one-token path, mirroring
`recurrent_kda_decode`: it consumes the packed post-convolution QKV buffer plus raw gate and
beta projections, computes the gate transform (`-exp(A_log) * softplus(raw + dt_bias)`), the
beta sigmoid, and the query/key L2 normalization in-kernel, supports grouped q/k heads, and
advances the paged FP32 state pool in place through `state_indices`, so no separate
elementwise kernels run per decode step.

::: attn_gym.linear.chunk_gdn

::: attn_gym.linear.paged_chunk_gdn

::: attn_gym.linear.recurrent_gdn

::: attn_gym.linear.recurrent_gdn_decode

## Stateful Short Convolution

`causal_conv1d` is the differentiable dense or packed operation. It accepts compact
per-sequence history and can return a newly allocated final history. Serving callers can keep
history in a shared `[num_slots, W - 1, C]` pool instead: `paged_causal_conv1d` reads selected
slots during multi-token prefill and advances them in place, while `causal_conv1d_decode` does
the same for one token per sequence. Both are inference-only and support padding between slots.
Positive `state_indices` must be unique and within the pool; non-positive routes return zero without
touching it. `has_initial_state=False` gives paged prefill a zero history before it overwrites a
fresh slot.

::: attn_gym.linear.causal_conv1d

::: attn_gym.linear.paged_causal_conv1d

::: attn_gym.linear.causal_conv1d_decode

## Kimi Delta Attention

The KDA references use token-major tensors: query, key, and per-channel gate are
`[batch, sequence, heads, key_dimension]`; value is
`[batch, sequence, heads, value_dimension]`; beta is
`[batch, sequence, heads]`. Both recurrent and chunked forms support ordinary
PyTorch autograd and an optional recurrent state shaped `[N, H, V, K]`. Paged prefill and decode
use the same axis order with the leading dimension interpreted as cache slots.

`examples/kda_training.py` builds these operations into a small trainable
`[B, T, hidden_size] -> [B, T, hidden_size]` attention module. To mirror the
main Kimi block structure, projected Q/K/V pass through a causal depthwise SiLU
convolution, the forget and output gates use two-stage factorized projections,
and the per-head output uses learned RMS normalization before sigmoid gating.
`--backend=reference` uses the PyTorch reference throughout. The optimized `chunk_kda`
core used by `--backend=fused` supports Ampere or newer and owns the BT64 scan behind the same
public boundary. The complete fused example currently requires Hopper because its separate
`bound_gate` producer uses TMA. Implementations may inline the scan, but cumulative gates and
chunk boundaries are not caller-visible representations.

The example explicitly runs projections in BF16 while retaining FP32 parameters and gate
math; no ambient autocast context is required. Distributed mixed-precision policies must
preserve `A_log` and `dt_bias` as FP32 when parameters are materialized, rather than only
casting activations inside `forward`. A module-wide BF16 FSDP policy violates that
contract. A correctness-first integration can keep the KDA unit under an FP32 policy while
its projections explicitly compute in BF16; isolating only the strict-FP32 decay state is
a future bandwidth optimization.
The optimized `chunk_kda` core requires Ampere or newer and `head_dim=128`; its public
boundary accepts FP16, BF16, or FP32 inputs. SM100/SM103 select the specialized CuTe path;
SM120 and earlier supported architectures use the portable Triton stages. Homogeneous FP16 and
BF16 Q/K/V stay in their input dtype, while
FP32 or mixed-dtype inputs retain the existing BF16 normalization. The core chunks internally at
64 tokens. Complete `B=1` inputs whose length is a multiple of the chunk size run on the
direct dense route; other dense `[B, T, H, D]` inputs are lowered internally to
equal-length packed sequences, while `chunk_kda(..., cu_seqlens=offsets)` accepts
explicitly packed `[1, T, H, D]` inputs. All forms carry sequence boundaries through the
forward, backward, and recurrent states; logical sequences may have tails or be empty. For
fixed-capacity execution, the terminal offset may be smaller than physical `T`; primitive
forward values outside `[0, cu_seqlens[-1])` are unspecified. `paged_chunk_kda` uses the same
Ampere-or-newer forward route while updating selected FP32 state-cache slots in place. The
internal reverse scan returns zero cotangents for inactive gate rows. This does not sanitize arbitrary
parameterized gate producers: callers still need the masking rules below because
``0 * NaN`` can poison their reductions.

!!! warning "FP16 intermediate range"

    Use L2-normalized Q/K with FP16, as the training example does, and keep states and state
    cotangents within the FP16 normal range. See "What the fused paths keep in FP32" in the
    Gated Delta Rule section for the boundary between the FP32 carry and the Q/K/V-dtype staging
    that this shares with GDN.

A captured graph with sequence capacity `N` keeps `cu_seqlens.shape == (N + 1,)`. If a
replay has `M <= N` nonempty sequences and `L <= T` active tokens, repeat the terminal
endpoint through the unused tail:

```text
[0, sequence_start_1, ..., L, L, ..., L]
```

The repeated ranges are ordinary empty sequences. Stateful APIs therefore retain `N`
state rows even when only `M` sequences are nonempty. Both `L` and `M` may change on
replay, but physical token capacity `T` and metadata capacity `N` may not.

Scheduling for this over-capture regime is automatic and selected independently for
each eligible chunk-parallel ragged kernel: a kernel switches to a bounded persistent
worker grid only when its capacity task count exceeds a few waves of that grid.
Persistent workers stride over the active chunk count built on device from
`cu_seqlens`, so chunk-compute launch overhead tracks active work rather than captured
capacity. Capacity-sized initialization and reduction work may remain. Exact or mildly
padded shapes keep capacity-sized STATIC grids whose padding CTAs return immediately; there is no
user-facing scheduling knob.

Callers that use dynamic active lengths within fixed-capacity tensors must opt in to
masking. Ragged primitives read only `[0, cu_seqlens[-1])` from token-shaped inputs,
including output cotangents, and leave the suffix of token-shaped outputs and input
gradients unspecified. Four edge rules make those primitives safe to compose:

- **Caller buffer → ordinary operation:** value-mask the buffer. A zero cotangent does
  not neutralize a NaN activation in a weight reduction: `0 * NaN` is still NaN.
- **Parameterized producer → ragged primitive:** add a gradient barrier so the
  primitive's unspecified input-gradient suffix cannot enter the producer's reduction.
- **Ragged primitive → ordinary operation:** value-mask the primitive output before the
  ordinary operation saves it for backward.
- **Ragged primitive → ragged primitive:** do nothing; neither primitive reads the
  inactive suffix.

Construct one device-resident predicate inside the captured graph and reuse it at every
boundary. This keeps all masks consistent and lets replay recompute them when
`cu_seqlens[-1]` changes without a host read or recapture.

```python
from attn_gym.linear.kda import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

active_mask = active_token_mask(hidden, cu_seqlens)
hidden = mask_inactive_tokens(hidden, active_mask)  # Caller → ordinary projection.
projected = input_projection(hidden)
projected = mask_inactive_token_gradients(projected, active_mask)
stage = ragged_primitive_one(projected, cu_seqlens=cu_seqlens)
output = ragged_primitive_two(stage, cu_seqlens=cu_seqlens)  # No mask between primitives.
output = mask_inactive_tokens(output, active_mask)
output = output_projection(output)
output = mask_inactive_token_gradients(output, active_mask)  # Model boundary.
```

`mask_inactive_token_gradients(x, active_mask)` preserves forward values while zeroing
inactive automatic-differentiation paths. For contiguous packed CUDA tensors in eager
mode its autograd path aliases `x` in the forward and row-masks tangents and
cotangents; compiled graphs and unsupported layouts keep the
`torch.where(mask, x, x.detach())` form (which materializes an elementwise result) so
Inductor can own fusion. Automatic-differentiation paths are zero on inactive rows,
and subsequent derivatives inherit the same mask. Recurrent
and convolution states have one row per logical sequence rather than one row per physical
token, so token masks must not be applied to them.

::: attn_gym.linear.kda.active_token_mask

::: attn_gym.linear.kda.mask_inactive_tokens

::: attn_gym.linear.kda.mask_inactive_token_gradients

`KDAAttention.forward` passes explicit offsets to its short convolution and `chunk_kda`;
the selected implementation owns its sequence-local scan and scheduling. Set
`mask_inactive_capacity=True` only when the packed tensor reserves physical rows beyond
`cu_seqlens[-1]`; dense and exact-packed
callers leave it disabled and pay no masking cost. The optimized boundaries are
first-order and do not support higher-order autograd. Run
`python examples/kda_training.py --backend=fused --packed --batch-size=4 --tokens=256`
to sample token-level lengths from a truncated Zipf distribution, pack them exactly
into one physical batch, print their `cu_seqlens`, and pass those offsets through the
complete training step. The complete composed core forward and backward use private custom
operators with fake-tensor registrations and first-order autograd wrappers, so fused
`chunk_kda` supports strict `torch.compile(fullgraph=True)` and CUDA Graph capture for
fixed physical token capacity and sequence count. Packed reference execution is
eager-only: it reads `cu_seqlens` on the host to run each logical sequence
independently. Boundary values and the active token count may change on replay;
changing the physical token capacity or sequence count
requires recompilation or recapture. Pass `--compile` to compile the complete example as
one full graph. This keeps the custom KDA core behind its registered operator boundary
while allowing
Inductor to fuse the surrounding PyTorch normalization and remaining pointwise work.
The bounded gate itself uses private CuTeDSL forward and backward operators. It can be
combined with `--profile`; compilation warmups run before the
trace starts. Like FLA's default training path, its backward recomputes the W/U,
gated Q/K, recurrent-state, and
corrected-value intermediates instead of retaining them across the forward/backward
boundary.

Graph-safe active-token replay does not by itself make complete model time proportional
to `L`. The ragged short convolution, gate scan, and KDA core avoid reading inactive
token values, but the example's projections, output normalization, output gate, and
output projection still process physical capacity `T`. An end-to-end integration needs
active-prefix-aware surrounding operations to turn smaller `L` into a comparable step-time
reduction. Million-token training also requires model-level activation checkpointing and
context parallelism; this single-device example implements neither distributed policy.

The module can sit behind a transformer layer's attention slot while state is
threaded explicitly:

```python
from examples.kda_training import KDAAttention

attention = KDAAttention(hidden_size=512, num_heads=4, head_dim=128).cuda()
first = attention(hidden_states[:, :128], return_final_state=True)
second = attention(
    hidden_states[:, 128:],
    first.final_state,
    initial_conv_state=first.final_conv_state,
    return_final_state=True,
)
```

The example intentionally is not checkpoint-compatible with Kimi K3. A model
adapter must still provide exact checkpoint parameter names and initialization,
the model's packed-sequence metadata, cache layout, and distributed execution policy. The
short convolution, factorized gates, and learned gated RMS normalization match
the production structure but remain ordinary PyTorch teaching implementations.

The two public KDA cores share one gate contract: `gate` is the per-token
natural-log decay before any prefix sum. `chunk_kda` owns the natural-log-to-log2
conversion and its sequence-local BT64 cumulative sum; `recurrent_kda` performs only the
conversion because recurrence consumes one token decay at a time. This keeps chunking out
of model code and lets callers switch execution modes without changing gate
representation. Custom producers should return finite, nonpositive values. The repo-local fused
backend requires approximately `[-5.914, 0]`. Mega BF16 instead requires every aligned 16-token
per-channel sum to exceed `-126 * ln(2)`; a uniform `lower_bound >= -5.45` is safe, including the
common `-5` bound. Mega stages these MMA operands in the Q/K dtype rather than TF32, so Mega FP16 does
not support usual model-range gates. These limits are not checked at runtime. The training example
uses the Kimi-style FP32 transform
`lower_bound * sigmoid(exp(A_log) * (raw_gate.float() + dt_bias))`, but that model policy is not
part of the public KDA API.

Both cores select their implementation with `impl`: `"fused"` runs the optimized kernels
and enforces their constraints (the chunked core requires `head_dim=128` and Ampere or newer,
preserves homogeneous FP16/BF16 Q/K/V, normalizes FP32 or mixed inputs to BF16, and chunks at
64 tokens; the fused recurrent scan is inference-only), while `"reference"`
runs the eager FP32 oracle behind the identical packed contract on any hardware and head
dimension, and stays differentiable. There is no automatic fallback between implementations, and
the chunk-versus-recurrent switch is caller policy (on B200 the scan wins below roughly 32 tokens
per sequence).

`recurrent_kda_decode` is the serving-specific one-token path. It consumes
channel-major post-convolution QKV (`[Q for all heads | K for all heads | V for all
heads]`), raw gate and beta projections, and a paged state cache. Q/K normalization,
gate activation, beta sigmoid, recurrence, output, and state-cache update run in one
Triton kernel. Callers may provide a stable output buffer for allocation-free CUDA
Graph replay.

`chunk_kda(..., kernel_options={"backend": "mega"})` selects an opt-in SM100/SM103
FP16/BF16 training backend for
applications that already hold per-token natural-log gate increments. Q/K/V share one dtype. It uses
public CuTeDSL 4.7 primitives and has no cuDNN Frontend runtime dependency. Like the other
implementations, it returns `(output, final_state)`; request the latter with
`output_final_state=True`. Dense no-state calls require T divisible by 64 and omit `cu_seqlens`,
while packed/stateful calls pass explicit contiguous int32 boundaries and FP32 `[N, H, V, K]`
states. Q/K/V/gate/state accept TMA-compatible innermost modes with aligned dynamic outer strides;
beta requires an element-aligned contiguous inner mode. By default the forward is exact and
unsplit: one persistent work item per sequence and head, which leaves the GPU underused for a few
long sequences at low head counts. FP16 and BF16 use exact backward execution by default; eligible
packed no-state long contexts may use the Mega backward with one unsplit work item per sequence and
head. Callers that guarantee normalized keys, post-sigmoid beta, and nonpositive decay increments may
set `split_backward` and/or `split_forward` to `True` in a Mega `kernel_options` mapping to opt into
approximate forgetting-horizon splitting: a sequence is cut only where its cumulative decay has
crossed the kernel's threshold, and each cut item rebuilds its entry state from zero over a short
warmup window. Gates that never forget produce no cuts and reproduce the unsplit result exactly; when
cuts are accepted the result stays within the low-precision error budget while exposing several
times the parallel work. The threshold is a margin of bits past the output dtype's half-ulp, not an
underflow; see `NOTE [Forgetting Horizon]` in
`attn_gym/linear/_delta_rule/mega/kernels/common/split_k.py`. `split_forward` affects only the
forward recurrence: unless `split_backward` is also enabled, backward computes the exact unsplit
recurrence's gradient rather than the derivative of the split forward. The implementation chooses
the split count from the input geometry; no public split-size knob is exposed. Split schedules
currently require a no-state call, so context parallelism never uses them. `paged_chunk_kda(..., kernel_options={"backend": "mega"})` uses the
same exact unsplit forward while updating selected cache slots directly; paged execution never uses
forgetting-horizon splitting. Install the `mega` extra to use this backend.

::: attn_gym.linear.chunk_kda

::: attn_gym.linear.paged_chunk_kda

::: attn_gym.linear.recurrent_kda

::: attn_gym.linear.recurrent_kda_decode

::: attn_gym.linear.Impl

## Context Parallelism

Every delta-rule token step is affine in the V-major recurrent state, so any token range
collapses to one FP32 map `H_out = H_in @ A + B`, packed as `[HV, V + K, K] = [bias; transition]`
(one map per value head; GQA key heads are expanded by the factor kernels).
Context parallelism is therefore a prefix scan over these summaries, and the same machinery serves
KDA and GDN (whose scalar per-head gate broadcasts onto the per-channel summary kernels). The
public surface has three tiers so that new partitionings or communication topologies never add
options to an op.

### Terminology and index spaces

Anything a plan is built from is **global**; anything that touches a tensor on a rank is
**local**. `attn_gym.linear.state_summary` carries the canonical `NOTE [Terminology]`.

| term | meaning | space |
|---|---|---|
| global stream | the whole packed token stream, `cu_seqlens_global` | global |
| sequence | one document in the global stream | global |
| fragment | one contiguous global token range a rank owns; may cover several sequences. A rank owns a list of them: one for a contiguous shard, two for zig-zag | global |
| subsequence | the tokens of one fragment that belong to one sequence; a fragment covering several sequences has several subsequences, and each becomes one `cu_seqlens` segment of the owner's span | global |
| span | the concatenation of a rank's fragments, in the order it listed them: the packed `q`/`k`/`v`/`output` tensors on that rank, with `cu_seqlens` marking its subsequence boundaries | local |
| chunk | the `attn_gym.linear.kda.stages.CHUNK_SIZE` (64) token block the fused kernels work in (WY factors and one state step per block); every subsequence is chunked on its own, starting at its first token, so its last chunk may be partial and no fragment cut needs to be chunk-aligned | local |
| summary | the `[bias; transition]` map of a token range of the span | local range |
| slot | `gathered[rank][i]`: the summary of that rank's `i`-th subsequence, or the identity | (rank, index) |
| predecessors / successors | subsequences of the same sequence earlier / later in global order | (rank, slot) |
| terminal | subsequences that end their sequence; true final states live there | local index |

Fragment cut points are arbitrary global tokens: chunking restarts at every subsequence, so they
need no alignment to sequences or chunks. With `cu_seqlens_global = (0, 40, 232, 384)` and the
zig-zag table `fragments_global = [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]`:

```text
global tokens   0     40             96       192   232      288      384
sequences       |- s0 -|------------ s1 --------------|-------- s2 --------|
rank 0           [====== frag A ======]                        [= frag B =]
rank 1                                 [ frag C ][= frag D ==]

rank 0 span:  A∩s0 | A∩s1 | B∩s2     local cu_seqlens (0, 40, 96, 192)
rank 1 span:  C∩s1 | D∩s1 | D∩s2     local cu_seqlens (0, 96, 136, 192)
chains:       s1 = A∩s1 -> C∩s1 -> D∩s1        s2 = D∩s2 -> B∩s2
```

`state_summary(start, stop)` takes **local** span offsets and is exact over whole chunks of one
subsequence: `start = sub_start + 64·i`, `stop = sub_start + 64·j` or the subsequence end, never
crossing a `cu_seqlens` boundary. The recipe always passes one whole subsequence,
`(cu_seqlens[i], cu_seqlens[i + 1])`.

**Primitives** (`attn_gym.linear.kda.chunk_kda_prepare` / `chunk_kda_prepare_backward`, and
`attn_gym.linear.gdn.chunk_gdn_prepare` / `chunk_gdn_prepare_backward`) split the fused core
around the communication point without exposing its WY factors:

```python
from attn_gym.linear.kda import chunk_kda_prepare, chunk_kda_prepare_backward

prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
summary = prepared.state_summary(start, stop)  # [bias; transition] of one local sequence
# ...exchange summaries and compose each sequence's entry state...
output, final_state = prepared.run(initial_state, output_final_state=True)

grads = chunk_kda_prepare_backward(prepared.saved, d_output, initial_state, scale=prepared.scale)
grad_summary = grads.state_grad_summary(start, stop)  # reverse map [bias; transition]
# ...exchange and compose each sequence's exit cotangent...
dq, dk, dv, dgate, dbeta, _ = grads.run(d_final_state)
```

`start`/`stop` are host integers naming exactly one local `cu_seqlens` segment, so CUDA Graph
capture never syncs. `attn_gym.linear.state_summary` holds the pure-PyTorch algebra on summaries
(`merge_state`, `compose_summaries`, `neutral_summary`).

**Ownership plans** (`attn_gym.linear.context_parallel.ContextParallelPlan.from_fragments`) take
every rank's fragments. You choose them in plain Python; the plan cuts each fragment at sequence
boundaries into `Subsequence`s, one span `cu_seqlens` segment each, and derives host-static routing:
which subsequences need a forward or reverse summary, which gathered slots to fold for each entry
state or exit cotangent, and which subsequences end their sequence and therefore hold true final
states. Two pieces of one document on the same rank are simply two span segments whose entry states
the routing supplies, so contiguous shards, zig-zag load balancing, and document-aligned partitions
differ only in the fragment lists. The fragments must tile `[0, total_tokens)` exactly once.
`summary_slots` / `grad_summary_slots` fill a rank's `[slots, ...]` buffer from a prepared handle,
and `compose_entry_states` / `compose_exit_cotangents` fold the gathered `[world, slots, ...]`
buffers into each subsequence's entry state or exit cotangent; the collective in between is the
caller's.
[`examples/kda_context_parallel.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/kda_context_parallel.py)
builds contiguous and zig-zag fragment lists in a dozen lines.

**Reference recipe** (`attn_gym.linear.context_parallel.context_parallel_chunk`, bound as
`attn_gym.linear.kda.context_parallel_kda` and `attn_gym.linear.gdn.context_parallel_gdn`) moves
summaries with one padded all-gather per direction and owns the autograd function; it is generic
over the op through a `StagedOp` pair of the two `prepare` entry points.
`context_parallel_conv_history` builds the short convolution's `initial_state` from the same plan.
The recipe is one composition, not an extension point: for a point-to-point pipeline, a
recursive-doubling scan over `compose_summaries`, DTensor, or communication overlap, copy the
autograd function and swap the collective; the primitives and plans are unchanged. Sharding does
not cost accuracy: against an FP32 reference, sharded gradients match the unsharded fused op's error
to within noise for both ops.

```python
from attn_gym.linear.context_parallel import ContextParallelPlan
from attn_gym.linear.kda import context_parallel_kda

# Two ranks, one fragment each: the global stream split down the middle.
total_tokens = cu_seqlens_global[-1]
half = total_tokens // 2
plan = ContextParallelPlan.from_fragments(
    cu_seqlens_global, [[(0, half)], [(half, total_tokens)]], cp_rank
)
token_ids = plan.global_token_ids(device)  # gather global tensors into this rank's span
cu_seqlens = torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=device)
output, final_state = context_parallel_kda(
    q, k, v, gate, beta, cu_seqlens=cu_seqlens, plan=plan, group=group
)
true_final_states = final_state[list(plan.terminal)]
```

The recipe starts each sequence from zero, always returns every subsequence's exit state, and does
not accept `initial_state` or `output_final_state=False`. A captured CUDA Graph is valid only for
its fixed plan.

::: attn_gym.linear.context_parallel.context_parallel_chunk

::: attn_gym.linear.kda.context_parallel.context_parallel_kda

::: attn_gym.linear.gdn.context_parallel.context_parallel_gdn

::: attn_gym.linear.context_parallel.ContextParallelPlan

::: attn_gym.linear.context_parallel.Subsequence

::: attn_gym.linear.kda.stages.chunk_kda_prepare

::: attn_gym.linear.kda.stages.chunk_kda_prepare_backward

::: attn_gym.linear.gdn.stages.chunk_gdn_prepare

::: attn_gym.linear.gdn.stages.chunk_gdn_prepare_backward
