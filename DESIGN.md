# Linear and Sparse Attention in Attention Gym

- **Status:** Draft
- **Audience:** Attention Gym contributors, PyTorch maintainers, TorchTitan integrators
- **Scope:** High-level product direction, package structure, API principles, and contribution requirements

## Summary

Attention Gym will expand beyond its current focus on FlexAttention masks and score
modifications to provide production-oriented implementations of popular linear and sparse
attention variants.

The new public namespaces will be:

```python
from attn_gym.linear import ...
from attn_gym.sparse import ...
```

The repository will provide a stable place in the Meta PyTorch ecosystem for rapidly adding,
testing, and evolving attention implementations without inheriting PyTorch core's backward- and
forward-compatibility constraints. APIs may break when warranted, and users should pin an
Attention Gym version.

Implementations will be Python-only. Optimized kernels should primarily use Triton or CuTeDSL.
Other Python kernel DSLs are welcome when they can be maintained and tested. Wrappers around
external implementations, especially Flash Linear Attention (FLA), are acceptable when they let
users train important models sooner. We should only do this when we have concrete value to add via
extensions or improvements.

This document is expected to change as the first implementations establish the right common
abstractions.

## Motivation

PyTorch aims to enable its users rather than compete with every specialized kernel library. Dense
attention illustrates both the value and the limits of putting fast attention directly in core:
`scaled_dot_product_attention` made FlashAttention broadly accessible, but its original interface
predated several capabilities now needed by modern models, including paged KV caches and many
structured attention patterns.

FlexAttention addressed part of this problem by pairing a programmable attention interface with
JIT-compiled kernel templates. It has been successful for expressing dense and block-sparse softmax
attention variants without adding a one-off operator for every new pattern. Yet it still is not
all encompasing.

Linear attention and specialized sparse attention now need a complementary strategy:

1. Ship model-enabling, SDPA-like operators quickly.
2. Continue looking for reusable abstractions that can cover families of algorithms.
3. Develop those APIs outside PyTorch core, where iteration and intentional BC breaks are easier.

The thesis is that these approaches are complementary. We should not delay useful concrete
operators while waiting for a universal linear- or sparse-attention abstraction, but each concrete
operator should help us discover that abstraction rather than create an unrelated API island.
A great result would be to develop an abstraction that encompasses much of the new variants we
add in this repo, land in pytorch core and then replace our implementations here.

## Goals

- Provide functional APIs for popular linear and sparse attention variants.
- Support training, prefill, and recurrent decoding where the algorithm permits it.
- Make eager/reference implementations first-class correctness specifications.
- Allow multiple implementations of an operation, including eager PyTorch, Triton, CuTeDSL, other
  Python kernel DSLs, and adapters to external libraries. Only when necessary.
- Operators should be designed for both training and inference in mind.
- Design toward variable-length inputs, explicit recurrent state, batch-invariant execution, and
  context parallelism.
- Co-design interfaces with TorchTitan and representative model integrations to prove validity.
- Establish a contribution path that lets a new model's required attention operator land quickly.
- Keep optimized implementation details out of the public API whenever possible.
- We should end up building a library of primitives from the first N implementations that make
  authoring the N+1 easier.

## Non-goals

- **PyTorch core stability guarantees.** Attention Gym APIs may change between releases. Breaking
  changes must still be motivated, documented, and accompanied by a migration path.
- **An optimized native implementation for every operator on day one.** A correct reference plus an
  FLA adapter may be the right first implementation.
- **Broad hardware coverage.** NVIDIA GPUs, with an initial focus on Blackwell and newer, are the
  primary optimization target. A backend that cannot be exercised in project CI is not part of the
  supported matrix.
- **Repository-owned C++ or CUDA C++ extensions.** Runtime and kernel code must be authored through
  Python or a Python DSL. A DSL may compile generated native GPU code, we will not accept native code.
- **Owning serving-framework kernels immediately.** vLLM and SGLang are likely to remain the primary
  inference runtimes. Attention Gym should make training-to-inference transitions clear, but it
  should not assume that serving frameworks will call these kernels directly.
- **A universal abstraction before we have evidence.** We will avoid a speculative framework that
  attempts to encode every linear or sparse attention algorithm before multiple implementations
  demonstrate shared structure.

## Design principles

### Separate semantics from implementation

An operation name identifies mathematical behavior. Execution form and kernel backend are
orthogonal choices:

- **Operation:** gated delta rule, GLA, NSA, and so on.
- **Execution form:** chunked, recurrent, or automatic selection.
- **Backend:** eager/reference, Triton, CuTeDSL, another Python DSL, FLA, or automatic selection.

A backend must not silently change the operation's documented semantics. Numerically different but
mathematically equivalent execution forms must have explicit tolerance and invariance contracts.
I am tempted to require bit exactness once we establish a trusted implementation for tweaks or new
imps but that is tbd.

### Prefer variant-specific public functions

Linear attention variants do not share a sufficiently uniform set of inputs, states, gates, and
normalizations to justify a single public function with a large parameter union.

Prefer:

```python
from attn_gym.linear import gated_delta_rule

result = gated_delta_rule(
    query,
    key,
    value,
    gate,
    beta,
    mode="chunked",
    backend="auto",
)
```

Do not begin with:

```python
linear_attention(kind="gated_delta_rule", ..., algorithm_options={...})
```

Shared types and dispatch behavior should be extracted only when multiple operators use them.

### Keep state explicit

This is an initial idea and subject to change.

Recurrent state is part of the mathematical API, not hidden backend state. An operation that supports
streaming should accept an initial state and optionally return a final state. This enables:

- chunk-by-chunk correctness tests;
- prefill-to-decode transitions;
- sequence or context parallel implementations;
- activation checkpointing policies;
- serving-framework adapters.

The initial proposal is a generic structured result:

```python
StateT = TypeVar("StateT")


@dataclass
class LinearAttentionOutput(Generic[StateT]):
    output: torch.Tensor
    final_state: StateT | None = None
```

The concrete state type may be a tensor or a variant-specific dataclass.

### Reference implementations define correctness

Every operation should have an eager PyTorch implementation that prioritizes clarity and correctness
over speed. The reference implementation should:

- run without optional kernel dependencies;
- work on CPU when its PyTorch operations permit it;
- support autograd or have an explicit limitation;
- define state transitions and masking behavior;
- serve as the oracle for optimized backends;
- support a deliberately slow batch-invariant mode if that contract cannot be met by optimized
  kernels.

A reference implementation is not required to use the same algorithmic decomposition as an optimized
chunked kernel, but differences must be covered by numerical tests.

### Do not expose a global registry prematurely

The first operators can use small per-operation dispatch functions. A shared backend registry should
be introduced only if it removes demonstrated duplication. Public registration of arbitrary
third-party backends is not an initial requirement.

## Proposed package structure

The package should be organized variant-first. This keeps the formula, reference, optimized kernels,
and tests for an operation understandable as a unit.

```text
attn_gym/
  linear/
    __init__.py
    <variant>/
      __init__.py
      api.py
      reference.py
      triton.py
      cute.py
  sparse/
    __init__.py
    <variant>/
      __init__.py
      api.py
      reference.py
      flex.py
      triton.py
      cute.py

test/
  linear/
    test_<variant>.py
  sparse/
    test_<variant>.py

benchmarks/
  linear/
  sparse/
```

Not every variant needs every backend file. Empty placeholder modules should not be created.

Public symbols are re-exported from the namespace root:

```python
from attn_gym.linear import gated_delta_rule
from attn_gym.sparse import compresed_sp
```

The operation names above are illustrative; this design does not choose the first variants.
Backend implementation functions are not public API unless there is a demonstrated need for direct
access.

## Relationship to existing FlexAttention APIs

The existing namespaces remain focused on FlexAttention building blocks:

- `attn_gym.masks` constructs or supports `BlockMask` patterns.
- `attn_gym.mods` defines FlexAttention score modifications.

The initial recommendation is that `attn_gym.sparse` own end-to-end specialized sparse attention
operators, not every sparse mask. Existing mask helpers should not be moved merely because their
patterns are sparse.

For example, VSA's block-mask construction can remain in `attn_gym.masks.vsa`, while a future
end-to-end `vsa_attention(...)` operator with selection, sparse attention, and output combination
could be exported from `attn_gym.sparse`. Its first backend could simply invoke FlexAttention if
FlexAttention provides the required semantics and performance; a specialized backend can be added
later without changing the public API.

This boundary avoids turning `attn_gym.sparse` into a second name for `attn_gym.masks`. The sparse
namespace identifies user-facing attention algorithms, not a requirement that every implementation
use a custom sparse kernel.

## Linear attention API shape

### Inputs and layout

Fixed-length public APIs use the SDPA/FlexAttention layout:

```text
[batch, heads, sequence, dimension]
```

Packed variable-length inputs use:

```text
[total_tokens, heads, dimension]
```

where `total_tokens` is the sum of all sequence lengths. Backends may transpose internally, but
public callers should not need backend-specific layouts. Outputs follow the corresponding input
layout.

### Execution form

The proposed public argument is:

```python
mode: Literal["auto", "chunked", "recurrent"] = "auto"
```

Each operation documents the modes it supports. Unsupported modes fail clearly rather than falling
back to a semantically different path.

The expected policy is:

- `chunked` for training and long prefill;
- `recurrent` for token-by-token decoding and a correctness-oriented batch-invariant path;
- `auto` for a documented shape- and autograd-aware choice.

An operation may expose an additional parallel formulation when it has one, but that is
operation-specific rather than part of the common contract.

Whether `auto` belongs in the first public release is still open. Explicit-only mode selection is
simpler for reproducibility and compilation; automatic selection is easier for model authors.

### Backend selection

The proposed public argument is:

```python
backend: Literal["auto", "eager", "triton", "cute"] = "auto"
```

Backend selection rules must be deterministic and inspectable. `auto` may consider device,
architecture, dtype, shape, execution mode, autograd requirements, and installed dependencies. It
must not hide an unsupported input by changing mathematical behavior.

The initial implementation should use straightforward dispatch rather than a general capability
solver. If backend selection grows complex, supported capabilities can later be represented as data.

### Variable-length inputs

Variable-length support is part of the intended base contract rather than an optional follow-up.
Packed inputs use `[total_tokens, heads, dimension]` tensors with cumulative sequence lengths:

```python
cu_seqlens: torch.Tensor | None = None
```

For `batch` logical sequences, `cu_seqlens` has `batch + 1` entries. Whether a backend also needs a
CPU copy or other metadata is an implementation detail that should not leak into the public API
unless required.

### Outputs and recurrent state

The public API should always return a structured result rather than switch between a tensor and tuple
based on `return_final_state`. A stable result shape is easier to compose and extend:

```python
result.output
result.final_state
```

The cost is a small departure from existing operator conventions. This should be validated with the
first TorchTitan integration before being finalized.

### Layers versus functional operators

Attention Gym will initially ship lower-level functional operators, not complete
projection/norm/gating `torch.nn.Module` layers. Users compose these primitives into their model
modules. Examples may demonstrate representative module integration without making those modules
part of the supported API.

## Sparse attention API shape

"Sparse attention" covers at least two different layers of functionality:

1. a sparse pattern or selected block representation; and
2. an end-to-end operation that may include token/block selection, sparse attention, and output
   combination.

The namespace should prioritize end-to-end operators needed by models while reusing existing
FlexAttention mask utilities where appropriate. FlexAttention itself is a valid backend when it
meets the operator's semantic and performance requirements.

We should not define a universal `SparsePattern` protocol until at least two specialized operators
show that they can share one without losing important information. Candidate common inputs that may
emerge include:

- selected KV block indices;
- per-query block counts;
- block sizes and sequence lengths;
- causal and document boundaries;
- routing scores or gates;
- local/dense fallback regions.

As with linear attention, sparse operators should expose variant-specific public functions and keep
backend selection orthogonal.

## Determinism and batch invariance

Each operation should have at least one deterministic, batch-invariant implementation intended for
correctness checks and debugging. For a given sequence, its output should not depend on:

- which batch slot contains it;
- the order of sequences in the batch;
- which unrelated sequences are processed alongside it.

A slow eager or recurrent implementation may provide this guarantee. Optimized chunked
implementations may advertise a weaker numerical contract if the distinction is explicit.

The exact first contract remains open: bitwise equality versus tolerance-based equality, whether it
covers gradients and final recurrent state, and which dtypes and hardware are guaranteed. We should
settle those details using the first operation rather than prematurely adding a public
`batch_invariant` flag. Batch invariance should not be inferred solely from
`mode="recurrent"`.

## Backend and dependency policy

### Accepted implementation types

- Eager PyTorch reference implementations.
- Triton kernels.
- CuTeDSL kernels.
- Other Python DSL implementations when there is a compelling benefit and maintainers can test and
  support them.
- Thin adapters to maintained external libraries.

### External adapters

An external adapter should:

- preserve the Attention Gym public contract;
- isolate imports and version-specific behavior;
- declare the tested dependency range;
- have correctness tests against the eager reference;
- make unsupported features explicit;
- avoid exposing the dependency's internal configuration surface wholesale.

If adapting a library becomes more code than maintaining the needed implementation, vendoring or a
native backend can be reconsidered.

### Hardware support

Each optimized backend must declare its supported:

- GPU architecture;
- CUDA version when relevant;
- dtype;
- head/state dimensions or alignment constraints;
- execution modes;
- forward/backward support;
- variable-length support;
- deterministic or batch-invariant guarantees.

"Unsupported" should produce a clear error or a documented `auto` fallback. Silent execution on an
untested architecture is not considered support.

## Testing requirements

Every new operation should include the applicable parts of this matrix.

### Correctness

- Eager/reference forward correctness.
- Gradients against the reference implementation.
- Optimized backend forward and backward correctness.
- Chunked versus recurrent equivalence.
- Full sequence versus state-carrying segmented execution.
- Fixed-length versus packed variable-length equivalence.
- Nontrivial random inputs and representative edge cases.
- Initial and final state behavior.
- Batch-invariance tests for any advertised guarantee.

### Integration

- `torch.compile` behavior where supported.
- Fake tensor or tracing behavior where required by downstream integration.
- Autocast behavior.
- Contiguous and intentionally supported non-contiguous layouts.
- TorchTitan integration for at least the first production-targeted operator.
- We want our functions to be compilable and cuda-graphable.

### Hardware

- CPU tests for references where possible.
- GPU correctness tests on every supported architecture available in CI.
- Architecture-specific skips with an explicit reason.
- No acceptance of an optimized support claim that cannot be run in CI.

### Performance

Benchmarks are separate from correctness tests. Each optimized backend should include a reproducible
benchmark with:

- shapes, dtype, hardware, software versions, and warmup/measurement procedure;
- forward and backward measurements when training is supported;
- comparison against the reference only for correctness, not as the sole performance baseline;
- comparison against the relevant external or model baseline when available.

Performance is a goal, but a contribution does not need to be globally fastest to be useful. It must
be clear where it is intended to be competitive.

## Contribution requirements

A PR adding a new attention operation should include:

1. A short description of the mathematical operation and a paper/model reference.
2. A public functional API under `attn_gym.linear` or `attn_gym.sparse`.
3. An eager/reference implementation or a documented reason one is impractical.
4. Tests covering forward behavior and training gradients when applicable.
5. A supported-capabilities section in the operation's documentation.
6. At least one runnable example or model integration.
7. Backend dependency and hardware requirements.
8. Benchmark coverage for optimized implementations.
9. A migration note for any breaking API change.

## Repository and release policy

- Attention Gym remains a single package with `linear` and `sparse` as namespaces, not separate
  distributions initially.
- Users should pin versions.
- Releases should document breaking changes and backend dependency changes.
- A formal model-support SLA should be defined only after measuring the first several integrations.
- The repository description, README, documentation navigation, and contribution guide should be
  updated when the first operator lands, rather than advertising empty namespaces.

## References

- PyTorch FlexAttention documentation: <https://pytorch.org/docs/main/nn.attention.flex_attention.html>
- Flash Linear Attention: <https://github.com/fla-org/flash-linear-attention>
- NVIDIA CUTLASS and CuTeDSL: <https://github.com/NVIDIA/cutlass>
