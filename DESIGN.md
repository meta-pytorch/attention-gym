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

## Repository and implementation structure

The repository is organized first by public concept and then, for end-to-end operators, by
attention variant. The existing `masks` and `mods` namespaces remain collections of FlexAttention
building blocks. The `linear` and `sparse` namespaces contain complete attention operations with a
stable public API and one or more interchangeable implementations.

```text
attn_gym/
  _backends/                     # Private infrastructure shared across attention variants
    triton/
      autotune.py                # Reusable Triton tuning and configuration machinery
      utils.py                   # Small reusable Triton helpers; split as it grows
    cute/
      compilation.py             # Reusable CuTeDSL compilation and caching machinery
      tensor_checks.py           # CuTeDSL-specific device, dtype, layout, and shape checks
  masks/                         # FlexAttention mask and BlockMask construction
  mods/                          # FlexAttention score modifications
  linear/                        # End-to-end linear attention operators
    __init__.py                  # Public exports for all linear variants
    <variant>/                   # One complete mathematical attention variant
      __init__.py                # Thin re-export of the variant's public symbols
      api.py                     # Public API, validation, and lazy backend dispatch
      impl/                      # Private interchangeable implementations
        __init__.py              # Marks the private implementation package
        common.py                # Shared semantic primitives used by multiple backends
        reference.py             # Readable eager PyTorch correctness oracle
        triton.py                # Triton kernels, launchers, and backend constraints
        cute.py                  # CuTeDSL kernels, launchers, and backend constraints
  sparse/                        # End-to-end specialized sparse attention operators
    __init__.py                  # Public exports for all sparse variants
    <variant>/                   # One complete mathematical attention variant
      __init__.py                # Thin re-export of the variant's public symbols
      api.py                     # Public API, validation, and lazy backend dispatch
      impl/                      # Private interchangeable implementations
        __init__.py              # Marks the private implementation package
        common.py                # Shared semantic primitives used by multiple backends
        reference.py             # Readable eager PyTorch correctness oracle
        flex.py                  # FlexAttention-based implementation and constraints
        triton.py                # Triton kernels, launchers, and backend constraints
        cute.py                  # CuTeDSL kernels, launchers, and backend constraints

test/
  linear/                        # Linear variant correctness and integration tests
    test_<variant>.py            # Public API and focused private-primitive tests
  sparse/                        # Sparse variant correctness and integration tests
    test_<variant>.py            # Public API and focused private-primitive tests

benchmarks/
  linear/                        # Reproducible linear attention benchmarks
  sparse/                        # Reproducible sparse attention benchmarks
```

Not every operation needs every backend file. Empty placeholders should not be created. A backend
may start as one module and become a package if its implementation becomes too large; for example,
a large Triton backend may evolve from `impl/triton.py` into
`impl/triton/{forward,backward}.py` without changing the public API.

### Module responsibilities

Each operation has one public API layer and a private implementation layer:

- `<variant>/__init__.py` is a thin re-export layer for the variant's documented public symbols.
- `api.py` is the sole public module for the variant. It may define a small cohesive set of public
  functions and types, such as the main operation, its output type, and a variant-specific state
  type. It owns backend-independent validation, mode validation, and lazy backend dispatch, and it
  must not import optional backend dependencies at module import time.
- `impl/common.py` contains named mathematical primitives, shared private types, and backend-neutral
  tensor transformations used unchanged by at least two implementations.
- `impl/reference.py` is the readable eager PyTorch correctness oracle.
- `impl/triton.py`, `impl/cute.py`, and `impl/flex.py` own backend constraints, backend-specific
  preparation, kernel launch logic, and custom autograd code. Their entry points are implementation
  details, not additional public APIs.

The allowed dependency direction is:

```text
namespace __init__.py
        |
        v
      api.py
        |
        +----> impl/reference.py ----+
        +----> impl/triton.py -------+----> impl/common.py
        +----> impl/cute.py ---------+
        +----> impl/flex.py ---------+
```

Implementation modules may import `common.py`, but they must not import one another. In particular,
an optimized backend must not import from `reference.py`. If reference and Triton need the exact
same helper, that helper belongs in `common.py`. If they merely implement the same mathematics using
different decompositions, each backend keeps its own implementation.

`common.py` is not a general utility drawer. Code belongs there only when both conditions hold:

1. the behavior is part of the operation's mathematical semantics; and
2. at least two implementations use the code unchanged.

Shared helpers should have semantic names. For compressed sparse attention, for example,
`compress_interleaved_blocks(...)` is preferable to `compress(...)`, and its docstring should state
the padding, shifted B branch, joint normalization, and output shape. Dense reference masks,
Triton launch configuration, contiguity checks, and kernel-specific indexing remain in their
respective implementation modules.

### Shared backend infrastructure

Backend-specific infrastructure that is reused unchanged across attention variants belongs in the
private `attn_gym._backends` package. Examples include Triton autotuning machinery, CuTeDSL
compilation caches, and backend-specific tensor capability checks. This code is shared because it
integrates a kernel DSL with PyTorch, not because it implements shared attention mathematics.

Use the narrowest ownership level that fits the behavior:

- mathematical behavior shared by two implementations of one variant belongs in
  `<variant>/impl/common.py`;
- Triton infrastructure shared by different variants belongs in `attn_gym/_backends/triton/`;
- CuTeDSL infrastructure shared by different variants belongs in `attn_gym/_backends/cute/`;
- public types shared by multiple linear or sparse operations belong in a namespace-level module
  such as `attn_gym/linear/types.py`, not in an implementation package.

The `_backends` package must remain private and dependency-oriented. While a backend has only a few
small reusable helpers, they may live together in `utils.py`; split them into responsibility-specific
modules such as `autotune.py`, `compilation.py`, or `tensor_checks.py` once coherent groups emerge.
Do not create a backend-wide `common.py`, and do not let `utils.py` absorb variant-specific behavior.
Variant implementation modules may import the corresponding shared backend package, but shared
backend packages must not import variant APIs or encode variant-specific formulas, state layouts, or
dispatch policy. Optional Triton and CuTeDSL dependencies must still be imported lazily.

### Compilation and custom operator boundaries

The public operation should remain an ordinary Python function that validates backend-independent
semantics and dispatches to a private implementation. Supporting `torch.compile` does not by itself
require turning the public API or every implementation into a custom operator.

Use the least opaque integration that provides the required compiler behavior:

1. Keep eager/reference implementations as traceable compositions of PyTorch operations whenever
   practical. Do not wrap traceable PyTorch code in a custom operator merely to hide it from the
   compiler.
2. Prefer compiler-visible Triton integration when the backend can be represented with PyTorch's
   Triton operator APIs. This lets the compiler reason about the implementation instead of treating
   the entire backend as an opaque call.
3. Use a private custom operator boundary for CuTeDSL launchers, external-library adapters, or other
   backend code that `torch.compile` cannot trace correctly. The custom operator is an implementation
   detail behind `backend=` and is not a second public Attention Gym API.
4. Keep variant semantics at the variant layer. Shared `_backends` code may provide registration,
   compilation, caching, or validation machinery, but the variant backend owns its operator schema,
   fake implementation, mutation declaration, autograd registration, and output/state metadata.

#### Register contracts without importing optional backends

When a public operation supports both a torch-only reference and an optional fused backend, put its
private operator contracts in a small torch-only module:

```text
<variant>/
  api.py                  # Imports ops.py, validates semantics, lazily selects a backend
  ops.py                  # Schemas, fake kernels, dispatch registration, torch.ops handles
  impl/reference.py       # Torch-only eager oracle
  impl/triton.py          # Optional kernels and launchers
  impl/cute.py            # Optional kernels and launchers
```

`ops.py` must not import Triton, CuTeDSL, or another optional implementation at module import time.
Its device dispatch functions may import the selected launcher locally when the dispatcher executes
the operator. This establishes schemas and fake implementations before Dynamo starts tracing while
keeping reference imports usable with only PyTorch installed:

```text
import public API
  register schemas, fake kernels, and lazy device dispatch

call reference
  run ordinary PyTorch without importing an optional backend

call fused
  trace an already-registered operator
  import the optional launcher only when its device implementation executes
```

Do not register schemas, fake kernels, or device implementations as a side effect of a fused
backend's first public call. A first-ever `torch.compile(..., fullgraph=True)` invocation must not
depend on tracing-time registration. Test both boundaries in fresh processes: block optional
backend imports and run the reference path on CPU, then make strict compilation the first fused
operation in a backend-enabled process.

Every custom operator used by an optimized backend must provide the compiler metadata required by
its supported contract, including a fake implementation for shape and dtype propagation. Training
backends must define an autograd formula or clearly reject gradient-requiring inputs. Mutation and
aliasing must be declared accurately. Custom operators should be validated with PyTorch operator
checks in addition to end-to-end eager-versus-compiled tests.

Compilation is a public-operation property, not merely a kernel-launch property. Tests should call
the documented function through `torch.compile`, including its validation, dispatch, structured
result, forward path, backward path when supported, and initial/final state behavior. A backend that
works only when its private launcher is invoked directly does not satisfy the compilation contract.

### Public exports

Public functions are re-exported from the namespace root:

```python
from attn_gym.linear import gated_delta_rule
from attn_gym.sparse import compressed_sparse_attention
```

Backend entry points are private and use a consistent internal name such as `forward`. A backend
may additionally provide a small private capability check when dispatch needs one. Users select an
implementation through the public `backend=` argument rather than importing implementation modules
directly.

The initial dispatch should remain explicit and local to `api.py`: map a documented backend name to
one lazily imported implementation, validate that implementation's capabilities, and call it. Do
not introduce registration side effects or a repository-wide priority registry until several
operations demonstrate that the same mechanism would remove meaningful duplication. Automatic
selection should use an inspectable ordered policy and report why explicitly requested backends are
unsupported.

### Applying the structure to compressed sparse attention

The current compressed sparse attention code should move to:

```text
attn_gym/sparse/compressed_sparse_attention/
  __init__.py
  api.py
  impl/
    __init__.py
    common.py
    reference.py
    triton.py
```

The refactor should be mechanical and behavior-preserving:

1. Move shared padding, interleaved block compression, and the shared RoPE formulation into
   `impl/common.py`. Give these helpers explicit names, type annotations, shape documentation, and
   tests. Preserve separate backend implementations when sharing would compromise readability,
   compilation, or backend requirements.
2. Move the eager implementation into `impl/reference.py`, rename `CSA` to the private backend
   entry point `forward`, and retain dense-only helpers such as mask materialization and sink
   softmax there.
3. Move the Triton implementation into `impl/triton.py`, rename its public-looking entry point to
   `forward`, and retain Triton kernels, launchers, custom autograd, backend constraints, and
   Triton-specific preparation there.
4. Update `api.py` to lazily load `impl.reference.forward` or `impl.triton.forward`. Keep all public
   argument and shape validation in `api.py`; keep device, dtype, contiguity, architecture, and
   backend capability validation in the selected implementation.
5. Keep `compressed_sparse_attention/__init__.py` and `attn_gym/sparse/__init__.py` as thin public
   re-export layers.
6. Update tests to import only the public API except for focused unit tests of private mathematical
   primitives. Run the existing eager-versus-Triton forward and backward matrix before and after
   the move to demonstrate that the refactor did not change behavior.

The initial refactor should not introduce a global backend registry, redesign the public signature,
or fuse additional work into kernels. Those are separate changes and should follow only after the
module boundaries are established.

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

- Public API execution through `torch.compile`, including full-graph capture where supported.
- Fake tensor behavior for every custom operator and tracing behavior required by downstream
  integration.
- PyTorch operator checks for custom operator schemas, fake implementations, mutation, and autograd.
- Compiled forward and backward correctness against eager execution for supported training paths.
- Autocast behavior.
- Contiguous and intentionally supported non-contiguous layouts.
- CUDA Graph capture and replay for supported optimized backends.
- TorchTitan integration for at least the first production-targeted operator.

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
- PyTorch custom operators: <https://docs.pytorch.org/docs/stable/library.html>
- PyTorch custom operators and `torch.compile` tutorial:
  <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>
- PyTorch user-defined Triton kernels with `torch.compile`:
  <https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>
- Flash Linear Attention: <https://github.com/fla-org/flash-linear-attention>
- NVIDIA CUTLASS and CuTeDSL: <https://github.com/NVIDIA/cutlass>
