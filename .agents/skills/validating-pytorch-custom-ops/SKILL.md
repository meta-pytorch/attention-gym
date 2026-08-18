---
name: validating-pytorch-custom-ops
description: Ensures new Attention Gym eager, Triton, CuTeDSL, and external-library implementations are torch.compile-friendly and correctly registered. Use when adding or reviewing an implementation or custom kernel; covers custom_op, triton_op, fake tensors, autograd, opcheck, full-graph compilation, dynamic shapes, and CUDA Graphs.
---

# Validate PyTorch Implementations and Custom Operators

Use this workflow whenever adding a backend under `attn_gym/linear/<variant>/impl/` or
`attn_gym/sparse/<variant>/impl/`.

## Choose the least opaque integration

1. **Eager/reference implementation:** keep ordinary PyTorch code traceable. Do not create a custom
   operator merely because the public function must support `torch.compile`.
2. **Triton implementation:** direct Triton kernels can work under `torch.compile`. Prefer
   `torch.library.triton_op` with `torch.library.wrap_triton` when a stable operator boundary is
   useful and compiler subsystems should remain able to inspect the implementation.
3. **CuTeDSL or external implementation:** use a private registered operator
   (`torch.library.define`/`impl`, below) when the launcher cannot be traced correctly.

The documented public function remains an ordinary Python function that owns semantic validation,
mode selection, and backend dispatch. Compiling only a private launcher does not prove that the
public operation is compile-friendly.

## Operator boundary CPU overhead

Every registered-operator boundary costs CPU time per call. Microbenchmark (torch 2.14 nightly,
B200, identical trivial launcher, CPU wall-clock per call, medians over interleaved rounds):

| boundary                        | inference | fwd, requires_grad=True     |
| ------------------------------- | --------- | --------------------------- |
| raw Python                      | 1.2 us    | 3.7 us (+autograd.Function) |
| `define`/`impl`                 | 2.6 us    | 5.3 us (+autograd.Function) |
| `custom_op`                     | 5.5 us    | 8.2 us (+autograd.Function) |
| `impl(op, "Autograd")` kernel   | 6.8 us    | 7.4 us                      |
| `custom_op` + register_autograd | —         | 10.2 us                     |

Nesting one registered op inside another pays the boundary again (nested `custom_op` 7.1 us,
nested `define`/`impl` 3.7 us). Wrapping the real `kda_l2norm_fwd` launcher in `custom_op` added
~11 us per forward call over invoking the launcher directly.

Consequences for launch-bound paths (small kernels, high call rate):

- Registration exists for compile and fake-tensor consumers; eager execution gains nothing from
  it. The eager floor is a plain `autograd.Function` over the launcher with no `torch.library`
  registration (real `l2norm` op: ~17 us cheaper per gradient-tracking forward and ~58 us per
  fwd+bwd than the `custom_op` boundary), at the cost of `fullgraph=True` support.
- Use the define/impl pattern below for repo operators; `torch.library.custom_op` costs ~3 us
  more per call for schema inference and mutation checking and is only preferable for
  prototypes.
- Prefer `autograd.Function` over `register_autograd` for hot training ops; it was the cheapest
  measured autograd boundary. Registering a Python kernel at the Autograd dispatch key
  (`torch.library.impl(op, "Autograd")`) makes the raw op differentiable but runs Python on
  every call, including inference; use it only when direct `torch.ops` calls must be
  differentiable.
- Give each hot path at most one registered-operator boundary. Nested custom ops double the
  dispatch tax; share the Python launcher between entrypoint ops instead of calling one op from
  another.
- A `define`/`impl` op has no autograd kernel: backprop through a direct call warns and produces
  no gradient, so route all differentiable use through the `autograd.Function` wrapper.
- `torch.library.opcheck` accepts `torch.ops.attn_gym.op.default`, so the validation workflow
  below is unchanged.

## Public result contract

Private backends should return tensors or a fixed tensor tuple. The public API may wrap that result
in a documented `NamedTuple` so users can use attributes or unpack it:

```python
from typing import NamedTuple

import torch


class OperationOutput(NamedTuple):
    output: torch.Tensor
    final_state: torch.Tensor | None = None


def operation(...) -> OperationOutput:
    output, final_state = backend_forward(...)
    return OperationOutput(output, final_state)
```

A registered operator's output structure and arity must agree with its schema. Do not switch between
returning a tensor and a tuple based on an argument. Represent optional outputs explicitly and match
them in the fake implementation.

## Opaque operator pattern (define/impl + autograd.Function)

The repo standard for opaque kernel boundaries. Write the schema string yourself; nothing is
inferred from annotations:

```python
import torch
from torch import Tensor

torch.library.define(
    "attn_gym::example_fwd",
    "(Tensor query, Tensor key, Tensor value, Tensor? initial_state) -> (Tensor, Tensor)",
)
torch.library.define("attn_gym::example_bwd", "(Tensor query, Tensor grad_output) -> Tensor")


def _example_fwd_cuda(
    query: Tensor, key: Tensor, value: Tensor, initial_state: Tensor | None
) -> tuple[Tensor, Tensor]:
    return launch_backend(query, key, value, initial_state)


# Register with an explicit call, not as a decorator: the decorator form returns None,
# and keeping the launcher callable lets benchmarks and tests bypass the boundary.
torch.library.impl("attn_gym::example_fwd", "CUDA", _example_fwd_cuda)


@torch.library.register_fake("attn_gym::example_fwd")
def _example_fwd_fake(
    query: Tensor, key: Tensor, value: Tensor, initial_state: Tensor | None
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(value), query.new_empty(query.shape[:2])


_example_fwd = torch.ops.attn_gym.example_fwd.default
_example_bwd = torch.ops.attn_gym.example_bwd.default  # impl/fake registered the same way


class _Example(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        output, state = _example_fwd(query, key, value, None)
        ctx.save_for_backward(query)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor):
        (query,) = ctx.saved_tensors
        return _example_bwd(query, grad_output), None, None
```

Schema rules:

- Optional tensors are `Tensor?`; scalars are `int`, `float`, `bool`; defaults may be embedded
  (`Tensor? cu_seqlens=None`).
- Encode mutation explicitly with alias annotations: `(Tensor(a!) state, Tensor value) -> ()`.
- The schema language is a superset of what `custom_op` annotation inference accepts, adding
  e.g. `ScalarType`, `Layout`, `MemoryFormat`, `Generator`, `SymInt`, and `int[2]`. Neither flow
  accepts arbitrary Python objects, dataclasses, or callables; flatten configs to scalars or
  specialize per-config outside the op.
- Optional *outputs* parse as `Tensor?` and, as of torch 2.15 nightly (verified 2026-08-16:
  plain fullgraph compile, `autograd.Function` wrapping + backward, and `opcheck` all pass for
  both the tensor and `None` branches), no longer hard-fail the compile stack; on older
  releases the `None` return handling was unreliable. Keep the fixed-arity convention anyway:
  for "N or N+1 returns depending on a flag", define two fixed-arity schemas sharing one
  launcher (see `kda_chunk_fwd` / `kda_chunk_fwd_with_state`) and let the `autograd.Function`
  branch on the flag; its output arity is free to vary. The bool flag specializes into separate
  graphs regardless, so a merged optional-output schema saves nothing under compile, while a
  `Tensor?` return forces `None`-narrowing on every caller and extra schema boxing costs real
  time on launch-bound eager paths (collapsing schema pairs measured 3.22% slower on the dense
  forward probe, PR #314).
- Optional *inputs* are fully compile-safe: the merged `kda_chunk_bwd` schema takes
  `Tensor? cu_seqlens, Tensor? chunk_offsets` for both dense and ragged and passes the strict
  fullgraph matrix. The forward keeps separate dense/ragged ops as a measured eager-dispatch
  decision, not a compile requirement.
- Tensors needed only by the backward (autograd tapes, packing metadata) should be op outputs
  saved by the `autograd.Function` via `ctx.save_for_backward`, not part of the wrapper's
  user-facing return. Unlike `register_autograd`, the Function is not limited to saving
  user-visible outputs.
- Bind `torch.ops.attn_gym.<op>.default` to a module-level name and call that; resolving the
  `torch.ops` attribute chain per call adds overhead.

The fake implementation describes output metadata without running the kernel. It must:

- return the same output structure as the real implementation;
- preserve shape, dtype, device, layout, and relevant strides;
- avoid reading tensor values, storage, data pointers, or calling `.item()`;
- use symbolic shape arithmetic rather than data-dependent Python branches.

For genuinely data-dependent output dimensions, use `torch.library.get_ctx()` and
`new_dynamic_size()` rather than inspecting input data.

Registration must exist before graph capture. Keep registration deterministic and lightweight; do
not rely on registration side effects occurring inside a compiled region. Continue importing
optional kernel dependencies lazily.

### Mutation and aliasing

Declare every mutated argument in the schema string:

```python
torch.library.define("attn_gym::update_state", "(Tensor(a!) state, Tensor value) -> ()")
```

Do not declare an operator functional if its launcher writes into an input, including recurrent
state, cache, workspace, or output buffers. Prefer functional operators when practical.

### Autograd

Training backends attach autograd with a `torch.autograd.Function` outside the registered ops, as
in the pattern above: `forward` calls the forward op (autograd is already disabled inside
`Function.forward`, so no redispatch guard is needed), `backward` calls the backward op, and the
public API routes through `Function.apply`. Notes:

- Mark first-order-only backwards with `@torch.autograd.function.once_differentiable`.
- Return one gradient entry per forward input and `None` for nondifferentiable inputs.
- Use `ctx.mark_non_differentiable(...)` for auxiliary outputs and
  `ctx.set_materialize_grads(False)` when the backward handles `None` grads.
- The raw forward op is not differentiable; direct `torch.ops` calls that backprop through it
  warn and produce no gradient. Route all differentiable use through the wrapper.
- Do not use `CustomOpDef.register_autograd` (slowest measured path) or an Autograd dispatch-key
  kernel (taxes every inference call) unless raw op calls must be differentiable.
- If training is unsupported, reject requiring-gradient inputs clearly instead of silently
  detaching them.

## Compiler-visible Triton pattern

```python
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, size, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@torch.library.triton_op("attn_gym::add", mutates_args={})
def add(x: Tensor, y: Tensor) -> Tensor:
    output = torch.empty_like(x)
    size = x.numel()
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK"]),)
    torch.library.wrap_triton(add_kernel)[grid](
        x,
        y,
        output,
        size,
        BLOCK=256,
    )
    return output
```

Keep cross-variant Triton infrastructure in `attn_gym._backends.triton` only after it has multiple
callers. Variant-specific kernels, schemas, fake behavior, and autograd remain in the variant
backend.

## Large-layout address width

Optimized kernels must preserve an int32 default path and select a separate int64 specialization
when relative element offsets can exceed signed int32. Inventory every input, output, optional
tensor, manually reconstructed view, and manual load/store offset; checking only the primary input
or `numel()` is insufficient.

For Triton, use `attn_gym._backends.triton.requires_int64_offsets` in a
`@triton.heuristics` `USE_INT64_OFFSETS` constexpr. It checks reachable storage cosize for the
project's nonnegative-strided layouts. In the wide branch, cast program IDs, loaded token/chunk
origins, and other address indices to `tl.int64` before the first potentially overflowing multiply
or addition. Casting a completed offset or pointer is too late. Bounded routing arrays may stay
int32, but widen their loaded values before using them in wide pointer arithmetic.

For CuTeDSL TVM-FFI, use `attn_gym._backends.cute.requires_int64_abi`. In addition to reachable
cosize, it checks every declared ABI stride because a size-one dimension can carry an unreachable
stride larger than `INT32_MAX` that the compiled signature must still represent. Carry
`use_int64_offsets` through the op, compile-cache key, and stable kernel name; use matching
`cute.sym_int64` fake-signature fields in the wide variant. Widen values before multiplication when
constructing layouts or adding to iterators.

Required validation has three layers:

1. Predicate tests on meta/fake tensors, including an oversized singleton stride without allocating
   the unreachable storage.
2. Forced-int64 equivalence with the normal int32 specialization on small forward and backward
   inputs.
3. When hardware memory permits, execution of an active offset beyond `INT32_MAX`, compared with an
   equivalent compact layout. Singleton-stride tests prove ABI routing but not wide device pointer
   arithmetic.

Also assert ordinary layouts select int32 and measure both variants before claiming no regression;
int64 address arithmetic can increase instructions and registers. Use `test/test_kda_int64_offsets.py`
as the project reference.

## Validate registration with `opcheck`

`torch.library.opcheck` validates operator registration. Its default utilities are:

- `test_schema`: runtime mutation and aliasing agree with the schema;
- `test_autograd_registration`: autograd is registered correctly;
- `test_faketensor`: fake execution matches real output metadata;
- `test_aot_dispatch_dynamic`: AOT dispatch works with dynamic shapes.

Run it through pytest after importing the registration module:

```python
import pytest
import torch


OPCHECK_UTILITIES = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_example_forward_registration(dtype, requires_grad):
    query, key, value = make_inputs(dtype=dtype, requires_grad=requires_grad)
    torch.library.opcheck(
        example_forward,
        (query, key, value),
        test_utils=OPCHECK_UTILITIES,
    )
```

Pass keyword arguments as the third argument:

```python
torch.library.opcheck(example_forward, args, {"causal": True})
```

Create separate cases for materially different registration paths:

- each supported device and dtype;
- inference and gradient-requiring inputs;
- optional arguments and outputs;
- contiguous and intentionally supported non-contiguous layouts;
- boundary, non-aligned, and zero-size shapes when supported;
- mutable inputs;
- dynamic dimensions.

Keep `raise_exception=True`, the default, in committed tests. Use `raise_exception=False` only while
diagnosing multiple failures:

```python
results = torch.library.opcheck(example_forward, args, raise_exception=False)
for utility, result in results.items():
    print(utility, result)
```

`opcheck` does not validate numerical correctness. A kernel can pass every utility and still compute
wrong values. Pair it with reference forward tests and gradient tests.

## Numerical correctness

Every optimized backend needs a trusted eager/reference oracle. Compare:

- outputs;
- final recurrent state;
- gradients for every differentiable input and initial state;
- segmented state-carrying execution against full-sequence execution.

Use `torch.autograd.gradcheck` when double precision is supported:

```python
def test_example_forward_gradcheck():
    inputs = make_inputs(dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(example_forward, inputs)
```

For lower-precision-only GPU kernels, compare gradients against the reference with explicit
atol/rtol values justified by dtype and reduction order.

## Compile the public operation

Compile the documented function with strict graph capture:

```python
def test_compiled_forward_and_backward():
    eager_inputs = make_inputs(requires_grad=True)
    compiled_inputs = clone_inputs(eager_inputs)

    expected = operation(*eager_inputs, backend="triton")
    compiled_operation = torch.compile(operation, fullgraph=True)
    actual = compiled_operation(*compiled_inputs, backend="triton")

    torch.testing.assert_close(actual.output, expected.output, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        actual.final_state,
        expected.final_state,
        atol=ATOL,
        rtol=RTOL,
    )

    expected_gradients = torch.autograd.grad(loss(expected), eager_inputs)
    actual_gradients = torch.autograd.grad(loss(actual), compiled_inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )
```

Include every mode that selects a materially different implementation, initial state present and
absent, and final state requested and omitted. A graph break under `fullgraph=True` fails the
advertised compilation contract.

### Dynamic shapes

If dynamic shapes are supported, compile once and reuse the callable across multiple sizes:

```python
compiled_operation = torch.compile(operation, fullgraph=True, dynamic=True)
for sequence_length in (127, 193):
    inputs = make_inputs(sequence_length)
    expected = operation(*inputs)
    actual = compiled_operation(*clone_inputs(inputs))
    assert_outputs_close(actual, expected)
```

Use `TORCH_LOGS="recompiles"` to verify dimensions advertised as dynamic are reused rather than
silently specialized.

### CUDA Graphs

When CUDA Graph support is claimed:

1. Compile, autotune, and warm up before capture.
2. Allocate static inputs and outputs before capture.
3. Ensure capture performs no lazy compilation, allocation, CPU synchronization, or data-dependent
   host work.
4. Capture and replay representative static-shape calls.
5. Compare replayed outputs and final state against eager execution.

Do not infer CUDA Graph compatibility from a successful `torch.compile` call.

## Failure triage

- **Graph break:** run with `TORCH_LOGS="graph_breaks,recompiles"`; locate whether validation,
  dispatch, registration, or the backend launcher caused it.
- **FakeTensor mismatch:** compare real and fake metadata field by field; remove data reads and
  data-dependent Python from fake code.
- **Schema failure:** inspect mutation and aliasing before changing the schema.
- **Autograd registration failure:** ensure inputs require gradients, then separately validate
  numerical gradients.
- **AOT dynamic failure:** remove Python specialization on symbolic dimensions or narrow the
  supported dynamic-shape contract.
- **Compile-only mismatch:** compare eager and compiled calls to the same selected backend before
  debugging the low-level kernel.

## Required completion report

Report these independently:

- reference forward and gradient tests;
- `opcheck` cases and utilities exercised;
- public `torch.compile(fullgraph=True)` forward/backward results;
- dynamic sizes tested and recompilation observations;
- CUDA Graph capture/replay results when claimed;
- unsupported contracts or unavailable hardware with exact reasons.

Do not present lint, imports, registration-only checks, or `opcheck` alone as behavioral correctness.

## References

- PyTorch custom Python operators:
  <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>
- `torch.library` APIs, including `custom_op`, `register_fake`, `register_autograd`, and `opcheck`:
  <https://docs.pytorch.org/docs/stable/library.html>
- User-defined Triton kernels with `torch.compile`:
  <https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>
