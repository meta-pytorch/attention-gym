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
3. **CuTeDSL or external implementation:** use a private `torch.library.custom_op` when the launcher
   cannot be traced correctly.

The documented public function remains an ordinary Python function that owns semantic validation,
mode selection, and backend dispatch. Compiling only a private launcher does not prove that the
public operation is compile-friendly.

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

## Opaque custom operator pattern

```python
import torch
from torch import Tensor


@torch.library.custom_op("attn_gym::example_forward", mutates_args=())
def example_forward(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return launch_backend(query, key, value)


@example_forward.register_fake
def example_forward_fake(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return torch.empty_like(value)
```

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

Declare every mutated argument:

```python
@torch.library.custom_op(
    "attn_gym::update_state",
    mutates_args={"state"},
)
def update_state(state: Tensor, value: Tensor) -> None:
    launch_state_update(state, value)
```

Do not declare an operator functional if its launcher writes into an input, including recurrent
state, cache, workspace, or output buffers. Prefer functional operators when practical.

### Autograd

Training backends must register an autograd formula:

```python
def setup_context(ctx, inputs, output):
    query, key, value = inputs
    ctx.save_for_backward(query, key, value)


def backward(ctx, grad_output):
    query, key, value = ctx.saved_tensors
    return backward_formula(query, key, value, grad_output)


example_forward.register_autograd(backward, setup_context=setup_context)
```

The backward formula must itself use operations understood by PyTorch. Return one gradient entry per
forward input and `None` for nondifferentiable inputs. If training is unsupported, reject
requiring-gradient inputs clearly instead of silently detaching them.

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
