---
name: cutedsl-tunable-kernel-template
description: Adds an Attention Gym tuning adapter to a CuTeDSL op using typed input-aware configs, cached fake-tensor TVM-FFI compilation, parallel candidate compilation, and sequential GPU benchmarking. Use after the core kernel exists and needs a public default, explicit-config, or autotune path.
---

# Tunable CuTeDSL Kernel Template

Use this after `cutedsl-kernel-template` establishes the core kernel. This skill owns the Attention Gym adapter and parent/compiler-process boundary. Load `cutedsl-performance` separately to design the search space, measure candidates, and accept a selector.

The canonical Attention Gym helpers are:

```python
from attn_gym._backends.cute import compile_tvm_ffi, jit_cache, run_tunable
```

If another repo has equivalent helpers, reuse them. Do not rebuild cache/process orchestration inside each op.

## Ownership Boundary

Keep each value in one domain:

| Value | Owner |
|---|---|
| Real `torch.Tensor` inputs and outputs | Parent process and public wrapper |
| Candidate generation from shapes/dtypes/device facts | `configs(*runtime_args)` in the parent |
| Static, pickleable specialization values | Config record and `compile_call(...)` result |
| Fake CuTe tensors matching the runtime ABI | Cached `compile(...)` method |
| Fake environment stream and typed TVM-FFI option | `compile_tvm_ffi(...)` |
| Compiled callable launch and benchmarking | Parent process through `launch(...)` |

Never pass real tensors to `compile(...)` or compiler workers. `compile_call(...)` is the explicit projection from runtime inputs to static compile arguments.

## Minimal Kernel Convention

Use a module-scope `NamedTuple` or frozen dataclass for codegen choices. Module scope makes the config importable and pickleable by fresh compiler processes.

```python
from typing import NamedTuple


class MyConfig(NamedTuple):
    threads: int
    tile_size: int
```

The internal op supplies five distinct pieces:

```python
class _MyOp:
    default_config = MyConfig(threads=128, tile_size=256)

    @staticmethod
    def configs(*runtime_args) -> tuple[MyConfig, ...]:
        """Return valid candidates derived from actual runtime inputs."""
        ...

    @staticmethod
    @jit_cache
    def compile(*static_args):
        """Construct the fake ABI and compile one specialization."""
        ...

    @staticmethod
    def compile_call(config: MyConfig, *runtime_args) -> tuple:
        """Project runtime inputs to static arguments for compile(...)."""
        ...

    @staticmethod
    def launch(compiled, config: MyConfig, *runtime_args):
        """Launch in the parent with real tensors and return the public result."""
        ...
```

These members should not be merged:

- `default_config` provides a no-tuning path.
- `configs(...)` may inspect runtime shape, stride, dtype, alignment, or device facts.
- `compile(...)` owns fake tensors and the cached artifact boundary.
- `compile_call(...)` prevents runtime tensors from leaking into cache keys or subprocess payloads.
- `launch(...)` binds the compiled callable to real tensors for correctness checks and timing.

## Public Wrapper

Users call one ordinary PyTorch function; they do not construct op objects or compiled callables.

```python
def my_op(
    x: torch.Tensor,
    *,
    config: MyConfig | None = None,
    tune: bool = False,
    configs: Iterable[MyConfig] | None = None,
) -> torch.Tensor:
    y = torch.empty_like(x)
    result, _selected = run_tunable(
        _MyOp,
        x,
        y,
        config=config,
        autotune=tune,
        configs=configs,
    )
    return result
```

This gives three intentional modes:

```python
my_op(x)                                      # conservative default
my_op(x, config=MyConfig(64, 128))            # force one specialization
my_op(x, tune=True)                           # input-aware candidate method
my_op(x, tune=True, configs=(cfg_a, cfg_b))   # explicit candidate override
```

Reject `config=` with tuning and reject `configs=` without tuning rather than silently ignoring either argument.

## Compile Contract

Inside `compile(...)`:

1. Accept only static values that completely determine generated code and the fake ABI.
2. Build fake compact tensors with runtime-compatible shapes, strides, dtype, and assumed alignment.
3. Instantiate any internal CuTeDSL op object there; never require public callers to construct it.
4. Call `compile_tvm_ffi(...)` with a stable lowercase config-derived name.
5. Return the compiled TVM-FFI callable directly.

```python
@staticmethod
@jit_cache
def compile(num_elements: int, config: MyConfig):
    source = cute.runtime.make_fake_compact_tensor(...)
    destination = cute.runtime.make_fake_compact_tensor(...)
    op = _MyOp()
    return compile_tvm_ffi(
        op._jit_entrypoint,
        source,
        destination,
        num_elements,
        config.threads,
        config.tile_size,
        name=op.get_name(num_elements, config),
    )
```

`compile_tvm_ffi` owns the typed TVM-FFI option and fake environment stream. Do not append another stream or pass string compiler options at call sites. `compile_call(...)` returns exactly one tuple of positional static arguments; use `(config,)` when `compile(...)` accepts only a config.

## Tune Flow

For `tune=True`, `run_tunable` should:

1. Use the explicit `configs=` iterable, otherwise call `kernel.configs(*runtime_args)` once.
2. Map each candidate through `compile_call(...)`.
3. Populate cold disk-cache entries in parallel compiler workers.
4. Load and benchmark candidates sequentially in the CUDA-owning parent.
5. Compile/load the winner through the same cache boundary and perform one final launch.

Before enabling tuning, force every generated candidate through `config=` in correctness tests and compare it with an independent reference; a benchmark cannot detect a wrong fast candidate. Direct benchmarking also assumes repeatable launches. Destructive or stateful ops must provide a `benchmark=` hook that restores inputs between measurements.

A fully warm run must not start the compiler process. It still benchmarks requested candidates unless a separate baked selector chooses one without tuning.

## Example

Read or run [copy_reads_example.py](copy_reads_example.py) for a complete toy kernel using an input-aware `ReadConfig` search space and a single public `copy_reads(...)` entrypoint.

## Validation

- Put the cache under a temporary directory in tests.
- Test default, explicit config, generated candidates, and explicit candidate override.
- Correctness-check every candidate against an independent reference before benchmarking.
- Verify all cold candidates produce distinct artifacts and a warm run launches no compiler process.
- Keep real GPU coverage tiny: compile a toy kernel, benchmark with Inductor's GPU benchmarker, launch the winner, and compare output with an independent reference.
- Measure cold and warm wall time locally, but do not make noisy scaling timing a correctness assertion.
