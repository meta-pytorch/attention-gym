---
name: cutedsl-tunable-kernel-template
description: Adds an Attention Gym tuning adapter to a CuTeDSL op using typed input-aware configs, cached fake-tensor TVM-FFI compilation, parallel candidate compilation, and sequential GPU benchmarking. Use after the core kernel exists and needs a public default, explicit-config, or autotune path.
---

# Tunable CuTeDSL Kernel Template

Use this after `cutedsl-kernel-template` establishes the core kernel. This skill owns the Attention Gym adapter and parent/compiler-process boundary. Load `cutedsl-performance` separately to design the search space, measure candidates, and accept a selector.

The canonical Attention Gym helpers are:

```python
from attn_gym._backends.cute import benchmark_gpu, compile_tvm_ffi, jit_cache, run_tunable, tune
```

If another repo has equivalent helpers, reuse them. Do not rebuild cache/process orchestration inside each op.

## Ownership Boundary

Keep each value in one domain:

| Value | Owner |
|---|---|
| Real `torch.Tensor` inputs and outputs | Parent process and public wrapper |
| Cohesive runtime tensor/output bundle | One parent-only `NamedTuple` or dataclass when the flat argument list grows |
| Candidate generation from shapes/dtypes/device facts | `configs(*runtime_args)` in the parent |
| Benchmark-winner reuse identity | Host-static `tuning_key(*runtime_args, target=...)` result |
| Static, pickleable specialization values | Config record and `compile_call(...)` result |
| Fake CuTe tensors matching the runtime ABI | Cached `compile(...)` method |
| Fake environment stream and typed TVM-FFI option | `compile_tvm_ffi(...)` |
| Compiled callable launch and benchmarking | Parent process through `launch(...)` |

Never pass real tensors to `compile(...)` or compiler workers. `compile_call(...)` is the explicit projection from runtime inputs to static compile arguments.

## Minimal Kernel Convention

Use a module-scope `NamedTuple` or frozen dataclass for codegen choices. Module scope makes the config importable and pickleable by fresh compiler processes.

When warps have named protocol responsibilities, represent them with a module-scope `IntEnum`
such as `WarpRole.TMA_PRODUCER`; do not compare warp indices with unexplained integer literals. Name
the responsibility precisely when one warp performs more than one role.

Make approximation choices such as `fastmath` explicit compile-time arguments. Default them to
`False` unless the public numerical contract deliberately chooses approximate math, encode them in
cache and profiler names, and correctness-test every exposed mode.

Put ordinary specialization defaults directly in the owning constructor or public function
signature. Avoid module constants that only alias those defaults or a one-kernel policy limit; keep
fixed compile-time expressions next to the device code or schedule that consumes them. Derive
one-use schedule values in the owning op rather than adding free helper functions.

Do not repeat dtype assertions inside a CuTeDSL entrypoint when the cached compiler boundary already
constructs an exact fake-tensor ABI and the runtime operator validates that ABI. Keep small helpers
only when they are reused or mark a real protocol, cache, or runtime-ABI boundary; use established
integer utilities such as `ceildiv` directly instead of wrapping one arithmetic expression.

```python
from typing import NamedTuple

from attn_gym._backends.cute.target import CompileTarget


class MyConfig(NamedTuple):
    threads: int
    tile_size: int
```

A tunable adapter supplies seven distinct responsibilities. When its launch ABI has several values,
make that ABI an `Args` type owned by the adapter rather than a separate module-level private type:

```python
class MyOp:
    class Args(NamedTuple):
        q: torch.Tensor
        k: torch.Tensor
        output: torch.Tensor
        workspace: torch.Tensor

    @staticmethod
    def default_config(args: Args, *, target: CompileTarget) -> MyConfig:
        """Return a deterministic valid config without benchmarking."""
        return MyConfig(threads=128, tile_size=256)

    @staticmethod
    def tuning_key(args: Args, *, target: CompileTarget) -> tuple[int]:
        """Identify workloads that may safely reuse one benchmark winner."""
        return (args.q.shape[1],)

    @staticmethod
    def configs(args: Args) -> tuple[MyConfig, ...]:
        """Return valid candidates derived from actual runtime inputs."""
        ...

    @staticmethod
    @jit_cache
    def compile(*static_args):
        """Construct the fake ABI and compile one specialization."""
        ...

    @staticmethod
    def compile_call(config: MyConfig, args: Args) -> tuple:
        """Project launch arguments to static arguments for compile(...)."""
        ...

    @staticmethod
    def launch(compiled, config: MyConfig, args: Args):
        """Launch in the parent with real tensors and return the public result."""
        ...
```

Keep these responsibilities distinct, but they need not all live on the CuTeDSL op class. A
constructor-heavy DSL op may stay focused on layouts and device code while a small adapter owns the
seven static/class methods above.

- `default_config(*runtime_args, target=...)` owns deterministic no-tune selection. It may inspect
  tensor metadata, static operation arguments, and target facts, and must return a valid config for
  every input accepted by the adapter without benchmarking or reading device-resident tensor values.
  Keep this policy on the adapter rather than selecting a default in the parent wrapper. A constant
  fallback simply ignores its arguments. Name canonical mode configs after the mode rather than
  calling an architecture-specific config the default.
- `tuning_key(*runtime_args, target=...)` defines when an autotuned winner may be reused. Return
  host-visible tensor metadata and target facts only; never read device values or synchronize, because
  the key must remain valid during CUDA Graph replay. Use `()` when `compile_call(...)` already
  distinguishes every relevant workload. Exact keys are the safe starting point; introduce buckets
  only after measurements establish stable winner regions.
- `configs(...)` may inspect runtime shape, stride, dtype, alignment, or device facts.
- `compile(...)` owns fake tensors and the cached artifact boundary.
- `compile_call(...)` prevents runtime tensors from leaking into cache keys or subprocess payloads.
- `launch(...)` binds the compiled callable to real tensors for correctness checks and timing.

The protocol accepts positional runtime arguments for small ABIs. Keep one or two naturally named
values positional; wrapping them in `Args` would add ceremony without clarifying ownership. Do not,
however, grow a positional list of inputs, outputs, and static semantics indefinitely. Once several
values form one launch ABI, pass one parent-only `Args` value. Nest `Args` in its adapter when no
other component owns that ABI; use a descriptive top-level type when multiple components genuinely
share it. Do not create a module-level `_Runtime` or `_Args` type merely to signal privacy. Reserve
“runtime” for an execution environment or lifecycle-bearing object, not a passive argument tuple.
`compile_call(...)` projects `Args` to static values, while `launch(...)` binds its real tensors to
the compiled callable.

Nesting communicates ownership, not public API status. If an adapter has a normal class name but is
an implementation detail, define the module's `__all__` explicitly and list only the supported
wrapper and configuration types. Do not use leading underscores as a substitute for deciding and
documenting the module's actual public surface.

## Public Wrapper

Users call one ordinary PyTorch function; they do not construct op objects or compiled callables.

```python
def my_op(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    config: MyConfig | None = None,
    tune: bool = False,
    configs: Iterable[MyConfig] | None = None,
) -> torch.Tensor:
    args = MyOp.Args(
        q,
        k,
        torch.empty_like(q),
        torch.empty_like(q, dtype=torch.float32),
    )
    return run_tunable(
        MyOp,
        args,
        config=config,
        autotune=tune,
        configs=configs,
    )[0]
```

This gives three intentional modes:

```python
my_op(q, k)                                      # conservative default
my_op(q, k, config=MyConfig(64, 128))            # force one specialization
my_op(q, k, tune=True)                           # input-aware candidate method
my_op(q, k, tune=True, configs=(cfg_a, cfg_b))   # explicit candidate override
```

Reject `config=` with tuning and reject `configs=` without tuning rather than silently ignoring either argument.
If target metadata is supplied explicitly, install it before generating candidates so `configs(...)`
and compilation observe the same target. On heterogeneous multi-GPU hosts, derive that target from
the runtime tensor's device rather than the process's current device. The installed target is
process-global and sticky; callers that temporarily override it must restore the previous target.

## Compile Contract

Inside `compile(...)`:

1. Accept only static values that completely determine generated code and the fake ABI. Keep them
   structurally stable so `jit_cache` can encode them for warm process-local lookups before
   constructing a persistent hash or cache path. When the call tuple is not the right specialization
   identity, pass `cache_key=` a pure function returning a complete hashable tuple or frozen config.
   Return the structural key itself, never `hash(key)`; `jit_cache` adds function, target, and source
   identity and uses the same structural key for memory and disk caching.
2. Keep batch-, token-, and head-count-like extents symbolic with `cute.sym_int()`/`cute.sym_int64()`
   when their dependent strides also remain dynamic. Bake a dimension into `compile_call(...)` only
   when it changes layout/stride address arithmetic, tiling, vectorization, block shape, shared-memory
   sizing, compile-time control flow, or another generated-code decision.
3. Build fake compact tensors with runtime-compatible shapes, strides, dtype, and assumed alignment.
4. Instantiate any internal CuTeDSL op object there; never require public callers to construct it.
5. Give `compile_tvm_ffi(...)` a stable lowercase name encoding every static compile argument.
   Class entrypoints may expose `get_name()` instead of passing `name=` explicitly.
6. Return the compiled TVM-FFI callable directly.

```python
@staticmethod
@jit_cache
def compile(config: MyConfig):
    num_elements = cute.sym_int()
    source = cute.runtime.make_fake_compact_tensor(
        ..., (num_elements,), stride_order=(0,), assumed_align=16
    )
    destination = cute.runtime.make_fake_compact_tensor(
        ..., (num_elements,), stride_order=(0,), assumed_align=16
    )
    op = _MyOp()
    return compile_tvm_ffi(
        op._jit_entrypoint,
        source,
        destination,
        config.threads,
        config.tile_size,
        name=op.get_name(config),
    )
```

`compile_tvm_ffi` owns the typed TVM-FFI option and fake environment stream. Do not append another stream or pass string compiler options at call sites. `compile_call(...)` returns exactly one tuple of positional static arguments; use `(config,)` when `compile(...)` accepts only a config.

### Int64 offset specialization

Keep ordinary layouts on the int32 address path. At the parent wrapper, call
`attn_gym._backends.cute.requires_int64_abi` on every ABI-visible input, output, and optional tensor,
then pass the result through `compile_call(...)` as a static `use_int64_offsets` argument. The CuTe
predicate is intentionally stricter than reachable cosize: TVM-FFI must represent every declared
stride, including a stride larger than `INT32_MAX` on a size-one mode that cannot reach it.

The width bool must participate in the `jit_cache` key and profiler/artifact name. In the wide
specialization, use `cute.sym_int64` for each dynamic fake-tensor dimension or stride involved in
addressing; the fake signature and runtime tensor ABI must agree. Inside the op, widen each dynamic
index or origin before its first potentially overflowing multiply or addition. Casting the final
layout stride, iterator offset, or pointer is too late. Audit manually rebuilt layouts, iterator
addition, chunk/program indices, and stores as well as ordinary tensor loads. Bounded routing arrays
may remain int32 when their values and their own addressing are independently proven safe.

Do not infer width from `numel()` and do not make int64 unconditional: wider arithmetic can add
instructions and register pressure. Validate predicate-only oversized singleton strides, force the
i64 specialization on small inputs and compare it with i32, and, when memory permits, execute an
active offset beyond `INT32_MAX` against an equivalent compact layout. See
`test/test_kda_int64_offsets.py` for the project pattern.

Validate reachable inputs and static semantics once at the eager boundary used by each path: the
public tune path or the private custom-op implementation. Validate again in `compile(...)`, because
cache/compiler-worker calls can bypass both wrappers; downstream launch helpers may then rely on the
allocated runtime ABI. Keep FakeTensor-incompatible checks such as `data_ptr()` alignment inside the
opaque/eager launcher rather than a trace-time validator.

## Tune Flow

For `tune=True`, `run_tunable` should:

1. Use the explicit `configs=` iterable, otherwise call `kernel.configs(*runtime_args)` once.
2. Map each candidate through `compile_call(...)`.
3. Populate cold disk-cache entries in parallel compiler workers.
4. Load and benchmark candidates in iteration order in the CUDA-owning parent.
5. Compile/load the winner through the same cache boundary and perform one final launch.

Before enabling tuning, force every generated candidate through `config=` in correctness tests and compare it with an independent reference; a benchmark cannot detect a wrong fast candidate. Direct benchmarking also assumes repeatable launches.

`run_tunable(...)` performs a final launch after benchmarking. For a destructive or accumulating op,
call `tune(...)` directly, restore all inputs and outputs after it returns, then execute the winner once
through the ordinary explicit-config path. A benchmark callback that restores only between samples is
insufficient because it cannot prepare state for the final launch.

A fully warm run must not start the compiler process. It still benchmarks requested candidates unless a separate baked selector chooses one without tuning.

## `torch.compile` Boundary

Cache lookup, target discovery, compilation, and tuning are eager host operations. When a public
function must support strict Dynamo capture, hide the ordinary no-tune launcher behind a private
functional `torch.library.custom_op` and register a fake implementation whose output shapes derive
symbolically from input shapes and static scalars.

Project a config to schema-supported scalars at that boundary; decode it inside the opaque op. Use an
optional scalar for target-resolved automatic selection rather than querying the target in the traced
wrapper. Keep `tune=True` eager-only and bypass the custom op. If output shape changes at a static
bucket boundary, Dynamo may legitimately compile another graph even with `dynamic=True`.

## Example

Read or run [copy_reads_example.py](copy_reads_example.py) for a complete toy kernel using an input-aware `ReadConfig` search space and a single public `copy_reads(...)` entrypoint.

## Validation

- Put the cache under a temporary directory in tests.
- Test default, explicit config, generated candidates, and explicit candidate override.
- Correctness-check every candidate against an independent reference before benchmarking.
- Verify all cold candidates produce distinct artifacts and a warm run launches no compiler process.
- Keep real GPU coverage tiny: compile a toy kernel, benchmark with Inductor's GPU benchmarker, launch the winner, and compare output with an independent reference.
- Measure cold and warm wall time locally, but do not make noisy scaling timing a correctness assertion.
- For address-width-specialized kernels, verify the compile projection/cache/name distinguish i32 and i64, ordinary inputs retain i32, and every generated config is correct under both widths.
