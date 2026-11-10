# Compile cache

[Back to warpdrive](index.md)

`attn_gym.warpdrive.cache` compiles CuTeDSL kernels once and reuses them in memory and on disk
across processes. Every CuTeDSL kernel in this repository, warpdrive or not, compiles through it.
Importing it does not import cutlass, so torch-only code paths can use it.

## Compile through `jit_cache`

Decorate a module-level function whose arguments are static values that fully determine the
generated code. Describe the runtime signature with fake tensors; runtime sizes stay
`cute.sym_int` so one build serves every shape:

```python
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor


@jit_cache(uncached=lambda op: op.trace_protocol)  # protocol logs need a real trace
def compile_gemm(op: Gemm):
    m = cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        name=op.get_name(),
    )
```

`compile_tvm_ffi` appends the TVM-FFI environment stream, so the compiled function runs on the
current PyTorch stream. It rejects real `torch.Tensor` arguments; use
`make_fake_compact_tensor` or `make_fake_strided_tensor`. Warm launches that already know their
static key can skip key hashing with `compile_fn.by_static_key(key, call)` or
`compile_fn.by_args(*args)`.

`compile_many(fn, calls)` compiles several variants in parallel worker processes and then
loads them in the caller; `precompile_many` only fills the disk cache. Arguments must pickle,
and config classes must live at module scope. warpdrive kernels pickle as their constructor
call.

## Where entries live

Entries are stored under, in order of precedence:

1. `set_cache_path(path)` (`None` restores the default);
2. `WARPDRIVE_CACHE_DIR`;
3. `$XDG_CACHE_HOME/warpdrive` (`~/.cache/warpdrive`).

`get_cache_path()` returns the active directory; `compile_many` workers fill the same one. Set
`CUTE_DSL_NO_CACHE=1` to bypass the disk cache; the in-process cache still serves repeated calls
(`compile_fn.cache_clear()` or `jit_cache(uncached=...)` forces a new trace). Concurrent processes
compiling the same key wait on a per-key file lock instead of compiling twice.

## What the key covers

A disk entry is reused until one of these changes:

- the compile function's arguments and the compile target (`CUTE_DSL_ARCH` reaches the key
  through the target);
- the Python, cutlass, tvm_ffi, torch and CUDA versions, and the codegen environment variables
  listed in `attn_gym/warpdrive/cache/_key.py`;
- the source of the compile function's module and of every `attn_gym` module it imports,
  directly or transitively (`_key.module_closure`). Unrelated modules do not invalidate it.

Imports are found statically, so keep the code a kernel traces reachable by import:

- Import helpers, constants and ops with ordinary `import`/`from` statements; local and
  relative imports count. A string literal module name, as in
  `importlib.import_module("attn_gym.x.y")`, also counts; a computed name does not.
- `import pkg.sub.module` does not hash `pkg/sub/__init__.py`. Keep package `__init__` files
  to plain re-exports.
- Pass `extra_sources=(...)` only for inputs the scan cannot see: files read while tracing, or
  code outside `attn_gym` other than the versioned dependencies above. Modules the compile
  module already imports are hashed; do not list them.
- A class passed as an argument keys by module and name only. An argument object can supply its
  own key with `__cute_cache_key__()`; to cover a class defined outside the compile module's
  imports, include `module_source_digest(cls.__module__)` there, as warpdrive kernels do.
- Host-only code that tracing never runs (report renderers, trace decoders) stays out of the key
  when imported at call time by relative name, `importlib.import_module(".mod", __package__)`.
  Never import traced code this way: edits to it would not recompile.

For a warpdrive kernel the key is its constructor call plus cluster, work streams, resolved
roles and trace settings; see [compile keys](roles.md#compile-keys). Pass static choices to the
constructor, not as attributes assigned afterwards.

## Cached builds were never traced

A build loaded from disk skips tracing, so anything tracing would have recorded on the
arguments is missing. An argument that defines `__cute_trace_record__()` (a picklable record)
and `__cute_restore_trace__(record)` gets the record of its build back on every memory or disk
hit; warpdrive uses this to restore `describe()`, TMA/alias reports and warnings.
`by_static_key` hits restore nothing. When the trace itself is the point, select those calls
with `jit_cache(uncached=predicate)`: they compile without reading or filling either cache.

## Test specialization reuse

Assert on `compile_fn.cache_info()`: `currsize` counts distinct keys built in this process and
`hits + misses` counts launches. Vary shapes across the specialization classes and check that
`currsize` stays put. Do not disable the disk cache with `CUTE_DSL_NO_CACHE` for this; it only
forces cold compiles. `compile_fn.cache_clear()` resets the in-process cache and counters.

## CuTeDSL versions

warpdrive supports `nvidia-cutlass-dsl` 4.6.2 through 4.8. Renamed or version-dependent
CuTeDSL APIs live in `attn_gym/warpdrive/compat.py`; import them from there rather than
branching on the version in a kernel. Version predicates (`cutedsl_version_matches`) belong in
host code, not in a kernel or a launch hot path. For example, advance a raw `PipelineState` by
several epochs with `attn_gym.warpdrive.advance_state(state, n)`: `state.advance(n)` is new in
4.8.

The key includes the CuTeDSL version, so several versions can share one cache directory; give
each environment its own `WARPDRIVE_CACHE_DIR` and `CUTE_DSL_CACHE_DIR` anyway to keep their runs
independent.
