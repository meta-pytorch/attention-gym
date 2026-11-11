# Compile cache

[Back to warpdrive](index.md)

Use `attn_gym.warpdrive.cache` to compile a CuTeDSL kernel once and reuse the compiled result.
It keeps results in memory for repeated calls and on disk for later processes. Every CuTeDSL
kernel in this repository uses it, including kernels that do not use warpdrive's warpspec features.

You normally need just two pieces: a function that describes how to compile your kernel, and
`@jit_cache` on that function. The cache handles saving and loading the result. Importing the
cache package does not import cutlass, so it is also safe to import from torch-only code paths.

## Compile through `jit_cache`

Keep **compile-time choices** separate from **runtime inputs**:

- The decorated function takes choices that determine the generated code, such as a kernel
  configuration. These arguments identify the compiled variant to reuse.
- Inside that function, fake tensors describe the tensors the kernel will accept at runtime.
  They specify shapes, layouts and dtypes without supplying real data. Use `cute.sym_int` for
  dimensions that should vary without compiling a new variant.

For example, given a `Gemm` kernel class:

```python
import cutlass
from cutlass import cute

from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor


@jit_cache
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

Here, `op` determines which kernel to build. The tensor dimensions do not re-compile:
`m` can vary at runtime, and `n` and `k` can vary while remaining multiples of eight. See the use of sym_int().
The kernel receives real tensors when you call the returned compiled function, not when you compile it.

Define the compile function at module scope, rather than inside another function. Its arguments
must capture all choices that affect the generated code. For warpdrive kernels, pass static
choices to the kernel constructor rather than assigning attributes afterwards.

`compile_tvm_ffi` accepts fake tensors, not real `torch.Tensor` inputs. Use
`make_fake_compact_tensor` for compact layouts or `make_fake_strided_tensor` for explicit strides.

### Launch on a different stream

The compiled function uses the current PyTorch stream **at launch time**, not the stream that
was current during compilation. `compile_tvm_ffi` arranges this through the TVM-FFI
environment-stream argument; you do not pass a stream when calling the compiled function.

To launch on another stream, use `torch.cuda.stream(...)`. Here, `a`, `b` and `out` are CUDA
tensors on the same device, allocated and prepared on that device's current stream:

```python
import torch

compiled = compile_gemm(op)
caller_stream = torch.cuda.current_stream(a.device)
work_stream = torch.cuda.Stream(device=a.device)

work_stream.wait_stream(caller_stream)  # wait for tensor preparation
with torch.cuda.stream(work_stream):
    compiled(a, b, out)
caller_stream.wait_stream(work_stream)  # later caller_stream work can safely use out
```

The context manager restores the previous stream on exit, but does not wait for the kernel.
The two `wait_stream` calls order work between streams without blocking Python. The same
compiled function can be reused on either stream; switching streams does not require a new
compile.

## Where entries live

On a normal call, the cache looks in memory first, then on disk. If neither has a matching
build, it compiles the kernel and saves the result. Concurrent processes coordinate through a
per-key file lock so they do not normally compile the same variant twice.

The disk directory is chosen in this order:

1. A path set with `set_cache_path(path)`. Call `set_cache_path(None)` to restore the default.
2. The `WARPDRIVE_CACHE_DIR` environment variable.
3. `$XDG_CACHE_HOME/warpdrive`, or `~/.cache/warpdrive` when `XDG_CACHE_HOME` is unset.

Call `get_cache_path()` to see the active directory. Parallel compilation workers use that same
directory.

There are two separate controls for bypassing cached results:

| Control | What it does |
| --- | --- |
| `CUTE_DSL_NO_CACHE=1` | Disables disk caching. Existing in-process entries can still be reused. |
| `compile_fn.cache_clear()` | Clears in-process entries and counters, but leaves disk entries. |

Neither control alone guarantees a new compile. To force one, combine them or use the
[`uncached` hook](#force-a-new-trace) for selected calls.

## What the key covers

A **cache key** identifies a compiled variant. The disk key accounts for:

- **What you compile:** the compile function's arguments and the compile target.
  `CUTE_DSL_ARCH` affects the key through that target.
- **What you compile with:** the Python, cutlass, tvm_ffi, torch and CUDA versions, plus the
  code-generation environment variables listed in `attn_gym/warpdrive/cache/_key.py`.
- **The source code it depends on:** the compile function's module and every `attn_gym` module
  it imports, directly or through other modules. The implementation calls this set
  `_key.module_closure`. Unrelated modules do not invalidate the entry.

Changing one of these inputs produces a different disk key, so the next process builds or
loads the corresponding variant. This is not live source reloading: source fingerprints are
cached within a process. Restart the process after editing source.

A warpdrive kernel contributes its constructor call, cluster, work streams, resolved roles and
trace settings to the key. See [compile keys](roles.md#compile-keys) for the full contract.

### Make source dependencies visible

The cache finds dependencies by reading import statements, not by observing which Python code
executes. Most code needs no special treatment, but the following rules keep source changes
from being missed:

- **Use ordinary imports for code the kernel traces.** Both `import` and `from` statements
  count, including local and relative imports. A literal call such as
  `importlib.import_module("attn_gym.x.y")` also counts; a computed module name does not.
- **Keep package `__init__.py` files to re-exports.** An import of `pkg.sub.module` does not
  automatically add the parent `pkg/sub/__init__.py` to the source hash.
- **Declare dependencies that imports cannot express.** Use `@jit_cache(extra_sources=(path,))`
  for files read during tracing or code outside `attn_gym` that is not covered by the dependency
  versions above. Do not list modules already reached by the compile module's imports.

An object passed to the compile function can define `__cute_cache_key__()` to describe its
static choices. A class passed directly as an argument is identified only by its module and
name, not by its source. If an object's key depends on a class outside the compile module's
imports, include `module_source_digest(cls.__module__)` in that key. Warpdrive kernels already
do this.

There is also a way to keep host-only utilities out of the source hash. A report renderer or
trace decoder that never runs during kernel tracing can be imported at call time with
`importlib.import_module(".mod", __package__)`. That relative dynamic import is not followed by
the source scan. **Do not use this for code that runs during tracing:** editing it would not
invalidate the cached build.

<a id="cached-builds-were-never-traced"></a>

## Debugging with cached builds

During compilation, CuTeDSL traces the Python kernel to generate code. That tracing can also
record useful information on the kernel object. A cache hit skips tracing, so those side
effects do not run again.

The cache can save and restore this information along with the build. An argument object opts
in with two methods:

- `__cute_trace_record__()` returns a picklable record of the information collected.
- `__cute_restore_trace__(record)` restores it onto the object receiving a cache hit.

Normal memory and disk hits restore these records. Warpdrive uses them to restore `describe()`,
TMA and alias reports, and warnings. The advanced `by_static_key` and `by_args` lookup paths
below do not restore records on their fast-path hits.

### Force a new trace

Sometimes you need tracing to run again, rather than restoring a saved report. The `uncached`
hook lets you select those calls without disabling caching for everything else.

For example, warpdrive's [protocol tracing](tracing.md#trace-the-protocol) records where a
kernel acquires, commits, waits on and releases channel slots. These are source locations
collected during compilation, not a timeline of GPU execution. The examples use this hook to
get a fresh protocol log when `op.trace_protocol` is enabled:

```python
@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_gemm(op: Gemm):
    ...  # same compile body as above
```

`uncached` receives the same arguments as the compile function. Returning `True` compiles fresh
without reading or filling either cache; returning `False` uses normal caching. The hook is
explicit: `jit_cache` does not check `trace_protocol` on its own. Use plain `@jit_cache` unless
your compile helper needs this behavior.

## Test specialization reuse

To check that changing runtime shapes does not create unnecessary variants, inspect
`compile_fn.cache_info()` before and after calls with different shapes:

| Field | Meaning |
| --- | --- |
| `currsize` | Distinct variants held in the function's normal in-process cache. |
| `hits` | Calls served by an existing memory or disk entry. |
| `misses` | Calls that compiled a result, including calls selected by `uncached`. |

`currsize` includes both locally compiled variants and variants loaded from disk. For shapes
that should share a variant, assert that `currsize` stays unchanged after the first call. Include
sizes of `1`, multiples of `16`, other values, and any relevant bucket boundaries to catch
accidental specialization. `hits + misses` counts calls through the cache lookup
APIs, not GPU launches of the returned function.

Leave disk caching enabled for this test: loading a saved build still lets you check whether
the shapes share a key. `CUTE_DSL_NO_CACHE` adds compilation work without helping that check.
Use `compile_fn.cache_clear()` to reset the in-process entries and counters between tests.

## Advanced lookup and parallel compilation

### Reuse a caller-supplied key

If a launch path already has a small static key, it can avoid reconstructing and hashing the
full compile arguments on each lookup:

- `compile_fn.by_static_key(key, call)` uses your key. On a miss, it invokes the zero-argument
  `call`, which must call the decorated compile function. Your key must fully determine that
  call's arguments; the cache adds the compile target.
- `compile_fn.by_args(*args)` uses the positional arguments themselves as the static key.
  They must be hashable. Top-level values are tagged by type; within nested tuples, keep each
  position's type consistent because those values compare by equality.

These fast-path hits return only the compiled function. They do not restore trace records or
reevaluate `uncached`, so use them only when reusing the same compiled arguments and when no
fresh trace is needed.

### Compile several variants in parallel

Use `compile_many(fn, calls)` to compile variants in worker processes and then load the results
in the caller. Use `precompile_many(fn, calls)` if you only want to fill the disk cache. Both
require disk caching to be enabled.

Each item in `calls` is a tuple of positional arguments, such as `(config,)` for a function
that takes one configuration. The arguments must be picklable, and configuration classes must
be defined at module scope so workers can import them. Warpdrive kernels are pickled as their
constructor call.

## CuTeDSL versions

Warpdrive supports `nvidia-cutlass-dsl` 4.6.2 through 4.8. Use the compatibility helpers in
`attn_gym/warpdrive/compat.py` for APIs that differ between versions, rather than putting
version branches in kernels. Keep version checks such as `cutedsl_version_matches` in host
code, outside kernels and launch hot paths.

For example, use `attn_gym.warpdrive.advance_state(state, n)` to advance a raw `PipelineState`
by several epochs. Calling `state.advance(n)` directly requires CuTeDSL 4.8.

The cache key includes the CuTeDSL version, so different versions can safely share a disk
cache. For independent environment runs, give each environment its own `WARPDRIVE_CACHE_DIR`
and `CUTE_DSL_CACHE_DIR` anyway.
