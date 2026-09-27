# Third-party notice

The upstream `NOTICE`, `LICENSE.txt` (Apache-2.0), `LICENSE-MIT.txt`, `LICENSING.md`, and
`THIRD_PARTY_LICENSES.txt` in this directory are copied unchanged from NVIDIA's `cudnn-frontend`
repository at tag `v1.30.0`.

## Attention Gym adaptation

The scalar-GDN and KDA linear-attention kernels (`kernel/`), their shared helpers (`common/`), and
the low-level tile helpers (`tile_dsl/`) are vendored from
`python/cudnn/linear_attention/frost/{kernel,common}` and `python/cudnn/frost/tile_dsl` at that tag.
Only the modules the GDN and KDA launch paths import are kept; GDN2, GDP (except the backward fork
the GDN chain prologue imports), SDPA, and the upstream engines are omitted.

Changes made for Attention Gym:

- Imports were moved from `cudnn.frost.*` into this package. Host device queries use PyTorch;
  compilation uses module-level `jit_cache` functions and fake TVM-FFI tensor signatures from
  `common/tvm_ffi.py`, without allocating live placeholder tensors or importing cuDNN host shims.

Modified upstream files carry an explicit modification notice and keep their original SPDX
identifiers.
