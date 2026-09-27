# Third-party notice

The upstream `NOTICE`, `LICENSE.txt` (Apache-2.0), and `LICENSE-MIT.txt` in this directory are
copied unchanged from NVIDIA's `cudnn-frontend` repository at tag `v1.30.0`.

## Attention Gym adaptation

The scalar-GDN forward and backward kernels, their chain/warmup launch hosts, and the supporting
`common/` and `tile_dsl/` helpers are vendored from `python/cudnn/linear_attention/frost/` at that
tag. Only the modules the GDN launch paths import are kept.

Changes made for Attention Gym:

- imports were moved from `cudnn.frost.*` into this package, and the cuDNN host buffer/device
  utilities were replaced by the Torch-backed shims in `_compat.py`;
- `gdn.py` is a new driver that mirrors the upstream engine's plan choice (uncut, d_v split, exact
  piece chain, or approximate warmup split) and adds a minimum piece length for the exact chain;
- `common/split_k.py` compacts empty `cu_seqlens` intervals out of the unsplit work table, and
  `common/thd.py` skips TMA descriptor builds for empty sequences;
- the launch compiles are persisted through Attention Gym's `jit_cache` (`_persist.py`).

Modified upstream files carry an explicit modification notice and keep their original SPDX
identifiers.
