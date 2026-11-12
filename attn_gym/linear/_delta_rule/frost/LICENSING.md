# Licensing

The frost kernels are derived from NVIDIA's `cudnn-frontend` v1.30.0
(`python/cudnn/linear_attention/frost/`) and rewritten for Attention Gym on warpdrive. Each
source file's header gives its licence (an SPDX `SPDX-License-Identifier:` tag, or the
repository's BSD header):

| Tag | Files | Terms |
| --- | --- | --- |
| `Apache-2.0` | the kernels, `work_tables.py` and `blockwise_inverse.py` | [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE) |
| `MIT` | `ptx.py`, `umma.py` (derived from the upstream tile helpers) | [LICENSE-MIT.txt](LICENSE-MIT.txt) |
| `BSD-3-Clause` | `contract.py`, `ragged.py` (Attention Gym code) | the repository [LICENSE](../../../../LICENSE) |

`LICENSE.txt`, `LICENSE-MIT.txt` and `NOTICE` are copied unchanged from `cudnn-frontend` at tag
`v1.30.0`. `NOTICE` also names upstream's `LICENSING.md` and `THIRD_PARTY_LICENSES.txt`; they
are not reproduced here because no file in this package derives from the third-party code they
cover.
