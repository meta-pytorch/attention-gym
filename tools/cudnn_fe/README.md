# cudnn-frontend maintenance tools

Tools for upgrading the vendored cuDNN GDN/KDA kernels in
`attn_gym/linear/_delta_rule/cudnn_fe/`. The upgrade procedure, stack layering, restyle rules
and the ledger of every Attention Gym modification are in
[MAINTENANCE.md](../../attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md); read it first.

| Step | Tool | Role |
|---|---|---|
| 1 churn | `vendor.py --diff-upstream OLD NEW` | per-file upstream churn of the kernel closure between two tags |
| 2 vendor | `vendor.py --rev TAG` (`closure.py`) | verbatim closure, import relocation, notices, licenses; `--verify COMMIT` checks it against a past drop (`auto`: the v1.30 drop, found by subject) |
| 3 restyle | `restyle.py --check/--write --root <candidate> --files <changed...>`, `audit.py --strict --root <candidate>` | apply the mechanical restyle; list upstream patterns still to restyle; `audit_allowlist.txt` holds the documented exceptions |
| 4 SASS gate | [`sass/`](sass/README.md) (`python -m tools.cudnn_fe.sass snapshot/diff`) | snapshot cubins through the AG drivers and diff SASS/resources per commit |
| 7 races | [`cutracer/`](cutracer/README.md) (`python -m tools.cudnn_fe.cutracer.oracle`, `python -m tools.cudnn_fe.cutracer.stress`) | CUTracer race and delay stress for the pipelined kernels |
| 8 perf | [`bench/`](bench/README.md) (`python -m tools.cudnn_fe.bench`) | bench gate against the previous stack tip (2% default threshold) |
| 5, 9 fixes | `verify_fixes.py`, `fixes.toml`, [`upstream/`](upstream/README.md) | run every ledger fix's guarding tests; run the DRAFT upstream repros against a stock cudnn frontend |

The vendor/closure tools read upstream from a cudnn-frontend clone (`--upstream` or
`$CUDNN_FE_UPSTREAM`) via `git show`, so no tag checkout is needed. `vendor.py` writes to `agent_space/cudnn_fe_vendor/<tag>`
unless `--dest` is given; it never writes into `attn_gym/` by default.

```bash
python -m tools.cudnn_fe.vendor --rev v1.30.0 --upstream ~/cudnn-frontend --verify auto
python -m pytest -n 6 test/test_cudnn_fe_tools_*.py   # CPU-only; --verify needs $CUDNN_FE_UPSTREAM
```

The editable install exposes `tools`, so imports work outside the repository root without
pytest `pythonpath` overrides; the runtime wheel still contains only `attn_gym`. After changing
packaging, reinstall with `uv pip install --python .venv/bin/python --no-deps -e .`.
Modal mounts tools, tests and package documentation too.

## Checking fixes after an upgrade

`fixes.toml` is the single source for every ledger row: commit subjects, guarding pytest node IDs
and, where one exists, a DRAFT upstream patch and repro in [`upstream/`](upstream/README.md) (nothing there is
filed). `verify_fixes.py` answers two questions after a rebase:

```bash
# Does every fix still hold on the rebased tree? (current interpreter; --tree-python to override)
gpu-run --timeout 900 auto -- timeout -k 10 1800 python -m tools.cudnn_fe.verify_fixes --tree .
# Which bugs does the new upstream still have? (--gpu-run reserves a GPU per repro)
python -m tools.cudnn_fe.verify_fixes --gpu-run --upstream-python <env>/bin/python
# Both at once, as in MAINTENANCE step 5:
python -m tools.cudnn_fe.verify_fixes --gpu-run --tree . --upstream-python <env>/bin/python
```

Reports go to stdout, JSON and per-run logs to `agent_space/cudnn_fe_verify/`. Tree verdicts:
`pass`, `no pytest guard` (SASS/bench/review-gated rows), `FAIL`, `MISSING` (a guarding test was
renamed: update `fixes.toml`), `NOT-RUN` (GPU busy or timeout). Upstream verdicts: "bug still
present", "fixed upstream", "no isolated bug" (hardening drafts), or an error to inspect. Add
`--upstream-pythonpath <patched checkout>/python` as a positive control. Compare with the v1.30
baseline in [WORKLOG.md](../../attn_gym/linear/_delta_rule/cudnn_fe/WORKLOG.md#verification-baseline-v130);
`test/test_cudnn_fe_tools_fixes.py` (CPU) checks the generated MAINTENANCE index and uses AST
parsing to validate function-level guarding node IDs without importing GPU test modules.
After editing the ledger, regenerate the index with:

```bash
python -m tools.cudnn_fe.verify_fixes --write-ledger
```

Upstream env recipe (swap in the new tag):

```bash
uv venv agent_space/cmp_env_<tag> --python 3.13
uv pip install --python agent_space/cmp_env_<tag>/bin/python --prerelease allow --index-url https://download.pytorch.org/whl/nightly/cu132 --extra-index-url https://pypi.org/simple torch "nvidia-cudnn-frontend==<tag>" nvidia-cutlass-dsl apache-tvm-ffi
```
