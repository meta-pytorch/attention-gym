# CUTracer race stress for the vendored cuDNN kernels

Step 7 of [MAINTENANCE.md](../../../attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md):
after the SASS gate passes, every warp-specialized kernel family must stay bitwise deterministic
under CUTracer `random_delay` perturbation (delays injected before mbarrier, TMA, MMA, SMEM and
TMEM instructions), and the small cases must trace to completion under `deadlock_detection`.

- `oracle.py` — record-then-compare bitwise oracle over the public `chunk_gdn` / `chunk_kda`
  cuDNN routes. The nine cases reach every kernel family (prefill/prep, d_v split, exact piece
  chain with state cotangents, the KDA `chain_nostate` chain, warmup split, prologues,
  `split_k`, `state_chain`, summaries, bprop/recompute). References go to a directory you
  choose, never into the repo. `oracle.py list` prints the cases.
- `stress.py` — Python driver replacing the old `stress.sh` + batch scripts: records the
  reference under the CUTracer launch logger, derives one filter per vendored kernel from the
  launch log (`frost_<kernel>_<Cfg>`, `frost_<kernel>_prologue`, `frost_split_k`,
  `frost_state_chain`), runs a delay ladder with N fresh patterns per delay, and writes
  `results.md` / `results.json` with per-attempt verdicts, enabled delay sites by SASS kind and
  kernel hashes. Every attempt runs under a process-level `timeout`
  (`tools/cudnn_fe/_common.py`); the oracle cases share their shapes and inputs with the bench
  gate through the same module.

## Setup

CUTracer is a native NVBit library plus a Python CLI; the CLI lives in its own env and the
traced program uses this worktree's `.venv`. Default CLI path: `~/.venvs/cutracer/bin/cutracer`
(`--cutracer` overrides). Follow the `cutracer` skill for `install_cutracer`.

aarch64 (GB200/GB300 hosts): `cutracer>=0.3` does not resolve normally because
`yscope-clp-core` has no aarch64 wheel, and Python 3.13 may silently downgrade to `cutracer==0.1.0`
(no CLI). Use a Python 3.12 env, install the CLI without dependencies, then the non-CLP
dependencies, then build the `.so`:

```bash
uv venv --python 3.12 ~/.venvs/cutracer
uv pip install --python ~/.venvs/cutracer/bin/python --no-deps 'cutracer==0.3.0'
uv pip install --python ~/.venvs/cutracer/bin/python 'click>=8' 'jsonschema>=4' 'zstandard>=0.20' 'tabulate>=0.9' 'tritonparse>=0.4.3'
source ~/.venvs/cutracer/bin/activate && install_cutracer
```

Random-delay and zstd/NDJSON tracing work without CLP. `stress.py` unsets
`CUDA_INJECTION64_PATH` for its children; the CLI owns that variable.

## Usage

Reserve the GPU around the whole run; the harness never queues on `gpu-run` itself.
`--gpu-run` instead claims a GPU per attempt with `gpu-run --no-wait auto` and stops the ladder
with exit 75 when every GPU is reserved (use it for long ladders on a shared host where holding
one reservation for hours is not acceptable).

```bash
PY=.venv/bin/python
# Full ladder for one family (3 delays x 3 patterns per filter, every filter in the launch log):
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.stress --family gdn --ref-dir /tmp/cudnn_refs --out /tmp/stress_gdn
# Per-attempt no-wait GPU claims instead of one outer reservation:
$PY -m tools.cudnn_fe.cutracer.stress --family kda --gpu-run --ref-dir /tmp/cudnn_refs --out /tmp/stress_kda
# One filter, one small case, short ladder (wiring check):
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.stress --family kda --case prep --filter frost_kda_prep_KdaPrepCfg --delays 5000 --patterns 1 --ref-dir /tmp/cudnn_refs --out /tmp/stress_prep
# Replay a dumped pattern bit-exactly:
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.stress --family kda --case prep --filter frost_kda_prep_KdaPrepCfg --skip-record --replay /tmp/stress_prep/kda/frost_kda_prep_KdaPrepCfg/d5000_a1.delay.json --ref-dir /tmp/cudnn_refs --out /tmp/replay
# Bounded deadlock detection on the smallest case that reaches a kernel (trace capped at 2 GB):
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.stress --family gdn --case uncut_packed --filter frost_gdn_prefill_GdnPrefillCfg --mode deadlock --trace-size-limit-mb 2048 --ref-dir /tmp/cudnn_refs --out /tmp/deadlock
# Only record + print the discovered filters:
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.stress --family all --dry-run --ref-dir /tmp/cudnn_refs --out /tmp/discover
# The oracle on its own:
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.cutracer.oracle record --ref-dir /tmp/cudnn_refs --family gdn
```

Verdict: `results.md` starts with `PASS` only if every attempt exited 0 with
`all N tensors bitwise identical to reference` **and** perturbed a kernel. An attempt fails when
its filter matched no launched vendored kernel (launch log available), when its delay dump
enabled no site (`no kernel instrumented`: typo'd `--filter`, or a `--case` subset that does not
launch the kernel; with `--skip-record --filter` there is no launch log, so only the dump check
applies), or when a `--replay` hit CUTracer's `No config found for kernel` / `NOT FOUND in
config` (the kernel changed since the dump; re-record the pattern). A `MISMATCH` line names the
tensor (`<family>_<case>_<index>`, outputs first, then input gradients in leaf order);
`no verdict` means the process crashed, hung past `--timeout`, or the reference was missing. In
deadlock mode any `Possible kernel hang` or `Deadlock sustained` report fails the attempt even
when the oracle matches. Trace directories are deleted after each attempt unless
`--keep-traces`; delay dumps (`d<ns>_a<k>.delay.json`) are always kept for replay.

Interpreting the site table: warp-specialized main kernels should show `SYNCS`, `UTCHMMA`,
`UTMALDG`/`UTMASTG`, `UTCBAR`, `LDS`/`STS`/`LDSM`/`STSM` and `LDTM`/`STTM` sites enabled; a filter
whose dumps show only `LDG`/`STG` is a plain SIMT prologue. Site kinds are base mnemonics with
the predicate stripped.

Kernel hashes in `results.md` are CUTracer cubin hashes: a filter whose hashes are unchanged
since its last clean ladder does not need to be re-stressed after an unrelated commit.

Known gaps: only bf16 with d_k = d_v = 128 and default gate flags is exercised; v1.30 KDA with
`initial_state` routes the backward to the cute path, so no stateful KDA bprop kernel exists to
stress; summary kernels on chain-size cases exceed the deadlock-detection trace budget
(random-delay coverage only).
