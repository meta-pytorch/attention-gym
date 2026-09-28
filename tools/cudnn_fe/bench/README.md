# Benchmark gate for the vendored cuDNN kernels

Step 8 of [MAINTENANCE.md](../../../attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md):
`python -m tools.cudnn_fe.bench` compares a candidate tree against a baseline tree (usually the
previous stack tip) on the same interpreter and GPU and fails when any cell regresses past the
threshold, the outputs drift, or the kernel launch set changes.

Measurement contract (same as the v1.30 migration gate in the PR stack):

- one interpreter for both trees; each child activates its tree (`[LABEL=]DIR`, the checkout
  that contains `attn_gym/`, see `tools/cudnn_fe/_common.py`) ahead of the installed package and
  asserts that `attn_gym` was imported from it;
- whole-op fixed-pointer CUDA-graph replay (prologues, `_prepare_ragged_chunk_offsets`, memsets
  and main kernels included; compile, capture, allocation and the host launch gap excluded):
  3 samples x 20 back-to-back replays between one event pair per phase, in-process median of
  the per-replay time;
- `--rounds` (default 3) process rounds per suite with the tree order alternated each round;
  table cell = median of the process-round medians, rounds shown in parentheses;
- factory DVFS; `nvidia-smi` SM clock / max, power and temperature of the GPU torch runs on
  (matched by UUID, so `CUDA_VISIBLE_DEVICES` renumbering does not pick a neighbour) are recorded
  per row and the clock set is printed per suite so a throttled run is visible;
- round 0 records the kernel launch set (full kernel names, so a specialization suffix change
  is visible) and dumps outputs (+ input gradients for `gdn`/`kda`); the parent compares them
  across trees and deletes the dumps (`--keep-dumps` to retain); no large files are written
  otherwise. Workloads and inputs come from `tools/cudnn_fe/_common.py`, shared with the
  CUTracer oracle.

Suites and checks:

| suite | op | phases | check |
|---|---|---|---|
| `gdn` | `chunk_gdn` cuDNN, HK16/H48/D128 | forward, backward, forward_backward | bitwise (o, dq, dk, dv, dg, db) |
| `kda` | `chunk_kda` cuDNN | forward, backward, forward_backward | relL2 <= `--rel-tol` (1e-2) |
| `summary` | CP affine state summaries, forward [B;A] / reverse [C;R] | forward, reverse | relL2 |
| `paged` | `paged_chunk_{gdn,kda}` forward, resumed routes | forward | gdn bitwise, kda relL2 |

Bitwise is the contract when the candidate keeps the same kernel generation; a new upstream
generation (different operand packing) is expected to move KDA at the 5e-3 level, which is why
the KDA suites use relL2. Override per run only by editing the `check` column in `CASES`.

## Usage

```bash
PY=.venv/bin/python
# Self-comparison smoke (ratios ~1.00, everything bitwise):
gpu-run --timeout 900 auto -- $PY -m tools.cudnn_fe.bench run --suite gdn kda summary paged --small --new . --base . --out /tmp/gate_self
# Full gate, candidate vs a detached worktree of the previous stack tip:
gpu-run --timeout 900 auto -- timeout -k 10 3000 $PY -m tools.cudnn_fe.bench run --suite gdn --new cand=. --base prev=/path/to/prev-worktree --out /tmp/gate_gdn
gpu-run --timeout 900 auto -- timeout -k 10 3000 $PY -m tools.cudnn_fe.bench run --suite kda summary paged --new cand=. --base prev=/path/to/prev-worktree --out /tmp/gate_kda
# Re-render the markdown with another threshold:
$PY -m tools.cudnn_fe.bench report --out /tmp/gate_gdn --threshold 0.03
```

Reserve the GPU around the run; the gate never claims one itself. Pin both trees (detached
worktree or `git archive`) before a long gate; the report records `git rev-parse --short HEAD`
of each tree in `config.json` and `report.md`. Run one suite per `gpu-run` reservation when the
full grid is used; `--child-timeout` bounds each child process.

`report.md` ends with `PASS` or `FAIL: <cells>`; a ratio above `1 + threshold`, a failed
correctness row, or a kernel launched on only one tree fails the gate (exit 1). Cells within
~1.3 % have historically been noise on GB200 (see the migration notes); attribute anything
larger per kernel with `torch.profiler` before changing the threshold.
