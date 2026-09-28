"""CPU-only tests of the benchmark gate's report logic (``tools/cudnn_fe/bench``) and of the
shared tool plumbing (``tools/cudnn_fe/_common.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.cudnn_fe import _common
from tools.cudnn_fe.bench import gate

TIMED = "GPU-66db3ba4-6a42-8484-7ac0-29cf38f1a1fe"
NVIDIA_SMI = """\
GPU-a8c7d79e-7e45-2c4c-0ddb-efebb20754a9, 2025 MHz, 2062 MHz, 987.12 W, 61
GPU-66db3ba4-6a42-8484-7ac0-29cf38f1a1fe, 1290 MHz, 2062 MHz, 78.40 W, 31
"""


def test_telemetry_row_is_selected_by_uuid_not_position() -> None:
    row = gate.parse_telemetry(NVIDIA_SMI, TIMED)
    assert row["clocks.sm"] == "1290 MHz" and row["power.draw"] == "78.40 W"
    # torch reports the UUID without the prefix and in either case.
    assert gate.parse_telemetry(NVIDIA_SMI, TIMED[4:].upper()) == row
    assert gate.parse_telemetry(NVIDIA_SMI, "GPU-0000") == {}


def test_launch_set_compares_full_kernel_names() -> None:
    long = "kda_prep_kernel_cutlass_frost_kda_prep_KdaPrepCfgio_dtype" + "x" * 80
    base = {"c": {long + "gate_scale_log2True": 1.0, "memset": 0.1}}
    assert gate.launch_set_changes(base, base) == {}
    new = {"c": {long + "gate_scale_log2False": 1.0, "memset": 0.1}}
    assert gate.launch_set_changes(new, base) == {
        "c": [f"+{long}gate_scale_log2False", f"-{long}gate_scale_log2True"]
    }


def write_run(out: Path, label: str, rounds: dict[int, list[float]], kernels: dict) -> None:
    for turn, samples in rounds.items():
        row = {
            "case": "prefill_2048",
            "label": label,
            "round": turn,
            "forward_us": samples,
            "telemetry": gate.parse_telemetry(NVIDIA_SMI, TIMED),
        }
        if turn == 0:
            row["kernels"] = kernels
        (out / f"gdn_{label}_{turn}.json").write_text(json.dumps({"meta": {}, "rows": [row]}))


def make_out(tmp_path: Path, new_rounds, base_rounds, new_kernels, base_kernels, compare_ok=True):
    config = {
        "new": {"label": "new", "tree": "/n", "commit": "abc"},
        "base": {"label": "base", "tree": "/b", "commit": "def"},
        "suites": {"gdn": ["prefill_2048"]},
        "rounds": len(new_rounds),
        "threshold": 0.02,
        "rel_tol": 0.01,
        "started": "now",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    write_run(tmp_path, "new", new_rounds, new_kernels)
    write_run(tmp_path, "base", base_rounds, base_kernels)
    (tmp_path / "gdn_compare.json").write_text(
        json.dumps(
            [
                {
                    "case": "prefill_2048",
                    "tensor": "o",
                    "check": "bitwise",
                    "bitwise": compare_ok,
                    "rel_l2": 0.0 if compare_ok else 1e-3,
                    "max_abs": 0.0,
                    "ok": compare_ok,
                }
            ]
        )
    )
    return tmp_path


KERNELS = {"frost_gdn_prefill_GdnPrefillCfg_x": 50.0, "prologue": 5.0}


def test_report_passes_within_threshold_and_renders_the_timed_gpu_clock(tmp_path) -> None:
    out = make_out(
        tmp_path, {0: [58.0, 58.5], 1: [58.2]}, {0: [58.0, 57.9], 1: [58.1]}, KERNELS, KERNELS
    )
    text, failures = gate.report(out)
    assert failures == [] and text.rstrip().endswith("PASS")
    assert "1290 MHz/2062 MHz" in text and "2025 MHz" not in text
    assert (out / "report.md").read_text() == text
    assert "identical on both trees" in text


def test_report_fails_on_regression_launch_set_change_and_correctness(tmp_path) -> None:
    changed = {"frost_gdn_prefill_GdnPrefillCfg_y": 50.0, "prologue": 5.0}
    out = make_out(
        tmp_path, {0: [60.0], 1: [60.0]}, {0: [58.0], 1: [58.0]}, changed, KERNELS, False
    )
    _, failures = gate.report(out)
    assert failures == [
        "gdn/prefill_2048/forward ratio 1.034",
        "gdn/prefill_2048 launch set changed",
        "gdn/prefill_2048/o bitwise check",
    ]
    _, loose = gate.report(out, threshold=0.05)
    assert loose == ["gdn/prefill_2048 launch set changed", "gdn/prefill_2048/o bitwise check"]


def test_select_small_preset_and_names() -> None:
    assert [c.name for c in gate.select("gdn", [], small=True)] == ["prefill_2048"]
    assert [c.name for c in gate.select("paged", ["kda_paged_8x2048"], small=True)] == [
        "kda_paged_8x2048"
    ]
    assert all(c.work.kind == "kda" for c in gate.select("summary", [], small=False))
    with pytest.raises(SystemExit):
        gate.select("kda", ["nope"], small=False)


def test_parse_tree_labels() -> None:
    assert _common.parse_tree("cand=/tmp/x", "new") == ("cand", Path("/tmp/x"))
    assert _common.parse_tree("/tmp/x", "new") == ("new", Path("/tmp/x"))


def test_run_logged_records_command_output_and_timeouts(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    rc, text = _common.run_logged(
        [sys.executable, "-c", "print('hi'); raise SystemExit(3)"], log, timeout=30
    )
    assert (rc, text) == (3, "hi\n")
    logged = log.read_text()
    assert logged.startswith("$ timeout -k 10 30 ") and logged.endswith("EXIT_STATUS=3\n")
    rc, _ = _common.run_logged(
        [sys.executable, "-c", "import time; time.sleep(30)"], log, timeout=1
    )
    assert rc == _common.TIMEOUT_EXIT and _common.describe_exit(rc) == "timeout (rc=124)"
    assert _common.describe_exit(_common.GPU_BUSY) == "GPU busy (gpu-run exit 75)"


def test_check_tree_rejects_a_foreign_checkout(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not to --tree"):
        _common.check_tree(tmp_path)
    assert _common.check_tree(None).name == "attn_gym"
