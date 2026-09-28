"""CPU-only tests of the CUTracer stress verdict logic (``tools/cudnn_fe/cutracer/stress.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cudnn_fe.cutracer import oracle, stress

PREFILL = (
    "cudnn_kernel_cutlass_frost_gdn_prefill_GdnPrefillCfgio_dtypecutlassbase_dsltypingBFloat16_1"
)
PROLOGUE = "cudnn_kernel_cutlass_frost_gdn_prefill_prologue_True_16_0"
LAUNCH_LOG = f"""\
LAUNCH - Kernel pc 0x7f - Kernel name {PREFILL} - kernel hash 0xabc
LAUNCH - Kernel pc 0x80 - Kernel name {PROLOGUE} - kernel hash 0xdef
LAUNCH - Kernel pc 0x81 - Kernel name {PREFILL} - kernel hash 0xabc
LAUNCH - Kernel pc 0x82 - Kernel name _Z5other_kernel - kernel hash 0x111
LAUNCH - Kernel pc 0x83 - Kernel name split_k_plan_kernel_cutlass_frost_split_k_plan_0 - kernel hash 0x222
"""
OK = "all 12 tensors bitwise identical to reference\n"


def test_launch_log_keeps_only_vendored_kernels_with_stable_filters() -> None:
    launches = stress.parse_launches(LAUNCH_LOG)
    assert launches == {
        PREFILL: {"0xabc"},
        PROLOGUE: {"0xdef"},
        "split_k_plan_kernel_cutlass_frost_split_k_plan_0": {"0x222"},
    }
    assert {stress.derive_filter(name) for name in launches} == {
        "frost_gdn_prefill_GdnPrefillCfg",
        "frost_gdn_prefill_prologue",
        "frost_split_k",
    }


def dump(tmp_path: Path, sites: list[tuple[str, bool]]) -> Path:
    path = tmp_path / "d5000_a1.delay.json"
    path.write_text(
        json.dumps(
            {
                "delay_ns": 5000,
                "kernels": {
                    "0": {
                        "kernel_name": PREFILL,
                        "kernel_checksum": "0xabc",
                        "instrumentation_points": {
                            str(i): {"sass": text, "on": on} for i, (text, on) in enumerate(sites)
                        },
                    }
                },
            }
        )
    )
    return path


def test_site_summary_counts_enabled_sites_by_opcode_ignoring_predicates(tmp_path) -> None:
    path = dump(
        tmp_path,
        [
            ("@!P3 SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4+0x100], UR5 ;", True),
            ("@!UP0 UTMALDG.3D [UR8], [UR10] ;", True),
            ("SYNCS.EXCH.64 URZ, [UR4], UR6 ;", True),
            ("LDS.128 R4, [R2] ;", False),
        ],
    )
    (summary,) = stress.site_summary(path).values()
    assert summary == {"hash": "0xabc", "enabled": 3, "kinds": {"SYNCS": 2, "UTMALDG": 1}}
    assert stress.site_summary(tmp_path / "missing.json") == {}


ENABLED = {PREFILL: {"hash": "0xabc", "enabled": 3, "kinds": {"SYNCS": 3}}}
EMPTY = {PREFILL: {"hash": "0xabc", "enabled": 0, "kinds": {}}}


@pytest.mark.parametrize(
    ("rc", "text", "sites", "replay", "matched", "ok", "expected"),
    [
        (0, OK, ENABLED, False, True, True, "all 12 tensors"),
        (0, OK, ENABLED, False, None, True, "all 12 tensors"),
        (0, OK, {}, False, True, False, "no kernel instrumented"),
        (0, OK, EMPTY, False, True, False, "no kernel instrumented"),
        (0, OK, ENABLED, False, False, False, "filter matched no launched kernel"),
        (
            0,
            f"Replay: No config found for kernel {PREFILL} (checksum: 0x9)\n{OK}",
            ENABLED,
            True,
            True,
            False,
            "replay pattern not applied",
        ),
        (
            0,
            f"kernel {PREFILL} NOT FOUND in config, skipping\n{OK}",
            ENABLED,
            True,
            True,
            False,
            "replay pattern not applied",
        ),
        (
            0,
            f"Replay: No config found for kernel x\n{OK}",
            ENABLED,
            False,
            True,
            True,
            "all 12 tensors",
        ),
        (
            1,
            "MISMATCH gdn_prep_3: max abs 1.5e-05, 1 elems\n",
            ENABLED,
            False,
            True,
            False,
            "MISMATCH gdn_prep_3",
        ),
        (
            2,
            "missing reference /tmp/refs/reference_gdn_all.pt; run first\n",
            ENABLED,
            False,
            True,
            False,
            "missing reference",
        ),
        (124, "", ENABLED, False, True, False, "no verdict (timeout (rc=124))"),
        (75, "", {}, False, True, False, "GPU busy"),
    ],
    ids=[
        "pass",
        "pass-without-launch-log",
        "no-sites",
        "sites-disabled",
        "filter-unmatched",
        "replay-no-config",
        "replay-not-found",
        "no-config-line-outside-replay",
        "mismatch",
        "missing-reference",
        "timeout",
        "gpu-busy",
    ],
)
def test_random_mode_verdicts(rc, text, sites, replay, matched, ok, expected) -> None:
    verdict, passed = stress.judge("random", rc, text, sites=sites, replay=replay, matched=matched)
    assert passed is ok
    assert expected in verdict


@pytest.mark.parametrize(
    ("text", "ok", "expected"),
    [
        (OK, True, "all 12 tensors"),
        (
            "Possible kernel hang: launch_id=3 state(looping=1, barrier=0, progressing=0) for "
            "12 seconds.\n" + OK,
            False,
            "DEADLOCK REPORT: Possible kernel hang",
        ),
        ("Deadlock sustained for 5 checks; sending SIGTERM.\n", False, "Deadlock sustained"),
    ],
    ids=["clean", "transient-hang", "sustained"],
)
def test_deadlock_mode_fails_on_any_hang_report(text, ok, expected) -> None:
    verdict, passed = stress.judge("deadlock", 0, text, sites={}, replay=False, matched=True)
    assert passed is ok
    assert expected in verdict


def test_results_fail_without_attempts_and_summarize_verdicts(tmp_path: Path) -> None:
    class Args:
        mode, delays, patterns = "random", [5000], 1

    assert stress.write_results(tmp_path, Args, {}, []) is False
    attempt = stress.Attempt(
        "gdn", "frost_gdn_prefill_GdnPrefillCfg", 5000, 1, "random", 0, OK.strip(), 3.0, "l", True
    )
    bad = stress.Attempt(
        "gdn",
        "frost_gdn_prefill_prologue",
        5000,
        1,
        "random",
        0,
        "no kernel instrumented",
        3.0,
        "l",
        False,
    )
    kernels = {"gdn": stress.parse_launches(LAUNCH_LOG)}
    assert stress.write_results(tmp_path, Args, kernels, [attempt]) is True
    assert (tmp_path / "results.md").read_text().startswith("# CUTracer random stress (PASS)")
    assert stress.write_results(tmp_path, Args, kernels, [attempt, bad]) is False
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["pass"] is False and [a["ok"] for a in results["attempts"]] == [True, False]


def test_oracle_reference_path_is_order_independent(tmp_path: Path) -> None:
    assert oracle.reference_path(tmp_path, "kda", ["prep", "chain"]) == oracle.reference_path(
        tmp_path, "kda", ["chain", "prep"]
    )
    assert oracle.reference_path(tmp_path, "gdn", []).name == "reference_gdn_all.pt"
    assert {c.name for c in oracle.select("kda", {"prep", "chain"})} == {"prep", "chain"}
    with pytest.raises(SystemExit):
        oracle.select("gdn", {"prep"})
