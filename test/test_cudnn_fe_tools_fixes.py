"""CPU-only checks that tools/cudnn_fe/fixes.toml cannot drift from the ledger and the suite."""

import ast

from tools.cudnn_fe import verify_fixes

MAINTENANCE = verify_fixes.maintenance_path()
UPSTREAM_STATUS = {
    "draft-not-filed",
    "hardening-draft",
    "superseded-by-v1.30",
    "excluded-no-upstream-abi",
    "ag-specific",
    "nv-surface-removed",
    "candidate-no-patch",
}


def test_fixes_match_the_maintenance_ledger():
    fixes = [f for f in verify_fixes.load_fixes() if f["group"] == "fix"]
    ids = [f["id"] for f in fixes]
    assert len(ids) == len(set(ids)), "duplicate fix ids"
    text = MAINTENANCE.read_text()
    table = text.split(verify_fixes.LEDGER_START, 1)[1].split(verify_fixes.LEDGER_END, 1)[0]
    assert (
        verify_fixes.LEDGER_START + table + verify_fixes.LEDGER_END == verify_fixes.ledger_table()
    )
    for f in fixes:
        assert f["title"] and f["kind"] and f["ag_subject"], f["id"]
        assert f["upstream_status"] in UPSTREAM_STATUS, f["id"]
        assert f["guarding_tests"] or f.get("gate"), f"{f['id']} has neither a test nor a gate"


def test_upstream_patches_and_repros_exist():
    for f in verify_fixes.load_fixes():
        for key in ("upstream_patch", "upstream_repro"):
            if key in f:
                assert (verify_fixes.HERE / f[key]).is_file(), f"{f['id']}: {f[key]}"
        if "upstream_repro" in f:
            assert f.get("upstream_expect") in {"bug", "smoke", "scope"}, f["id"]


def test_every_guarding_test_is_collected():
    nodes = {n for f in verify_fixes.load_fixes() for n in f["guarding_tests"]}
    files = sorted({n.split("::")[0] for n in nodes})
    absent = [path for path in files if not (verify_fixes.REPO_ROOT / path).is_file()]
    assert absent == []
    # Check the ledger's function-level node IDs without importing GPU modules or starting
    # nested pytest processes. Actual parametrizations are collected by verify_fixes --tree.
    collected = {
        f"{path}::{node.name}"
        for path in files
        for node in ast.parse((verify_fixes.REPO_ROOT / path).read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    missing = sorted(nodes - collected)
    assert missing == [], "renamed or removed guarding tests; update tools/cudnn_fe/fixes.toml"
