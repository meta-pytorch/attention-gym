"""CPU-only behavior tests for the PTX compiler-hazard gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.warpdrive.warpdrive_ptx_audit import audit_ptx

REPRO = """.version 8.7
.target sm_90
.address_size 64
.visible .entry repro(.param .u64 out, .param .u32 n) {
.reg .pred %p<3>;
.reg .b32 %r<40>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1, [out];
ld.param.u32 %r8, [n];
mov.u32 %r20, smem;
mov.u32 %r33, %tid.x;
setp.lt.u32 %p0, %r33, 32;
@%p0 bra FINISH;
add.s32 %r4, %r33, -32;
mov.b32 %r36, 0;
setp.lt.s32 %p1, %r8, 1;
@%p1 bra DONE;
neg.s32 %r34, %r8;
LOOP:
add.s32 %r36, %r36, %r33;
add.s32 %r34, %r34, 1;
setp.ne.b32 %p2, %r34, 0;
add.s32 %r33, %r33, 64;
@%p2 bra LOOP;
DONE:
shl.b32 %r25, %r4, 2;
add.s32 %r26, %r20, %r25;
st.shared.b32 [%r26+128], %r36;
FINISH:
ret;
}
"""


def test_reports_saved_index_and_loop_advanced_address_site():
    findings = audit_ptx(REPRO)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kernel == "repro"
    assert finding.saved_register == "%r4"
    assert finding.carried_register == "%r33"
    assert finding.offset == -32
    assert finding.memory_opcode == "st.shared.b32"
    assert finding.address == "%r26+128"
    assert finding.tid_line < finding.offset_line < finding.loop_line
    assert finding.loop_line < finding.increment_line < finding.memory_line


@pytest.mark.parametrize(
    "memory", ["st.shared.v2.b32 [%r26+128], {%r36, %r34};", "ld.global.b32 %r36, [%r26];"]
)
def test_vector_store_and_global_load_addresses_are_checked(memory):
    findings = audit_ptx(REPRO.replace("st.shared.b32 [%r26+128], %r36;", memory))
    assert len(findings) == 1
    assert findings[0].memory_opcode == memory.split()[0]


@pytest.mark.parametrize(
    "workaround",
    [
        "zero_offset",
        "address_before_loop",
        "fresh_iv",
        "same_block",
        "reread_after_loop",
        "no_post_loop_address",
    ],
)
def test_verified_workarounds_do_not_report_candidates(workaround):
    ptx = REPRO
    match workaround:
        case "zero_offset":
            ptx = ptx.replace("%r33, -32", "%r33, 0")
        case "address_before_loop":
            address = "shl.b32 %r25, %r4, 2;\nadd.s32 %r26, %r20, %r25;\n"
            ptx = ptx.replace(address, "").replace("LOOP:", address + "LOOP:")
        case "fresh_iv":
            ptx = ptx.replace("LOOP:", "mov.u32 %r39, %r33;\nLOOP:")
            ptx = ptx.replace("%r36, %r36, %r33", "%r36, %r36, %r39")
            ptx = ptx.replace("%r33, %r33, 64", "%r39, %r39, 64")
        case "same_block":
            offset = "add.s32 %r4, %r33, -32;\n"
            ptx = ptx.replace(offset, "").replace("setp.lt.u32", offset + "setp.lt.u32")
        case "reread_after_loop":
            ptx = ptx.replace("DONE:", "DONE:\nmov.u32 %r38, %tid.x;\nadd.s32 %r4, %r38, -32;")
        case "no_post_loop_address":
            ptx = ptx.replace("st.shared.b32 [%r26+128], %r36;", "add.s32 %r36, %r36, %r4;")
    assert audit_ptx(ptx) == []


def test_reused_inline_asm_labels_and_unconditional_backedges():
    wait = """{
.reg .pred P;
LAB_WAIT:
mbarrier.try_wait.parity.shared.b64 P, [%r20], %r34, 10000000;
@P bra DONE;
bra LAB_WAIT;
DONE:
}
"""
    # Local predicate names need not start with %, unlike NVVM virtual registers.
    ptx = REPRO.replace("LOOP:", wait + "LOOP:\n" + wait)
    ptx = ptx.replace("@%p2 bra LOOP;", "@%p2 bra AGAIN;\nbra DONE;\nAGAIN:\nbra.uni LOOP;")
    findings = audit_ptx(ptx)
    assert len(findings) == 1
    assert findings[0].address == "%r26+128"


@pytest.mark.parametrize("terminator", ["ret", "exit"])
def test_predicated_exit_preserves_the_non_exiting_role_path(terminator):
    ptx = REPRO.replace("add.s32 %r4, %r33, -32;", f"@%p0 {terminator};\nadd.s32 %r4, %r33, -32;")
    assert len(audit_ptx(ptx)) == 1


@pytest.mark.parametrize("terminator", ["ret", "exit"])
def test_predicated_return_dispatch_is_a_basic_block_boundary(terminator):
    ptx = REPRO.replace("@%p0 bra FINISH;", f"@%p0 {terminator};")
    assert len(audit_ptx(ptx)) == 1


@pytest.mark.parametrize(
    "barrier",
    [
        "mbarrier.expect_tx.shared::cta.b64 [%r4], 4;",
        "mbarrier.init.shared.b64 [%r4], 4;",
        "mbarrier.inval.shared.b64 [%r4];",
    ],
)
def test_barrier_address_operand_does_not_redefine_the_saved_index(barrier):
    ptx = REPRO.replace("LOOP:", f"LOOP:\n{barrier}")
    assert len(audit_ptx(ptx)) == 1


@pytest.mark.parametrize("safe_path", ["exit_before_store", "reset_induction"])
def test_unrelated_address_paths_and_reset_values_are_not_loop_carried(safe_path):
    ptx = REPRO
    if safe_path == "exit_before_store":
        ptx = ptx.replace("DONE:", "ret;\nDONE:")
    else:
        ptx = ptx.replace("LOOP:", "LOOP:\nmov.b32 %r33, 0;")
    assert audit_ptx(ptx) == []


@pytest.mark.parametrize(
    "ptx",
    [
        "// .entry empty()\nret;",
        ".visible .entry empty() { ret;",
        ".visible .entry empty() {}\nret;",
        ".visible .entry empty() {}",
    ],
)
def test_missing_or_incomplete_entry_bodies_are_not_a_passing_gate(ptx):
    with pytest.raises(ValueError):
        audit_ptx(ptx)


def test_malformed_branch_is_not_a_passing_gate():
    with pytest.raises(ValueError, match="unresolved branch label MISSING"):
        audit_ptx(REPRO.replace("@%p2 bra LOOP", "@%p2 bra MISSING"))


def test_cli_reports_json_and_distinguishes_findings_from_empty_inventory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/warpdrive/warpdrive_ptx_audit.py"
    ptx = tmp_path / "repro.ptx"
    ptx.write_text(REPRO)
    result = subprocess.run(
        [sys.executable, str(script), "--json", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)[str(ptx)][0]["saved_register"] == "%r4"
    ptx.unlink()
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "no PTX files found" in result.stderr


def test_cli_filesystem_failure_is_an_input_error(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/warpdrive/warpdrive_ptx_audit.py"
    (tmp_path / "directory.ptx").mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "directory.ptx" in result.stderr
    assert "Traceback" not in result.stderr
