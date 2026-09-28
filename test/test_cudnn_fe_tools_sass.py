"""CPU-only tests of the SASS gate's parsing and classification (``tools/cudnn_fe/sass``)."""

from __future__ import annotations

import re
import struct
from collections import Counter
from pathlib import Path

import pytest

from tools.cudnn_fe.sass import diff, sass

SASS_TEXT = """\
\t.target\tsm_100a

//--------------------- .text.gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1 ----------
\t.section\t.text.gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1,"ax",@progbits
.text.gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   UMOV UR4, 0x1 ;
        /*0030*/                   SYNCS.EXCH.64 URZ, [UR13+0x24000], UR4 ;
        /*0040*/              @!P0 SYNCS.EXCH.64 URZ, [UR13+0x24008], UR4 ;
        /*0050*/                   SYNCS.EXCH.64 URZ, [UR13], UR4 ;
        /*0060*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR13+0x24000], UR5 ;
        /*0070*/                   NOP;
        /*0080*/                   EXIT ;
.L_x_1:
        /*0090*/                   BRA `(.L_x_1);
.text.gdn_kernel_cutlass_frost_gdn_prefill_prologue_True_16_0:
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   EXIT ;
"""

RESOURCE_TEXT = """\
Resource usage:
 Common:
  GLOBAL:0
 Function gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1:
  REG:168 STACK:0 SHARED:1024 LOCAL:0
 Function gdn_kernel_cutlass_frost_gdn_prefill_prologue_True_16_0:
  REG:52 STACK:0 SHARED:1024 LOCAL:0
"""

LAUNCH_CONFIG = "!llvm.struct<(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32)>"
IR_TEXT = f"""\
module {{
  llvm.func @_cudaLaunchKernelEx(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @cuda_load_to_device(%arg0: !llvm.ptr, %arg1: i32) -> i32 {{
    %7 = llvm.mlir.addressof @kernels_gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1 : !llvm.ptr
    llvm.return %arg1 : i32
  }}
  llvm.func @host(%arg0: !llvm.ptr) -> i32 {{
    %7 = llvm.mlir.constant(232448 : i64) : i64
    %9 = llvm.mlir.constant(32776 : i64) : i64
    %78 = llvm.alloca %15 x {LAUNCH_CONFIG[1:]} : (i32) -> !llvm.ptr
    %84 = llvm.getelementptr %78[0, 2] : (!llvm.ptr) -> !llvm.ptr, {LAUNCH_CONFIG}
    llvm.store %9, %84 : i64, !llvm.ptr
    %90 = llvm.mlir.addressof @kernels_gdn_kernel_cutlass_frost_gdn_prefill_prologue_True_16_0 : !llvm.ptr
    %91 = llvm.call @_cudaLaunchKernelEx(%78, %90, %92) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    %184 = llvm.getelementptr %178[0, 2] : (!llvm.ptr) -> !llvm.ptr, {LAUNCH_CONFIG}
    llvm.store %7, %184 : i64, !llvm.ptr
    %187 = llvm.mlir.addressof @kernels_gdn_kernel_cutlass_frost_gdn_prefill_GdnCfgio_1 : !llvm.ptr
    %189 = llvm.call @_cudaLaunchKernelEx(%178, %187, %138) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return %189 : i32
  }}
}}
"""

# Shape of the kda_recompute mainloop as captured in the KDA restyle pair (agent_space/sass
# k4_base -> k4_restyle): the barrier block stays at UR13+0x0, the STS.128 tile writes sit at
# +0x1f000/+0x23000 and the LDS reads at +0x1ec00 before the restyle. The restyle moved the
# tiles to +0x1ec00/+0x22c00 and +0x2ec00 with the same instruction histogram apart from one
# IMAD.MOV.U32 -> IMAD.U32 and one LOP3 -> ULOP3 swap.
RECOMPUTE_TEXT = """\
.text.kda_kernel_cutlass_frost_kda_recompute_KdaCfg_0:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   SYNCS.EXCH.64 URZ, [UR13], UR4 ;
        /*0020*/                   SYNCS.EXCH.64 URZ, [UR13+0x8], UR4 ;
        /*0030*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR13], UR5 ;
        /*0040*/                   STS.128 [R4+0x1f000], R8 ;
        /*0050*/                   STS.128 [R4+0x1f010], R12 ;
        /*0060*/                   STS.128 [R4+0x23000], R16 ;
        /*0070*/                   LDS.64 R20, [R6+0x1ec00] ;
        /*0080*/                   LDS R22, [R6+0x1ec20] ;
        /*0090*/                   LDSM.16.M88.4 R24, [R7+0x1ec00] ;
        /*00a0*/                   IMAD.MOV.U32 R30, RZ, RZ, R31 ;
        /*00b0*/                   LOP3.LUT R32, R33, 0x7, RZ, 0xc0, !PT ;
        /*00c0*/                   FFMA R34, R35, R36, R37 ;
        /*00d0*/                   UTMALDG.3D [UR8], [UR10] ;
        /*00e0*/                   EXIT ;
"""

RECOMPUTE_RESOURCES = """\
 Function kda_kernel_cutlass_frost_kda_recompute_KdaCfg_0:
  REG:128 STACK:8 SHARED:1024 LOCAL:0
"""


def relocate_tiles(text: str) -> str:
    moves = {"0x1f000": "0x1ec00", "0x1f010": "0x1ec10", "0x23000": "0x22c00"}
    text = re.sub(r"STS\.128 \[R4\+(0x[0-9a-f]+)\]", lambda m: f"STS.128 [R4+{moves[m[1]]}]", text)
    text = re.sub(r"(LDS\S* R\d+, \[R[67]\+)0x1ec", r"\g<1>0x2ec", text)
    text = text.replace("IMAD.MOV.U32 R30, RZ, RZ, R31", "IMAD.U32 R30, R31, 0x1, RZ")
    return text.replace("LOP3.LUT R32, R33", "ULOP3.LUT UR32, UR33")


def test_parse_sass_keeps_modifiers_layout_and_mbarrier_offsets() -> None:
    prefill, prologue = sass.parse_sass(SASS_TEXT).values()
    assert prefill.ops == Counter(
        {
            "MOV": 1,
            "S2R": 1,
            "UMOV": 1,
            "SYNCS.EXCH.64": 3,
            "SYNCS.PHASECHK.TRANS64.TRYWAIT": 1,
            "NOP": 1,
            "EXIT": 1,
            "BRA": 1,
        }
    )
    # Only the EXCH (init) stores count as barriers, predicated or not; a bare base register is
    # offset 0. The layout also holds the TRYWAIT.
    assert prefill.mbarriers == [0x0, 0x24000, 0x24008]
    assert prefill.layout == Counter(
        {
            ("SYNCS.EXCH.64", 0x24000): 1,
            ("SYNCS.EXCH.64", 0x24008): 1,
            ("SYNCS.EXCH.64", 0x0): 1,
            ("SYNCS.PHASECHK.TRANS64.TRYWAIT", 0x24000): 1,
        }
    )
    assert prologue.ops == Counter({"IMAD.MOV.U32": 1, "EXIT": 1})
    assert prologue.layout == Counter() and prologue.mbarriers == []


def test_digest_ignores_addresses_and_label_numbers_only() -> None:
    def digest(text: str) -> str:
        return next(iter(sass.parse_sass(text).values())).digest

    renumbered = SASS_TEXT.replace("/*0070*/", "/*0470*/").replace(".L_x_1", ".L_x_7")
    assert digest(renumbered) == digest(SASS_TEXT)
    assert digest(SASS_TEXT.replace("R0, SR_TID.X", "R2, SR_TID.X")) != digest(SASS_TEXT)


def test_records_combine_resources_dynamic_smem_and_stems() -> None:
    records = sass.build_records(SASS_TEXT, RESOURCE_TEXT, sass.parse_dynamic_smem(IR_TEXT))
    by_stem = {record.stem: record for record in records}
    assert set(by_stem) == {"gdn_prefill", "gdn_prefill_prologue"}
    prefill = by_stem["gdn_prefill"]
    assert prefill.resources == {
        "REG": "168",
        "STACK": "0",
        "SHARED": "1024",
        "LOCAL": "0",
        "DSMEM": "232448",
    }
    assert by_stem["gdn_prefill_prologue"].resources["DSMEM"] == "32776"
    assert prefill.instructions == 10


@pytest.mark.parametrize(
    ("symbol", "stem"),
    [
        ("kda_cudnn_summary_kernel_cutlass_frost_kda_summary_KdaCfg_0", "kda_summary"),
        ("split_k_plan_16_16_kernel_cutlass_plan_kernel_0", "split_k_plan_16_16"),
        # A struct argument mangled ahead of the config must not extend the stem.
        (
            "cudnn_kernel_cutlass_frost_state_chain_cutlasscutecorestructobjectat_0_8__128_8__2",
            "state_chain",
        ),
        ("cudnn_kernel_cutlass_frost_state_chain__128_128_8__False_tensorptrf32_2", "state_chain"),
    ],
)
def test_kernel_stem(symbol: str, stem: str) -> None:
    assert sass.kernel_stem(symbol) == stem


def test_carve_cubins_sizes_each_elf_from_its_headers() -> None:
    def elf(machine: int, payload: bytes) -> bytes:
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, machine)
        shoff, shentsize, shnum = 64 + len(payload), 64, 1
        struct.pack_into("<QQ", header, 32, 0, shoff)
        struct.pack_into("<HHHHH", header, 52, 64, 0, 0, shentsize, shnum)
        return bytes(header) + payload + bytes(shentsize * shnum)

    cubin = elf(190, b"kernel")
    host = elf(62, b"x86 host object")
    blob = b"junk" + host + b"pad" + cubin + b"tail" + cubin
    assert list(sass.carve_cubins(blob)) == [cubin, cubin]


def snapshot(directory: Path, label: str, text: str, resources: str, dsmem: str = "65536"):
    directory.mkdir(exist_ok=True)
    records = sass.build_records(text, resources)
    for record in records:
        record.resources["DSMEM"] = dsmem
    sass.write_label(directory, label, records, text)


def classify(tmp_path: Path, before: str, after: str, resources=RECOMPUTE_RESOURCES, **kw):
    snapshot(tmp_path / "a", "case__host_0", before, resources)
    snapshot(tmp_path / "b", "case__host_0", after, resources, **kw)
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    (label,) = report.labels
    return report, label


def test_identical_snapshots_roundtrip_through_files(tmp_path: Path) -> None:
    report, label = classify(tmp_path, RECOMPUTE_TEXT, RECOMPUTE_TEXT)
    assert label.verdict is diff.Verdict.IDENTICAL
    assert not report.failed(strict=True)
    (record,) = sass.read_label(tmp_path / "a", "case__host_0").values()
    assert record.stem == "kda_recompute" and record.resources["STACK"] == "8"
    assert record.layout[("STS.128", 0x1F000)] == 1


def test_rescheduled_code_with_equal_histogram_is_same_histogram_not_identical(tmp_path) -> None:
    lines = RECOMPUTE_TEXT.splitlines(keepends=True)
    lines[11], lines[12] = lines[12], lines[11]  # swap LOP3 and FFMA
    report, label = classify(tmp_path, RECOMPUTE_TEXT, "".join(lines))
    assert label.verdict is diff.Verdict.SAME_HISTOGRAM
    assert not report.failed(strict=True)
    rendered = diff.render(report, tmp_path / "a", tmp_path / "b", quiet=True)
    assert rendered.startswith("summary: 0 identical, 1 same-histogram")


def test_kda_recompute_tile_relocation_is_real(tmp_path: Path) -> None:
    report, label = classify(tmp_path, RECOMPUTE_TEXT, relocate_tiles(RECOMPUTE_TEXT))
    assert label.verdict is diff.Verdict.REAL
    assert report.failed(strict=False)
    (kernel,) = label.kernels
    layout, histogram = kernel.reasons
    assert layout.startswith("smem layout 12 entries: LDS 0x1ec20 -1 0x2ec20 +1, LDS.64")
    assert "STS.128 0x1ec00 +1 0x1ec10 +1 0x1f000 -1" in layout
    assert histogram.startswith("noise 15 -> 15 instrs: IMAD.MOV.U32 -1 IMAD.U32 +1")


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("LDS.64 R20", "LDS.128 R20", "LDS.128 +1 LDS.64 -1"),
        ("TRYWAIT P0, [UR13]", "TRYWAIT P0, [UR13+0x8]", "smem layout 2 entries"),
        ("FFMA R34", "FFMA.FTZ R34", "FFMA -1 FFMA.FTZ +1"),
        ("UTMALDG.3D", "UTMALDG.2D", "real opcodes UTMALDG"),
        ("EXIT ;", "EXIT ;\n        /*00f0*/                   FMUL2 R2, R4, R6 ;", "FMUL2 +1"),
    ],
    ids=["width", "wait-offset", "ftz", "tma-rank", "packed-fp32"],
)
def test_modifier_offset_and_real_opcode_changes_are_real(tmp_path, before, after, expected):
    _, label = classify(tmp_path, RECOMPUTE_TEXT, RECOMPUTE_TEXT.replace(before, after))
    assert label.verdict is diff.Verdict.REAL, label.kernels[0].reasons
    assert any(expected in reason for reason in label.kernels[0].reasons)


def test_noise_band_allows_small_address_arithmetic_deltas(tmp_path: Path) -> None:
    def with_imads(count: int) -> str:
        extra = "".join(f"        /*0{i}f0*/  IMAD R{i}, R1, R2, R3 ;\n" for i in range(count))
        return RECOMPUTE_TEXT.replace("        /*00e0*/", extra + "        /*00e0*/")

    report, label = classify(tmp_path, RECOMPUTE_TEXT, with_imads(diff.NOISE_BUDGET))
    assert label.verdict is diff.Verdict.NOISE
    assert not report.failed(strict=False) and report.failed(strict=True)
    _, label = classify(tmp_path, RECOMPUTE_TEXT, with_imads(diff.NOISE_BUDGET + 1))
    assert label.verdict is diff.Verdict.REAL


@pytest.mark.parametrize(
    ("field", "value"),
    [("REG:128", "REG:130"), ("STACK:8", "STACK:16"), ("SHARED:1024", "SHARED:2048")],
)
def test_resource_changes_are_real_without_code_changes(tmp_path, field, value) -> None:
    snapshot(tmp_path / "a", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES)
    snapshot(
        tmp_path / "b", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES.replace(field, value)
    )
    (label,) = diff.compare(tmp_path / "a", tmp_path / "b").labels
    assert label.verdict is diff.Verdict.REAL
    assert label.kernels[0].reasons == [f"{field.replace(':', ' ')} -> {value.split(':')[1]}"]


@pytest.mark.parametrize("dsmems", [("65536", "?"), ("?", "?")], ids=["one-side", "both"])
def test_unknown_dynamic_smem_is_reported(tmp_path: Path, dsmems) -> None:
    snapshot(tmp_path / "a", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES, dsmems[0])
    snapshot(tmp_path / "b", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES, dsmems[1])
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    assert report.failed(strict=False)
    assert "DSMEM" in report.labels[0].kernels[0].reasons[0]


def test_read_label_joins_stale_res_stems_through_symbols(tmp_path: Path) -> None:
    """An older snapshot written under a different stem rule still pairs its resources."""
    snapshot(tmp_path / "a", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES)
    for suffix in (".fns", ".res", ".ops", ".layout"):
        path = tmp_path / "a" / f"case__host_0{suffix}"
        path.write_text(path.read_text().replace("kda_recompute ", "kda_recompute_oldstem "))
    (record,) = sass.read_label(tmp_path / "a", "case__host_0").values()
    assert record.stem == "kda_recompute" and record.resources["REG"] == "128"


def test_missing_labels_kernels_and_failed_cases_fail(tmp_path: Path) -> None:
    two = RECOMPUTE_TEXT + SASS_TEXT.split("\n", 4)[4]  # add the prefill sections
    snapshot(tmp_path / "a", "case__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES)
    snapshot(tmp_path / "b", "case__host_0", two, RECOMPUTE_RESOURCES + RESOURCE_TEXT)
    snapshot(tmp_path / "b", "other__host_0", RECOMPUTE_TEXT, RECOMPUTE_RESOURCES)
    (tmp_path / "a" / "CASES.txt").write_text("ok case 3s\n")
    (tmp_path / "b" / "CASES.txt").write_text("ok case 3s\nFAIL other 1s ValueError\n")
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    verdicts = {label.label: label.verdict for label in report.labels}
    assert verdicts == {"case__host_0": diff.Verdict.REAL, "other__host_0": diff.Verdict.REAL}
    assert report.case_problems == ["case other: absent -> FAIL"]
    assert report.failed(strict=False)
    rendered = diff.render(report, tmp_path / "a", tmp_path / "b", quiet=True)
    assert "kernel only in b" in rendered
    assert "summary: 0 identical, 0 same-histogram, 0 noise, 2 real" in rendered
