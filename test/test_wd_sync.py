"""Every warpdrive GDN/KDA kernel configuration carries the tcgen05 thread-sync fences.

Each case compiles one kernel class over a static configuration (state flags, head dims, the
d_v split, the gate dtype, the prefill variant) without either cache and runs
``ptx_check.check_tcgen05_sync`` on every path of its PTX: a tcgen05 operation after a wait
needs ``tcgen05.fence::after_thread_sync``, a signal after a tcgen05 operation needs
``::before_thread_sync`` (PTX ISA 9.7.18.6.3). The layer emits both on managed synchronization;
the kernels' raw signals emit their own. The configurations are those the public ops and drivers
build (recorded over test/gdn, test/kda and test/linear). Launch bundles run the same kernel
classes; the paged prefill states compile only inside a warmup-forward bundle, so those bundles
are cases too. Compile-only: a few seconds each.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("cutlass.cute")

import cutlass

from attn_gym.warpdrive.ptx_check import capture_ptx, check_tcgen05_sync, read_ptx

PACKAGE = "attn_gym.linear._delta_rule.warpdrive"
COMPILE = {
    "gdn_bprop": "compile_gdn_bprop",
    "gdn_bprop_summary": "compile_gdn_bprop_summary",
    "gdn_prefill": "compile_gdn_prefill",
    "gdn_recompute": "compile_gdn_recompute",
    "gdn_summary": "compile_gdn_summary",
    "gdn_tinv": "compile_gdn_tinv",
    "kda_bprop": "compile_kda_bprop",
    "kda_bprop_summary": "compile_kda_bprop_summary",
    "kda_prefill": "compile_kda_prefill",
    "kda_prep": "compile_kda_prep",
    "kda_prep_prefill": "compile_kda_prep_prefill",
    "kda_recompute": "compile_kda_recompute",
    "kda_summary": "compile_kda_summary",
}
# (module, kernel class, constructor arguments, compile-function arguments after the op)
KERNELS = [
    ("gdn_bprop", "GdnBprop", ("bf16", 128, False, False, False), ("Float32", None, None)),
    ("gdn_bprop", "GdnBprop", ("bf16", 128, True, False, True), ("Float32", None, "Float32")),
    ("gdn_bprop", "GdnBprop", ("bf16", 128, True, True, False), ("Float32", "Float32", None)),
    ("gdn_bprop", "GdnBprop", ("bf16", 128, True, True, True), ("Float32", "Float32", "Float32")),
    ("gdn_bprop", "GdnBprop", ("bf16", 64, True, True, True), ("Float32", "Float32", "Float32")),
    ("gdn_bprop", "GdnBprop", ("fp16", 64, True, False, True), ("Float32", None, "Float32")),
    ("gdn_bprop_summary", "GdnBpropSummary", ("bf16", 128, 128), ()),
    ("gdn_bprop_summary", "GdnBpropSummary", ("bf16", 128, 64), ()),
    ("gdn_bprop_summary", "GdnBpropSummary", ("fp16", 64, 128), ()),
    ("gdn_bprop_summary", "GdnBpropSummary", ("fp16", 64, 64), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 128, 1, False, False, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 128, 1, True, False, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 128, 1, True, True, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 64, 2, False, False, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 64, 2, False, True, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 64, 2, True, False, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "fp32", 128, 64, 2, True, True, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("fp16", "fp32", 128, 128, 1, False, True, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 128, 1, False, False, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 128, 1, True, False, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 128, 1, True, True, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 64, 2, False, False, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 64, 2, False, True, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 64, 2, True, False, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("bf16", "fp32", 128, 64, 2, True, True, False), ()),
    ("gdn_prefill", "GdnPrefillGmem", ("fp16", "fp32", 128, 128, 1, False, True, False), ()),
    ("gdn_recompute", "GdnRecompute", ("bf16", 128, 128, False), (None,)),
    ("gdn_recompute", "GdnRecompute", ("bf16", 128, 128, True), ("Float32",)),
    ("gdn_recompute", "GdnRecompute", ("bf16", 128, 64, True), ("BFloat16",)),
    ("gdn_recompute", "GdnRecompute", ("bf16", 64, 64, False), (None,)),
    ("gdn_recompute", "GdnRecompute", ("bf16", 64, 64, True), ("Float32",)),
    ("gdn_summary", "GdnSummary", ("bf16", 128, 128), ()),
    ("gdn_summary", "GdnSummary", ("bf16", 128, 64), ()),
    ("gdn_summary", "GdnSummary", ("bf16", 64, 64), ()),
    ("gdn_summary", "GdnSummary", ("fp16", 64, 128), ()),
    ("gdn_tinv", "GdnTinv", ("bf16", 128), ("fp32",)),
    ("gdn_tinv", "GdnTinv", ("bf16", 64), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "bf16", 64, 64, False, True, False, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, False, False, False, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, False, True, False, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, True, False, True, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, True, True, False, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, True, True, True, 148), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, True, True, True, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 64, True, False, True, 152), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("fp16", "fp32", 128, 128, False, False, False, 4), ("fp32",)),
    ("kda_bprop", "KdaBprop", ("fp16", "fp32", 64, 128, True, False, True, 152), ("fp32",)),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 64, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 64, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 64, 64, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 128, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 128, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 64, True), ()),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "bf16", 128, 64, 2, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, False, False, 0, 152),
        ("fp32", None),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, False, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, False, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 0, 148),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 2, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 64, 2, False, False, 0, 152),
        ("fp32", None),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 64, 2, True, False, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 64, 2, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 64, 64, 1, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("fp16", "fp32", 128, 128, 1, False, False, 0, 4),
        ("fp32", None),
    ),
    ("kda_prep", "KdaPrep", ("bf16", "bf16", 128, 152), ("bf16",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 148), ("fp32",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 152), ("fp32",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 64, 152), ("fp32",)),
    ("kda_prep", "KdaPrep", ("fp16", "fp16", 128, 152), ("fp16",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, False, False, 0, 152), (None,)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, False, True, 0, 152), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, False, 0, 152), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 0, 1), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 0, 152), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 1, 152), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 2, 152), ("fp32",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 64, 2, True, True, 0, 152), ("bf16",)),
    ("kda_recompute", "KdaRecompute", ("bf16", "bf16", 128, 64, True, 152), ("fp32", "fp32")),
    ("kda_recompute", "KdaRecompute", ("bf16", "bf16", 64, 64, True, 152), ("fp32", "bf16")),
    ("kda_recompute", "KdaRecompute", ("bf16", "fp32", 128, 128, False, 152), ("fp32", None)),
    ("kda_recompute", "KdaRecompute", ("bf16", "fp32", 128, 128, True, 148), ("fp32", "fp32")),
    ("kda_recompute", "KdaRecompute", ("bf16", "fp32", 128, 128, True, 152), ("fp32", "fp32")),
    ("kda_recompute", "KdaRecompute", ("fp16", "fp32", 128, 128, False, 4), ("fp32", None)),
    ("kda_recompute", "KdaRecompute", ("fp16", "fp32", 64, 128, False, 152), ("fp32", None)),
    ("kda_summary", "KdaSummary", ("bf16", "bf16", 128, 64, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 148), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 64, 64, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("fp16", "fp16", 128, 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("fp16", "fp32", 64, 128, 152), ("fp32",)),
]
BF16, FP32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32


def _gdn_paged(has_initial_state: bool, tiles_per_head: int) -> tuple:
    """``gdn_warmup_forward._compile_warmup_forward`` arguments: bf16, fp32 states in and out,
    seed indices, a paged state, natural-log gates, d 128."""
    states = (True, True, True, False, True, has_initial_state, False, True)
    return (BF16, FP32, FP32, FP32, None, 152, *states, 128, 128, tiles_per_head, False)


def _kda_paged(paged_state: int, tiles_per_head: int) -> tuple:
    """``kda_warmup_forward._compile_warmup_forward`` arguments: bf16, fp32 states in and out,
    d 128; the d_v split runs the prep prefill."""
    dtypes = (BF16, FP32, FP32, I32, FP32, FP32)
    return (None, 152, *dtypes, 128, 128, tiles_per_head, tiles_per_head > 1, False, paged_state)


# (bundle module, _compile_warmup_forward arguments): the paged prefill states.
BUNDLES = [
    ("gdn_warmup_forward", _gdn_paged(True, 1)),
    ("gdn_warmup_forward", _gdn_paged(False, 2)),
    ("kda_warmup_forward", _kda_paged(2, 1)),
    ("kda_warmup_forward", _kda_paged(1, 2)),
    ("kda_warmup_forward", _kda_paged(2, 2)),
]


def _kernel_id(case) -> str:
    module, cls, args, _ = case
    return "-".join([module, cls, *map(str, args)])


def _bundle_id(case) -> str:
    module, args = case
    return "-".join([module, *(getattr(a, "__name__", str(a)) for a in args)])


def _check(tmp_path, build, *args):
    with capture_ptx(tmp_path) as out:
        build.__wrapped__(*args)  # uncached: a cached kernel has no PTX
    (path,) = out.glob("*.ptx")
    findings = check_tcgen05_sync(read_ptx(path))
    assert not findings, "\n".join(f.describe() for f in findings)


@pytest.mark.parametrize("case", KERNELS, ids=_kernel_id)
def test_kernel_ptx_is_fenced(requires_blackwell, tmp_path, case):
    module, cls, args, compile_args = case
    mod = importlib.import_module(f"{PACKAGE}.{module}")
    _check(tmp_path, getattr(mod, COMPILE[module]), getattr(mod, cls)(*args), *compile_args)


@pytest.mark.parametrize("case", BUNDLES, ids=_bundle_id)
def test_bundle_ptx_is_fenced(requires_blackwell, tmp_path, case):
    module, args = case
    mod = importlib.import_module(f"{PACKAGE}.bundles.{module}")
    _check(tmp_path, mod._compile_warmup_forward, *args)
