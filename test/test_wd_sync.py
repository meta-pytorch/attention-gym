"""Every warpdrive GDN/KDA kernel configuration carries the tcgen05 thread-sync fences.

Each case compiles one kernel class over a static configuration (state flags, head dims, the
d_v split, the gate dtype, the prefill variant) without either cache and runs
``ptx_check.check_tcgen05_sync`` on every path of its PTX: a tcgen05 operation after a wait
needs ``tcgen05.fence::after_thread_sync``, a signal after a tcgen05 operation needs
``::before_thread_sync`` (PTX ISA 9.7.18.6.3), and every path reaching the TMEM teardown
rendezvous completes its loads and stores. The layer emits the fences on managed
synchronization and the drain at teardown; the kernels' raw signals emit their own fences.
Calls that skip a fence another one already provides (``skip_fence=``, a raw helper's
``fence=False``) are proven only here, per configuration. The configurations are those the public ops and drivers
build (recorded over test/gdn, test/kda and test/linear). Each kernel case also traces with
strict alias and fence diagnostics and allows only the audited fence entries in
``FENCE_UNVERIFIED``. Launch bundles run the same kernel
classes; the paged prefill states compile only inside a warmup-forward bundle, so those bundles
are cases too. Compile-only: a few seconds each.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

pytest.importorskip("cutlass.cute")

import cutlass

from attn_gym.warpdrive.fences import FENCE_KINDS
from attn_gym.warpdrive.named import TMEM_TEARDOWN_BARRIER_ID
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
    ("kda_bprop", "KdaBprop", ("fp16", "fp32", 128, 128, True, True, True, 152), ("fp32",)),
    (
        "kda_bprop",
        "KdaBprop",
        ("bf16", "fp32", 128, 128, True, True, True, 152, False),
        ("fp32",),
    ),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 64, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 64, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 64, 64, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 128, False), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 128, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 64, 64, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("fp16", 128, 128, True), ()),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, True, False), ()),
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
    (
        "kda_prefill",
        "KdaPrefill",
        ("fp16", "fp32", 128, 64, 2, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 0, 152, False),
        ("fp32", "fp32"),
    ),
    ("kda_prep", "KdaPrep", ("bf16", "bf16", 128, 152), ("bf16",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 148), ("fp32",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 152), ("fp32",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 64, 152), ("fp32",)),
    ("kda_prep", "KdaPrep", ("fp16", "fp16", 128, 152), ("fp16",)),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 152, False), ("fp32",)),
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
    ("kda_recompute", "KdaRecompute", ("fp16", "fp32", 128, 128, True, 152), ("fp32", "fp32")),
    (
        "kda_recompute",
        "KdaRecompute",
        ("bf16", "fp32", 128, 128, True, 152, False),
        ("fp32", "fp32"),
    ),
    ("kda_summary", "KdaSummary", ("bf16", "bf16", 128, 64, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 148), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 64, 64, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("fp16", "fp16", 128, 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("fp16", "fp32", 64, 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 152, False), ("fp32",)),
]
KERNELS += [
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "bf16", 128, 128, 1, True, True, False), ()),
    ("gdn_prefill", "GdnPrefillCompute", ("bf16", "bf16", 128, 64, 2, True, True, False), ()),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 2, 152),
        ("fp32", "bf16"),
    ),
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 64, 2, True, True, 2, 152),
        ("fp32", "bf16"),
    ),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 1, 152), ("bf16",)),
    ("kda_prep_prefill", "KdaPrepPrefill", ("bf16", 128, 2, True, True, 2, 152), ("bf16",)),
]
BF16, FP32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32


def _gdn_paged(has_initial_state: bool, tiles_per_head: int, state_dtype=FP32) -> tuple:
    """``gdn_warmup_forward._compile_warmup_forward`` arguments: BF16 inputs,
    selectable state dtype, seed indices, paged state, natural-log gates, d 128."""
    states = (True, True, True, False, True, has_initial_state, False, True)
    return (BF16, state_dtype, FP32, FP32, None, 152, *states, 128, 128, tiles_per_head, False)


def _kda_paged(paged_state: int, tiles_per_head: int, state_dtype=FP32) -> tuple:
    """``kda_warmup_forward._compile_warmup_forward`` arguments: BF16 inputs,
    selectable input/output state dtype, d 128; the d_v split runs the prep prefill."""
    dtypes = (BF16, FP32, FP32, I32, state_dtype, state_dtype)
    return (None, 152, *dtypes, 128, 128, tiles_per_head, tiles_per_head > 1, False, paged_state)


# (bundle module, _compile_warmup_forward arguments): the paged prefill states.
BUNDLES = [
    ("gdn_warmup_forward", _gdn_paged(True, 1)),
    ("gdn_warmup_forward", _gdn_paged(False, 2)),
    ("kda_warmup_forward", _kda_paged(2, 1)),
    ("kda_warmup_forward", _kda_paged(1, 2)),
    ("kda_warmup_forward", _kda_paged(2, 2)),
    ("gdn_warmup_forward", _gdn_paged(True, 1, BF16)),
    ("gdn_warmup_forward", _gdn_paged(False, 2, BF16)),
    ("kda_warmup_forward", _kda_paged(2, 1, BF16)),
    ("kda_warmup_forward", _kda_paged(1, 2, BF16)),
    ("kda_warmup_forward", _kda_paged(2, 2, BF16)),
]


# Fence-check entries that stay "unverified" on these kernels, each audited against the code and
# keyed by its report line (channel, side, effective fences): the fence is the one PTX requires,
# and the check cannot see why. Accesses go through raw inline PTX (stmatrix, st.shared,
# tcgen05.ld/st) or raw tmem.ptr/cols column lookups, so no annotation is recorded while the
# handle is open; TMEM accesses count against every open handle (``fences._seen``), so another
# channel's TMEM traffic inside a hold looks unmatched. Proxy rule: generic SMEM accesses
# followed by async-proxy ones (TMA, tcgen05.mma operands) need fence.proxy.async in each
# accessing thread before its signal (PTX ISA 9.7.10.28.2, 9.7.18.6.5). TMEM: a tcgen05.st handed
# on needs tcgen05.wait::st, a tcgen05.ld handed to another warp's writer needs tcgen05.wait::ld
# (9.7.18.6.2, 9.7.18.6.4.4). Any other unverified entry (another fence at an audited site among
# them), an unverified requirement the check names by fence kind, any missing fence, or any alias
# finding fails.
_GENERIC_TO_UMMA = "generic stmatrix/st.shared writes read by tcgen05.mma need smem_async"
_GENERIC_TO_TMA = "generic stores read by a TMA store need smem_async"
_TILES = (
    "the tinv/a/da/dm tiles share the Inter stage at fixed offsets and bind no buffer (binding "
    "Inter to all four would alias them); stmatrix writes read by tcgen05.mma need smem_async"
)
FENCE_UNVERIFIED = {
    ("gdn_bprop", "q release_fence none"): (
        "cg1's ldmatrix reads of sQ are followed by its explicit fence_async_shared before the "
        "release, so the TMA refill needs no further fence"
    ),
    ("gdn_bprop", "v release_fence none"): (
        "cg1 reads sV, writes U over it, and the u commit's smem_async runs in every thread "
        "before v's release: the TMA refill is ordered after a fence.proxy.async"
    ),
    ("gdn_bprop", "do release_fence none"): (
        "as v: the u commit's smem_async precedes do's release"
    ),
    ("gdn_bprop", "dm commit_fence smem_async"): _GENERIC_TO_UMMA + " (dM, no buffer bound)",
    ("gdn_bprop", "u commit_fence smem_async"): _GENERIC_TO_UMMA + " (U over sV, no buffer bound)",
    ("gdn_bprop", "z commit_fence smem_async"): _GENERIC_TO_UMMA + " (Z in sdQ, no buffer bound)",
    ("gdn_bprop", "dq commit_fence smem_async"): _GENERIC_TO_TMA,
    ("gdn_bprop", "dk commit_fence smem_async"): (
        _GENERIC_TO_TMA + " (cg2); cg1 uses sdK as scratch and signals raw without a fence, "
        "but cg2 overwrites every scratch byte after dgate_cg1 and fences before its commit, so "
        "the TMA read cannot observe cg1's writes"
    ),
    ("gdn_bprop", "dv commit_fence smem_async"): _GENERIC_TO_UMMA + " and by a TMA store",
    ("gdn_bprop_summary", "a_acc release_fence tmem_load"): (
        "pairwise's tcgen05.ld of a precedes the next QK MMA write (wait::ld); the do_prime "
        "stores inside the hold are handed on by do_prime's own wait::st"
    ),
    ("gdn_bprop_summary", "do_prime commit_fence tmem_store"): (
        "tcgen05.st read by tcgen05.mma needs wait::st; the MMA side's tmem.ptr is raw"
    ),
    ("gdn_bprop_summary", "trip release_fence tmem_load+tmem_store"): (
        "du loads dvdk/dy before MMA overwrites (wait::ld) and stores du/dvdk the next MMAs "
        "read (wait::st); the MMA side's tmem.ptr is raw"
    ),
    ("gdn_bprop_summary", "dstate release_fence tmem_load+tmem_store"): (
        "state loads dstate before MMA writes (wait::ld) and stores the rescaled dstate the MMA "
        "reads (wait::st); its dstate_in stores in the same loop are dstate_in's"
    ),
    ("gdn_bprop_summary", "dstate_in commit_fence tmem_store"): (
        "tcgen05.st read by tcgen05.mma needs wait::st; the dstate accesses in the same loop are "
        "handed on by dstate's release"
    ),
    ("gdn_tinv", "tile commit_fence smem_async"): (
        "the tile's generic writes are verified; the unmatched access is the TMEM acc load inside "
        "the hold, ordered by acc's own tmem_load release"
    ),
    ("gdn_prefill", "cg0 release_fence tmem_load"): (
        "compute0's tcgen05.ld (raw tmem.ptr column) precedes the MMA warp's overwrite: wait::ld"
    ),
    ("gdn_prefill", "o commit_fence smem_async"): (
        _GENERIC_TO_TMA + "; sO is allocated ahead of the layer for the SMEM order, so o binds "
        "no buffer"
    ),
    ("kda_prefill", "kk commit_fence smem_async"): (
        _GENERIC_TO_UMMA + " (Kdecay); the epilogue consumer reads only Kinv, which the check "
        "reports as a missing Kdecay access"
    ),
    ("kda_prefill", "qk commit_fence smem_async"): (
        _GENERIC_TO_UMMA + " (Krestore); the epilogue reads only Qdecay"
    ),
    ("kda_prep", "raw release_fence smem_async"): (
        "warp 0 refills the stage by TMA after the empty wait (external producer); consumers "
        "read it, and with an fp32 gate write it, generically: smem_async"
    ),
    ("kda_prep_prefill", "o_acc release_fence tmem_load"): (
        "cg1's tcgen05.ld of qs precedes the MMA's qs writes, with no pipelining st: wait::ld"
    ),
    ("kda_bprop", "decay commit_fence smem_async"): (
        _GENERIC_TO_UMMA + " (Qdecay/Kdecay); qk_raw's tcgen05.st inside the hold is qk_raw's"
    ),
    ("kda_bprop", "qk_raw commit_fence tmem_store"): (
        "tcgen05.st read by cg2's tcgen05.ld: wait::st"
    ),
    ("kda_bprop", "qk_raw release_fence tmem_load"): (
        "cg2's tcgen05.ld precedes cg0's overwrite four chunks later: wait::ld; the state_in and "
        "dkd loads inside the hold are those channels'"
    ),
    ("kda_bprop", "state_in commit_fence tmem_store"): (
        "tcgen05.st read by tcgen05.mma and cg2's tcgen05.ld: wait::st"
    ),
    ("kda_bprop", "state_in release_fence tmem_load"): (
        "cg2's tcgen05.ld before cg0's overwrite: wait::ld"
    ),
    ("kda_bprop", "tinv commit_fence smem_async"): _TILES,
    ("kda_bprop", "a commit_fence smem_async"): _TILES,
    ("kda_bprop", "da commit_fence smem_async"): _TILES,
    ("kda_bprop", "dm commit_fence smem_async"): _TILES,
    ("kda_bprop", "dqk release_fence tmem_load"): (
        "cg2's tcgen05.ld of dq/dkr/dki/dkd precede the next MMA writes: wait::ld completes all"
    ),
    ("kda_bprop", "dstate commit_fence smem_async"): (
        _GENERIC_TO_UMMA + " and by cg2's generic loads"
    ),
    ("kda_bprop", "dq commit_fence smem_async"): _GENERIC_TO_TMA,
    ("kda_bprop", "dk commit_fence none"): (
        "no fence of its own: the same threads' dq commit runs fence.proxy.async after these "
        "stores and before dk's arrival, which orders every prior generic access"
    ),
    ("kda_bprop", "dgate commit_fence smem_async"): _GENERIC_TO_TMA,
    ("kda_bprop", "dv commit_fence smem_async"): _GENERIC_TO_UMMA + " and by a TMA store",
}


def _kernel_id(case) -> str:
    module, cls, args, _ = case
    return "-".join([module, cls, *map(str, args)])


def _bundle_id(case) -> str:
    module, args = case
    return "-".join([module, *(getattr(a, "__name__", str(a)) for a in args)])


def _unaudited(module: str, op) -> list[str]:
    """The fence results of ``op``'s trace that ``FENCE_UNVERIFIED`` does not audit."""
    out = []
    for r in op._ws_fences.results if op._ws_fences else []:
        site = f"{r.channel} {r.key} {'+'.join(r.fences) or 'none'}"
        named = [u for u in r.unverified if u.startswith(FENCE_KINDS)]  # "tmem_load: ..."
        if r.status != "covered" and ((module, site) not in FENCE_UNVERIFIED or named):
            out.append(r.describe())
    return out


def _check(tmp_path, build, *args):
    with capture_ptx(tmp_path) as out:
        build.__wrapped__(*args)  # uncached: a cached kernel has no PTX
    (path,) = out.glob("*.ptx")
    findings = check_tcgen05_sync(read_ptx(path), teardown_barrier=TMEM_TEARDOWN_BARRIER_ID)
    assert not findings, "\n".join(f.describe() for f in findings)


@pytest.mark.parametrize("case", KERNELS, ids=_kernel_id)
def test_kernel_ptx_and_reports(requires_blackwell, tmp_path, case):
    """Every PTX path is fenced and drained; with strict diagnostics (a missing fence or alias
    hazard raises) the trace leaves no alias finding and only allowlisted fence entries."""
    module, cls, args, compile_args = case
    mod = importlib.import_module(f"{PACKAGE}.{module}")
    op = getattr(mod, cls)(*args)
    op.alias_check = "strict"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the reports are asserted below
        _check(tmp_path, getattr(mod, COMPILE[module]), op, *compile_args)
    unaudited = _unaudited(module, op)
    assert not unaudited, "\n".join(unaudited)
    flagged = [r.describe() for r in op._ws_alias.results if r.status != "no hazard found"]
    assert not flagged, "\n".join(flagged)


def test_dropped_audited_fence_fails(requires_blackwell, tmp_path):
    """cg2 loads Q/K from the qk_raw TMEM ring and releases it to cg0's overwrite; without its
    wait::ld the PTX stays fenced and the entry stays unverified, but with no fence: the
    allowlist must reject it."""
    from attn_gym.linear._delta_rule.warpdrive.kda_bprop import KdaBprop, compile_kda_bprop

    op = KdaBprop("bf16", "fp32", 128, 128, True, True, True, 152)
    op.qk_raw.release_fence = ()
    op.alias_check = "strict"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _check(tmp_path, compile_kda_bprop, op, "fp32")
    unaudited = _unaudited("kda_bprop", op)
    assert len(unaudited) == 1 and "qk_raw release_fence none" in unaudited[0], unaudited


@pytest.mark.parametrize("case", BUNDLES, ids=_bundle_id)
def test_bundle_ptx_is_fenced(requires_blackwell, tmp_path, case):
    module, args = case
    mod = importlib.import_module(f"{PACKAGE}.bundles.{module}")
    _check(tmp_path, mod._compile_warmup_forward, *args)
