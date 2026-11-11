"""Pin liveness coverage of the shipped compile configurations across CuTeDSL versions."""

import importlib

import cutlass
import pytest
import torch

from attn_gym import warpdrive as wd
from attn_gym.warpdrive.kernel import WarpSpecialized

FOUND = "liveness: no deadlock found ("
CLUSTER = "liveness: unverified (cross-CTA placement)"
RAW = "liveness: unverified (raw barriers: "
COUNTS = "liveness: unverified (loop counts on "
EPOCHS = "liveness: unverified (unknown epoch correspondence on "
BRANCH = "liveness: unverified (runtime branch or unknown region: "
SCHEDULER = "liveness: unverified (scheduler channel "
# What stops each configuration, or FOUND. A CuTeDSL upgrade that changes a verdict fails here.
EXPECTED = {
    "gemm_oneshot": FOUND,
    "pgemm_static_single": FOUND,
    "pgemm_static_pair": CLUSTER,
    "pgemm_clc_pair": CLUSTER,
    "ex_pgemm_clc_single": SCHEDULER,
    "fa4_single": COUNTS,  # softmax fills n_blocks - 1 in its loop, correction takes n_blocks
    "fa4_clc_pair": CLUSTER,
    "fa4_clc_pair_splitp": CLUSTER,
    "ex_fa4_static": COUNTS,
    "ex_fa4_bwd": BRANCH,
    "ex_fa4_bwd_causal": BRANCH,
    "ex_fa4_bwd_2cta": CLUSTER,
    "ex_fa4_decode": EPOCHS,
    "ex_decode_p2": EPOCHS,
    "ex_decode_perseq": EPOCHS,
    "ex_decode_shared": EPOCHS,
    "ex_decode_shared_tiny": EPOCHS,
    "ex_grouped_gemm": FOUND,
    "ex_qwen_megakernel": BRANCH,
    "ex_raw_interop": RAW,
    "affine_summary_fwd": BRANCH,
    "affine_summary_rev": BRANCH,
    "delta_h_bwd": BRANCH,
    "wy_dqkg": RAW,
    "wy_dqkg_ragged": RAW,
    "kda_intra_fwd": BRANCH,
    "indexer_score_packed": FOUND,
    "indexer_score_generic": FOUND,
}

EXAMPLES = [
    ("gemm_oneshot", "gemm", "Gemm", "compile_gemm", {}),
    (
        "pgemm_static_single",
        "persistent_gemm",
        "PersistentGemm",
        "compile_persistent_gemm",
        {"schedule": wd.StaticPersistent(), "tile_n": 256},
    ),
    (
        "pgemm_static_pair",
        "persistent_gemm",
        "PersistentGemm",
        "compile_persistent_gemm",
        {"schedule": wd.StaticPersistent(), "cluster": wd.CtaPair(), "tile_n": 256},
    ),
    (
        "pgemm_clc_pair",
        "persistent_gemm",
        "PersistentGemm",
        "compile_persistent_gemm",
        {"schedule": wd.ClusterLaunchControl(), "cluster": wd.CtaPair(), "tile_n": 256},
    ),
    (
        "ex_pgemm_clc_single",
        "persistent_gemm",
        "PersistentGemm",
        "compile_persistent_gemm",
        {"schedule": wd.ClusterLaunchControl()},
    ),
    ("fa4_single", "fa4_fwd", "FlashAttentionForward", "compile_fa4_fwd", {}),
    (
        "fa4_clc_pair",
        "fa4_fwd",
        "FlashAttentionForward",
        "compile_fa4_fwd",
        {"schedule": wd.ClusterLaunchControl(), "cluster": wd.CtaPair()},
    ),
    (
        "fa4_clc_pair_splitp",
        "fa4_fwd",
        "FlashAttentionForward",
        "compile_fa4_fwd",
        {"schedule": wd.ClusterLaunchControl(), "cluster": wd.CtaPair(), "split_p": True},
    ),
    (
        "ex_fa4_static",
        "fa4_fwd",
        "FlashAttentionForward",
        "compile_fa4_fwd",
        {"schedule": wd.StaticPersistent()},
    ),
    ("ex_fa4_bwd", "fa4_bwd", "FlashAttentionBackward", "compile_fa4_bwd", {}),
    (
        "ex_fa4_bwd_causal",
        "fa4_bwd",
        "FlashAttentionBackward",
        "compile_fa4_bwd",
        {"causal": True},
    ),
    (
        "ex_fa4_bwd_2cta",
        "fa4_bwd",
        "FlashAttentionBackward2Cta",
        "compile_fa4_bwd",
        {"causal": False, "roles": None},
    ),
    (
        "ex_fa4_decode",
        "fa4_decode_swapab",
        "SwapABDecode",
        "compile_decode",
        {"tile_hq": 8},
    ),
    (
        "ex_decode_p2",
        "fa4_decode_swapab",
        "SwapABDecode",
        "compile_decode",
        {"tile_hq": 8, "p_stages": 2},
    ),
    (
        "ex_decode_perseq",
        "fa4_decode_swapab",
        "SwapABDecode",
        "compile_decode",
        {"tile_hq": 8, "per_sequence": True},
    ),
    (
        "ex_decode_shared",
        "fa4_decode_swapab",
        "SharedKVDecode",
        "compile_decode",
        {"tile_hq": 8},
    ),
    (
        "ex_decode_shared_tiny",
        "fa4_decode_swapab",
        "SharedKVDecode",
        "compile_decode",
        {"tile_hq": 8, "kv_stages": (1, 1)},
    ),
    ("ex_grouped_gemm", "grouped_gemm", "GroupedGemm", "compile_grouped_gemm", {}),
    (
        "ex_qwen_megakernel",
        "qwen_megakernel",
        "QwenDecodeMegakernel",
        "compile_megakernel",
        {},
    ),
    (
        "ex_raw_interop",
        "raw_interop",
        "RawInterop",
        "compile_raw_interop",
        {},
    ),
]
PORTED = [
    (
        "affine_summary_fwd",
        "linear._delta_rule.cute.affine_summary_fwd",
        "_compile_affine_summary",
        ("bf16", 2, 64, False, True),
        {"fastmath": True},
    ),
    (
        "affine_summary_rev",
        "linear._delta_rule.cute.affine_summary_rev",
        "_compile_affine_summary_rev",
        ("bf16", 2, False, True, True),
        {},
    ),
    (
        "delta_h_bwd",
        "linear.kda.bwd.cute.chunk_delta_h_bwd",
        "_compile_delta_h_bwd",
        (4, 32, cutlass.BFloat16, False, False, True, False),
        {},
    ),
    (
        "wy_dqkg",
        "linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused",
        "_compile_chunk_kda_bwd_wy_dqkg",
        (),
        {
            "heads": 2,
            "head_dim": 128,
            "chunk_size": 64,
            "io_dtype": torch.bfloat16,
            "scale": 128**-0.5,
            "fastmath": True,
            "ragged": False,
        },
    ),
    (
        "wy_dqkg_ragged",
        "linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused",
        "_compile_chunk_kda_bwd_wy_dqkg",
        (),
        {
            "heads": 2,
            "head_dim": 128,
            "chunk_size": 64,
            "io_dtype": torch.bfloat16,
            "scale": 128**-0.5,
            "fastmath": True,
            "ragged": True,
        },
    ),
    (
        "kda_intra_fwd",
        "linear.kda.fwd.cute.chunk_kda_fwd_intra_engine",
        "_compile_intra_engine_fwd",
        (4, 128, False, False),
        {},
    ),
    (
        "indexer_score_packed",
        "sparse.indexer.impl.cute",
        "_compile_scores",
        (torch.bfloat16, 64, 128, True, 1, False, True, False),
        {},
    ),
    (
        "indexer_score_generic",
        "sparse.indexer.impl.cute",
        "_compile_scores",
        (torch.bfloat16, 16, 64, True, 1, False, True, False),
        {},
    ),
]


@pytest.mark.parametrize("config", EXAMPLES + PORTED, ids=lambda c: c[0])
def test_shipped_liveness_coverage(config, monkeypatch):
    reports = []
    finish = WarpSpecialized._finish_trace

    def capture(self):
        finish(self)
        reports.append(self._ws_liveness.line())

    monkeypatch.setattr(WarpSpecialized, "_finish_trace", capture)
    if config in EXAMPLES:
        _, module, constructor, compile_name, kwargs = config
        if module.startswith("fa4"):
            pytest.importorskip("flash_attn.cute")
        mod = importlib.import_module(f"attn_gym.warpdrive.examples.{module}")
        args, kwargs = (getattr(mod, constructor)(**kwargs),), {}
    else:
        _, module, compile_name, args, kwargs = config
        mod = importlib.import_module(f"attn_gym.{module}")
    expected = EXPECTED[config[0]]
    getattr(mod, compile_name).__wrapped__(*args, **kwargs)
    assert reports and all(line.startswith(expected) for line in reports), reports
