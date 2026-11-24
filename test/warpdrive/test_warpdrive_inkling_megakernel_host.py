"""Host-side contracts of the Inkling 2 Mini megakernel that need no GPU: the PyTorch paths
import without CuTeDSL, the kernels' compile keys hash no Inkling host module, and the runners
reject shapes the kernels cannot run before anything launches."""

import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from attn_gym.warpdrive.cache._key import module_closure
from attn_gym.warpdrive.examples.inkling_megakernel import ref

CONFIG = ref.InklingConfig()


def test_pytorch_paths_import_without_cutedsl():
    """The oracle, the quantized-expert loader, the torch drafter and speculative step, and the
    chat's PyTorch backends import no CuTeDSL: with ``cutlass`` blocked, importing them
    succeeds."""
    probe = (
        "import sys\n"
        "sys.modules['cutlass'] = None\n"
        "from attn_gym.warpdrive.examples.inkling_megakernel import (\n"
        "    dspark, fast, quant, ref, spec\n"
        ")\n"
        "from attn_gym.warpdrive.examples.megakernel import formats\n"
        "from attn_gym.warpdrive.examples.megakernel.spec import SpecStep\n"
        "from attn_gym.warpdrive.examples.inkling_megakernel.chat import reply_parts\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_kernels_compile_from_kernel_modules_only():
    """The compile keys of the Inkling launches hash no Inkling host module: editing the
    oracle, the quantized-expert loader, the host runners, the drafters and speculative step,
    the benches or the chat recompiles nothing (``test_warpdrive_megakernel_package`` checks they
    hash no other model either)."""
    package = Path(ref.__file__).parent
    host = {"ref", "act_quant", "quant", "megakernel", "fast", "bench", "chat"}
    host |= {"dspark", "spec", "draft", "draft_bench"}
    host = {package / f"{name}.py" for name in host}
    kernels = ("visit_kernel", "msg_kernel", "head_kernel", "multi_kernel")
    drafts = ("draft_common", "draft_ctx_kernel", "draft_backbone_kernel", "draft_head_kernel")
    for kernel in kernels + drafts:
        assert not host & set(module_closure(package / f"{kernel}.py")), kernel
    # Nor does the visit launch hash the multi-token GEMVs, which only its block form runs.
    multi_gemv = package.parent / "megakernel" / "multi_gemv.py"
    assert multi_gemv not in module_closure(package / "visit_kernel.py")


@pytest.mark.parametrize(
    "change,match",
    [
        # Routing reads 3 logits per lane: odd lanes would issue misaligned vector loads.
        ({"num_experts": 96}, "routing reads"),
        ({"num_experts": 2048}, "at most 1024 experts"),
        # An o_proj split merges 8 heads x 128 = 1024 elements into a 768-element input row.
        ({"hidden_size": 768, "num_q_heads": 32, "moe_intermediate_size": 512,
          "rel_extent": 512, "sliding_window": 512}, r"hidden-size \(768\) input row"),
    ],
)  # fmt: skip
def test_unsupported_geometry_is_rejected(change, match):
    """Shapes the kernels cannot run are rejected when a runner builds its kernels, for both
    layer kinds, before anything launches."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import visit_kernel_for

    config = dataclasses.replace(CONFIG, **change)
    for full in (False, True):
        with pytest.raises(ValueError, match=match):
            visit_kernel_for(config, full)


@pytest.mark.parametrize(
    "change,match",
    [
        # The union counts one expert per compute thread (256), and routing packs a lane's
        # chosen experts below its slot's bit 8.
        ({"num_experts": 512}, "at most 256 experts"),
        # Rows 0 and 7 of a block are 7 positions apart: outside a 4-key window and bias extent.
        ({"sliding_window": 4, "rel_extent": 4}, "window and relative-bias extent"),
    ],
)  # fmt: skip
def test_unsupported_block_geometry_is_rejected(change, match):
    """Shapes the one-token kernels run but an 8-token block cannot are rejected when a
    runner builds the multi-token kernels, for both layer kinds, before anything launches."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import visit_kernel_for

    config = dataclasses.replace(CONFIG, **change)
    for full in (False, True):
        visit_kernel_for(config, full)
        with pytest.raises(ValueError, match=match):
            visit_kernel_for(config, full, tokens=8)


@pytest.mark.parametrize("tokens", [1, 3])
def test_decoder_rejects_unsupported_block_sizes(tokens):
    """The block step runs the tested blocks of 2, 4 and 8 tokens; a one-token block would
    take the block path's shapes for a one-token launch. Rejected before any weight is read."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingDecoder

    with pytest.raises(ValueError, match=r"block_tokens must be one of \(2, 4, 8\)"):
        InklingDecoder(SimpleNamespace(config=CONFIG), block_tokens=tokens)


@pytest.mark.parametrize(
    "change,match",
    [
        # The reduced decoder tests' eight layers: 16 visits.
        ({"num_layers": 8, "full_layers": (1, 3, 5, 7), "block_layers": 2, "loop_start": 2,
          "loop_end": 6}, "more than 16 visits"),
        # A core of layers 5-22 restarts its second and third passes at visits 23 and 41.
        ({"loop_start": 5, "loop_end": 23}, "must not restart a core pass"),
    ],
)  # fmt: skip
def test_block_features_need_the_drafter_visits(change, match):
    """``decode_block``'s features read the inputs of visits 5, 23 and 41 as the previous
    visits' outputs: model shapes without those visits, or restarting a pass at one, are
    rejected (the decoder itself takes them; ``decode_block`` checks before launching)."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import check_feature_visits

    check_feature_visits(CONFIG)
    with pytest.raises(ValueError, match=match):
        check_feature_visits(dataclasses.replace(CONFIG, **change))
