"""Host-side contracts of the Inkling 2 Mini megakernel that need no GPU: the PyTorch paths
import without CuTeDSL, the kernels' compile keys hash no Inkling host module, and the runners
reject shapes the kernels cannot run before anything launches."""

import dataclasses
import subprocess
import sys
from pathlib import Path

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
    oracle, the quantized-expert loader, the host runners, the drafter and speculative step,
    the bench or the chat recompiles nothing (``test_warpdrive_megakernel_package`` checks they
    hash no other model either)."""
    package = Path(ref.__file__).parent
    host = {"ref", "quant", "megakernel", "fast", "bench", "chat", "dspark", "spec"}
    host = {package / f"{name}.py" for name in host}
    for kernel in ("visit_kernel", "msg_kernel", "head_kernel"):
        assert not host & set(module_closure(package / f"{kernel}.py")), kernel


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
