"""Hardware row-max loads preserve the unmasked FA4 example's numerical contract."""

import re
import shutil
import subprocess

import pytest
import torch


@pytest.fixture
def requires_sm103():
    if not torch.cuda.is_available():
        pytest.skip("requires an SM103-family GPU")
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor < 3:
        pytest.skip("requires an SM103-family GPU")


@pytest.mark.parametrize("mode", ["single", "pair_clc", "pair_static"])
def test_fa4_hardware_rowmax_matches_software(requires_sm103, mode):
    pytest.importorskip("flash_attn.cute")
    from test_warpdrive import fa4_inputs

    from attn_gym.warpdrive import (
        ClusterLaunchControl,
        CtaPair,
        StaticPersistent,
    )
    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    kwargs = {}
    if mode != "single":
        kwargs = {
            "cluster": CtaPair(),
            "schedule": ClusterLaunchControl() if mode == "pair_clc" else StaticPersistent(),
            "split_p": True,
            "max_ctas": 4,
        }
    # Ragged Q, growing K-block magnitudes, and distinct head/tile scales force rescaling.
    # The small persistent cap makes the same CTAs wrap their channel phases across items.
    for b, h, sq, sk in [(1, 2, 640, 384), (2, 3, 1024, 512)]:
        q, k, v = fa4_inputs(b, h, sq, sk, nonuniform=True)
        expected, expected_lse = fa4_fwd(q, k, v, **kwargs)
        for _ in range(3):
            actual, actual_lse = fa4_fwd(q, k, v, use_ldred_rowmax=True, **kwargs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)


def test_fa4_hardware_rowmax_rejects_sm100(monkeypatch):
    pytest.importorskip("flash_attn.cute")
    from cutlass.base_dsl import BaseDSL

    from attn_gym._backends.cute.compat import Arch
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    monkeypatch.setattr(BaseDSL._get_dsl(), "get_arch_enum", lambda: Arch.sm_100a)
    with pytest.raises(ValueError, match="SM103-family"):
        FlashAttentionForward(use_ldred_rowmax=True)


def test_fa4_hardware_rowmax_preserves_rescale_flush_to_zero(requires_sm103):
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    q = torch.ones(1, 256, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    k[:, 128:] = 7.8125  # row-max growth makes the rescale exponent about -127.5
    v = torch.zeros_like(q)
    v[:, :128] = 1e36  # amplify a subnormal rescale without overflowing the first PV
    expected, expected_lse = fa4_fwd(q, k, v)
    actual, actual_lse = fa4_fwd(q, k, v, use_ldred_rowmax=True)
    # The software path's fast exp2 deliberately flushes this scale to zero.
    assert torch.count_nonzero(expected).item() == 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)


def test_fa4_rowmax_compiles_distinct_hardware_load(requires_sm103, tmp_path):
    pytest.importorskip("flash_attn.cute")
    from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support

    from attn_gym.utils import cuda_kernel_profiler
    from attn_gym.warpdrive.examples.fa4_fwd import (
        FlashAttentionForward,
        compile_fa4_fwd,
        fa4_fwd,
    )

    disassembler = shutil.which("nvdisasm")
    if disassembler is None:
        pytest.skip("generated hardware-load check requires nvdisasm")
    software = FlashAttentionForward()
    hardware = FlashAttentionForward(use_ldred_rowmax=True)
    assert software.__cute_cache_key__() != hardware.__cute_cache_key__()
    for op in (software, hardware):
        compiled = compile_fa4_fwd.__wrapped__(op)  # the compiled IR, not a cached artifact
        obj = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        elf_starts = [match.start() for match in re.finditer(b"\x7fELF", obj)]
        assert len(elf_starts) == 2  # host object followed by one device cubin
        cubin = tmp_path / "kernel.cubin"
        cubin.write_bytes(obj[elf_starts[1] :])
        sass = subprocess.check_output([disassembler, str(cubin)], text=True)
        assert ("LDTM.STAT.x32.MAX.F32" in sass) == op.use_ldred_rowmax
    # Also verify the public wrapper reaches the compiled hardware path, not a cached default.
    q = torch.ones(1, 256, 1, 128, device="cuda", dtype=torch.bfloat16)
    fa4_fwd(q, q, q, use_ldred_rowmax=True)
    with cuda_kernel_profiler("_ldred_") as recorded:
        fa4_fwd(q, q, q, use_ldred_rowmax=True)
    assert recorded["found"], recorded["kernel_names"]
