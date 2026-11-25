"""``checked=True`` builds: tensor indices and view offsets are checked.

A CuTe layout maps any coordinate to an offset, so a wrong index or a view built before its
tensor addresses other memory without an error. The checked build of a kernel compares each
index with its extent and aborts the launch that runs one outside; ``unchecked("bounds")``
keeps a deliberate one. Every other kernel is traced by the unmodified library.
"""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from checked_kernels import MistakenGemm, gemm_ptx
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym.warpdrive import Cluster, unchecked
from attn_gym.warpdrive.bounds import indexing
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive.compat import TENSOR_CLASS
from attn_gym.warpdrive.examples import grouped_gemm
from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm
from attn_gym.warpdrive.examples.grouped_gemm import GroupedGemm, compile_grouped_gemm
from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm
from attn_gym.warpdrive.protocol import compile_scope, take_opt_outs
from attn_gym.warpdrive.ptx_check import capture_ptx, read_ptx

SCRIPT = Path(__file__).with_name("checked_kernels.py")
ASSERT = "Assertion `checked_kernels.py:"


def _script(*args: str, **env: str) -> subprocess.CompletedProcess:
    """``checked_kernels.py`` in its own process."""
    environ = {k: v for k, v in os.environ.items() if "COREDUMP" not in k} | env
    command = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(
        command, capture_output=True, text=True, timeout=600, env=environ, check=False
    )


def _launch(mistake: str, checked: bool) -> subprocess.CompletedProcess:
    return _script(mistake, str(int(checked)))


def _asserts(ptx: str) -> int:
    return ptx.count("__assertfail")


def test_checked_is_a_bool():
    with pytest.raises(ValueError, match="checked must be a bool"):
        Gemm(checked=1)


def test_checked_is_a_separate_build_with_the_same_result(requires_blackwell, tmp_path):
    plain, checked = Gemm(tile_n=64), Gemm(tile_n=64, checked=True)
    assert plain.__cute_cache_key__() != checked.__cute_cache_key__()
    assert _asserts(gemm_ptx(tmp_path / "plain", plain)) == 0
    assert _asserts(gemm_ptx(tmp_path / "checked", checked)) > 0
    a = torch.randn(300, 136, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(72, 136, device="cuda", dtype=torch.bfloat16)
    outs = [torch.empty(300, 72, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    for op, out in zip((plain, checked), outs, strict=True):
        compile_gemm(op)(a, b, out)
    assert torch.equal(*outs)


def test_a_checked_compile_leaves_later_plain_builds_alone(requires_blackwell, tmp_path):
    """With line information on, the PTX names the source line of every op: a plain build after
    a checked one is the same text, so no op is attributed to the checks."""
    result = _script("ptx", str(tmp_path), CUTE_DSL_LINEINFO="1")
    assert result.returncode == 0, result.stderr
    assert "plain builds equal: True; checked differs: True" in result.stdout
    assert ".loc" in (tmp_path / "after").glob("*.ptx").__next__().read_text(errors="ignore")


def test_wrong_tile_index_aborts_the_checked_launch(requires_blackwell):
    """B's tile is taken by the M tile index, which runs past B's one tile. TMA zero-fills the
    tile, so the plain build returns a wrong product; the checked one names the index."""
    plain = _launch("tile", checked=False)
    assert plain.returncode == 0, plain.stderr
    assert float(plain.stdout.split()[-1]) > 1
    checked = _launch("tile", checked=True)
    assert checked.returncode != 0
    assert ASSERT in checked.stderr and "index of mode 1 out of range" in checked.stderr


def test_unsigned_index_past_the_signed_range_aborts(requires_blackwell):
    checked = _launch("unsigned", checked=True)
    assert checked.returncode != 0
    assert ASSERT in checked.stderr and "linear index out of range" in checked.stderr


@pytest.mark.parametrize(
    "mistake, message",
    [
        ("static", "linear index 4 outside [0, 4)"),
        ("base", "domain_offset -1 of mode 0 is before the tensor"),
    ],
)
def test_mistakes_known_at_trace_time_abort_the_launch_that_runs_them(
    requires_blackwell, mistake, message
):
    checked = _launch(mistake, checked=True)
    assert checked.returncode != 0
    assert ASSERT in checked.stderr and message in checked.stderr


def test_a_mistake_in_a_branch_not_taken_launches(requires_blackwell):
    """A kernel traces branches that a given launch never takes (an interpreter's handlers)."""
    checked = _launch("untaken", checked=True)
    assert checked.returncode == 0, checked.stderr
    assert float(checked.stdout.split()[-1]) < 1


def test_unchecked_keeps_a_deliberate_index(requires_blackwell):
    """The same index inside unchecked("bounds") launches, and describe() lists what it kept."""
    assert _launch("opted_out", checked=True).returncode == 0
    op = MistakenGemm("opted_out", checked=True)
    compile_gemm(op)
    assert "unchecked bounds @ checked_kernels.py" in op.describe()
    assert "'the wrong tile, on purpose' -> index @ checked_kernels.py" in op.describe()


def test_an_enclosing_opt_out_does_not_reach_a_kernel_compiled_inside(
    requires_blackwell, tmp_path
):
    """An opt-out is about the code inside its block, not about a kernel that happens to be
    compiled while it is open: that build would be cached without its checks."""
    op = MistakenGemm("tile", checked=True)
    alone = _asserts(gemm_ptx(tmp_path / "alone", op))
    with unchecked("bounds", reason="the caller's own index"):
        inside = _asserts(gemm_ptx(tmp_path / "inside", MistakenGemm("tile", checked=True)))
    assert inside == alone > 0
    take_opt_outs()


def test_opt_outs_stay_with_their_compile():
    with unchecked("tma", reason="before"):
        pass
    with compile_scope():
        assert take_opt_outs() == []
        with unchecked("tma", reason="inside"):
            pass
    assert [opt.reason for opt in take_opt_outs()] == ["before"]


def _fakes() -> list:
    shapes = ((512, 128), (64, 128), (512, 64))
    return [make_fake_compact_tensor(cutlass.BFloat16, shape) for shape in shapes]


def _asserts_launched_by_a_host_function(directory, checked: bool) -> int:
    op = Gemm(tile_n=64, checked=checked)

    @cute.jit
    def launch(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        op(a, b, c, stream)

    with capture_ptx(directory) as out:
        compile_tvm_ffi(launch, *_fakes(), name=f"launch_checked{int(checked)}")
    return _asserts(read_ptx(next(out.glob("*.ptx"))))


def test_a_kernel_launched_by_a_host_function_keeps_its_setting(requires_blackwell, tmp_path):
    """The entrypoint is a plain function, as in the linear bundles: its kernel's role code is
    checked or not by the kernel's own setting."""
    assert _asserts_launched_by_a_host_function(tmp_path / "plain", checked=False) == 0
    assert _asserts_launched_by_a_host_function(tmp_path / "checked", checked=True) > 0


def test_a_checked_kernel_does_not_compile_outside_compile_tvm_ffi(requires_blackwell):
    """Nothing there would take the checks out again if the trace failed."""
    with pytest.raises(Exception, match="a checked=True kernel compiles through compile_tvm_ffi"):
        cute.compile(Gemm(tile_n=64, checked=True), *_fakes(), cuda.CUstream(0))
    cute.compile(Gemm(tile_n=64), *_fakes(), cuda.CUstream(0))


def test_a_failed_checked_trace_leaves_the_library_unmodified(requires_blackwell):
    indexing_before = TENSOR_CLASS.__getitem__
    with pytest.raises(Exception, match="planted"):
        compile_gemm.__wrapped__(MistakenGemm("raises", checked=True))
    assert TENSOR_CLASS.__getitem__ is indexing_before


def test_a_launch_function_cannot_check_an_index_it_does_not_know(requires_blackwell):
    @cute.jit
    def last_row(a: cute.Tensor, stream: cuda.CUstream):
        return a[a.shape[0] - 1, 0]

    fake = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(), 128))
    with indexing(True), pytest.raises(IndexError, match="a launch function cannot check it"):
        cute.compile(last_row, fake, cuda.CUstream(0))


def test_view_before_its_tensor_fails_the_compile_of_a_launch_function(
    requires_blackwell, monkeypatch
):
    """The grouped GEMM builds its ragged output view in ``__call__``, which always runs and
    cannot assert; without its opt-out the checked build does not compile."""
    monkeypatch.setattr(grouped_gemm, "unchecked", lambda *_, **__: contextlib.nullcontext())
    compile_grouped_gemm.__wrapped__(GroupedGemm())
    with pytest.raises(IndexError, match=r"grouped_gemm\.py:\d+: domain_offset -256 of mode 0"):
        compile_grouped_gemm.__wrapped__(GroupedGemm(checked=True))


def test_clipped_tiles_and_shifted_views_are_opted_out(requires_blackwell):
    """The 2x2 cluster GEMM indexes MMA tiles past a ragged edge, which TMA clips, and the
    grouped GEMM's ragged output view starts before the output: both say so and run checked."""
    a = torch.randn(300, 136, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(520, 136, device="cuda", dtype=torch.bfloat16)
    kwargs = {"cluster": Cluster(2, 2, pair=True), "tile_n": 224}
    checked = persistent_gemm(a, b, checked=True, **kwargs)
    assert torch.equal(checked, persistent_gemm(a, b, **kwargs))
    op = GroupedGemm(checked=True)
    compile_grouped_gemm(op)
    assert "domain_offset @ grouped_gemm.py" in op.describe()
