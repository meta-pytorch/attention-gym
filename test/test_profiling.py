"""Portable coverage for optional example instrumentation, without CUDA or CuTeDSL."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from attn_gym.testing import annotate_kernels, profile_trace, profiling, record_function


class AnnotatedModule(torch.nn.Module):
    """A non-delta-rule module exercising the shared method decorator."""

    def __init__(self, enabled: bool):
        super().__init__()
        self.enable_graph_annotations = enabled
        self.label = "example"

    @annotate_kernels("{module.label}/forward")
    def forward(self, value: torch.Tensor, *, offset: float = 1.0) -> torch.Tensor:
        """Keep keyword arguments and autograd intact under the decorator."""
        return value.square() + offset


def unexpected_optional_dependency(*args, **kwargs):
    """Fail if disabled instrumentation probes or invokes an optional dependency."""
    pytest.fail("disabled instrumentation accessed an optional dependency")


@pytest.mark.parametrize("compiled", [False, True])
def test_disabled_annotations_preserve_fullgraph_autograd(monkeypatch, compiled):
    monkeypatch.setattr(profiling, "mark_kernels", unexpected_optional_dependency)
    module = AnnotatedModule(False)
    # Disabled decoration must not even try to format its label.
    del module.label
    call = torch.compile(module, backend="eager", fullgraph=True) if compiled else module
    value = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    result = call(value, offset=2.0)
    (grad,) = torch.autograd.grad(result.sum(), value)
    torch.testing.assert_close(result, value.square() + 2.0)
    torch.testing.assert_close(grad, 2.0 * value)


def test_enabled_annotations_preserve_labels_and_autograd(monkeypatch):
    labels = []

    @contextmanager
    def mark(name):
        labels.append(name)
        yield

    monkeypatch.setattr(profiling, "mark_kernels", mark)
    value = torch.tensor([-1.5, 2.0], requires_grad=True)
    result = AnnotatedModule(True)(value, offset=3.0)
    (grad,) = torch.autograd.grad(result.sum(), value)
    assert labels == ["example/forward"]
    assert AnnotatedModule.forward.__name__ == "forward"
    torch.testing.assert_close(result, value.square() + 3.0)
    torch.testing.assert_close(grad, 2.0 * value)


def test_disabled_range_does_not_enter_profiler(monkeypatch):
    monkeypatch.setattr(torch.profiler, "record_function", unexpected_optional_dependency)
    with record_function(False, "ignored"):
        pass


def test_record_function_labels_cpu_trace():
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as active,
        record_function(True, "example/forward"),
    ):
        torch.arange(8).square()
    assert "example/forward" in [event.key for event in active.key_averages()]


def test_native_trace_reports_missing_dependency(monkeypatch, tmp_path):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(profiling, "import_module", missing)
    with (
        pytest.raises(RuntimeError, match="requires transformer-nuggets"),
        profile_trace(tmp_path / "missing.pftrace"),
    ):
        pytest.fail("missing profiler must not silently fall back to another format")


def test_native_trace_uses_pftrace_and_forwards_warmup(monkeypatch, tmp_path):
    calls = []
    token = object()

    @contextmanager
    def profiler(path, *, record_shapes, trace_format, warmup):
        calls.append((path, record_shapes, trace_format, warmup))
        yield token

    monkeypatch.setattr(
        profiling, "import_module", lambda name: SimpleNamespace(profiler=profiler)
    )
    trace = tmp_path / "profiles" / "training"
    with profile_trace(trace, warmup=2) as active:
        assert active is token
    assert trace.parent.is_dir()
    assert calls == [(trace.with_suffix(".pftrace"), True, "track_event", 2)]


def test_native_trace_rejects_outdated_profiler(monkeypatch, tmp_path):
    def legacy_profiler(path, record_shapes=True):
        pytest.fail("unsupported profiler must be rejected before entering it")

    monkeypatch.setattr(
        profiling, "import_module", lambda name: SimpleNamespace(profiler=legacy_profiler)
    )
    with (
        pytest.raises(RuntimeError, match="lacks native Perfetto support"),
        profile_trace(tmp_path / "outdated.pftrace"),
    ):
        pytest.fail("outdated profiler must not silently fall back to Chrome JSON")


@pytest.mark.parametrize("available", [None, False, True])
def test_graph_annotation_capability_is_optional(monkeypatch, available):
    module = None if available is None else SimpleNamespace(is_available=lambda: available)
    monkeypatch.setitem(sys.modules, "torch.cuda.graph_annotations", module)
    assert profiling.graph_annotations_available() is bool(available)


def test_distributed_native_merge_preserves_timestamps(monkeypatch, tmp_path):
    path = tmp_path / "trace"
    path.with_name("trace_rank_0.pftrace").write_bytes(b"rank0")
    merge = Mock()
    monkeypatch.setattr(
        profiling, "import_module", lambda name: SimpleNamespace(merge_traces=merge)
    )
    monkeypatch.setattr(profiling, "profile_trace", lambda *args, **kwargs: nullcontext(Mock()))
    monkeypatch.setattr(profiling.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(profiling.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(profiling.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def gather(value, outputs, dst):
        assert value == b"rank0" and dst == 0
        outputs[:] = [b"rank0", b"rank1"]

    monkeypatch.setattr(profiling.dist, "gather_object", gather)
    merged = profiling.record_distributed_profile(Mock(), path, "step", torch.device("cpu"))
    assert merged == path.with_name("trace_merged.pftrace")
    assert path.with_name("trace_rank_1.pftrace").read_bytes() == b"rank1"
    merge.assert_called_once()
    # Native merge rejects JSON's re-zeroing option; retain the native clock timestamps.
    assert merge.call_args.kwargs.get("align_timestamps", False) is False


def test_kernel_stage_forwards_annotation_direction(monkeypatch):
    labels = []

    @contextmanager
    def mark(name, *, backward):
        labels.append((name, backward))
        yield

    monkeypatch.setattr(profiling, "mark_kernels", mark)
    with profiling.kernel_stage("backward", True, backward=False):
        pass
    with profiling.kernel_stage("unannotated", False):
        pass
    assert labels == [("backward", False)]
