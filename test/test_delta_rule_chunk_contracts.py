"""Shared chunk ownership and torch-only operator metadata contracts."""

import ast
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear._delta_rule import chunk_ops, chunk_schedule


def test_chunk_ops_preserve_registered_graph_targets() -> None:
    """Moving contracts must not create alternate schemas or operator boundaries."""
    assert (
        chunk_ops.prepare_chunk_offsets_op is torch.ops.attn_gym.kda_prepare_chunk_offsets.default
    )
    assert chunk_ops._plain_gate_scan_op is torch.ops.attn_gym._kda_plain_gate_scan.default


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_shared_gate_scan_fake_preserves_compact_output(reverse: bool, packed: bool) -> None:
    """Both routes register a compact output for a noncompact CUDA input without Triton."""
    with FakeTensorMode():
        values = torch.empty(1, 2, 65, 3, device="cuda").transpose(1, 2)
        offsets = torch.empty(4, dtype=torch.int32, device="cuda") if packed else None
        output = chunk_ops._plain_gate_scan_op(values, offsets, offsets, reverse)
        assert output.shape == values.shape
        assert output.dtype == values.dtype
        assert output.device == values.device
        assert output.is_contiguous()
        assert output is not values


@pytest.mark.parametrize("tokens", [0, 65, 256])
def test_shared_metadata_builder_fake(tokens: int) -> None:
    """The production metadata builder can trace without importing a GPU DSL."""
    with FakeTensorMode():
        offsets = torch.empty(4, dtype=torch.int32, device="cuda")
        metadata = chunk_schedule.prepare_ragged_chunk_metadata(offsets, tokens, 64)
        assert metadata.cu_seqlens is offsets
        assert metadata.chunk_offsets.shape == offsets.shape
        assert metadata.chunk_offsets.dtype == offsets.dtype
        assert metadata.chunk_offsets.device == offsets.device
        assert metadata.chunk_offsets.is_contiguous()
        assert metadata.chunk_offsets is not offsets
        assert metadata.capacity == chunk_schedule.chunk_capacity(tokens, 3, 64)
        assert metadata.chunk_size == 64


@pytest.mark.parametrize(
    "module_path",
    [
        "chunk_ops.py",
        "chunk_schedule.py",
        "constants.py",
        "span.py",
        "triton/chunk_scheduler.py",
        "triton/plain_gate.py",
    ],
)
def test_shared_chunk_infrastructure_does_not_import_variants(module_path: str) -> None:
    """Keep the extracted shared layer independent of KDA/GDN implementation ownership."""
    shared_root = Path(chunk_schedule.__file__).parent
    tree = ast.parse((shared_root / module_path).read_text())
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
        ):
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                modules.append(argument.value)
        for module in modules:
            assert not module.startswith(("attn_gym.linear.kda", "attn_gym.linear.gdn")), (
                f"{module_path}:{node.lineno} imports variant-owned {module}"
            )


requires_warpdrive = pytest.mark.skipif(
    not cuda_runs_sm100_kernels() or importlib.util.find_spec("cutlass") is None,
    reason="the warpdrive backward requires an SM100-family GPU and CuTeDSL",
)


@requires_warpdrive
@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("plan", ["uncut", "chain"])
@pytest.mark.parametrize(("family", "heads"), [("kda", 3), ("kda", 40), ("gdn", 3)])
def test_warpdrive_backward_zeroes_only_the_dgate_tail(monkeypatch, family, heads, plan):
    """The backward prologue zeroes ``dgate`` from ``cu_seqlens[-1]`` on. KDA rows of 3 and 40
    heads are narrower and wider than one prologue CTA; GDN rows hold one value per head."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda
    from attn_gym.linear._delta_rule.warpdrive.launchers.backward import (
        chunk_delta_rule_bwd_warpdrive_packed,
    )
    from attn_gym.linear._delta_rule.warpdrive.launchers.gdn_backward import (
        chunk_gdn_bwd_warpdrive_packed,
    )

    driver, backward = (
        (kda, chunk_delta_rule_bwd_warpdrive_packed)
        if family == "kda"
        else (gdn, chunk_gdn_bwd_warpdrive_packed)
    )
    build = driver.BackwardPlan.build
    pieces = 4 if plan == "chain" else 0
    monkeypatch.setattr(
        driver.BackwardPlan, "build", lambda *args: replace(build(*args), pieces=pieces)
    )
    cases = ((256, (191, 256, 255, 0)),)
    if plan == "uncut":
        cases = ((1, (0, 1)), (16, (16, 3)), (65, (64, 0)), *cases)
    generator = torch.Generator(device="cuda").manual_seed(0)
    for tokens, ends in cases:
        q, k, v = (
            torch.nn.functional.normalize(
                torch.randn(1, tokens, heads, 128, device="cuda", generator=generator), dim=-1
            ).bfloat16()
            for _ in range(3)
        )
        gate_shape = q.shape if family == "kda" else q.shape[:3]
        gate = -torch.rand(gate_shape, device="cuda", generator=generator)
        beta = torch.rand(q.shape[:3], device="cuda", generator=generator)
        for end in ends:
            cu_seqlens = torch.tensor([0, end // 2, end], dtype=torch.int32, device="cuda")
            dgate = backward(q, k, v, gate, beta, torch.ones_like(v), cu_seqlens)[3]
            assert dgate.shape == gate.shape
            assert not dgate[:, end:].any(), (tokens, end)
            assert dgate[:, :end].isfinite().all(), (tokens, end)
