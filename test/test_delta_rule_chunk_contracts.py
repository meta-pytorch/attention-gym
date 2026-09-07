"""Shared chunk ownership, compatibility, and torch-only operator metadata contracts."""

import ast
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear._delta_rule import chunk_ops, chunk_schedule
from attn_gym.linear.kda import chunk_schedule as legacy_schedule
from attn_gym.linear.kda import ops as kda_ops


@pytest.mark.parametrize(
    "name",
    [
        "RaggedChunkMetadata",
        "ResolvedSchedule",
        "ScheduleKind",
        "ScheduleRequest",
        "chunk_capacity",
        "prepare_ragged_chunk_metadata",
        "validate_schedule_request",
    ],
)
def test_kda_schedule_exports_share_one_owner(name: str) -> None:
    """Legacy imports preserve enum, metadata type, and callable identity."""
    shared = getattr(chunk_schedule, name)
    assert getattr(legacy_schedule, name) is shared
    assert shared.__module__ == chunk_schedule.__name__


def test_chunk_ops_preserve_registered_graph_targets() -> None:
    """Moving contracts must not create alternate schemas or operator boundaries."""
    assert kda_ops.prepare_chunk_offsets_op is chunk_ops.prepare_chunk_offsets_op
    assert kda_ops._plain_gate_scan_op is chunk_ops._plain_gate_scan_op
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
