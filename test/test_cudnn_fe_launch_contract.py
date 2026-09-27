"""Invalid common-host metadata must fail before any GPU compilation or launch."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.linear._delta_rule.cudnn_fe.common import (
    gate_bwd,
    head_reduce,
    l2norm,
    piece_chain,
    split_k,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")


def unexpected_compile(*args, **kwargs):
    raise AssertionError("invalid metadata reached the compiler")


def test_head_reduction_rejects_nondividing_heads(monkeypatch):
    monkeypatch.setattr(head_reduce, "_compile_head_reduce", unexpected_compile)
    source = torch.empty(8, 4, 64, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(8, 3, 64, dtype=source.dtype, device="cuda")
    with pytest.raises(ValueError, match="head count must divide"):
        head_reduce.head_group_reduce(source, output, stream=0)


def test_l2norm_rejects_partial_vector_rows(monkeypatch):
    monkeypatch.setattr(l2norm, "_compile_l2norm_qk", unexpected_compile)
    source = torch.empty(8, 2, 63, dtype=torch.bfloat16, device="cuda")
    norm = torch.empty(8, 2, device="cuda")
    with pytest.raises(ValueError, match="head dimension must be 64 or 128"):
        l2norm.build_l2norm_qk(source, source, source, source, norm, norm, stream=0)


def test_channel_gate_rejects_short_partial_workspace(monkeypatch):
    monkeypatch.setattr(gate_bwd, "_compile_gate_bwd", unexpected_compile)
    source = torch.empty(8, 2, 64, device="cuda")
    parameter = torch.empty(2, device="cuda")
    partial = torch.empty(1, device="cuda")
    with pytest.raises(ValueError, match="part_a requires at least"):
        gate_bwd.channel_gate_bwd(
            source, source, parameter, None, parameter, None, partial, None, -1.0, stream=0
        )


def test_state_chain_rejects_empty_row_tiles(monkeypatch):
    monkeypatch.setattr(piece_chain, "_compile_state_chain", unexpected_compile)
    with pytest.raises(ValueError, match="rows_per_cta"):
        piece_chain.build_state_chain(
            heads_out=2,
            dim_v=64,
            dim_k=64,
            pieces=2,
            rows_per_cta=0,
            transpose=False,
            has_seed=False,
            has_tail=True,
            emit_summary=False,
            device=0,
            opt_level=2,
        )


@pytest.mark.parametrize("invalid", ["tiles", "chunks"])
def test_split_table_rejects_invalid_launch_geometry(monkeypatch, invalid):
    monkeypatch.setattr(split_k, "_compile_split_table", unexpected_compile)
    gate = torch.empty(32, 2, device="cuda")
    cu = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    items = torch.empty(64, 10, dtype=torch.int32, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    rows = 1 if invalid == "chunks" else split_k.chunk_scratch_rows(32, 1, 16)
    chunks = torch.empty(rows, 2, device="cuda")
    message = "n_tiles must equal" if invalid == "tiles" else "chunk_scratch requires at least"
    with pytest.raises(ValueError, match=message):
        split_k.build_split_table(
            gate,
            cu,
            items,
            count,
            ideal_chunks=32,
            n_tiles=3 if invalid == "tiles" else 2,
            num_sms=148,
            b_t=16,
            chunk_scratch=chunks,
            item_scratch=items,
            scheduler_counter=None,
            split=True,
            opt_level=2,
            stream=0,
        )
