"""Invalid common-host metadata must fail before any GPU compilation or launch."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.linear._delta_rule.cudnn_fe import gdn
from attn_gym.linear._delta_rule.cudnn_fe.common import (
    gate_bwd,
    head_reduce,
    l2norm,
    piece_chain,
    split_k,
)
from attn_gym.linear._delta_rule.cudnn_fe.common.host import tensormap_workspace_bytes
from attn_gym.linear._delta_rule.cudnn_fe.common.launch import (
    validate_named_barriers,
    validate_warp_roles,
)
from attn_gym.linear._delta_rule.cudnn_fe.kernel import (
    gdn_bprop_f16,
    gdn_bprop_summary_f16,
    gdn_chain_backward_f16,
    gdn_recompute_f16,
    gdn_warmup_backward_f16,
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


# ---- GDN backward kernels (S12) -----------------------------------------------------------

TOKENS, HEADS, DIM, B_T = 128, 2, 128, 64


def reached_compiler(*args, **kwargs):
    raise RuntimeError("reached the compiler")


def test_warp_role_and_named_barrier_tables_reject_conflicts():
    groups = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    assert validate_warp_roles(groups, (12, 13, 14, 15)) == 16
    with pytest.raises(ValueError, match="disjoint"):
        validate_warp_roles(groups, (12, 13, 14, 11))
    with pytest.raises(ValueError, match="aligned four-warp"):
        validate_warp_roles(((1, 2, 3, 4),), (0, 5))
    with pytest.raises(ValueError, match="distinct"):
        validate_named_barriers(512, a=(1, 128), b=(1, 128))
    with pytest.raises(ValueError, match="whole-warp"):
        validate_named_barriers(512, a=(2, 48))


@pytest.mark.parametrize("module", [gdn_bprop_f16, gdn_recompute_f16, gdn_bprop_summary_f16])
@pytest.mark.parametrize("invalid", ["d_k", "dtype", "clusters"])
def test_gdn_backward_cfgs_reject_unsupported_geometry(module, invalid):
    import cutlass

    dim = 96 if invalid == "d_k" else DIM
    io = cutlass.Float32 if invalid == "dtype" else cutlass.BFloat16
    kwargs = {"max_active_clusters": 0 if invalid == "clusters" else 148, "d_k": dim, "d_v": dim}
    args = (io, cutlass.Float32) if module is gdn_recompute_f16 else (io,)
    if module is gdn_recompute_f16:
        kwargs["use_initial_state"] = True
    with pytest.raises(ValueError):
        module.build_cfg(*args, **kwargs)


def workspace(module, batch=1):
    return torch.empty(
        tensormap_workspace_bytes(module, batch) // 8, dtype=torch.int64, device="cuda"
    )


def misaligned(tensor):
    storage = torch.empty(tensor.numel() + 8, dtype=tensor.dtype, device=tensor.device)
    return storage[1 : 1 + tensor.numel()].view_as(tensor)


def packed(*shape, dtype=torch.bfloat16):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def work_table():
    return {
        "work_items": packed(HEADS, 10, dtype=torch.int32),
        "work_count": packed(1, dtype=torch.int32),
        "scheduler_counter": packed(2, dtype=torch.int32),
    }


def bprop_args():
    qkv = [packed(TOKENS, HEADS, DIM) for _ in range(7)]
    scalars = [packed(TOKENS, HEADS, dtype=torch.float32) for _ in range(4)]
    q, k, v, do, dq, dk, dv = qkv
    gate, beta, dgate, dbeta = scalars
    checkpoints = packed(TOKENS // B_T + 1, HEADS, DIM, DIM)
    cu = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    positional = {
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "do": do,
        "state_checkpoints": checkpoints,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dgate": dgate,
        "dbeta": dbeta,
        "cu_seqlens": cu,
    }
    keywords = {"workspace": workspace(gdn_bprop_f16), "device": 0, "num_sm": 148, "stream": 0}
    return positional, {**work_table(), **keywords}


@pytest.mark.parametrize("invalid", [None, "checkpoint_rows", "workspace", "work_fields", "dq"])
def test_gdn_bprop_host_rejects_invalid_launch_metadata(monkeypatch, invalid):
    monkeypatch.setattr(gdn_bprop_f16, "_compile_gdn_bprop", reached_compiler)
    positional, keywords = bprop_args()
    if invalid == "checkpoint_rows":
        positional["state_checkpoints"] = positional["state_checkpoints"][:1]
    elif invalid == "workspace":
        keywords["workspace"] = keywords["workspace"][:8]
    elif invalid == "work_fields":
        keywords["work_items"] = packed(HEADS, 8, dtype=torch.int32)
    elif invalid == "dq":
        positional["dq"] = misaligned(positional["dq"])
    error = RuntimeError if invalid is None else ValueError
    with pytest.raises(error):
        gdn_bprop_f16.chunk_gdn_bwd(*positional.values(), DIM**-0.5, **keywords)


@pytest.mark.parametrize("invalid", [None, "interval", "checkpoint_rows", "tinv_rows"])
def test_gdn_recompute_host_rejects_invalid_launch_metadata(monkeypatch, invalid):
    monkeypatch.setattr(gdn_recompute_f16, "_compile_gdn_recompute", reached_compiler)
    k, v = packed(TOKENS, HEADS, DIM), packed(TOKENS, HEADS, DIM)
    gate = packed(TOKENS, HEADS, dtype=torch.float32)
    cu = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    output_state = packed(1, HEADS, DIM, DIM, dtype=torch.float32)
    checkpoints = packed(TOKENS // B_T + 1, HEADS, DIM, DIM)
    tinv = packed(TOKENS // B_T + 1, HEADS, B_T, B_T)
    every_n = B_T // 2 if invalid == "interval" else B_T
    if invalid == "checkpoint_rows":
        checkpoints = checkpoints[:1]
    if invalid == "tinv_rows":
        tinv = tinv[:1]
    error = RuntimeError if invalid is None else ValueError
    with pytest.raises(error):
        gdn_recompute_f16.chunk_gdn_recompute(
            k,
            v,
            gate,
            cu,
            None,
            output_state,
            every_n,
            checkpoints,
            **work_table(),
            tinv=tinv,
            workspace=workspace(gdn_recompute_f16),
            device=0,
            num_sm=148,
            stream=0,
        )


@pytest.mark.parametrize("invalid", [None, "state_shape"])
def test_gdn_bprop_summary_host_rejects_invalid_launch_metadata(monkeypatch, invalid):
    monkeypatch.setattr(gdn_bprop_summary_f16, "_compile_gdn_bprop_summary", reached_compiler)
    q, k, do = (packed(TOKENS, HEADS, DIM) for _ in range(3))
    gate = packed(TOKENS, HEADS, dtype=torch.float32)
    width = DIM // 2 if invalid == "state_shape" else DIM
    d_initial_state = packed(1, HEADS, DIM, width, dtype=torch.float32)
    cu = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
    tinv = packed(TOKENS // B_T + 1, HEADS, B_T, B_T)
    error = RuntimeError if invalid is None else ValueError
    with pytest.raises(error):
        gdn_bprop_summary_f16.chunk_gdn_bwd_summary(
            q,
            k,
            gate,
            do,
            d_initial_state,
            cu,
            DIM**-0.5,
            **work_table(),
            tinv=tinv,
            workspace=workspace(gdn_bprop_summary_f16),
            device=0,
            num_sm=148,
            stream=0,
        )


def run_gdn_backward(tokens):
    q, k, v, do = (packed(tokens, HEADS, DIM) for _ in range(4))
    gate = packed(tokens, HEADS, dtype=torch.float32)
    beta = packed(tokens, HEADS, dtype=torch.float32)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    gdn.gdn_backward(q, k, v, gate, beta, do, cu, scale=DIM**-0.5)


@pytest.mark.parametrize("invalid", [None, "n_tiles", "checkpoint_rows"])
def test_gdn_warmup_backward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real uncut plan passes; corrupting one of its buffers fails before compiling."""
    monkeypatch.setattr(gdn_warmup_backward_f16, "_compile_warmup_backward", reached_compiler)
    monkeypatch.setattr(gdn.BackwardPlan, "chain", property(lambda self: False))
    build = gdn.build_warmup_backward

    def corrupt(**kwargs):
        if invalid == "n_tiles":
            kwargs["n_tiles"] += 1
        elif invalid == "checkpoint_rows":
            kwargs["checkpoints"] = kwargs["checkpoints"][:1]
        return build(**kwargs)

    monkeypatch.setattr(gdn, "build_warmup_backward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_gdn_backward(TOKENS)


@pytest.mark.parametrize("invalid", [None, "chain_rows", "summary_items"])
def test_gdn_chain_backward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real chain plan passes; corrupting one of its buffers fails before compiling."""
    monkeypatch.setattr(gdn_chain_backward_f16, "_compile_chain_backward_head", reached_compiler)
    monkeypatch.setattr(gdn, "MIN_CHAIN_TOKENS_PER_PIECE_BWD", 0)
    build = gdn.build_chain_backward

    def corrupt(**kwargs):
        assert kwargs["pieces"] > 1
        if invalid == "chain_rows":
            kwargs["chain_rows"] = 3
        elif invalid == "summary_items":
            kwargs["work_items_summary"] = kwargs["work_items_summary"][:1]
        return build(**kwargs)

    monkeypatch.setattr(gdn, "build_chain_backward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_gdn_backward(16384)
