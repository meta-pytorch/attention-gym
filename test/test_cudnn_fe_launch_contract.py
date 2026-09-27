"""Invalid common-host metadata must fail before any GPU compilation or launch."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.linear._delta_rule.cudnn_fe import gdn, kda, plan
from attn_gym.linear._delta_rule.cudnn_fe.common import split_k
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
    kda_chain_forward_f16,
    kda_prefill_f16,
    kda_prep_f16,
    kda_prep_prefill_f16,
    kda_summary_f16,
    kda_warmup_forward_f16,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")


def reached_compiler(*args, **kwargs):
    raise RuntimeError("reached the compiler")


@pytest.mark.parametrize("invalid", ["tiles", "chunks"])
def test_split_table_rejects_invalid_launch_geometry(monkeypatch, invalid):
    monkeypatch.setattr(split_k, "_compile_split_table", reached_compiler)
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


def workspace(module):
    return plan.workspace(module, 1, "cuda")


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


# ---- KDA forward kernels (S12) ------------------------------------------------------------


def kda_cfg_args(module, io, dim, clusters):
    import cutlass

    flags = {
        "l2norm": True,
        "safe_gate": False,
        "gate_scale_log2": 0.0,
        "beta_sigmoid": False,
        "allow_neg_eigval": False,
    }
    if module is kda_prep_f16:
        return (io, cutlass.Float32), {**flags, "num_sm": clusters, "d_k": dim}
    sizes = {"max_active_clusters": clusters, "d_k": dim, "d_v": dim}
    if module is kda_summary_f16:
        return (io, cutlass.Float32), {**flags, **sizes, "use_initial_state": True}
    states = {"use_initial_state": True, "store_final_state": True, "enable_checkpoints": False}
    return (io, cutlass.Float32, cutlass.Float32), {**flags, **sizes, **states}


@pytest.mark.parametrize(
    "module", [kda_prefill_f16, kda_prep_prefill_f16, kda_summary_f16, kda_prep_f16]
)
@pytest.mark.parametrize("invalid", [None, "d_k", "dtype", "clusters"])
def test_kda_forward_cfgs_reject_unsupported_geometry(module, invalid):
    import cutlass

    io = cutlass.Float32 if invalid == "dtype" else cutlass.BFloat16
    dim = 96 if invalid == "d_k" else DIM
    args, kwargs = kda_cfg_args(module, io, dim, 0 if invalid == "clusters" else 148)
    if invalid is None:
        cfg = module.build_cfg(*args, **kwargs)
        assert cfg.threads_per_cta == (128 if module is kda_prep_f16 else 512)
        return
    with pytest.raises(ValueError):
        module.build_cfg(*args, **kwargs)


def run_kda_forward(tokens):
    q, k, v = (packed(tokens, HEADS, DIM) for _ in range(3))
    gate = packed(tokens, HEADS, DIM, dtype=torch.float32)
    beta = packed(tokens, HEADS, dtype=torch.float32)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    kda.kda_forward(q, k, v, gate, beta, cu, scale=DIM**-0.5)


@pytest.mark.parametrize("invalid", [None, "n_tiles", "interval", "work_rows", "workspace"])
def test_kda_warmup_forward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real uncut plan passes; corrupting one of its buffers fails before compiling."""
    monkeypatch.setattr(kda_warmup_forward_f16, "_compile_warmup_forward", reached_compiler)
    monkeypatch.setattr(kda.ForwardPlan, "chain", property(lambda self: False))
    build = kda.build_warmup_forward

    def corrupt(**kwargs):
        if invalid == "n_tiles":
            kwargs["n_tiles"] += 1
        elif invalid == "interval":
            kwargs["checkpoint_every_n_tokens"] = 8
        elif invalid == "work_rows":
            kwargs["work_items"] = kwargs["work_items"][:1]
        elif invalid == "workspace":
            kwargs["workspace"] = kwargs["workspace"][:8]
        return build(**kwargs)

    monkeypatch.setattr(kda, "build_warmup_forward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_kda_forward(TOKENS)


@pytest.mark.parametrize("invalid", [None, "chain_rows", "state_rows"])
def test_kda_chain_forward_rejects_invalid_plan_buffers(monkeypatch, invalid):
    """The real chain plan passes; corrupting one of its buffers fails before compiling."""
    monkeypatch.setattr(kda_chain_forward_f16, "_compile_chain_forward", reached_compiler)
    monkeypatch.setattr(kda, "MIN_CHAIN_TOKENS_PER_PIECE_FWD", 0)
    build = kda.build_chain_forward

    def corrupt(**kwargs):
        assert kwargs["pieces"] > 1
        if invalid == "chain_rows":
            kwargs["chain_rows"] = 3
        elif invalid == "state_rows":
            kwargs["state_m"] = kwargs["state_m"][:1]
        return build(**kwargs)

    monkeypatch.setattr(kda, "build_chain_forward", corrupt)
    with pytest.raises(RuntimeError if invalid is None else ValueError):
        run_kda_forward(16384)
