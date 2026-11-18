"""``kernel_options={"backend": "cudnn"}`` is a supported alias of ``"warpdrive"``."""

import warnings

import pytest
import torch
import torch.nn.functional as F

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.linear._delta_rule.validation import resolve_chunk_kernel_options
from attn_gym.linear.kda.validation import resolve_kernel_options
from attn_gym.linear.types import GDN_STAGE_OPTION_SPEC, SPLIT_OPTION_SPEC

ALIAS = {"backend": "cudnn"}
RESOLVERS = {
    "chunk_kda": lambda options: resolve_kernel_options(options)._asdict(),
    "chunk_kda_prepare": lambda options: resolve_kernel_options(options, staged=True)._asdict(),
    "chunk_kda_prepare_backward": lambda options: resolve_kernel_options(
        options, staged=True, tape_backend="warpdrive"
    )._asdict(),
    "chunk_gdn": lambda options: resolve_chunk_kernel_options(
        options, SPLIT_OPTION_SPEC, op_name="chunk_gdn"
    ),
    "chunk_gdn_prepare": lambda options: resolve_chunk_kernel_options(
        options, GDN_STAGE_OPTION_SPEC, op_name="chunk_gdn_prepare"
    ),
}


@pytest.mark.parametrize(
    "extra",
    [{}, {"split_forward": True}, {"split_backward": True}, {"batch_invariant": True}],
    ids=["plain", "split_forward", "split_backward", "batch_invariant"],
)
@pytest.mark.parametrize("op_name", RESOLVERS)
def test_cudnn_backend_resolves_like_warpdrive_without_warning(op_name, extra):
    """Either spelling resolves to the same options, including the warpdrive-only ones."""
    if "batch_invariant" in extra and "prepare" not in op_name:
        pytest.skip("only the staged handles take batch_invariant")
    resolve = RESOLVERS[op_name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve({**ALIAS, **extra})
    assert resolved == resolve({"backend": "warpdrive", **extra})
    assert resolved["backend"] == "warpdrive"


def test_cudnn_backend_dispatches_to_warpdrive_on_cpu():
    """The alias reaches the warpdrive backend's own validation, not the fused path."""
    q = torch.randn(1, 4, 1, 128)
    k = F.normalize(torch.randn_like(q), dim=-1)
    gate = F.logsigmoid(torch.randn(1, 4, 1))
    beta = torch.sigmoid(torch.randn(1, 4, 1))
    with pytest.raises(ValueError, match="warpdrive GDN backend requires CUDA"):
        chunk_gdn(q, k, q, gate, beta, kernel_options=ALIAS)
    with pytest.raises(ValueError, match="warpdrive KDA backend requires CUDA"):
        chunk_kda(q, k, q, gate[..., None].expand_as(q), beta, kernel_options=ALIAS)


def test_cudnn_backend_compiles_fullgraph():
    """The alias resolves inside a full graph exactly as outside it."""

    split = {"split_forward": True}

    def uses_alias(x):
        options = resolve_kernel_options({**ALIAS, **split})
        return x + (options == resolve_kernel_options({"backend": "warpdrive", **split}))

    torch._dynamo.reset()
    compiled = torch.compile(uses_alias, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))


def test_cudnn_staged_type_names_are_aliases():
    pytest.importorskip("triton")
    from attn_gym.linear.kda import ChunkKDACudnnSaved, ChunkKDAWarpdriveSaved, stages

    assert ChunkKDACudnnSaved is ChunkKDAWarpdriveSaved
    assert stages.ChunkKDACudnnPrepared is stages.ChunkKDAWarpdrivePrepared
    assert stages.ChunkKDACudnnBackward is stages.ChunkKDAWarpdriveBackward
    assert {"ChunkKDACudnnPrepared", "ChunkKDACudnnBackward"} <= set(stages.__all__)


@pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive backend requires an SM100-family GPU",
)
def test_cudnn_backend_matches_warpdrive_bitwise():
    pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")
    from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

    q, k, v, gate, beta = make_kda_test_inputs(
        96, gate_scale=0.5, sigmoid_beta=True, normalize_qk=True
    )
    cu_seqlens = cumulative_sequence_offsets((33, 63))
    scalar_gate = gate[..., 0].contiguous()
    for op, op_gate in ((chunk_kda, gate), (chunk_gdn, scalar_gate)):
        expected, _ = op(
            q, k, v, op_gate, beta, cu_seqlens=cu_seqlens, kernel_options={"backend": "warpdrive"}
        )
        actual, _ = op(q, k, v, op_gate, beta, cu_seqlens=cu_seqlens, kernel_options=ALIAS)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
