"""``kernel_options={"backend": "cudnn"}`` stays a deprecated alias of ``"warpdrive"``."""

import json
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
import torch.nn.functional as F

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda

DEPRECATED = {"backend": "cudnn"}


def test_cudnn_backend_warns_once_per_process():
    """Every chunk op resolves ``"cudnn"`` to ``"warpdrive"``; only the first call warns."""
    probe = textwrap.dedent(
        """
        import json
        import warnings

        from attn_gym.linear._delta_rule.validation import resolve_chunk_kernel_options
        from attn_gym.linear.kda.validation import resolve_kernel_options
        from attn_gym.linear.types import SPLIT_OPTION_SPEC

        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as caught:
            backends = [
                resolve_kernel_options({"backend": "cudnn"}).backend,
                resolve_kernel_options({"backend": "cudnn", "split_forward": True}).backend,
                resolve_chunk_kernel_options(
                    {"backend": "cudnn"}, SPLIT_OPTION_SPEC, op_name="chunk_gdn"
                )["backend"],
            ]
        print(json.dumps([backends, [[w.category.__name__, str(w.message)] for w in caught]]))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    )
    backends, caught = json.loads(result.stdout)
    assert backends == ["warpdrive"] * 3
    assert len(caught) == 1
    category, message = caught[0]
    assert category == "FutureWarning" and "'warpdrive'" in message


def test_cudnn_backend_dispatches_to_warpdrive_on_cpu():
    """The alias reaches the warpdrive backend's own validation, not the fused path."""
    q = torch.randn(1, 4, 1, 128)
    k = F.normalize(torch.randn_like(q), dim=-1)
    gate = F.logsigmoid(torch.randn(1, 4, 1))
    beta = torch.sigmoid(torch.randn(1, 4, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(ValueError, match="warpdrive GDN backend requires CUDA"):
            chunk_gdn(q, k, q, gate, beta, kernel_options=DEPRECATED)
        with pytest.raises(ValueError, match="warpdrive KDA backend requires CUDA"):
            chunk_kda(q, k, q, gate[..., None].expand_as(q), beta, kernel_options=DEPRECATED)


def test_cudnn_backend_compiles_fullgraph():
    """Dynamo cannot trace ``warnings.warn``: the alias must resolve without warning there."""
    from attn_gym.linear.kda.validation import resolve_kernel_options

    def uses_alias(x):
        return x + (resolve_kernel_options(DEPRECATED).backend == "warpdrive")

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for op, op_gate in ((chunk_kda, gate), (chunk_gdn, scalar_gate)):
            expected, _ = op(
                q,
                k,
                v,
                op_gate,
                beta,
                cu_seqlens=cu_seqlens,
                kernel_options={"backend": "warpdrive"},
            )
            actual, _ = op(
                q, k, v, op_gate, beta, cu_seqlens=cu_seqlens, kernel_options=DEPRECATED
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
