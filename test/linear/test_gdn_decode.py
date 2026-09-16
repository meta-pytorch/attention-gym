"""Correctness, registration, and capture tests for fused raw-ABI GDN decode."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_gdn, recurrent_gdn_decode
from attn_gym.linear._delta_rule import decode as decode_module
from attn_gym.linear._delta_rule.decode import _decode_launch_config, _spec_decode_launch_config
from attn_gym.linear._delta_rule.recurrent import GateKind
from attn_gym.linear.gdn.ops import recurrent_decode_op, recurrent_spec_decode_op
from attn_gym.testing import strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_gdn_decode requires CUDA"
)


@pytest.mark.parametrize(
    ("value_dim", "sequence_heads", "tuned_major", "gate_kind", "expected"),
    [
        (32, 128, None, GateKind.SCALAR, (8, 4)),
        (128, 8, 9, GateKind.SCALAR, (8, 2)),
        (128, 96, 9, GateKind.SCALAR, (8, 1)),
        (128, 104, 9, GateKind.SCALAR, (16, 1)),
        (128, 224, 9, GateKind.VECTOR, (8, 1)),
        (128, 232, 9, GateKind.VECTOR, (16, 1)),
        (128, 32, 10, GateKind.SCALAR, (8, 4)),
        (128, 34, 10, GateKind.SCALAR, (8, 2)),
        (128, 128, 10, GateKind.SCALAR, (8, 2)),
        (128, 130, 10, GateKind.SCALAR, (8, 1)),
        (128, 96, None, GateKind.SCALAR, (16, 1)),
    ],
)
def test_decode_launch_config(
    value_dim: int,
    sequence_heads: int,
    tuned_major: int | None,
    gate_kind: GateKind,
    expected: tuple[int, int],
):
    assert _decode_launch_config(value_dim, sequence_heads, tuned_major, gate_kind) == expected


@pytest.mark.parametrize(
    ("sequence_heads", "speculative_width", "expected_block_v"),
    [
        (16, 1, 4),
        (16, 4, 2),
        (16, 8, 1),
        (32, 4, 4),
        (32, 8, 2),
        (32, 16, 1),
        (64, 1, 8),
        (64, 8, 4),
        (64, 16, 2),
        (128, 1, 8),
        (128, 4, 4),
        (256, 1, 16),
        (1024, 32, 16),
    ],
)
def test_spec_decode_launch_config(
    sequence_heads: int,
    speculative_width: int,
    expected_block_v: int,
):
    assert _spec_decode_launch_config(128, sequence_heads, speculative_width, 10) == (
        expected_block_v,
        1,
    )


def test_spec_decode_launch_config_preserves_generic_fallback():
    assert _spec_decode_launch_config(128, 16, 32, None) == (16, 1)
    assert _spec_decode_launch_config(24, 8, 32, 10) == (8, 4)


def make_decode_inputs(
    *,
    batch: int = 3,
    heads: int = 4,
    key_heads: int | None = None,
    key_dim: int = 32,
    value_dim: int = 24,
    num_slots: int = 6,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create raw decode operands; ``key_heads`` enables grouped heads."""
    torch.manual_seed(seed)
    key_heads = key_heads or heads
    q = torch.randn(batch, key_heads, key_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, heads, value_dim, device="cuda", dtype=dtype)
    _storage, pool = strided_state_pool(num_slots, heads, key_dim, value_dim)
    return {
        "packed_qkv": torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1),
        "raw_gate": torch.randn(1, batch, heads, device="cuda", dtype=dtype),
        "raw_beta": torch.randn(1, batch, heads, device="cuda", dtype=dtype),
        "A_log": 0.1 * torch.randn(heads, device="cuda", dtype=torch.float32),
        "dt_bias": 0.1 * torch.randn(heads, device="cuda", dtype=torch.float32),
        "state_cache": pool,
        "state_indices": torch.tensor([5, 1, 3], device="cuda", dtype=torch.int32)[:batch],
        "_storage": _storage,
    }


def cooked_operands(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Recover token-major operands with the decode preprocessing applied in FP32."""
    pool = inputs["state_cache"]
    heads, value_dim, key_dim = pool.shape[1:]
    packed = inputs["packed_qkv"].float()
    qk_channels = packed.shape[1] - heads * value_dim
    key_heads = qk_channels // (2 * key_dim)
    q, k, v = packed.split((key_heads * key_dim, key_heads * key_dim, heads * value_dim), dim=1)
    q = q.view(-1, key_heads, key_dim)
    k = k.view(-1, key_heads, key_dim)
    v = v.view(-1, heads, value_dim)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = -inputs["A_log"].exp() * F.softplus(inputs["raw_gate"][0].float() + inputs["dt_bias"])
    beta = torch.sigmoid(inputs["raw_beta"][0].float())
    return q, k, v, gate, beta


def reference_decode(
    inputs: dict[str, torch.Tensor], scale: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a cloned pool through the eager reference, gathering active slots."""
    q, k, v, gate, beta = cooked_operands(inputs)
    pool = inputs["state_cache"].clone()
    slots = inputs["state_indices"]
    heads = pool.shape[1]
    key_heads = q.shape[1]
    if key_heads != heads:
        q, k = (t.repeat_interleave(heads // key_heads, dim=1) for t in (q, k))
    output = torch.zeros(v.shape, device="cuda", dtype=torch.float32)
    for row, slot in enumerate(slots.tolist()):
        if slot <= 0:
            continue
        state = pool[slot].unsqueeze(0).clone()
        row_output, final_state = recurrent_gdn(
            q[row : row + 1, None],
            k[row : row + 1, None],
            v[row : row + 1, None],
            gate[row : row + 1, None],
            beta[row : row + 1, None],
            state,
            output_final_state=True,
            scale=pool.shape[-1] ** -0.5 if scale is None else scale,
            impl="reference",
        )
        output[row] = row_output[0, 0]
        pool[slot] = final_state[0]
    return output, pool


def reference_spec_decode(
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ordinary decode per token and save every speculative prefix state."""
    pool = inputs["state_cache"].clone()
    state_indices = inputs["state_indices"]
    heads, value_dim = pool.shape[1:3]
    output = inputs["packed_qkv"].new_zeros(inputs["packed_qkv"].shape[0], heads, value_dim)
    for sequence, (start, end) in enumerate(pairwise(cu_seqlens.tolist())):
        source_column = int(num_accepted_tokens[sequence].item()) - 1
        source_slot = int(state_indices[sequence, source_column].item())
        if source_slot <= 0:
            continue
        working_pool = pool.new_zeros((2, *pool.shape[1:]))
        working_pool[1].copy_(pool[source_slot])
        working_index = torch.tensor([1], device="cuda", dtype=torch.int32)
        for token_offset, token in enumerate(range(start, end)):
            token_output = recurrent_gdn_decode(
                inputs["packed_qkv"][token : token + 1],
                inputs["raw_gate"][:, token : token + 1],
                inputs["raw_beta"][:, token : token + 1],
                inputs["A_log"],
                inputs["dt_bias"],
                working_pool,
                working_index,
                scale=scale,
            )
            output[token] = token_output[0, 0]
            destination_slot = int(state_indices[sequence, token_offset].item())
            if destination_slot > 0:
                pool[destination_slot].copy_(working_pool[1])
    return output, pool


@pytest.mark.parametrize("scale", [None, 0.25])
@pytest.mark.parametrize("key_heads", [None, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_decode_matches_reference(key_heads: int | None, dtype: torch.dtype, scale: float | None):
    inputs = make_decode_inputs(key_heads=key_heads, dtype=dtype)
    expected_output, expected_pool = reference_decode(inputs, scale)

    output = recurrent_gdn_decode(
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_cache"],
        inputs["state_indices"],
        scale=scale,
    )

    tolerance = 1e-5 if dtype is torch.float32 else 3e-2
    assert output.dtype == dtype and output.shape[0] == 1
    torch.testing.assert_close(output[0].float(), expected_output, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(inputs["state_cache"], expected_pool, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("accepted", [(1, 1), (3, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_spec_decode_matches_token_reference(accepted: tuple[int, int], dtype: torch.dtype):
    """Speculative decode resumes the accepted checkpoint and saves every new prefix."""
    inputs = make_decode_inputs(batch=7, num_slots=10, dtype=dtype, seed=11)
    state_indices = torch.empty_strided(
        (2, 4),
        (9, 2),
        device="cuda",
        dtype=torch.int32,
    )
    state_indices.copy_(torch.tensor([[9, 8, 7, 6], [5, 4, 3, 2]], device="cuda"))
    inputs["state_indices"] = state_indices
    cu_seqlens = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int32)
    num_accepted_tokens = torch.tensor(accepted, device="cuda", dtype=torch.int32)
    expected_output, expected_pool = reference_spec_decode(
        inputs,
        cu_seqlens,
        num_accepted_tokens,
    )

    output = recurrent_gdn_decode(
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_cache"],
        inputs["state_indices"],
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted_tokens,
    )

    # The speculative kernel retains state in registers across tokens, while the
    # oracle launches once per token. Triton may schedule their FP32 arithmetic
    # differently, so checkpoint states can differ by a few FP32 ULPs.
    torch.testing.assert_close(output[0], expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(inputs["state_cache"], expected_pool, rtol=1e-5, atol=5e-7)


def test_spec_decode_int64_offsets_match_default(monkeypatch):
    """The wide-address specialization matches the ordinary int32 path exactly."""
    inputs = make_decode_inputs(batch=4, num_slots=6, dtype=torch.bfloat16, seed=13)
    inputs["state_indices"] = torch.tensor(
        [[5, 4], [3, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    initial_pool = inputs["state_cache"].clone()

    def run() -> torch.Tensor:
        return recurrent_gdn_decode(
            inputs["packed_qkv"],
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
        )

    expected = run()
    expected_pool = inputs["state_cache"].clone()
    inputs["state_cache"].copy_(initial_pool)
    calls = 0

    def force_int64(*_tensors):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(decode_module, "requires_int64_offsets", force_int64)
    actual = run()

    assert calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state_cache"], expected_pool, rtol=0, atol=0)


@pytest.mark.parametrize(("major", "heads"), [(9, 8), (10, 16)])
def test_tuned_decode_schedule_matches_reference(major: int, heads: int):
    if torch.cuda.get_device_capability()[0] != major:
        pytest.skip(f"SM{major} decode schedule")
    inputs = make_decode_inputs(
        batch=1,
        heads=heads,
        key_dim=128,
        value_dim=128,
        dtype=torch.bfloat16,
    )
    inputs["state_indices"] = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected_output, expected_pool = reference_decode(inputs)

    output = recurrent_gdn_decode(
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_cache"],
        inputs["state_indices"],
    )

    torch.testing.assert_close(output[0].float(), expected_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(inputs["state_cache"], expected_pool, rtol=1e-5, atol=1e-5)


def test_decode_padding_and_fresh_slots():
    """Nonpositive slots produce zero output; fresh slots start from zero and overwrite."""
    inputs = make_decode_inputs()
    inputs["state_indices"] = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], device="cuda")
    original_pool = inputs["state_cache"].clone()
    zeroed = dict(inputs)
    zeroed["state_cache"] = inputs["state_cache"].clone()
    zeroed["state_cache"][2] = 0.0
    expected_output, expected_pool = reference_decode(zeroed)

    output = recurrent_gdn_decode(
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_cache"],
        inputs["state_indices"],
        has_initial_state=has_initial_state,
    )

    torch.testing.assert_close(output[0, 0], torch.zeros_like(output[0, 0]), rtol=0, atol=0)
    torch.testing.assert_close(output[0].float(), expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(inputs["state_cache"][2], expected_pool[2], rtol=1e-5, atol=1e-5)
    preserved = [0, 1, 3, 5]
    torch.testing.assert_close(
        inputs["state_cache"][preserved], original_pool[preserved], rtol=0, atol=0
    )


def test_decode_out_buffer_contract():
    inputs = make_decode_inputs()
    heads, value_dim = inputs["state_cache"].shape[1:3]
    out = torch.empty(1, 3, heads, value_dim, device="cuda")
    returned = recurrent_gdn_decode(
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_cache"],
        inputs["state_indices"],
        out=out,
    )
    assert returned is out

    with pytest.raises(ValueError, match="must not alias"):
        recurrent_gdn_decode(
            inputs["packed_qkv"],
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
            # Slice the pool's backing storage; flattening the strided view would copy.
            out=inputs["_storage"].flatten()[: out.numel()].view_as(out),
        )


def test_decode_rejects_bad_inputs():
    inputs = make_decode_inputs()
    with pytest.raises(ValueError, match="positive divisor"):
        recurrent_gdn_decode(
            inputs["packed_qkv"][:, :-1],
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
        )
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_gdn_decode(
            inputs["packed_qkv"].clone().requires_grad_(),
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
        )


def test_decode_custom_op_registration():
    inputs = make_decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8, num_slots=3)
    inputs["state_indices"] = torch.tensor([1], device="cuda", dtype=torch.int32)
    out = inputs["packed_qkv"].new_empty(1, 1, 1, 8)
    torch.library.opcheck(
        recurrent_decode_op,
        (
            inputs["packed_qkv"],
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
            None,
            out,
            0.25,
        ),
    )


def test_spec_decode_custom_op_registration():
    inputs = make_decode_inputs(batch=3, heads=1, key_dim=16, value_dim=8, num_slots=5)
    state_indices_storage = torch.tensor([[4, 0, 3, 0, 2]], device="cuda", dtype=torch.int32)
    inputs["state_indices"] = state_indices_storage[:, ::2]
    cu_seqlens = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    num_accepted_tokens = torch.tensor([2], device="cuda", dtype=torch.int32)
    out = inputs["packed_qkv"].new_empty(1, 3, 1, 8)
    torch.library.opcheck(
        recurrent_spec_decode_op,
        (
            inputs["packed_qkv"],
            inputs["raw_gate"],
            inputs["raw_beta"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["state_cache"],
            inputs["state_indices"],
            num_accepted_tokens,
            cu_seqlens,
            out,
            0.25,
        ),
    )


@pytest.mark.parametrize(
    "decode_kwargs",
    [
        pytest.param({}, id="generic"),
        pytest.param(
            {
                "batch": 1,
                "heads": 16,
                "key_dim": 128,
                "value_dim": 128,
                "dtype": torch.bfloat16,
            },
            id="blackwell",
        ),
    ],
)
def test_decode_fullgraph_and_cuda_graph(decode_kwargs: dict[str, object]):
    """The decode op compiles fullgraph and replays under CUDA graph capture."""
    if decode_kwargs and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Blackwell-specific decode schedule")
    inputs = make_decode_inputs(**decode_kwargs)
    initial_pool = inputs["state_cache"].clone()
    args = (
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
    )
    slots = inputs["state_indices"]

    def run(pool: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return recurrent_gdn_decode(*args, pool, slots, out=out)

    with torch.no_grad():
        eager_pool = inputs["state_cache"]
        expected = run(eager_pool)

        heads, value_dim, key_dim = inputs["state_cache"].shape[1:]
        _compiled_storage, compiled_pool = strided_state_pool(6, heads, key_dim, value_dim)
        compiled_pool.copy_(initial_pool)
        compiled = torch.compile(recurrent_gdn_decode, fullgraph=True)
        compiled_out = torch.empty_like(expected)
        actual = compiled(*args, compiled_pool, slots, out=compiled_out)
        assert actual is compiled_out
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(compiled_pool, eager_pool, rtol=0, atol=0)

        _graph_storage, graph_pool = strided_state_pool(6, heads, key_dim, value_dim)
        graph_pool.copy_(initial_pool)
        out = torch.empty_like(expected)
        run(graph_pool, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(graph_pool, out=out)
        # Reset to the pre-decode snapshot so replay reproduces the eager call.
        graph_pool.copy_(initial_pool)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(graph_pool, eager_pool, rtol=0, atol=0)


def test_spec_decode_fullgraph_and_cuda_graph():
    """Speculative metadata remains dynamic across compile and CUDA graph replay."""
    inputs = make_decode_inputs(
        batch=5,
        heads=16,
        key_dim=128,
        value_dim=128,
        num_slots=8,
        dtype=torch.bfloat16,
        seed=12,
    )
    state_indices_storage = torch.empty_strided(
        (2, 3),
        (7, 2),
        device="cuda",
        dtype=torch.int32,
    )
    state_indices_storage.copy_(torch.tensor([[7, 6, 5], [4, 3, 2]], device="cuda"))
    state_indices = state_indices_storage
    cu_seqlens = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    initial_pool = inputs["state_cache"].clone()
    args = (
        inputs["packed_qkv"],
        inputs["raw_gate"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
    )

    def fresh_pool() -> torch.Tensor:
        _storage, pool = strided_state_pool(8, 16, 128, 128)
        pool.copy_(initial_pool)
        return pool

    def run(pool: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return recurrent_gdn_decode(
            *args,
            pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            out=out,
        )

    with torch.no_grad():
        compiled = torch.compile(run, fullgraph=True)
        warmup_pool = fresh_pool()
        compiled(warmup_pool)
        torch.cuda.synchronize()

        graph_pool = fresh_pool()
        graph_out = args[0].new_empty(1, args[0].shape[0], 16, 128)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(graph_pool, out=graph_out)

        graph_pool.copy_(initial_pool)
        num_accepted_tokens.copy_(torch.tensor([2, 1], device="cuda", dtype=torch.int32))
        inputs["packed_qkv"].mul_(0.9)
        eager_pool = fresh_pool()
        compiled_pool = fresh_pool()
        expected = run(eager_pool)
        actual = compiled(compiled_pool)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(compiled_pool, eager_pool, rtol=0, atol=0)
        torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)
        torch.testing.assert_close(graph_pool, eager_pool, rtol=0, atol=0)
