"""Fused Mamba2 inference paths: recurrent scan, paged prefill, and one-token paged decode."""

import importlib

import pytest
import torch

from attn_gym.linear import (
    chunk_mamba2,
    paged_chunk_mamba2,
    recurrent_mamba2,
    recurrent_mamba2_decode,
)
from attn_gym.linear.mamba2.ops import (
    chunk_fwd_paged_op,
    decode_op,
    recurrent_fwd_no_state_op,
    recurrent_fwd_op,
    recurrent_fwd_paged_op,
)
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HEADS, GROUPS, HEAD_DIM, STATE_DIM = 6, 2, 64, 128


def make_tokens(batch, tokens, *, dtype=torch.bfloat16, seed=0, heads=HEADS, groups=GROUPS):
    """Mamba2-like token-major inputs plus per-head parameters."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    return {
        "x": randn(batch, tokens, heads, HEAD_DIM, dtype=dtype),
        "dt": randn(batch, tokens, heads) - 2.0,
        "A": -torch.exp(torch.rand(heads, generator=generator, device="cuda") * 2.77),
        "B": randn(batch, tokens, groups, STATE_DIM, dtype=dtype),
        "C": randn(batch, tokens, groups, STATE_DIM, dtype=dtype),
        "D": randn(heads),
        "z": randn(batch, tokens, heads, HEAD_DIM, dtype=dtype),
        "dt_bias": randn(heads) * 0.5,
    }


TRANSFORM = {"dt_softplus": True, "dt_limit": (0.0, float("inf"))}


def reference(inputs, initial_state=None, **kwargs):
    """FP64 chunk reference on the same rounded inputs."""
    upcast = {name: value.double() for name, value in inputs.items()}
    if initial_state is not None:
        initial_state = initial_state.double()
    return chunk_mamba2(
        **upcast, initial_state=initial_state, impl="reference", output_final_state=True, **kwargs
    )


def relative_error(actual, expected):
    expected = expected.double()
    return ((actual.double() - expected).norm() / expected.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_matches_reference(dtype, packed):
    lengths = [37, 0, 1, 70] if packed else None
    batch, tokens = (1, sum(lengths) + 3) if packed else (2, 45)
    inputs = make_tokens(batch, tokens, dtype=dtype)
    states = len(lengths) if packed else batch
    initial_state = torch.randn(states, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    kwargs = TRANSFORM | {
        "cu_seqlens": None if lengths is None else cumulative_sequence_offsets(lengths)
    }
    with torch.no_grad():
        output, state = recurrent_mamba2(
            **inputs, initial_state=initial_state, output_final_state=True, **kwargs
        )
    expected, expected_state = reference(inputs, initial_state, **kwargs)
    active = tokens if lengths is None else sum(lengths)
    # BF16 activations: FP32 math with the reference rounding only at the output.
    budget = 2e-3 if dtype is torch.bfloat16 else 1e-5
    assert relative_error(output[:, :active], expected[:, :active]) < budget
    assert relative_error(state, expected_state) < budget


def routed_pool(num_slots, routes, has_initial, *, dtype=torch.float32):
    """A pool with garbage in fresh slots and the expected per-sequence initial states."""
    pool = torch.randn(num_slots, HEADS, HEAD_DIM, STATE_DIM, device="cuda").to(dtype)
    pool[0] = float("nan")  # slot 0 is never routed to
    initial = []
    for route, load in zip(routes, has_initial):
        resumed = route > 0 and load
        initial.append(
            pool[route].float() if resumed else torch.zeros_like(pool[0], dtype=torch.float32)
        )
    state_indices = torch.tensor(routes, dtype=torch.int32, device="cuda")
    mask = torch.tensor(has_initial, dtype=torch.bool, device="cuda")
    return pool, state_indices, mask, torch.stack(initial)


ROUTES = [3, 0, 5, -1, 2]
HAS_INITIAL = [True, True, False, True, True]
LENGTHS = [40, 12, 70, 9, 0]


@pytest.fixture
def isolated_winners(tmp_path, monkeypatch):
    """Force autotuning, whose candidate launches must not advance a paged pool repeatedly."""
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})


@pytest.mark.parametrize("op", ["recurrent", "chunk"])
@pytest.mark.parametrize("pool_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("resume_all", [False, True])
def test_paged_routes(op, pool_dtype, resume_all, isolated_winners):
    inputs = make_tokens(1, sum(LENGTHS) + 4)
    has_initial = [True] * len(ROUTES) if resume_all else HAS_INITIAL
    pool, state_indices, mask, initial = routed_pool(8, ROUTES, has_initial, dtype=pool_dtype)
    mask = None if resume_all else mask
    before = pool.clone()
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    with torch.no_grad():
        if op == "recurrent":
            output, _ = recurrent_mamba2(
                **inputs,
                initial_state=pool,
                state_indices=state_indices,
                has_initial_state=mask,
                cu_seqlens=cu_seqlens,
                **TRANSFORM,
            )
        else:
            output = paged_chunk_mamba2(
                **inputs,
                state_cache=pool,
                state_indices=state_indices,
                has_initial_state=mask,
                cu_seqlens=cu_seqlens,
                **TRANSFORM,
            )
    expected, expected_state = reference(inputs, initial, cu_seqlens=cu_seqlens, **TRANSFORM)
    # The chunk path rounds its MMA operands to BF16 (~2.4e-3); BF16 pools round the state.
    budget = (2e-3 if op == "recurrent" else 5e-3) if pool_dtype is torch.float32 else 1e-2
    for sequence, route in enumerate(ROUTES):
        span = slice(int(cu_seqlens[sequence]), int(cu_seqlens[sequence + 1]))
        if route <= 0:
            assert output[:, span].eq(0).all(), "null routes produce zero output"
            continue
        if span.start < span.stop:
            assert relative_error(output[:, span], expected[:, span]) < budget
        assert relative_error(pool[route], expected_state[sequence]) < budget
    untouched = [slot for slot in range(8) if slot not in [r for r in ROUTES if r > 0]]
    torch.testing.assert_close(pool[untouched], before[untouched], equal_nan=True)


def test_decode_matches_reference_step_from_strided_views():
    batch = 5
    # x, B, and C sliced out of one post-convolution projection buffer, as serving stacks keep.
    x_width, bc_width = HEADS * HEAD_DIM, GROUPS * STATE_DIM
    packed = torch.randn(batch, x_width + 2 * bc_width, device="cuda", dtype=torch.bfloat16)
    x = packed[:, :x_width].view(batch, HEADS, HEAD_DIM)
    B = packed[:, x_width : x_width + bc_width].view(batch, GROUPS, STATE_DIM)
    C = packed[:, x_width + bc_width :].view(batch, GROUPS, STATE_DIM)
    params = make_tokens(batch, 1)
    dt, z = params["dt"][:, 0], params["z"][:, 0]
    pool, state_indices, mask, initial = routed_pool(8, ROUTES, HAS_INITIAL)
    before = pool.clone()
    out = torch.empty_like(z)
    with torch.no_grad():
        result = recurrent_mamba2_decode(
            x,
            dt,
            params["A"],
            B,
            C,
            pool,
            state_indices,
            params["D"],
            z,
            has_initial_state=mask,
            dt_bias=params["dt_bias"],
            out=out,
            **TRANSFORM,
        )
    assert result is out
    step = {
        "x": x.unsqueeze(1),
        "dt": dt.unsqueeze(1),
        "A": params["A"],
        "B": B.unsqueeze(1),
        "C": C.unsqueeze(1),
        "D": params["D"],
        "z": z.unsqueeze(1),
        "dt_bias": params["dt_bias"],
    }
    expected, expected_state = reference(step, initial, **TRANSFORM)
    for sequence, route in enumerate(ROUTES):
        if route <= 0:
            assert out[sequence].eq(0).all()
            continue
        assert relative_error(out[sequence], expected[sequence, 0]) < 2e-3
        assert relative_error(pool[route], expected_state[sequence]) < 2e-3
    torch.testing.assert_close(pool[[0, 1, 4, 6, 7]], before[[0, 1, 4, 6, 7]], equal_nan=True)


def test_paged_prefill_then_decode_continues_the_sequence():
    prompt, steps = 100, 6
    inputs = make_tokens(1, prompt + steps, seed=3)
    pool = torch.zeros(4, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    state_indices = torch.tensor([2], dtype=torch.int32, device="cuda")
    fresh = torch.tensor([False], device="cuda")
    with torch.no_grad():
        prefix = {
            name: value[:, :prompt] if value.ndim > 1 else value for name, value in inputs.items()
        }
        outputs = [
            paged_chunk_mamba2(
                **prefix,
                state_cache=pool,
                state_indices=state_indices,
                has_initial_state=fresh,
                **TRANSFORM,
            )
        ]
        for token in range(prompt, prompt + steps):
            step = outputs[-1].new_empty(1, HEADS, HEAD_DIM)
            recurrent_mamba2_decode(
                inputs["x"][:, token],
                inputs["dt"][:, token],
                inputs["A"],
                inputs["B"][:, token],
                inputs["C"][:, token],
                pool,
                state_indices,
                inputs["D"],
                inputs["z"][:, token],
                dt_bias=inputs["dt_bias"],
                out=step,
                **TRANSFORM,
            )
            outputs.append(step.unsqueeze(1))
    expected, expected_state = reference(inputs, **TRANSFORM)
    # The prefill rounds its MMA operands to BF16 (~2.4e-3 relative L2).
    assert relative_error(torch.cat(outputs, dim=1), expected) < 5e-3
    assert relative_error(pool[2], expected_state[0]) < 5e-3


def test_decode_launches_beyond_grid_y_limit():
    # batch * heads > 65535 overflowed grid axis 1 when (sequence, head) programs lived there.
    batch, heads, head_dim, state_dim = 1100, 64, 16, 16
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(batch, heads, head_dim, generator=generator, device="cuda")
    dt = torch.rand(batch, heads, generator=generator, device="cuda") * 0.1
    A = -torch.rand(heads, generator=generator, device="cuda")
    B = torch.randn(batch, 1, state_dim, generator=generator, device="cuda")
    C = torch.randn(batch, 1, state_dim, generator=generator, device="cuda")
    pool = torch.randn(batch + 1, heads, head_dim, state_dim, generator=generator, device="cuda")
    expected_state = pool[1:].clone()
    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        out = recurrent_mamba2_decode(x, dt, A, B, C, pool, state_indices)
    decay = torch.exp(dt * A)[..., None, None]
    expected_state = decay * expected_state + (dt[..., None] * x)[..., None] * B[:, :, None]
    expected = (expected_state @ C[:, 0, :, None].unsqueeze(1)).squeeze(-1)
    torch.testing.assert_close(pool[1:], expected_state, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_decode_reads_independently_strided_projections():
    # B sliced from a wider buffer and C contiguous: each must use its own strides.
    params = make_tokens(3, 1)
    wide = torch.randn(3, GROUPS, 2 * STATE_DIM, device="cuda", dtype=torch.bfloat16)
    B, C = wide[..., :STATE_DIM], params["C"][:, 0].contiguous()
    pool = torch.randn(4, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    state_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    expected_pool = pool.clone()
    args = (params["x"][:, 0], params["dt"][:, 0], params["A"])
    with torch.no_grad():
        out = recurrent_mamba2_decode(*args, B, C, pool, state_indices)
        expected = recurrent_mamba2_decode(*args, B.contiguous(), C, expected_pool, state_indices)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


def test_paged_ops_reject_pool_aliasing():
    inputs = make_tokens(1, 4)
    pool = torch.zeros(4, HEADS, HEAD_DIM, STATE_DIM, device="cuda", dtype=torch.bfloat16)
    # x viewed out of the pool: the kernel would overwrite input it has not read yet.
    inputs["x"] = pool.view(-1)[: inputs["x"].numel()].view_as(inputs["x"])
    state_indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    with torch.no_grad(), pytest.raises(ValueError, match="must not alias"):
        paged_chunk_mamba2(**inputs, state_cache=pool, state_indices=state_indices)
    with torch.no_grad(), pytest.raises(ValueError, match="must not alias"):
        recurrent_mamba2(**inputs, initial_state=pool, state_indices=state_indices)


def test_decode_rejects_bad_inputs():
    params = make_tokens(2, 1)
    pool = torch.zeros(4, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    args = (params["dt"][:, 0], params["A"], params["B"][:, 0], params["C"][:, 0])
    with pytest.raises(ValueError, match="x must have shape"):
        recurrent_mamba2_decode(params["x"], *args, pool, state_indices)
    with pytest.raises(ValueError, match="state_indices"):
        recurrent_mamba2_decode(params["x"][:, 0], *args, pool, state_indices[:1])
    with pytest.raises(ValueError, match="out must have shape"):
        recurrent_mamba2_decode(
            params["x"][:, 0], *args, pool, state_indices, out=params["x"][:1, 0]
        )
    with pytest.raises(ValueError, match="out must not alias"):
        recurrent_mamba2_decode(
            params["x"][:, 0], *args, pool, state_indices, out=params["x"][:, 0]
        )
    params["x"].requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_mamba2_decode(params["x"][:, 0], *args, pool, state_indices)


def test_inference_ops_pass_opcheck():
    inputs = make_tokens(1, 70)
    cu_seqlens = cumulative_sequence_offsets([30, 40])
    pool = torch.randn(4, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    tensors = [inputs[name] for name in ("x", "dt", "A", "B", "C", "D", "z")]
    transform = (True, 0.0, float("inf"))
    torch.library.opcheck(
        recurrent_fwd_op,
        (*tensors, pool[1:3].clone(), inputs["dt_bias"], cu_seqlens, *transform),
    )
    torch.library.opcheck(
        recurrent_fwd_no_state_op,
        (*tensors, None, inputs["dt_bias"], cu_seqlens, *transform),
    )
    has_initial_state = torch.tensor([True, False], device="cuda")
    for op in (chunk_fwd_paged_op, recurrent_fwd_paged_op):
        paged = (pool.clone(), state_indices, has_initial_state, inputs["dt_bias"], cu_seqlens)
        torch.library.opcheck(op, (*tensors, *paged, *transform))
    step = {name: value[0] for name, value in inputs.items() if value.ndim > 1}
    torch.library.opcheck(
        decode_op,
        (
            step["x"][:2],
            step["dt"][:2],
            inputs["A"],
            step["B"][:2],
            step["C"][:2],
            inputs["D"],
            step["z"][:2],
            pool,
            state_indices,
            None,
            inputs["dt_bias"],
            *transform,
            torch.empty_like(step["x"][:2]),
        ),
    )


def test_decode_compiles_fullgraph():
    params = make_tokens(3, 1)
    pool = torch.randn(5, HEADS, HEAD_DIM, STATE_DIM, device="cuda")
    state_indices = torch.tensor([4, 1, 2], dtype=torch.int32, device="cuda")
    args = (
        params["x"][:, 0],
        params["dt"][:, 0],
        params["A"],
        params["B"][:, 0],
        params["C"][:, 0],
    )

    def step(state_cache):
        return recurrent_mamba2_decode(
            *args, state_cache, state_indices, params["D"], dt_bias=params["dt_bias"], **TRANSFORM
        )

    eager_pool, compiled_pool = pool.clone(), pool.clone()
    with torch.no_grad():
        expected = step(eager_pool)
        actual = torch.compile(step, fullgraph=True)(compiled_pool)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(compiled_pool, eager_pool)
