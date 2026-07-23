from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse import compressed_sparse_attention


# These are the tensor arguments for which the current implementations expose
# gradients. This matches the existing Triton and CuTe backward tests.
#
# Argument order:
#
#   0  Q
#   1  Q_I
#   2  KV
#   3  C_a
#   4  C_b
#   5  Z_a
#   6  Z_b
#   7  B_a
#   8  B_b
#   9  W_I
#   10 K_Ia
#   11 K_Ib
#   12 Z_Ia
#   13 Z_Ib
#   14 B_Ia
#   15 B_Ib
#   16 q_norm_weight
#   17 compressed_indices_norm_weight
#   18 compressed_kv_norm_weight
#   19 attention_sink
#   20 compression_rate
#   21 num_topk_blocks
#   22 sliding_window_size
#   23 rope_dims
#   24 share_kv
#
# The discrete top-k selection path does not expose useful gradients through
# Q_I, W_I, or the indexer tensors, so those arguments are intentionally absent.
DIFFERENTIABLE_INPUTS = frozenset({0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19})

TRITON_MAX_ABS_ERROR = 3e-2
CUTE_MAX_ABS_ERROR = 3e-2


@dataclasses.dataclass(frozen=True)
class BackendCase:
    """One legal torch.compile test configuration for a CSA backend."""

    id: str
    backend: str
    dtype: torch.dtype
    share_kv: bool
    batch: int
    heads: int
    sequence_length: int
    head_dim: int
    index_heads: int
    index_dim: int
    compression_rate: int
    topk: int
    window: int
    rope_dims: int
    output_atol: float
    output_rtol: float
    gradient_atol: float
    gradient_rtol: float
    requires_cuda: bool = True
    requires_sm100: bool = False


BACKEND_CASES = (
    pytest.param(
        BackendCase(
            id="reference",
            backend="eager",
            dtype=torch.float32,
            share_kv=False,
            batch=2,
            heads=3,
            sequence_length=17,
            head_dim=32,
            index_heads=2,
            index_dim=16,
            compression_rate=4,
            topk=3,
            window=5,
            rope_dims=8,
            output_atol=1e-5,
            output_rtol=1e-5,
            gradient_atol=1e-5,
            gradient_rtol=1e-5,
        ),
        id="reference",
    ),
    pytest.param(
        BackendCase(
            id="triton",
            backend="triton",
            dtype=torch.bfloat16,
            share_kv=False,
            batch=2,
            heads=4,
            sequence_length=19,
            head_dim=32,
            index_heads=2,
            index_dim=16,
            compression_rate=4,
            topk=3,
            window=5,
            rope_dims=8,
            output_atol=TRITON_MAX_ABS_ERROR,
            output_rtol=0.0,
            gradient_atol=TRITON_MAX_ABS_ERROR,
            gradient_rtol=0.0,
        ),
        id="triton",
    ),
    pytest.param(
        BackendCase(
            id="cute",
            backend="cute",
            dtype=torch.bfloat16,
            share_kv=True,
            batch=1,
            heads=64,
            sequence_length=35,
            head_dim=512,
            index_heads=4,
            index_dim=64,
            compression_rate=16,
            topk=2,
            window=16,
            rope_dims=64,
            output_atol=CUTE_MAX_ABS_ERROR,
            output_rtol=0.0,
            gradient_atol=CUTE_MAX_ABS_ERROR,
            gradient_rtol=0.0,
            requires_sm100=True,
        ),
        id="cute",
    ),
)


def _skip_if_backend_is_unavailable(case: BackendCase) -> None:
    """Skip only the unavailable backend instead of skipping the whole file."""

    if case.requires_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    if case.backend == "triton":
        pytest.importorskip("triton")

    if case.backend == "cute":
        pytest.importorskip("flash_attn.cute.interface")

        if torch.cuda.get_device_capability() != (10, 0):
            pytest.skip("the CuTe backend targets SM100 exclusively")

        # Import the backend here so missing optional CuTe/CUTLASS dependencies
        # skip only this parameterized invocation.
        try:
            importlib.import_module(
                "attn_gym.sparse.compressed_sparse_attention.cute"
            )
        except (ImportError, RuntimeError) as error:
            pytest.skip(f"CuTe backend is unavailable: {error}")


def _make_inputs(case: BackendCase) -> tuple[torch.Tensor | int | bool, ...]:
    """Construct a valid public-API argument tuple for one backend."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(123)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device=device,
                dtype=case.dtype,
                generator=generator,
            )
            * scale
        )

    def query(*shape: int) -> torch.Tensor:
        return F.normalize(randn(*shape), dim=-1)

    kv_heads = 1 if case.share_kv else case.heads
    index_kv_heads = 1 if case.share_kv else case.index_heads

    return (
        # Q
        query(
            case.batch,
            case.heads,
            case.sequence_length,
            case.head_dim,
        ),
        # Q_I
        query(
            case.batch,
            case.index_heads,
            case.sequence_length,
            case.index_dim,
        ),
        # KV
        randn(
            case.batch,
            kv_heads,
            case.sequence_length,
            case.head_dim,
        ),
        # C_a
        randn(
            case.batch,
            kv_heads,
            case.sequence_length,
            case.head_dim,
        ),
        # C_b
        randn(
            case.batch,
            kv_heads,
            case.sequence_length,
            case.head_dim,
        ),
        # Z_a
        randn(
            case.batch,
            kv_heads,
            case.sequence_length,
            case.head_dim,
        ),
        # Z_b
        randn(
            case.batch,
            kv_heads,
            case.sequence_length,
            case.head_dim,
        ),
        # B_a
        randn(case.compression_rate, case.head_dim),
        # B_b
        randn(case.compression_rate, case.head_dim),
        # W_I
        randn(
            case.batch,
            case.sequence_length,
            case.index_heads,
        ),
        # K_Ia
        randn(
            case.batch,
            index_kv_heads,
            case.sequence_length,
            case.index_dim,
        ),
        # K_Ib
        randn(
            case.batch,
            index_kv_heads,
            case.sequence_length,
            case.index_dim,
        ),
        # Z_Ia
        randn(
            case.batch,
            index_kv_heads,
            case.sequence_length,
            case.index_dim,
        ),
        # Z_Ib
        randn(
            case.batch,
            index_kv_heads,
            case.sequence_length,
            case.index_dim,
        ),
        # B_Ia
        randn(case.compression_rate, case.index_dim),
        # B_Ib
        randn(case.compression_rate, case.index_dim),
        # q_norm_weight
        1.0 + randn(case.head_dim, scale=0.05),
        # compressed_indices_norm_weight
        1.0 + randn(case.index_dim, scale=0.05),
        # compressed_kv_norm_weight
        1.0 + randn(case.head_dim, scale=0.05),
        # attention_sink
        randn(case.heads),
        # Scalar/static arguments
        case.compression_rate,
        case.topk,
        case.window,
        case.rope_dims,
        case.share_kv,
    )


def _differentiable_copy(
    inputs: tuple[torch.Tensor | int | bool, ...],
) -> tuple[torch.Tensor | int | bool, ...]:
    """Clone inputs and enable gradients only for differentiable arguments."""

    return tuple(
        value.detach().clone().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(inputs)
    )


def _gradient_targets(
    inputs: tuple[torch.Tensor | int | bool, ...],
) -> tuple[torch.Tensor, ...]:
    """Return differentiable tensors in a deterministic argument order."""

    return tuple(
        inputs[index]
        for index in sorted(DIFFERENTIABLE_INPUTS)
        if isinstance(inputs[index], torch.Tensor)
    )


def _make_backend_function(case: BackendCase) -> Callable[..., torch.Tensor]:
    """Close over the backend so it is static during Dynamo tracing."""

    backend = case.backend

    def run(*args: Any) -> torch.Tensor:
        return compressed_sparse_attention(*args, backend=backend)

    return run


def _make_grad_output(output: torch.Tensor) -> torch.Tensor:
    """Create a stable nontrivial cotangent for backward comparisons."""

    generator = torch.Generator(device=output.device).manual_seed(456)
    return (
        torch.randn(
            output.shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        * 0.01
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    message: str,
) -> None:
    """Give a useful maximum-error message when a comparison fails."""

    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as error:
        max_abs_error = (
            actual.detach().float() - expected.detach().float()
        ).abs().max().item()
        raise AssertionError(
            f"{message}; max absolute error = {max_abs_error}"
        ) from error


def _warm_backend(
    run: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor | int | bool, ...],
) -> None:
    """Load optional modules and JIT-compile backend kernels before Dynamo.

    The warmup is outside the compiled region. It avoids testing Python module
    import machinery or first-use kernel compilation instead of testing whether
    the already-resolved public operation is graph-capturable.
    """

    warm_inputs = _differentiable_copy(inputs)
    warm_targets = _gradient_targets(warm_inputs)

    output = run(*warm_inputs)
    grad_output = _make_grad_output(output)
    torch.autograd.grad(output, warm_targets, grad_output)

    torch.cuda.synchronize()


@pytest.mark.parametrize("case", BACKEND_CASES)
def test_compiled_sparse_attention_compiled(case: BackendCase) -> None:
    """Compile forward and backward for all three CSA backends.

    `fullgraph=True` causes any Dynamo graph break in the public API or backend
    path to fail the test.

    The test compares:
      1. eager and compiled forward outputs;
      2. eager and compiled gradients for every differentiable input;
      3. a second compiled invocation, ensuring the captured graph is reusable.

    This test deliberately invokes the public `compressed_sparse_attention`
    function rather than importing backend-private functions directly.
    """

    _skip_if_backend_is_unavailable(case)

    base_inputs = _make_inputs(case)
    run = _make_backend_function(case)

    # Resolve lazy imports and first-use JIT compilation before torch.compile.
    _warm_backend(run, base_inputs)

    # Ensure earlier tests or warmups do not affect this test's Dynamo state.
    torch._dynamo.reset()

    compiled_run = torch.compile(
        run,
        fullgraph=True,
        dynamic=False,
    )

    eager_inputs = _differentiable_copy(base_inputs)
    compiled_inputs = _differentiable_copy(base_inputs)

    eager_targets = _gradient_targets(eager_inputs)
    compiled_targets = _gradient_targets(compiled_inputs)

    expected = run(*eager_inputs)
    actual = compiled_run(*compiled_inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    _assert_close(
        actual,
        expected,
        atol=case.output_atol,
        rtol=case.output_rtol,
        message=f"{case.id} compiled forward does not match eager",
    )

    grad_output = _make_grad_output(expected)

    expected_gradients = torch.autograd.grad(
        expected,
        eager_targets,
        grad_output,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        compiled_targets,
        grad_output,
    )

    assert len(actual_gradients) == len(expected_gradients)

    for input_index, actual_gradient, expected_gradient in zip(
        sorted(DIFFERENTIABLE_INPUTS),
        actual_gradients,
        expected_gradients,
    ):
        _assert_close(
            actual_gradient,
            expected_gradient,
            atol=case.gradient_atol,
            rtol=case.gradient_rtol,
            message=(
                f"{case.id} compiled gradient for input {input_index} "
                "does not match eager"
            ),
        )

    # Invoke the compiled function again using fresh tensor objects with the
    # same static shapes. This verifies that the graph can be reused rather
    # than only surviving its initial compilation.
    second_inputs = _differentiable_copy(base_inputs)
    second_actual = compiled_run(*second_inputs)

    _assert_close(
        second_actual,
        expected.detach(),
        atol=case.output_atol,
        rtol=case.output_rtol,
        message=f"{case.id} second compiled forward does not match eager",
    )

    torch.cuda.synchronize()