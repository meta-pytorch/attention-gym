"""Integration tests for the GDN training example."""

import math

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("triton")
pytest.importorskip("typer")

from examples.gdn_training import GDNAttention, training_loop

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the GDN training example requires CUDA"
)

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def small_loop(backend: str, *, compile_module: bool = False) -> list[float]:
    """Run a tiny training loop with test-sized defaults."""
    return training_loop(
        backend=backend,
        batch_size=2,
        tokens=64,
        hidden_size=128,
        num_heads=6,
        num_key_heads=2,
        head_dim=128 if backend == "fused" else 32,
        steps=12,
        learning_rate=3e-3,
        compile_module=compile_module,
    )


def test_reference_training_overfits():
    losses = small_loop("reference")
    assert all(map(math.isfinite, losses))
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.skipif(not BLACKWELL, reason="the fused backend requires CUDA capability 10.0")
@pytest.mark.parametrize("compile_module", [False, True])
def test_fused_training_overfits(compile_module: bool):
    losses = small_loop("fused", compile_module=compile_module)
    assert all(map(math.isfinite, losses))
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.skipif(not BLACKWELL, reason="the fused backend requires CUDA capability 10.0")
def test_backends_agree_at_initialization():
    """One forward/backward step matches across backends from identical parameters."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    kwargs = {"hidden_size": 128, "num_heads": 6, "num_key_heads": 2, "head_dim": 128}
    reference = GDNAttention(
        backend="reference", compute_dtype=torch.bfloat16, device=device, **kwargs
    )
    fused = GDNAttention(backend="fused", device=device, **kwargs)
    fused.load_state_dict(reference.state_dict())

    hidden_states = torch.randn(2, 64, 128, device=device)
    expected = reference(hidden_states, return_final_state=True)
    actual = fused(hidden_states, return_final_state=True)
    torch.testing.assert_close(actual.final_state, expected.final_state, rtol=2e-2, atol=2e-2)
    expected = expected.hidden_states
    actual = actual.hidden_states
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, reference_param), fused_param in zip(
        reference.named_parameters(), fused.parameters(), strict=True
    ):
        assert fused_param.grad is not None, f"missing {name} gradient"
        torch.testing.assert_close(fused_param.grad, reference_param.grad, rtol=5e-2, atol=5e-2)


def test_rejects_indivisible_head_groups():
    with pytest.raises(ValueError, match="multiple of num_key_heads"):
        GDNAttention(128, 5, 2, 32)
