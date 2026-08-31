"""Tests for the indexer_loss function in examples/compressed_sparse_attention.py.

Covers:
- Multi-head end-to-end computation (einsum rank fix).
- Validity masking: invalid selections are excluded from KL divergence.
- Early positions where no blocks are causally complete (all-invalid rows).
- Gradient flow through selected_indexer_logits for valid positions.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


def _load_csa_example():
    example_path = (
        Path(__file__).resolve().parents[1] / "examples" / "compressed_sparse_attention.py"
    )
    spec = spec_from_file_location("_compressed_sparse_attention_example", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load example from {example_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csa_example = _load_csa_example()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexer_loss_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_selected: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
):
    """Build synthetic inputs for indexer_loss with correct shapes."""
    if device is None:
        device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(42)

    main_query = torch.randn(
        batch_size, num_heads, seq_len, head_dim, generator=gen, dtype=dtype, device=device
    )
    selected_compressed_kv = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        num_selected,
        head_dim,
        generator=gen,
        dtype=dtype,
        device=device,
    )
    attention_lse = torch.randn(
        batch_size, num_heads, seq_len, generator=gen, dtype=dtype, device=device
    )
    selected_indexer_logits = torch.randn(
        batch_size, seq_len, num_selected, generator=gen, dtype=dtype, device=device
    )
    selected_is_valid = torch.ones(
        batch_size, seq_len, num_selected, dtype=torch.bool, device=device
    )
    return (
        main_query,
        selected_compressed_kv,
        attention_lse,
        selected_indexer_logits,
        selected_is_valid,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 4, 8])
def test_multi_head_produces_finite_scalar(num_heads):
    """The einsum must handle the 5-D selected_compressed_kv across multiple heads."""
    inputs = _make_indexer_loss_inputs(
        batch_size=2, num_heads=num_heads, seq_len=16, head_dim=32, num_selected=4
    )
    loss = csa_example.indexer_loss(*inputs)

    assert loss.shape == ()
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss >= 0, f"KL divergence should be non-negative, got {loss.item()}"


def test_multi_head_matches_loop_over_heads():
    """Cross-check the vectorised einsum against a per-head loop."""
    B, H, S, D, K = 1, 4, 8, 16, 3
    inputs = _make_indexer_loss_inputs(
        batch_size=B, num_heads=H, seq_len=S, head_dim=D, num_selected=K
    )
    main_query, selected_compressed_kv, attention_lse, sel_logits, _sel_valid = inputs

    vectorised = csa_example.indexer_loss(*inputs)

    softmax_scale = D**-0.5
    eps = torch.finfo(torch.float32).tiny
    per_head_logits = (
        torch.stack(
            [
                torch.einsum(
                    "bsd,bskd->bsk",
                    main_query[:, h],
                    selected_compressed_kv[:, h],
                )
                for h in range(H)
            ],
            dim=1,
        )
        * softmax_scale
    )

    full_lse = attention_lse.float()
    per_head_probs = torch.exp(per_head_logits - full_lse[..., None])
    teacher_mass = per_head_probs.sum(dim=1)
    teacher_probs = teacher_mass / teacher_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    indexer_probs = torch.softmax(sel_logits.float(), dim=-1)
    kl = (
        teacher_probs * (teacher_probs.clamp_min(eps).log() - indexer_probs.clamp_min(eps).log())
    ).sum(dim=-1)
    expected = kl.mean()

    torch.testing.assert_close(vectorised, expected, atol=1e-10, rtol=1e-6)


def test_invalid_selections_excluded():
    """Marking some selections invalid should change the loss and keep it finite."""
    inputs_all_valid = _make_indexer_loss_inputs(
        batch_size=1, num_heads=2, seq_len=8, head_dim=16, num_selected=4
    )
    loss_all_valid = csa_example.indexer_loss(*inputs_all_valid)

    mq, skv, lse, sil, siv = inputs_all_valid
    siv_partial = siv.clone()
    siv_partial[:, :, -1] = False
    loss_partial = csa_example.indexer_loss(mq, skv, lse, sil, siv_partial)

    assert torch.isfinite(loss_partial)
    assert loss_partial >= 0
    assert not torch.allclose(loss_all_valid, loss_partial), (
        "Masking out a selection should change the loss value"
    )


def test_all_invalid_returns_zero():
    """When every selection is invalid (early positions), loss should be zero."""
    inputs = _make_indexer_loss_inputs(
        batch_size=2, num_heads=4, seq_len=8, head_dim=16, num_selected=3
    )
    mq, skv, lse, sil, _ = inputs
    all_invalid = torch.zeros_like(inputs[-1])

    loss = csa_example.indexer_loss(mq, skv, lse, sil, all_invalid)

    assert torch.isfinite(loss), f"all-invalid loss should be finite, got {loss}"
    assert loss.item() == 0.0, f"all-invalid loss should be exactly 0, got {loss.item()}"


def test_early_positions_no_completed_blocks():
    """Simulate compression_rate=4: the first 3 query positions have no valid blocks.

    The loss should be finite and only reflect valid (later) positions.
    """
    B, H, S, D, K = 1, 2, 8, 16, 2
    compression_rate = 4
    inputs = _make_indexer_loss_inputs(
        batch_size=B, num_heads=H, seq_len=S, head_dim=D, num_selected=K
    )
    mq, skv, lse, sil, siv = inputs
    siv = siv.clone()
    siv[:, : compression_rate - 1, :] = False

    loss = csa_example.indexer_loss(mq, skv, lse, sil, siv)

    assert torch.isfinite(loss), f"loss with early invalid rows should be finite, got {loss}"
    assert loss >= 0


def test_gradient_flows_through_valid_positions():
    """selected_indexer_logits should receive gradients only at valid positions."""
    B, H, S, D, K = 1, 2, 6, 8, 3
    inputs = _make_indexer_loss_inputs(
        batch_size=B, num_heads=H, seq_len=S, head_dim=D, num_selected=K
    )
    mq, skv, lse, sil, siv = inputs
    sil = sil.clone().requires_grad_(True)
    siv = siv.clone()
    siv[:, 0, :] = False

    loss = csa_example.indexer_loss(mq, skv, lse, sil, siv)
    loss.backward()

    assert sil.grad is not None, "No gradient on selected_indexer_logits"
    assert (sil.grad[:, 0, :] == 0).all(), "Invalid row (position 0) should have zero gradient"
    valid_grad = sil.grad[:, 1:, :]
    assert valid_grad.abs().sum() > 0, "Valid positions should have non-zero gradients"
