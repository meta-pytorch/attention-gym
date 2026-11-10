"""Bitwise parity of the scheduler-driven short-convolution kernels with their previous versions.

The port of decode, the paged state update and the initial-state gradient changes only how a CTA
finds its (channel block, row, sequence) item, so outputs and mutated state must match the legacy
kernels bit for bit on both grids. Removed together with ``cute_legacy.py``.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.short_conv import cute as cute_backend
from attn_gym.linear.short_conv import cute_legacy
from attn_gym.linear.short_conv.cute import (
    ShortConvConfig,
    causal_conv1d,
    causal_conv1d_decode,
    paged_causal_conv1d,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the CuTeDSL short convolution requires CUDA capability 8.0 or newer",
)

# Position of ``grid`` in each ported compile function; the legacy ones take the rest.
_GRID_ARGUMENT = {
    "_compile_decode": 7,
    "_compile_paged_state_update": 7,
    "_compile_initial_state_gradient": 8,
}


def _use_legacy(monkeypatch) -> list[str]:
    """Route the three launchers to the legacy kernels; returns the log of legacy compiles."""
    used = []
    for name, position in _GRID_ARGUMENT.items():
        legacy = getattr(cute_legacy, name)

        def compile_legacy(*args, _legacy=legacy, _position=position):
            used.append(_legacy.__name__)
            return _legacy(*args[:_position], *args[_position + 1 :])

        monkeypatch.setattr(cute_backend, name, compile_legacy)
    return used


def _compare(monkeypatch, run, grid: str, int64: bool) -> None:
    """``run`` returns tensors; the legacy kernels must reproduce the ported ones exactly."""
    if int64:
        monkeypatch.setattr(cute_backend, "requires_int64_abi", lambda *tensors: True)
    with monkeypatch.context() as patch:
        if grid == "fold_l":  # the variant the launcher only picks above 65535 sequences
            patch.setattr(cute_backend, "_sequence_grid", lambda *args, **kwargs: "fold_l")
        actual = run()
    used = _use_legacy(monkeypatch)
    expected = run()
    assert used, "the reference run did not reach a legacy kernel"
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


def _generator(*seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(hash(seed) % 2**31)


GRIDS = pytest.mark.parametrize("grid", ["tiles", "fold_l"])
DTYPES = pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])


@GRIDS
@DTYPES
@pytest.mark.parametrize(
    ("sequences", "channels", "width", "config"),
    [
        pytest.param(1, 6, 2, None, id="minimum"),
        pytest.param(5, 330, 4, ShortConvConfig(64, 2, 1), id="partial-channel-block"),
        pytest.param(37, 2048, 4, None, id="default-config"),
        pytest.param(65535, 12, 3, None, id="last-tile-grid-sequence"),
    ],
)
@pytest.mark.parametrize("routing", ["implicit", "paged", "paged-fresh"])
def test_decode_matches_legacy_bitwise(
    monkeypatch, grid, dtype, sequences, channels, width, config, routing
):
    generator = _generator(sequences, channels, width)
    slots = sequences if routing == "implicit" else sequences + 3

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    x, weight, state = (
        randn(sequences, channels),
        randn(channels, width),
        randn(slots, width - 1, channels),
    )
    indices = mask = None
    if routing != "implicit":
        indices = torch.randperm(slots - 1, device="cuda", generator=generator)[:sequences]
        indices = (indices + 1).to(torch.int32)
        indices[::7] = 0  # padding sequences write zeros and leave the pool alone
    if routing == "paged-fresh":
        mask = torch.arange(sequences, device="cuda") % 3 != 0
        state[indices[~mask].long()] = float("nan")

    def run():
        advanced = state.clone()
        output = causal_conv1d_decode(
            x,
            weight,
            advanced,
            activation="silu",
            state_indices=indices,
            has_initial_state=mask,
            forward_config=config,
        )
        return output, advanced

    _compare(monkeypatch, run, grid, int64=False)  # decode has no wide ABI


@GRIDS
@DTYPES
@pytest.mark.parametrize("int64", [False, True], ids=["int32", "int64"])
@pytest.mark.parametrize(
    ("boundaries", "batch", "tokens", "channels", "width"),
    [
        pytest.param(None, 1, 1, 6, 2, id="dense-minimum"),
        pytest.param(None, 5, 2, 330, 4, id="dense-shorter-than-history"),
        pytest.param(None, 3, 67, 512, 4, id="dense"),
        pytest.param((0, 0, 1, 3, 3, 40, 41), 1, 48, 330, 5, id="packed-short-and-empty"),
        pytest.param("unit", 1, 65535, 12, 3, id="packed-last-tile-grid-sequence"),
    ],
)
@pytest.mark.parametrize("with_mask", [False, True], ids=["resumed", "fresh-mask"])
def test_paged_state_update_matches_legacy_bitwise(
    monkeypatch, grid, dtype, int64, boundaries, batch, tokens, channels, width, with_mask
):
    generator = _generator(batch, tokens, channels, width)
    if boundaries == "unit":
        boundaries = range(tokens + 1)
    sequences = batch if boundaries is None else len(boundaries) - 1
    offsets = (
        None
        if boundaries is None
        else torch.tensor(list(boundaries), device="cuda", dtype=torch.int32)
    )
    slots = sequences + 3

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    x, weight, state = (
        randn(batch, tokens, channels),
        randn(channels, width),
        randn(slots, width - 1, channels),
    )
    indices = torch.randperm(slots - 1, device="cuda", generator=generator)[:sequences]
    indices = (indices + 1).to(torch.int32)
    indices[::7] = 0
    mask = torch.arange(sequences, device="cuda") % 3 != 0 if with_mask else None

    def run():
        advanced = state.clone()
        with torch.no_grad():
            output = paged_causal_conv1d(
                x,
                weight,
                advanced,
                indices,
                activation="silu",
                cu_seqlens=offsets,
                has_initial_state=mask,
            )
        active = tokens if offsets is None else int(offsets[-1])
        return output[:, :active], advanced

    _compare(monkeypatch, run, grid, int64)


@GRIDS
@DTYPES
@pytest.mark.parametrize("int64", [False, True], ids=["int32", "int64"])
@pytest.mark.parametrize(
    ("boundaries", "batch", "tokens", "channels", "width"),
    [
        pytest.param(None, 1, 1, 6, 2, id="dense-minimum"),
        pytest.param(None, 5, 2, 330, 4, id="dense-shorter-than-history"),
        pytest.param(None, 3, 67, 512, 6, id="dense-wide"),
        pytest.param((0, 0, 1, 3, 3, 40, 41), 1, 48, 330, 5, id="packed-short-and-empty"),
        pytest.param("unit", 1, 65535, 12, 3, id="packed-last-tile-grid-sequence"),
    ],
)
def test_initial_state_gradient_matches_legacy_bitwise(
    monkeypatch, grid, dtype, int64, boundaries, batch, tokens, channels, width
):
    generator = _generator(batch, tokens, channels, width)
    if boundaries == "unit":
        boundaries = range(tokens + 1)
    sequences = batch if boundaries is None else len(boundaries) - 1
    offsets = (
        None
        if boundaries is None
        else torch.tensor(list(boundaries), device="cuda", dtype=torch.int32)
    )

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    x, weight = randn(batch, tokens, channels), randn(channels, width)
    initial_state = randn(sequences, width - 1, channels).requires_grad_()
    incoming = randn(batch, tokens, channels)

    def run():
        output = causal_conv1d(
            x, weight, activation="silu", cu_seqlens=offsets, initial_state=initial_state
        )
        return torch.autograd.grad(output, initial_state, incoming)

    _compare(monkeypatch, run, grid, int64)
