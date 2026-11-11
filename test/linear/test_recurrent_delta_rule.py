# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for the shared recurrent delta-rule scan."""

import itertools

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym._backends import tune as tune_module
from attn_gym.linear import recurrent_gdn, recurrent_kda
from attn_gym.linear._delta_rule import recurrent as recurrent_module
from attn_gym.linear._delta_rule.recurrent import GateKind, launch_recurrent_delta_rule_fwd
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused recurrent scan requires CUDA"
)


def _make_inputs(
    *,
    tokens: int,
    dtype: torch.dtype,
    initial_state: bool,
    key_dim: int = 64,
    value_dim: int = 48,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Create stable scalar-gated inputs in the public GDN layout."""
    torch.manual_seed(0)
    batch, heads = 2, 2
    query = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, tokens, heads, value_dim, device="cuda")
    gate = F.logsigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, value_dim, key_dim, device="cuda") if initial_state else None
    return query.to(dtype), key.to(dtype), value.to(dtype), gate, beta, state


def _launch_scalar_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    store_final_state: bool,
    batch_invariant: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the scalar-gate specialization with public GDN inputs."""
    output, final_state = launch_recurrent_delta_rule_fwd(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        None,
        scale=scale,
        gate_kind=GateKind.SCALAR,
        store_final_state=store_final_state,
        batch_invariant=batch_invariant,
    )
    return output, final_state


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize(
    "dtype,tokens,use_initial_state,return_final_state,scale,key_dim,value_dim",
    [
        (torch.float32, 37, True, True, 0.37, 64, 48),
        (torch.float32, 3, True, True, None, 40, 37),
        (torch.float32, 5, False, False, None, 64, 48),
        (torch.bfloat16, 37, True, True, None, 64, 48),
    ],
)
def test_scalar_gate_matches_recurrent_gdn(
    dtype,
    tokens,
    use_initial_state,
    return_final_state,
    scale,
    key_dim,
    value_dim,
    batch_invariant,
):
    query, key, value, gate, beta, state = _make_inputs(
        tokens=tokens,
        dtype=dtype,
        initial_state=use_initial_state,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    kernel_scale = query.shape[-1] ** -0.5 if scale is None else scale
    with torch.no_grad():
        expected = recurrent_gdn(
            query,
            key,
            value,
            gate,
            beta,
            state,
            scale=scale,
            output_final_state=return_final_state,
            impl="reference",
        )
        output, final_state = _launch_scalar_gdn(
            query,
            key,
            value,
            gate,
            beta,
            state,
            scale=kernel_scale,
            store_final_state=return_final_state,
            batch_invariant=batch_invariant,
        )

    assert output.dtype == dtype
    if return_final_state:
        assert final_state is not None and final_state.dtype == torch.float32
    else:
        assert final_state is None and expected[1] is None
    if dtype == torch.bfloat16:
        high_precision = recurrent_gdn(
            query.double(),
            key.double(),
            value.double(),
            gate.double(),
            beta.double(),
            scale=scale,
            initial_state=None if state is None else state.double(),
            output_final_state=return_final_state,
            impl="reference",
        )
        assert_matches_low_precision_reference(output, high_precision[0], expected[0], "output")
        if return_final_state:
            assert_matches_low_precision_reference(
                final_state, high_precision[1], expected[1], "final_state"
            )
    else:
        torch.testing.assert_close(output, expected[0], rtol=1e-5, atol=1e-5)
        if return_final_state:
            torch.testing.assert_close(final_state, expected[1], rtol=1e-5, atol=1e-5)


def test_scalar_and_vector_gate_specializations_agree():
    query, key, value, gate, beta, state = _make_inputs(
        tokens=37, dtype=torch.float32, initial_state=True
    )
    scale = query.shape[-1] ** -0.5
    with torch.no_grad():
        scalar_output, scalar_state = _launch_scalar_gdn(
            query, key, value, gate, beta, state, scale=scale, store_final_state=True,
            batch_invariant=False,
        )  # fmt: skip
        vector_gate = gate.unsqueeze(-1).expand(*gate.shape, query.shape[-1])
        vector_output, vector_state = launch_recurrent_delta_rule_fwd(
            query,
            key,
            value,
            vector_gate.contiguous(),
            beta,
            state,
            None,
            scale=scale,
            gate_kind=GateKind.VECTOR,
            store_final_state=True,
            batch_invariant=False,
        )

    torch.testing.assert_close(scalar_output, vector_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scalar_state, vector_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("gate_kind", [GateKind.SCALAR, GateKind.VECTOR])
@pytest.mark.parametrize(
    "batch,tokens,key_dim",
    [
        (1, 5, 128),  # BV=2 one-warp tiles
        (2, 5, 128),  # BV=4
        (5, 5, 128),  # BV=16 under the register cap
        (5, 5, 192),  # padded keys at the K > 128 cap (BV=8)
        (5, 1, 256),  # one-token one-warp tile (BV=4)
    ],
)
def test_wide_key_fixed_tiles_match_reference(gate_kind, batch, tokens, key_dim, batch_invariant):
    """Each fixed one-warp tile for K >= 128 matches the eager recurrence, in both arithmetic
    modes."""
    torch.manual_seed(0)
    heads, value_dim = 16, 128
    query = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, tokens, heads, value_dim, device="cuda")
    gate_shape = (batch, tokens, heads) + ((key_dim,) if gate_kind is GateKind.VECTOR else ())
    gate = F.logsigmoid(torch.randn(gate_shape, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, value_dim, key_dim, device="cuda")
    scale = key_dim**-0.5
    with torch.no_grad():
        output, final_state = launch_recurrent_delta_rule_fwd(
            query,
            key,
            value,
            gate,
            beta,
            state,
            None,
            scale=scale,
            gate_kind=gate_kind,
            store_final_state=True,
            batch_invariant=batch_invariant,
        )
        reference = recurrent_kda if gate_kind is GateKind.VECTOR else recurrent_gdn
        expected = reference(
            query,
            key,
            value,
            gate,
            beta,
            state,
            scale=scale,
            output_final_state=True,
            impl="reference",
        )

    torch.testing.assert_close(output, expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, expected[1], rtol=1e-5, atol=1e-5)


def test_recurrent_reuses_kernel_across_token_counts(triton_compiles):
    """Aligned and unaligned token counts launch the scan compiled for the first one."""
    # Single-token scans select a different prefetch policy, so they are not part of this set.
    for tokens in (37, 16, 48):
        inputs = _make_inputs(tokens=tokens, dtype=torch.bfloat16, initial_state=False)
        recurrent_gdn(*inputs[:5], kernel_options={"autotune": False})
    torch.cuda.synchronize()
    assert triton_compiles["_recurrent_delta_rule_fwd_kernel"] == 1


def test_recurrent_reuses_tuning_across_token_counts(isolated_winners, monkeypatch):
    """Scans of 64 or more tokens at one sequence count share one autotuning result, which a
    single-token scan does not reuse."""
    tune = tune_module.tune
    sweeps = []

    def counting_tune(*args, **kwargs):
        sweeps.append(None)
        return tune(*args, **kwargs)

    monkeypatch.setattr(tune_module, "tune", counting_tune)
    for tokens in (64, 91, 160):
        inputs = _make_inputs(tokens=tokens, dtype=torch.bfloat16, initial_state=False)
        recurrent_gdn(*inputs[:5], kernel_options={"autotune": True})
    torch.cuda.synchronize()
    assert len(sweeps) == 1
    inputs = _make_inputs(tokens=1, dtype=torch.bfloat16, initial_state=False)
    recurrent_gdn(*inputs[:5], kernel_options={"autotune": True})
    assert len(sweeps) == 2


def test_paged_launch_never_benchmarks(no_benchmarking):
    """A paged launch asked to autotune is pinned: each candidate would advance the pool."""
    query, key, value, gate, beta, _ = _make_inputs(
        tokens=5, dtype=torch.bfloat16, initial_state=False
    )
    pool = torch.randn(3, value.shape[2], value.shape[3], query.shape[3], device="cuda")
    slots = torch.tensor([2, 1], device="cuda", dtype=torch.int32)

    def launch(autotune: bool) -> tuple[torch.Tensor, torch.Tensor]:
        advanced = pool.clone()
        output, _ = launch_recurrent_delta_rule_fwd(
            query, key, value, gate, beta, advanced, None,
            scale=1.0,
            gate_kind=GateKind.SCALAR,
            store_final_state=True,
            state_indices=slots,
            autotune=autotune,
            batch_invariant=False,
        )  # fmt: skip
        return output, advanced

    for tuned, pinned in zip(launch(True), launch(False)):
        assert torch.equal(tuned, pinned)


def test_tuning_key_determines_pinned_configuration():
    """Shapes that share a tuning key pin the same configuration.

    ``TritonTuner`` remembers the pinned candidate per key, so a key coarser than the shape
    heuristic would hand one shape another's configuration.
    """
    operand = torch.empty(0, dtype=torch.bfloat16)
    pinned_by_key: dict[tuple, set] = {}
    shapes = itertools.product(
        (*range(70), 128, 4096), (1, 2, 16, 64, 200), (64, 128, 256), (1, 48, 128), (False, True)
    )
    for tokens, num_sequences, key_dim, value_dim, is_varlen in shapes:
        args = {
            "T": tokens * num_sequences if is_varlen else tokens,
            "N": num_sequences,
            "H": 16,
            "HK": 16,
            "K": key_dim,
            "V": value_dim,
            "USE_INITIAL_STATE": False,
            "STORE_FINAL_STATE": False,
            "IS_VARLEN": is_varlen,
            "SCALAR_GATE": True,
            "PREFETCH": False,
            "BATCH_INVARIANT": False,
            **dict.fromkeys(("q", "k", "v", "gate", "beta"), operand),
            "h0": None,
            "cu_seqlens": operand if is_varlen else None,
        }
        pinned = recurrent_module._recurrent_scan.configs(None, args)[0]
        pinned_by_key.setdefault(recurrent_module._recurrent_tuning_key(args), set()).add(pinned)
    assert all(len(pinned) == 1 for pinned in pinned_by_key.values())


def _batch_invariant_paged_scan(scan, q, k, v, gate, beta, lens, pool, slots, fresh):
    """Run ``scan`` with ``batch_invariant`` over packed sequences of ``lens`` tokens, reading and
    writing ``pool[slots]``."""
    cu_seqlens = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda")
    output, _ = scan(
        q, k, v, gate, beta, pool,
        cu_seqlens=cu_seqlens.to(torch.int32),
        state_indices=slots,
        has_initial_state=~fresh,
        kernel_options={"batch_invariant": True, "autotune": False},
    )  # fmt: skip
    return output


@pytest.mark.parametrize("gate_kind", [GateKind.SCALAR, GateKind.VECTOR])
def test_batch_invariant_bits_do_not_depend_on_batch_shape(gate_kind):
    """With ``batch_invariant``, a sequence gets the same output and state bits alone, in a padded
    batch, and decoded.

    The launcher picks BV, num_warps and PREFETCH from the batch shape. Batch-invariant
    serving needs the bits to stay fixed anyway: a trainer scores one packed sequence while the
    server prefills it among empty slots and then decodes it one token per step.
    """
    torch.manual_seed(0)
    tokens, heads, dim, padding = 150, 16, 128, 64
    scan = recurrent_gdn if gate_kind is GateKind.SCALAR else recurrent_kda
    q = F.normalize(torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
    gate_shape = (1, tokens, heads) if gate_kind is GateKind.SCALAR else (1, tokens, heads, dim)
    gate = F.logsigmoid(torch.randn(gate_shape, device="cuda"))
    beta = torch.rand(1, tokens, heads, device="cuda").bfloat16()

    def run(lens, slots):
        pool = torch.zeros(len(lens) + 1, heads, dim, dim, device="cuda")
        slots = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fresh = torch.ones(len(lens), dtype=torch.bool, device="cuda")
        output = _batch_invariant_paged_scan(scan, q, k, v, gate, beta, lens, pool, slots, fresh)
        return output, pool[1]

    alone_output, alone_state = run([tokens], [1])
    padded_output, padded_state = run([tokens] + [0] * padding, [1] + [0] * padding)
    assert torch.equal(padded_output, alone_output)
    assert torch.equal(padded_state, alone_state)

    # Decode: each step passes this sequence's next token plus `padding` empty rows.
    pool = torch.zeros(2, heads, dim, dim, device="cuda")
    slots = torch.tensor([1] + [0] * padding, dtype=torch.int32, device="cuda")
    for step in range(tokens):
        fresh = torch.zeros(padding + 1, dtype=torch.bool, device="cuda")
        fresh[0] = step == 0
        token = slice(step, step + 1)
        output = _batch_invariant_paged_scan(
            scan, q[:, token], k[:, token], v[:, token], gate[:, token], beta[:, token],
            [1] + [0] * padding, pool, slots, fresh,
        )  # fmt: skip
        assert torch.equal(output, alone_output[:, token]), step
    assert torch.equal(pool[1], alone_state)

    # Unpaged with autotuning: whichever config the autotuner picks gives the same bits.
    output, _ = scan(q, k, v, gate, beta, kernel_options={"batch_invariant": True})
    assert torch.equal(output, alone_output)


@pytest.mark.parametrize("gate_kind", [GateKind.SCALAR, GateKind.VECTOR])
def test_batch_invariant_bits_do_not_depend_on_launch_config(monkeypatch, gate_kind):
    """With ``batch_invariant``, every (BV, num_warps, PREFETCH) the launcher can pick gives the
    same output and state bits."""
    kernel = recurrent_module._recurrent_delta_rule_fwd_kernel
    forced = {}

    class ForcedConfig:
        def __getitem__(self, grid):
            return lambda *args, autotune, **kwargs: kernel[grid](*args, **{**kwargs, **forced})

    monkeypatch.setattr(recurrent_module, "_recurrent_scan", ForcedConfig())
    torch.manual_seed(0)
    tokens, heads, dim = 64, 4, 128
    q = F.normalize(torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, tokens, heads, dim, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
    gate_shape = (1, tokens, heads) if gate_kind is GateKind.SCALAR else (1, tokens, heads, dim)
    gate = F.logsigmoid(torch.randn(gate_shape, device="cuda"))
    beta = torch.rand(1, tokens, heads, device="cuda").bfloat16()
    scan = recurrent_gdn if gate_kind is GateKind.SCALAR else recurrent_kda

    configs = list(itertools.product((2, 4, 8, 16, 32), (1, 2, 4), (False, True)))
    results = []
    for block_v, num_warps, prefetch in configs:
        forced.update(BV=block_v, num_warps=num_warps, PREFETCH=prefetch)
        results.append(
            scan(
                q, k, v, gate, beta,
                output_final_state=True,
                kernel_options={"batch_invariant": True, "autotune": False},
            )
        )  # fmt: skip
    for config, (output, state) in zip(configs[1:], results[1:]):
        assert torch.equal(output, results[0][0]), config
        assert torch.equal(state, results[0][1]), config
