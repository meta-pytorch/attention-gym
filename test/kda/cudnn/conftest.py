"""Compare ownership changes within a fixed plan and across automatic KDA plans."""

from dataclasses import replace

import pytest
import torch


@pytest.fixture(params=["uncut", "automatic"])
def kda_plan_comparison(request, monkeypatch):
    """Keep the bitwise uncut contract; allow <1% relative L2 between different plans.

    Prep and piece-chain plans change rounding order and are selected from shape.
    Their independent forward/gradient correctness is covered by the FP64 reference tests.
    """
    from attn_gym.linear._delta_rule.cudnn_fe import kda

    pinned = request.param == "uncut"
    if pinned:
        forward = kda.ForwardPlan.build
        backward = kda.BackwardPlan.build
        monkeypatch.setattr(
            kda.ForwardPlan,
            "build",
            lambda *args: replace(forward(*args), pieces=0, tiles_per_head=1),
        )
        monkeypatch.setattr(
            kda.BackwardPlan, "build", lambda *args: replace(backward(*args), pieces=0)
        )

    def compare(actual: torch.Tensor, expected: torch.Tensor, name: str = "result") -> None:
        if pinned:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
        else:
            error = torch.linalg.vector_norm(actual.float() - expected.float())
            norm = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
            assert error / norm < 1e-2, f"{name}: relative L2 {float(error / norm):.5g} >= 1e-2"

    return compare
