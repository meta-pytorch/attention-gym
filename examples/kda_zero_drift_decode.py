"""Match KDA training numerics during decode by replaying partial chunks.

Token-at-a-time ``recurrent_kda`` groups floating-point additions differently than 64-row
``chunk_kda``, and the difference compounds in the FP32 recurrent state. Replay uses
``chunk_kda`` on the current partial chunk from the last FP32 boundary state. Determinism,
prefix invariance, and FP32 boundary chaining make its output bitwise-identical to training.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from attn_gym.linear import bounded_gate_cumsum, chunk_kda, recurrent_kda

CHUNK_SIZE = 64
TOKENS = 256
HEADS = 2
HEAD_DIM = 128  # Required by fused chunk_kda.


@dataclass
class Inputs:
    """KDA inputs and the parameters used to build their gates."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    beta: torch.Tensor

    def slice(self, start: int, stop: int) -> Inputs:
        """Return the token range needed for one replay step."""
        return Inputs(
            self.q[:, start:stop],
            self.k[:, start:stop],
            self.v[:, start:stop],
            self.raw_gate[:, start:stop],
            self.a_log,
            self.dt_bias,
            self.beta[:, start:stop],
        )


def make_inputs() -> Inputs:
    """Create a fixed workload so the example is repeatable."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    shape = (1, TOKENS, HEADS, HEAD_DIM)
    return Inputs(
        q=randn(*shape),
        k=randn(*shape),
        v=randn(*shape),
        raw_gate=randn(*shape),
        a_log=torch.zeros(HEADS, device="cuda"),
        dt_bias=torch.zeros(HEADS, HEAD_DIM, device="cuda"),
        beta=torch.rand(
            1, TOKENS, HEADS, device="cuda", dtype=torch.bfloat16, generator=generator
        ),
    )


def run_chunk(
    inputs: Inputs,
    impl: str,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the training kernel after rebuilding gates for this partial chunk."""
    cumulative_gate = bounded_gate_cumsum(
        inputs.raw_gate, inputs.a_log, inputs.dt_bias, chunk_size=CHUNK_SIZE
    )
    return chunk_kda(
        inputs.q,
        inputs.k,
        inputs.v,
        cumulative_gate,
        inputs.beta,
        initial_state,
        output_final_state=output_final_state,
        autotune=False,  # Fixed heuristics keep kernel selection repeatable.
        impl=impl,
    )


def replay_decode(inputs: Inputs, impl: str) -> torch.Tensor:
    """Decode each token by replaying its partial chunk from the last boundary."""
    boundary_state: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []

    for token in range(inputs.q.shape[1]):
        chunk_start = (token // CHUNK_SIZE) * CHUNK_SIZE
        completes_chunk = (token + 1) % CHUNK_SIZE == 0
        output, final_state = run_chunk(
            inputs.slice(chunk_start, token + 1),
            impl,
            initial_state=boundary_state,
            output_final_state=completes_chunk,
        )
        outputs.append(output[:, -1:])

        if completes_chunk:
            assert final_state is not None
            boundary_state = final_state  # Keep the exact FP32 state produced by training.

    return torch.cat(outputs, dim=1)


def main() -> None:
    """Compare replay and recurrent decode with one chunked training pass."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--impl", choices=["auto", "fused", "reference"], default="auto")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    impl = args.impl
    if impl == "auto":
        impl = "fused" if torch.cuda.get_device_capability()[0] >= 10 else "reference"

    inputs = make_inputs()
    training_output, _ = run_chunk(inputs, impl)
    replay_output = replay_decode(inputs, impl)

    per_token_gate = bounded_gate_cumsum(
        inputs.raw_gate, inputs.a_log, inputs.dt_bias, chunk_size=1
    )
    recurrent_output, _ = recurrent_kda(
        inputs.q, inputs.k, inputs.v, per_token_gate, inputs.beta, impl=impl
    )

    replay_matches = torch.equal(replay_output, training_output)
    recurrent_drifts = not torch.equal(recurrent_output, training_output)
    max_drift = (recurrent_output.float() - training_output.float()).abs().max().item()

    print(f"impl: {impl}")
    print(f"replay decode matches training bitwise: {replay_matches}")
    print(f"recurrent baseline drifts: {recurrent_drifts} (max |difference|: {max_drift:.3e})")


if __name__ == "__main__":
    main()
