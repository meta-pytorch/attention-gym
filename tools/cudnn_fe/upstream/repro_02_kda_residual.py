# SPDX-License-Identifier: MIT
"""Exact KDA cancellation regression through the upstream cuDNN frontend API.

Requires SM100-family CUDA, PyTorch, cuDNN frontend 1.30.0 and CuTeDSL >= 4.7.
Run with the desired cudnn package on PYTHONPATH. No Attention Gym imports.

Tokens 0/1 write S[V0,K0]=4096 and S[V0,K1]=2. Token 16 reads
S @ (e0+e1)=4098 against V0=4096 with beta=1/4. The residual is -2,
so S[V0,K1] and outputs at tokens 16/32 must be 1.5. Rounding the
contraction to BF16/FP16 before subtracting incorrectly gives 2.0.
With dO16=dO32=64, dBeta16=-2*(64+64)=-256 and dQ32,K1=1.5*64=96.
The default backward recomputes checkpoints (none are saved by forward).

--scheme forces an internal forward schedule for branch coverage, without
changing arithmetic or replacing kernels. The default uncut case uses only
the public batch_invariant=True option. Non-uncut cases are forward-only;
run each scheme in a fresh process to avoid reusing cached frontend plans.
"""

import argparse
import contextlib
import importlib.metadata
import json
from unittest.mock import patch

import cudnn
import torch
from cudnn.linear_attention import kimi_delta_attention


def run_case(dtype: torch.dtype, with_state: bool, scheme: str) -> dict:
    q = torch.zeros(64, 1, 128, device="cuda", dtype=dtype)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    g = torch.zeros_like(q, dtype=torch.float32)
    beta = torch.zeros(64, 1, device="cuda", dtype=torch.float32)
    k[0, 0, 0] = k[1, 0, 1] = 1
    v[0, 0, 0], v[1, 0, 0] = 4096, 2
    beta[0, 0] = beta[1, 0] = 1
    q[16, 0, 1] = q[32, 0, 1] = 1
    k[16, 0, :2] = 1
    v[16, 0, 0] = 4096
    beta[16, 0] = 0.25
    cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    initial = torch.zeros(1, 1, 128, 128, device="cuda") if with_state else None
    if scheme == "uncut":
        q.requires_grad_()
        beta.requires_grad_()
    print(f"RUN dtype={dtype} state={with_state} scheme={scheme}", flush=True)
    with contextlib.ExitStack() as stack:
        if scheme != "uncut":
            from cudnn.linear_attention.frost import kda_engine
            from cudnn.linear_attention.frost.common import piece_chain

            stack.enter_context(
                patch.object(
                    piece_chain, "choose_pieces", return_value=(2 if scheme == "chain" else 0, 1)
                )
            )
            stack.enter_context(
                patch.object(piece_chain, "is_dv_split", return_value=scheme in ("dv", "prep"))
            )
            stack.enter_context(
                patch.object(kda_engine, "PREP_TILE_FRACTION", 1.0 if scheme == "prep" else 0.0)
            )
        output, _ = kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu,
            scale=1.0,
            initial_state=initial,
            batch_invariant=scheme == "uncut",
            plan_name="kda_frost",
        )
        actual = {"output16": output[16, 0, 0].item(), "output32": output[32, 0, 0].item()}
        expected = {"output16": 1.5, "output32": 1.5}
        if scheme == "uncut":
            d_output = torch.zeros_like(output)
            d_output[16, 0, 0] = d_output[32, 0, 0] = 64
            dq, dbeta = torch.autograd.grad(output, (q, beta), d_output)
            actual.update(dbeta16=dbeta[16, 0].item(), dq32=dq[32, 0, 1].item())
            expected.update(dbeta16=-256.0, dq32=96.0)
    result = {
        "dtype": str(dtype),
        "initial_state": with_state,
        "scheme": scheme,
        "actual": actual,
        "expected": expected,
        "passed": actual == expected,
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("uncut", "dv", "prep", "chain"), default="uncut")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "cudnn": cudnn.__version__,
                "cudnn_source": cudnn.__file__,
                "torch": torch.__version__,
                "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
                "triton": importlib.metadata.version("triton"),
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    assert torch.cuda.get_device_capability()[0] == 10, "requires SM100-family GPU"
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    selected = tuple(dtypes.values()) if args.dtype == "both" else (dtypes[args.dtype],)
    results = [
        run_case(dtype, with_state, args.scheme)
        for dtype in selected
        for with_state in (False, True)
    ]
    failed = sum(not result["passed"] for result in results)
    print(f"RESULT: {len(results) - failed} passed, {failed} failed", flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
