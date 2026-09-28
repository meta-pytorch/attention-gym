"""Public-API KDA multi-wave smoke for derived scheduler arrival counts.

Default legal role maps have the same counts before and after. Both must pass;
this does not establish a current upstream bug or validate alternative role maps.
"""

import argparse

import cudnn
import torch
from cudnn.linear_attention import kimi_delta_attention

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=32)
args = parser.parse_args()
print("frontend:", cudnn.__file__, "GPU:", torch.cuda.get_device_name(), flush=True)
sequences, length, heads, dim = 32, 128, 32, 128
q = torch.zeros(sequences * length, heads, dim, device="cuda", dtype=torch.bfloat16)
k = torch.zeros_like(q)
v = torch.zeros_like(q)
q[..., 0] = 1
k[..., 0] = 1
v[..., 0] = 1
gate = torch.zeros(sequences * length, heads, dim, device="cuda")
beta = torch.ones(sequences * length, heads, device="cuda")
state = torch.zeros(sequences, heads, dim, dim, device="cuda")
cu = torch.arange(sequences + 1, device="cuda", dtype=torch.int32) * length
inputs = tuple(x.requires_grad_() for x in (q, k, v, gate, beta, state))
for step in range(args.iterations):
    out, final = kimi_delta_attention(
        q,
        k,
        v,
        gate,
        beta,
        cu,
        scale=1.0,
        initial_state=state,
        output_final_state=True,
        batch_invariant=True,
        plan_name="kda_frost",
    )
    torch.testing.assert_close(out, v, atol=0, rtol=0)
    expected_final = torch.zeros_like(final)
    expected_final[..., 0, 0] = 1
    torch.testing.assert_close(final, expected_final, atol=0, rtol=0)
    grads = torch.autograd.grad(out, inputs, torch.ones_like(out))
    assert all(bool(torch.isfinite(grad).all()) for grad in grads)
    torch.cuda.synchronize()
    if step % 8 == 0 or step + 1 == args.iterations:
        print(
            f"iteration {step + 1}/{args.iterations}: finite gradients, exact output/state",
            flush=True,
        )
print("PASS (stress only; no isolated failure-before claim)")
