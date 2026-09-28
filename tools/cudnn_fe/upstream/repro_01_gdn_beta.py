"""Reproduce GDN's lost beta-zero gradient using only the upstream frontend API.

Run with a Blackwell GPU and nvidia-cudnn-frontend 1.30.0 (no Attention Gym imports):
    python repro_01_gdn_beta.py
Optional tiny-beta sweep:
    python repro_01_gdn_beta.py --beta 0 1e-12 1e-10 1e-8
The default recompute path precomputes beta-scaled factors (tinv_source="gmem").
Use --checkpoints to supply the known entering checkpoint and exercise "compute".

Two tokens have q=k=e_0, v=(e_0, 2*e_0), log-gate=0, beta=(1,b).
With loss=o[1,0,0], the scalar recurrence gives loss=1+b and dBeta[1]=1,
including b=0. Every gradient is compared exactly to a CPU FP64 recurrence.
The script deliberately chooses postactivation beta and disables QK normalization.
"""

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path

import cudnn
import torch


def reference_gradients(
    inputs: tuple[torch.Tensor, ...], do: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    q, k, value, gate, beta = [x.cpu().double().requires_grad_() for x in inputs]
    state = torch.zeros(1, 128, 128, dtype=torch.float64)
    outputs = []
    for token in range(2):
        decayed = gate[token].exp()[:, None, None] * state
        residual = value[token] - torch.einsum("hk,hkv->hv", k[token], decayed)
        state = decayed + beta[token, :, None, None] * torch.einsum(
            "hk,hv->hkv", k[token], residual
        )
        outputs.append(torch.einsum("hk,hkv->hv", q[token], state))
    loss = (torch.stack(outputs) * do.cpu().double()).sum()
    return torch.autograd.grad(loss, (q, k, value, gate, beta))


def run_case(dtype: torch.dtype, beta_value: float, checkpoints: bool) -> list[str]:
    q = torch.zeros((2, 1, 128), device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    value = torch.zeros_like(q)
    q[..., 0] = 1
    k[..., 0] = 1
    value[0, 0, 0] = 1
    value[1, 0, 0] = 2
    gate = torch.zeros((2, 1), device="cuda", dtype=torch.float32)
    beta = torch.tensor([[1.0], [beta_value]], device="cuda", dtype=torch.float32)
    do = torch.zeros_like(q)
    do[1, 0, 0] = 1
    cu = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    inputs = (q, k, value, gate, beta)

    graph = cudnn.pygraph()
    io_dtype = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph_dtypes = (io_dtype, io_dtype, io_dtype, cudnn.data_type.FLOAT, cudnn.data_type.FLOAT)
    ports = [
        graph.tensor(list(x.shape), data_type=dt, name=name)
        for x, dt, name in zip(inputs, graph_dtypes, ("q", "k", "v", "g", "beta"), strict=True)
    ]
    cu_port = graph.tensor([2], data_type=cudnn.data_type.INT32, name="cu_seqlens")
    do_port = graph.tensor([2, 1, 128], data_type=io_dtype, name="dO")
    checkpoint_port = (
        graph.tensor([1, 1, 128, 128], data_type=io_dtype, name="state_checkpoints")
        if checkpoints
        else None
    )
    grad_ports = graph.gdn_bwd(
        q=ports[0],
        k=ports[1],
        v=ports[2],
        g=ports[3],
        beta=ports[4],
        cu_seqlens=cu_port,
        dO=do_port,
        state_checkpoints=checkpoint_port,
        scale=1.0,
        use_beta_sigmoid=False,
        use_qk_l2norm=False,
        name="gdn_beta_zero",
    )[:5]
    for port, dt in zip(grad_ports, graph_dtypes, strict=True):
        port.set_output(True).set_data_type(dt)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    plans = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    graph.select_plan(plans.index("gdn_frost"))
    graph.check_support()
    graph.build_plans()
    actual = tuple(torch.empty_like(x) for x in inputs)
    pack = dict(zip(ports, inputs, strict=True))
    pack.update(zip(grad_ports, actual, strict=True))
    pack.update({cu_port: cu, do_port: do})
    if checkpoints:
        # One chunk, no initial state: its entering checkpoint is exactly zero.
        pack[checkpoint_port] = torch.zeros((1, 1, 128, 128), device="cuda", dtype=dtype)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    print(
        f"RUN dtype={dtype} beta={beta_value:g} checkpoints={checkpoints} plan=gdn_frost",
        flush=True,
    )
    graph.execute(pack, workspace)
    torch.cuda.synchronize()

    expected = reference_gradients(inputs, do)
    failures = []
    for name, out, ref in zip(("dQ", "dK", "dV", "dGate", "dBeta"), actual, expected, strict=True):
        out = out.cpu()
        target = ref.to(out.dtype)
        exact = torch.equal(out, target)
        error = (out.double() - target.double()).abs().max().item()
        print(f"  {name}: exact={exact} max_abs_error={error:g}", flush=True)
        if not exact:
            failures.append(f"{dtype}, beta={beta_value:g}, {name}")
    print(
        f"  dBeta[1]={actual[4][1, 0].item():.10g}; expected={expected[4][1, 0].item():.10g}",
        flush=True,
    )
    assert expected[4][1, 0].item() == 1.0
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--checkpoints", action="store_true", help="supply the known zero entering checkpoint"
    )
    args = parser.parse_args()
    print(f"frontend={version('nvidia-cudnn-frontend')} cudnn={cudnn.__file__}", flush=True)
    print(
        f"torch={torch.__version__} CUDA={torch.version.cuda} cutlass={version('nvidia-cutlass-dsl')}",
        flush=True,
    )
    print(
        f"GPU={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}",
        flush=True,
    )
    source = Path(cudnn.__file__).parent / "linear_attention/frost/kernel/gdn_bprop_f16.py"
    print(f"kernel={source} sha256={hashlib.sha256(source.read_bytes()).hexdigest()}", flush=True)
    failures = []
    for dtype in (torch.bfloat16, torch.float16):
        for beta_value in args.beta:
            failures.extend(run_case(dtype, beta_value, args.checkpoints))
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("PASS: all five gradients match the FP64 reference exactly", flush=True)


if __name__ == "__main__":
    main()
