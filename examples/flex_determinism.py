import torch
from torch.nn.attention import flex_attention
from tabulate import tabulate
import torch._inductor.config
import torch._inductor.utils
import warnings
from tqdm import tqdm
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore", message=".*dynamo_pgo force disabled.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.mkldnn.deterministic = True


@dataclass
class TestConfig:
    name: str
    B: int
    Hq: int
    Hkv: int
    Q: int
    KV: int
    Dqk: int
    Dv: int


def test_bitwise_determinism(
    dynamic, backend, mode, num_runs=3, pbar=None, kernel_options=None, inductor_config=None
):
    """Test bitwise determinism across multiple runs with various shapes for given compilation settings."""

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return []

    if kernel_options is None:
        kernel_options = {}

    if inductor_config is None:
        inductor_config = {}

    device = torch.device("cuda:0")

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    test_configs = [
        TestConfig("Standard", B=2, Hq=32, Hkv=32, Q=2048, KV=2048, Dqk=128, Dv=128),
        TestConfig("Decode-single", B=1, Hq=32, Hkv=32, Q=1, KV=2048, Dqk=128, Dv=128),
        TestConfig("Decode-small-batch", B=4, Hq=8, Hkv=8, Q=1, KV=1024, Dqk=64, Dv=64),
        TestConfig("GQA-decode", B=1, Hq=32, Hkv=8, Q=1, KV=4096, Dqk=128, Dv=128),
        TestConfig("Long-context", B=1, Hq=16, Hkv=16, Q=4096, KV=4096, Dqk=128, Dv=128),
    ]

    results = []

    for config in test_configs:
        if pbar is not None:
            pbar.set_postfix_str(f"Testing {config.name}")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        B, Hq, Hkv, Q, KV, Dqk, Dv = (
            config.B,
            config.Hq,
            config.Hkv,
            config.Q,
            config.KV,
            config.Dqk,
            config.Dv,
        )

        q = torch.randn(B, Hq, Q, Dqk, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, Hkv, KV, Dqk, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, Hkv, KV, Dv, device=device, dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn(B, Hq, Q, Dv, device=device, dtype=torch.bfloat16)
        with torch._inductor.config.patch({"deterministic": True}):
            block_mask = torch.compile(flex_attention.create_block_mask)(
                mask_mod=causal_mask,
                B=B,
                H=Hq,
                Q_LEN=Q,
                KV_LEN=KV,
                device=device,
            )

        def flex_forward(q, k, v, bm):
            return flex_attention.flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=bm,
                enable_gqa=q.size(1) != k.size(1),
                kernel_options=kernel_options,
            )

        fwd_deterministic = True
        bwd_q_deterministic = True
        bwd_k_deterministic = True
        bwd_v_deterministic = True

        first_fwd_output = None
        first_grad_q = None
        first_grad_k = None
        first_grad_v = None

        for run in range(num_runs):
            torch._dynamo.reset()

            with torch._inductor.utils.fresh_inductor_cache():
                compiled_fn = torch.compile(
                    flex_forward, fullgraph=True, dynamic=dynamic, backend=backend, mode=mode
                )

                output = compiled_fn(q, k, v, block_mask)

                grad_q, grad_k, grad_v = torch.autograd.grad(
                    output, inputs=[q, k, v], grad_outputs=[grad_out]
                )

            if run == 0:
                first_fwd_output = output.detach().clone()
                first_grad_q = grad_q.clone()
                first_grad_k = grad_k.clone()
                first_grad_v = grad_v.clone()
            else:
                if not torch.equal(output, first_fwd_output):
                    fwd_deterministic = False
                if not torch.equal(grad_q, first_grad_q):
                    bwd_q_deterministic = False
                if not torch.equal(grad_k, first_grad_k):
                    bwd_k_deterministic = False
                if not torch.equal(grad_v, first_grad_v):
                    bwd_v_deterministic = False

        bwd_deterministic = bwd_q_deterministic and bwd_k_deterministic and bwd_v_deterministic

        status_fwd = "✓ PASS" if fwd_deterministic else "✗ FAIL"
        status_bwd = "✓ PASS" if bwd_deterministic else "✗ FAIL"

        grad_notes = []
        if not bwd_q_deterministic:
            grad_notes.append("grad_q")
        if not bwd_k_deterministic:
            grad_notes.append("grad_k")
        if not bwd_v_deterministic:
            grad_notes.append("grad_v")
        grad_status = ", ".join(grad_notes) if grad_notes else ""

        result = {
            **asdict(config),
            "Dynamic": str(dynamic),
            "Backend": backend,
            "Mode": mode,
            "Forward": status_fwd,
            "Backward": status_bwd,
            "Notes": grad_status,
        }
        if kernel_options:
            result["KernelOptions"] = str(kernel_options)
        if inductor_config:
            result["InductorConfig"] = str(inductor_config)
        results.append(result)

        torch.cuda.synchronize()
        del q, k, v, grad_out, block_mask
        del output, grad_q, grad_k, grad_v
        del first_fwd_output, first_grad_q, first_grad_k, first_grad_v
        torch.cuda.empty_cache()

    return results


def main():
    """Main function to run determinism tests across multiple compilation settings."""

    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.test_configs.distort_benchmarking_result = "random"

    # fmt: off
    compile_settings = [
        # This is full eager - it is deterministic
        {"dynamic": False, "backend": "eager", "mode": "default"},

        # Default mode is not deterministic in the backwards since it is possible for `delta` calc's
        # reduction to choose different config based off of noisy neighbor
        {"dynamic": False, "backend": "inductor", "mode": "default"},

        # Force the reduction order to be the same for delta
        {"dynamic": False, "backend": "inductor", "mode": "default", "inductor_config": {"test_configs.force_filter_reduction_configs": True}},

        # Run again to show that it is honest to goodness fix
        {"dynamic": False, "backend": "inductor", "mode": "default"},

        # Using Inductor's deterministic mode also enforces consistent reduction loops
        # {"dynamic": False, "backend": "inductor", "mode": "default", "inductor_config": {"deterministic": True}},

        # Noisy neighbor problem also can be found here
        # Not running since takes some time
        # {"dynamic": False, "backend": "inductor", "mode": "max-autotune"},

    ]
    # fmt: on

    print("BITWISE DETERMINISM TEST".center(80, "="))

    all_results = []

    pbar = tqdm(compile_settings, desc="Compile settings")
    for compile_setting in pbar:
        dynamic = compile_setting["dynamic"]
        backend = compile_setting["backend"]
        mode = compile_setting["mode"]
        kernel_options = compile_setting.get("kernel_options")
        inductor_config = compile_setting.get("inductor_config")

        print(
            f"\n--- Testing with dynamic={dynamic}, backend={backend}, mode={mode}, kernel_options={kernel_options}, inductor_config={inductor_config} ---"
        )

        with torch._inductor.config.patch(inductor_config):
            results = test_bitwise_determinism(
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                pbar=pbar,
                kernel_options=kernel_options,
                inductor_config=inductor_config,
            )
        all_results.extend(results)

        columns = ["name", "Dynamic", "Backend", "Mode", "B", "Hq", "Hkv", "Q", "KV", "Dqk", "Dv", "Forward", "Backward", "Notes"]  # fmt: skip
        if inductor_config:
            columns.append("InductorConfig")
        if kernel_options:
            columns.append("KernelOptions")
        table_data = [[r.get(col, "") for col in columns] for r in results]  # fmt: skip
        print()
        print(tabulate(table_data, headers=columns, tablefmt="grid"))

    print("\n" + "=" * 80)

    print("SUMMARY".center(80, "="))
    total = len(all_results)
    fwd_pass = sum(1 for r in all_results if "PASS" in r["Forward"])
    bwd_pass = sum(1 for r in all_results if "PASS" in r["Backward"])
    print(f"Forward determinism:  {fwd_pass}/{total} passed")
    print(f"Backward determinism: {bwd_pass}/{total} passed")
    print("=" * 80 + "\n")

    if not all("PASS" in r["Forward"] for r in all_results):
        print("ERROR: Some forward passes failed bitwise determinism check")
        return False
    if not all("PASS" in r["Backward"] for r in all_results):
        print("ERROR: Some backward passes failed bitwise determinism check")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
