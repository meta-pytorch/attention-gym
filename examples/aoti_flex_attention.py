import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
from tabulate import tabulate
from torch._dynamo import lookup_backend
from torch._dynamo.aot_compile import AOTCompiledModel, ModelInput, aot_compile_module
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.hooks import Hooks
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)


warnings.filterwarnings(
    "ignore",
    message="`isinstance\\(treespec, LeafSpec\\)` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="You are calling torch.compile inside torch.export region",
    category=UserWarning,
)


class FlexAttentionForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask: BlockMask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, aux = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            return_aux=AuxRequest(lse=True),
        )
        return output, aux.lse


def serializable_inductor_backend(gm, example_inputs):
    compiled = lookup_backend("inductor")(gm, example_inputs)
    return BundledAOTAutogradSerializableCallable(compiled)


def build_aot_flex_attention(
    block_mask: BlockMask,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hooks = Hooks()
    example_inputs = [
        ModelInput(
            (query, key, value),
            {"block_mask": block_mask},
            [
                nullcontext(),
            ],
        ),
    ]
    module = FlexAttentionForward()
    aot_compiled_flex = aot_compile_module(
        module, example_inputs, hooks=hooks, backend=serializable_inductor_backend
    )
    return aot_compiled_flex


def save_aot_module(aot_model: AOTCompiledModel, path: Path) -> None:
    payload = aot_model.serialize()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def load_aot_module(path: Path) -> AOTCompiledModel:
    payload = path.read_bytes()
    module = FlexAttentionForward()
    return AOTCompiledModel.deserialize(module, payload)


def report_differences(
    reference_outputs: tuple[torch.Tensor, torch.Tensor],
    reference_grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]],
    grad_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    out_ref, lse_ref = reference_outputs
    grad_q_ref, grad_k_ref, grad_v_ref = reference_grads
    rows: list[dict[str, float]] = []
    for tag, out_test, lse_test in comparisons:
        max_output_diff = (out_test - out_ref).abs().max()
        max_lse_diff = (lse_test - lse_ref).abs().max()
        grad_q_test, grad_k_test, grad_v_test = torch.autograd.grad(
            out_test, grad_inputs, grad_outputs=torch.ones_like(out_test)
        )
        rows.append(
            {
                "tag": tag,
                "max|out diff|": max_output_diff.item(),
                "max|lse diff|": max_lse_diff.item(),
                "max|grad_q diff|": (grad_q_test - grad_q_ref).abs().max().item(),
                "max|grad_k diff|": (grad_k_test - grad_k_ref).abs().max().item(),
                "max|grad_v diff|": (grad_v_test - grad_v_ref).abs().max().item(),
            }
        )
    print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4e"))


def build_and_run() -> None:
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(2, 2, 128, 64, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    def causal_mask(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return kv_idx <= q_idx

    block = create_block_mask(
        mask_mod=causal_mask,
        B=q.shape[0],
        H=q.shape[1],
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=device,
    )

    compiled_flex = torch.compile(flex_attention, backend="inductor")
    out_compile, out_aux = compiled_flex(
        q, k, v, block_mask=block, return_aux=AuxRequest(lse=True)
    )
    out_lse = out_aux.lse

    aot_module = build_aot_flex_attention(block, q, k, v)
    out_aot, lse_aot = aot_module(q, k, v, block)

    grad_q, grad_k, grad_v = torch.autograd.grad(
        out_compile, (q, k, v), grad_outputs=torch.ones_like(out_compile)
    )
    print("\n--- Testing Exported Backward ---")
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("fresh", out_aot, lse_aot),
    ]

    from transformer_nuggets.utils.benchmark import profiler, record_function

    with (
        profiler("aot_run", with_stack=True),
        record_function("aot_flex_attention_save_load"),
    ):
        artifact_path = Path("data/aot_flex_attention_artifact.bin")
        with record_function("aot_flex_attention_save_load"):
            save_aot_module(aot_module, artifact_path)
        print(f"AOT artifact saved to {artifact_path}")
        with record_function("aot_flex_attention_reload"):
            reloaded_aot = load_aot_module(artifact_path)
        print(f"Reloaded artifact from {artifact_path}")

        with record_function("aot_flex_attention_run_reloaded"):
            out_aot_reloaded, lse_aot_reloaded = reloaded_aot(q, k, v, block)
        comparisons.append(("reloaded", out_aot_reloaded, lse_aot_reloaded))
    report_differences(
        (out_compile, out_lse),
        (grad_q, grad_k, grad_v),
        comparisons,
        (q, k, v),
    )


if __name__ == "__main__":
    from transformer_nuggets import init_logging

    init_logging()
    # with profiler("build_and_run") as p:
    build_and_run()
