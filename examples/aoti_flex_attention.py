import warnings
from contextlib import nullcontext

import torch
from torch._dynamo import lookup_backend
from torch._dynamo.aot_compile import aot_compile_module, ModelInput

# from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.hooks import Hooks
from torch.nn.attention import flex_attention as flex_attention_module
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)

flex_attention_module._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

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


def build_aot_flex_attention(
    block_mask: BlockMask,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hooks = Hooks()
    example_inputs = [
        ModelInput(
            (query, key, value, block_mask),
            {},
            [
                nullcontext(),
            ],
        ),
    ]
    module = FlexAttentionForward()
    backend = lookup_backend("inductor")
    aot_compiled_flex = aot_compile_module(module, example_inputs, hooks=hooks, backend=backend)
    return aot_compiled_flex


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

    compiled_flex = torch.compile(flex_attention_module.flex_attention, backend="inductor")
    aot_module = build_aot_flex_attention(block, q, k, v)
    out_compile, out_lse = compiled_flex(q, k, v, block, return_aux=AuxRequest(lse=True))
    out_aot, lse_aot = aot_module(q, k, v, block)

    print(f"compile output_norm={out_compile.norm().item():.4f}")
    print(f"compile lse_norm={out_lse.norm().item():.4f}")

    max_output_diff = (out_aot - out_compile).abs().max()
    max_lse_diff = (lse_aot - out_lse).abs().max()
    print(f"max |output diff|={max_output_diff.item():.4e}")
    print(f"max |lse diff|={max_lse_diff.item():.4e}")

    print("\n--- Testing Exported Backward ---")
    grad_q, grad_k, grad_v = torch.autograd.grad(
        out_compile, (q, k, v), grad_outputs=torch.ones_like(out_compile)
    )
    grad_q_aot, grad_k_aot, grad_v_aot = torch.autograd.grad(
        out_aot, (q, k, v), grad_outputs=torch.ones_like(out_aot)
    )
    max_grad_q_diff = (grad_q_aot - grad_q).abs().max()
    max_grad_k_diff = (grad_k_aot - grad_k).abs().max()
    max_grad_v_diff = (grad_v_aot - grad_v).abs().max()

    print(f"max |grad_q diff|={max_grad_q_diff.item():.4e}")
    print(f"max |grad_k diff|={max_grad_k_diff.item():.4e}")
    print(f"max |grad_v diff|={max_grad_v_diff.item():.4e}")


if __name__ == "__main__":
    from transformer_nuggets import init_logging

    init_logging()
    # with profiler("build_and_run") as p:
    build_and_run()
