import os
import warnings
from typing import Callable, Optional, Tuple

import torch
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


MaskPayload = Tuple[
    tuple[int, int],
    tuple[int, int],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def _pack_block_mask(block_mask: BlockMask) -> MaskPayload:
    return (
        block_mask.seq_lengths,
        block_mask.BLOCK_SIZE,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.full_kv_num_blocks,
        block_mask.full_kv_indices,
        block_mask.q_num_blocks,
        block_mask.q_indices,
        block_mask.full_q_num_blocks,
        block_mask.full_q_indices,
    )


def _unpack_block_mask(mask_payload: MaskPayload, mask_mod: Callable) -> BlockMask:
    (
        seq_lengths,
        block_size,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        q_num_blocks,
        q_indices,
        full_q_num_blocks,
        full_q_indices,
    ) = mask_payload

    return BlockMask(
        seq_lengths=seq_lengths,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )


class FlexAttentionForward(torch.nn.Module):
    def __init__(self, mask_mod: Callable) -> None:
        super().__init__()
        self._mask_mod = mask_mod

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_payload: MaskPayload,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = _unpack_block_mask(mask_payload, self._mask_mod)
        output, aux = flex_attention(
            query,
            key,
            value,
            block_mask=block,
            return_aux=AuxRequest(lse=True),
        )
        if aux.lse is None:
            raise RuntimeError("Expected log-sum-exp output when requesting AuxRequest(lse=True.)")
        return output, aux.lse


class FlexAttentionBackward(torch.nn.Module):
    def __init__(self, mask_mod: Callable) -> None:
        super().__init__()
        self._mask_mod = mask_mod

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        grad_out: torch.Tensor,
        mask_payload: MaskPayload,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (query.requires_grad and key.requires_grad and value.requires_grad):
            raise RuntimeError("Inputs to FlexAttentionBackward must require gradients.")

        block = _unpack_block_mask(mask_payload, self._mask_mod)
        result = flex_attention(query, key, value, block_mask=block)
        loss = (result * grad_out).sum()
        dq, dk, dv = torch.autograd.grad(loss, (query, key, value), allow_unused=False)
        return dq, dk, dv


def _run_exported_forward(
    module: FlexAttentionForward,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_payload: MaskPayload,
    package_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    exported = torch.export.export(module, (query, key, value, mask_payload))
    compiled_path = torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
    runtime = torch._inductor.aoti_load_package(compiled_path)
    return runtime(query, key, value, mask_payload)


def _run_compiled_forward(
    module_ctor: Callable[[], FlexAttentionForward],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_payload: MaskPayload,
) -> tuple[torch.Tensor, torch.Tensor]:
    compiled_module = torch.compile(module_ctor(), fullgraph=True, backend="inductor")
    output, lse = compiled_module(query, key, value, mask_payload)
    return output, lse


def build_and_run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    mask_payload = _pack_block_mask(block)

    import torch.autograd.profiler as profiler

    with profiler.record_function("exported_forward"):
        fwd = FlexAttentionForward(mask_mod=causal_mask).to(device)
        fwd_path = os.path.join(os.getcwd(), "data/flex_attention_fwd.pt2")
        out_aoti, lse_aoti = _run_exported_forward(fwd, q, k, v, mask_payload, fwd_path)
        print(f"AOTI   output_norm={out_aoti.norm().item():.4f}")
        print(f"AOTI   lse_norm={lse_aoti.norm().item():.4f}")

    def compiled_module_ctor() -> FlexAttentionForward:
        return FlexAttentionForward(mask_mod=causal_mask).to(device)

    with profiler.record_function("compiled_forward"):
        out_compiled, lse_compiled = _run_compiled_forward(
            compiled_module_ctor,
            q,
            k,
            v,
            mask_payload,
        )
    print(f"compile output_norm={out_compiled.norm().item():.4f}")
    print(f"compile lse_norm={lse_compiled.norm().item():.4f}")

    max_output_diff = (out_aoti - out_compiled).abs().max()
    max_lse_diff = (lse_aoti - lse_compiled).abs().max()
    print(f"max |output diff|={max_output_diff.item():.4e}")
    print(f"max |lse diff|={max_lse_diff.item():.4e}")


if __name__ == "__main__":
    from transformer_nuggets import init_logging
    from transformer_nuggets.utils.benchmark import profiler

    init_logging()
    with profiler("build_and_run") as p:
        build_and_run()
