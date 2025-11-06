import os
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
)
from torch.export import Dim
from typing import Callable
from attn_gym.masks.causal import causal_mask
from functools import lru_cache

import warnings

# Suppress FutureWarning about LeafSpec deprecation in torch._dynamo
warnings.filterwarnings(
    "ignore", message=".*isinstance.*treespec.*LeafSpec.*", category=FutureWarning
)


def block_mask_equivalent(bm_1: BlockMask, bm_2: BlockMask) -> bool:
    if not isinstance(bm_1, BlockMask) or not isinstance(bm_2, BlockMask):
        return False
    for attr in BlockMask._CONTEXT_ATTRS:
        left = getattr(bm_1, attr)
        right = getattr(bm_2, attr)
        if attr == "mask_mod":
            if left is not right:
                return False
            continue
        if left != right:
            return False
    for attr in BlockMask._TENSOR_ATTRS:
        left = getattr(bm_1, attr)
        right = getattr(bm_2, attr)
        if left is None or right is None:
            if left is not None or right is not None:
                return False
            continue
        if left.dtype != right.dtype or left.device != right.device:
            return False
        if left.shape != right.shape:
            return False
        if not torch.equal(left, right):
            return False
    return True


ModOut = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class BlockMaskProgram(torch.nn.Module):
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        mask_mod: Callable,
        block_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.seq_lengths = (q_len, kv_len)
        self.block_size = block_size
        self.mask_mod = mask_mod

    def forward(
        self,
        batch_size: torch.Tensor,
        head_count: torch.Tensor,
        # device: torch.device, # TODO figure this out
    ) -> ModOut:
        n_batch = batch_size.shape[0]
        n_heads = head_count.shape[0]
        torch._check(n_batch != 0)
        torch._check(n_heads != 0)
        block = create_block_mask(
            mask_mod=self.mask_mod,
            B=n_batch,
            H=n_heads,
            Q_LEN=self.seq_lengths[0],
            KV_LEN=self.seq_lengths[1],
            device=torch.accelerator.current_accelerator(),
            # device=device, # TODO figure this out
        )
        if self.block_size is None:
            self.block_size = block.BLOCK_SIZE
        kv_num_blocks = block.kv_num_blocks
        kv_indices = block.kv_indices
        full_kv_num_blocks = block.full_kv_num_blocks
        full_kv_indices = block.full_kv_indices
        q_num_blocks = block.q_num_blocks
        q_indices = block.q_indices
        full_q_num_blocks = block.full_q_num_blocks
        full_q_indices = block.full_q_indices
        return (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        )


@lru_cache(maxsize=128)
def _load_aoti_package_cached(path: str, device_index: int):
    """Cached loader for AOTI packages based on path and device_index."""
    return torch._inductor.aoti_load_package(path, device_index=device_index)


def produce_block_mask_aoti(
    path: str,
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device,
    block_size: tuple[int, int] | None = None,
) -> str:
    """Produce a block mask AOTI module and compile it to a package returns the path to the package"""
    block_size = (
        (_DEFAULT_SPARSE_BLOCK_SIZE, _DEFAULT_SPARSE_BLOCK_SIZE)
        if block_size is None
        else block_size
    )
    bm_mod = BlockMaskProgram(q_len=Q_LEN, kv_len=KV_LEN, mask_mod=mask_mod, block_size=block_size)
    batch_stub = torch.empty(B, dtype=torch.int32)
    head_stub = torch.empty(H, dtype=torch.int32)
    wrapped_ints = (batch_stub, head_stub)
    dynamic_shapes = {
        "batch_size": {0: Dim.DYNAMIC},
        "head_count": {0: Dim.DYNAMIC},
    }
    exported = torch.export.export(bm_mod, wrapped_ints, dynamic_shapes=dynamic_shapes)
    compiled_path = torch._inductor.aoti_compile_and_package(exported, package_path=path)
    return compiled_path


def create_block_mask_aoti(
    path: str,
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device,
    block_size: tuple[int, int] | None = None,
) -> BlockMask:
    """Bootleg verison that doesnnt jit compile"""
    device_index = device.index if device.index is not None else 0
    block_size = (
        (_DEFAULT_SPARSE_BLOCK_SIZE, _DEFAULT_SPARSE_BLOCK_SIZE)
        if block_size is None
        else block_size
    )
    batch_stub = torch.empty(B, dtype=torch.int32)
    head_stub = torch.empty(H, dtype=torch.int32)
    wrapped_ints = (batch_stub, head_stub)
    runtime = _load_aoti_package_cached(path, device_index)
    outputs = runtime(*wrapped_ints)
    return BlockMask(
        seq_lengths=(Q_LEN, KV_LEN),
        kv_num_blocks=outputs[0],
        kv_indices=outputs[1],
        full_kv_num_blocks=outputs[2],
        full_kv_indices=outputs[3],
        q_num_blocks=outputs[4],
        q_indices=outputs[5],
        full_q_num_blocks=outputs[6],
        full_q_indices=outputs[7],
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example.")
    device = torch.device("cuda")

    # Initial values
    B_val = 2
    H_val = 4

    path = produce_block_mask_aoti(
        path=os.path.join(os.getcwd(), "data/block_mask.pt2"),
        mask_mod=causal_mask,
        B=B_val,
        H=H_val,
        Q_LEN=128,
        KV_LEN=128,
        device=device,
    )
    # Test with different batch sizes
    for test_B, test_H in [(1, 2), (3, 8), (4, 4)]:
        jit = create_block_mask(causal_mask, test_B, test_H, 128, 128, device)
        aoti = create_block_mask_aoti(
            path, mask_mod=causal_mask, B=test_B, H=test_H, Q_LEN=128, KV_LEN=128, device=device
        )

        if not block_mask_equivalent(jit, aoti):
            raise RuntimeError(f"BlockMask mismatch for B={test_B}, H={test_H}")

        print(f"✓ AOTI runtime test passed for B={test_B}, H={test_H}")

    # Test multi-GPU if available
    if torch.cuda.device_count() < 2:
        print("\nSkipping multi-GPU tests (requires 2+ GPUs)")
    else:
        for device_idx in [0, 1]:
            jit = create_block_mask(
                causal_mask, test_B, test_H, 128, 128, torch.device("cuda", device_idx)
            )
            aoti = create_block_mask_aoti(
                path,
                mask_mod=causal_mask,
                B=test_B,
                H=test_H,
                Q_LEN=128,
                KV_LEN=128,
                device=torch.device("cuda", device_idx),
            )
            if not block_mask_equivalent(jit, aoti):
                raise RuntimeError(
                    f"BlockMask mismatch for B={test_B}, H={test_H} on device {device_idx}"
                )
            print(f"✓ AOTI runtime test passed for B={test_B}, H={test_H} on device {device_idx}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
