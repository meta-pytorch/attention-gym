"""Triton row masking for fixed-capacity packed KDA tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_inactive_rows_kernel(x, active_mask, out, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < cols
    base = row.to(tl.int64) * cols
    if tl.load(active_mask + row):
        values = tl.load(x + base + offsets, mask=in_bounds)
        tl.store(out + base + offsets, values, mask=in_bounds)
    else:
        zero = tl.zeros((BLOCK,), dtype=out.dtype.element_ty)
        tl.store(out + base + offsets, zero, mask=in_bounds)


@torch.library.triton_op("attn_gym::kda_mask_inactive_rows", mutates_args={})
def mask_inactive_rows(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Copy active token rows and write-only zero the inactive suffix."""
    out = torch.empty_like(x)
    tokens = x.shape[1]
    if tokens == 0:
        return out
    cols = x.numel() // tokens
    block = 2048
    grid = (tokens, triton.cdiv(cols, block))
    torch.library.wrap_triton(_mask_inactive_rows_kernel)[grid](
        x, active_mask, out, cols, BLOCK=block
    )
    return out


__all__ = ["mask_inactive_rows"]
