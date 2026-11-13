# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral metadata for mutable paged recurrent state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Pool base and slot-origin alignment that the warpdrive paged kernels require.
WARPDRIVE_POOL_ALIGNMENT_BYTES = 16


def state_pool_is_aligned(cache: torch.Tensor, alignment_bytes: int) -> bool:
    """Return whether a pool's base and slot origins are ``alignment_bytes``-aligned.

    Fake tensors have no address, so under ``torch.compile`` the view offset stands in for the
    base: allocations themselves are aligned.
    """
    element_bytes = cache.element_size()
    if alignment_bytes % element_bytes:
        raise ValueError("alignment must be a multiple of the state element size")
    base = (
        cache.storage_offset() * element_bytes
        if torch.compiler.is_compiling()
        else cache.data_ptr()
    )
    return (
        base % alignment_bytes == 0 and cache.stride(0) % (alignment_bytes // element_bytes) == 0
    )


def validate_has_initial_state(
    has_initial_state: torch.Tensor | None,
    num_sequences: int,
    device: torch.device,
) -> None:
    """Validate the optional per-sequence fresh-slot mask."""
    if has_initial_state is None:
        return
    if (
        has_initial_state.shape != (num_sequences,)
        or has_initial_state.dtype != torch.bool
        or not has_initial_state.is_contiguous()
        or has_initial_state.device != device
    ):
        raise ValueError(
            "has_initial_state must be a contiguous bool tensor with one entry "
            "per sequence on the inputs' device"
        )


@dataclass(frozen=True, slots=True)
class PagedState:
    """A mutable state pool and its per-sequence device routing metadata.

    Attributes:
        cache: FP32 or BF16 state pool shaped ``[slots, heads, value_dim, key_dim]``.
            Active routes update their selected slots in place.
        indices: Contiguous int32 route for each sequence. Positive values
            select cache slots; non-positive values produce zero output and
            leave the cache untouched. Positive routes must be in bounds and
            unique among concurrently processed sequences.
        has_initial_state: Optional bool mask with one value per sequence.
            ``True`` resumes the selected slot. ``False`` starts from zero and
            overwrites the slot; if omitted, every active route resumes.
    """

    cache: torch.Tensor
    indices: torch.Tensor
    has_initial_state: torch.Tensor | None

    @classmethod
    def validate(
        cls,
        cache: torch.Tensor,
        indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        *,
        num_sequences: int,
        heads: int,
        value_dim: int,
        key_dim: int,
        device: torch.device,
        read_only_inputs: tuple[torch.Tensor, ...] = (),
    ) -> PagedState:
        """Validate the common pool ABI and return its routing metadata."""
        expected_inner_strides = (value_dim * key_dim, key_dim, 1)
        slot_elements = heads * value_dim * key_dim
        if cache.ndim != 4 or cache.shape[1:] != (heads, value_dim, key_dim):
            raise ValueError(
                f"the paged state pool must have shape [slots, {heads}, {value_dim}, {key_dim}]"
            )
        if cache.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("the paged state pool must use float32 or bfloat16")
        if cache.device != device:
            raise ValueError("the paged state pool must be on q.device")
        if cache.stride()[1:] != expected_inner_strides:
            raise TypeError("the paged state pool must be contiguous within each [H, V, K] slot")
        if cache.stride(0) < slot_elements:
            raise ValueError("paged state pool slots must not overlap")
        if (
            indices.shape != (num_sequences,)
            or indices.dtype != torch.int32
            or not indices.is_contiguous()
            or indices.device != device
        ):
            raise ValueError(f"state_indices must be contiguous int32 of shape ({num_sequences},)")
        validate_has_initial_state(has_initial_state, num_sequences, device)
        # Alias checks use a non-traceable C++ predicate, so only opaque backend
        # implementations request them. Public graph-visible validation stays structural.
        if read_only_inputs:
            read_only_inputs += (indices,)
            if has_initial_state is not None:
                read_only_inputs += (has_initial_state,)
            if any(torch._C._overlaps(cache, tensor) for tensor in read_only_inputs):
                raise ValueError("the paged state pool must not alias read-only inputs")
        return cls(cache, indices, has_initial_state)

    def require_alignment(self, alignment_bytes: int) -> PagedState:
        """Refine the common contract with a backend's pointer-alignment requirement."""
        if not state_pool_is_aligned(self.cache, alignment_bytes):
            raise TypeError(
                f"the paged state pool base and slot origins must be {alignment_bytes}-byte aligned"
            )
        return self

    @property
    def byte_mask(self) -> torch.Tensor | None:
        """Expose the bool mask through byte-oriented foreign-function ABIs."""
        return None if self.has_initial_state is None else self.has_initial_state.view(torch.uint8)


__all__ = [
    "WARPDRIVE_POOL_ALIGNMENT_BYTES",
    "PagedState",
    "state_pool_is_aligned",
    "validate_has_initial_state",
]
