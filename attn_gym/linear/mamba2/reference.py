"""Readable eager PyTorch implementations of the Mamba2 SSD scan.

Each head ``h`` keeps a ``[P, N]`` state. Token ``t`` decays it by ``exp(dt[t, h] * A[h])``,
writes ``dt[t, h] * outer(x[t, h], B[t, g])`` and reads it with ``C[t, g]``, where ``g`` is the
B/C group of head ``h``. These cores return that state read; the public operations add the
``D`` skip and ``silu(z)`` gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CHUNK_SIZE = 64


def expand_groups(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """Repeat each B/C group across its ``heads // groups`` consecutive heads."""
    return tensor.repeat_interleave(heads // tensor.shape[2], dim=2)


def recurrent_forward(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the SSD recurrence one token at a time."""
    batch, sequence, heads, head_dim = x.shape
    B, C = expand_groups(B, heads), expand_groups(C, heads)
    state = (
        x.new_zeros(batch, heads, head_dim, B.shape[-1])
        if initial_state is None
        else initial_state
    )
    outputs = []
    for token in range(sequence):
        decay = torch.exp(dt[:, token] * A)[..., None, None]
        write = torch.einsum("bhp,bhn->bhpn", x[:, token] * dt[:, token, :, None], B[:, token])
        state = decay * state + write
        outputs.append(torch.einsum("bhpn,bhn->bhp", state, C[:, token]))
    return torch.stack(outputs, dim=1), state if output_final_state else None


def chunk_forward(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the SSD recurrence with the chunk-parallel decomposition.

    Within each ``CHUNK_SIZE`` block with in-chunk cumulative log decay ``g``:

    1. intra-chunk output ``((C B^T) * exp(g_i - g_j) * [i >= j]) (dt x)``;
    2. chunk state write ``sum_j exp(g_last - g_j) (dt x)_j B_j^T``;
    3. a sequential pass that decays the entry state by ``exp(g_last)`` and adds the write;
    4. inter-chunk output ``exp(g_i) C_i S_entry^T``.
    """
    batch, sequence, heads, head_dim = x.shape
    B, C = expand_groups(B, heads), expand_groups(C, heads)
    state_dim = B.shape[-1]
    # Zero dt makes padded tokens identity steps: no decay and no write.
    padding = (-sequence) % CHUNK_SIZE
    scaled_x, B, C = (
        F.pad(tensor, (0, 0, 0, 0, 0, padding)) for tensor in (x * dt[..., None], B, C)
    )
    log_decay = F.pad(dt * A, (0, 0, 0, padding))
    chunks = scaled_x.shape[1] // CHUNK_SIZE

    def to_chunks(tensor: torch.Tensor) -> torch.Tensor:
        """``[batch, T, H, D] -> [batch, H, chunks, CHUNK_SIZE, D]``."""
        return tensor.reshape(batch, chunks, CHUNK_SIZE, heads, tensor.shape[-1]).permute(
            0, 3, 1, 2, 4
        )

    scaled_x, B, C = to_chunks(scaled_x), to_chunks(B), to_chunks(C)
    g = log_decay.reshape(batch, chunks, CHUNK_SIZE, heads).permute(0, 3, 1, 2).cumsum(-1)

    causal = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=x.device).tril()
    segment_decay = torch.where(causal, g[..., :, None] - g[..., None, :], -torch.inf).exp()
    output = ((C @ B.transpose(-1, -2)) * segment_decay) @ scaled_x

    g_last = g[..., -1:]
    chunk_writes = (scaled_x * (g_last - g).exp()[..., None]).transpose(-1, -2) @ B

    state = (
        x.new_zeros(batch, heads, head_dim, state_dim) if initial_state is None else initial_state
    )
    entry_states = []
    for chunk in range(chunks):
        entry_states.append(state)
        state = g_last[:, :, chunk, 0, None, None].exp() * state + chunk_writes[:, :, chunk]
    entry = torch.stack(entry_states, dim=2)

    output = output + (C * g.exp()[..., None]) @ entry.transpose(-1, -2)
    output = output.permute(0, 2, 3, 1, 4).reshape(batch, -1, heads, head_dim)
    return output[:, :sequence], state if output_final_state else None


__all__ = ["CHUNK_SIZE", "chunk_forward", "expand_groups", "recurrent_forward"]
