"""The host side of the megakernels' launches: reading their tagged scratch."""

import torch


def index_rows(rows: list[list[int]], first: int = 0) -> torch.Tensor:
    """``rows`` as an int32 table, each ending in its index ``first + i`` (the trace's row ID)."""
    return torch.tensor([row + [first + i] for i, row in enumerate(rows)], dtype=torch.int32)


def tagged_values(words: torch.Tensor) -> torch.Tensor:
    """The fp32 values of int64 tagged words (``tagged.pack_tagged``: value bits low), ignoring
    their tags."""
    return words.view(torch.int32)[..., 0::2].view(torch.float32)
