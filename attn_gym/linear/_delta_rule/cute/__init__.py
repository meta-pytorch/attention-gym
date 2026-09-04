"""Native CuTeDSL kernels shared by delta-rule attention variants."""

from .affine_summary_fwd import build_state_summaries
from .affine_summary_rev import build_state_grad_summaries

__all__ = [
    "build_state_grad_summaries",
    "build_state_summaries",
]
