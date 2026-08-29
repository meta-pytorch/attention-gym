"""Native CuTeDSL kernels shared by delta-rule attention variants."""

from .affine_summary_fwd import build_state_summary
from .affine_summary_rev import build_state_grad_summary

__all__ = ["build_state_grad_summary", "build_state_summary"]
