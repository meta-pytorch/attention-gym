"""CuTe DSL kernels for selected attention (SM100).

fa4_selected.py — Forked FlashAttention-4 MLA kernel with fused sink correction.
launcher.py     — Thin compile/launch wrapper matching the selected_attention API.
"""

from .launcher import selected_attention_forward

__all__ = ["selected_attention_forward"]
