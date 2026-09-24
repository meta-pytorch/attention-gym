"""Gather attention over explicit KV indices and an optional causal local window."""

from attn_gym.types import Impl

from .api import AuxRequest, GatherAttnAux, gather_attn

__all__ = ["AuxRequest", "GatherAttnAux", "Impl", "gather_attn"]
