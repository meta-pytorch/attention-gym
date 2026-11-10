"""Alias of :mod:`attn_gym.warpdrive.cache.jit`, kept until its callers import it there."""

import sys

import attn_gym.warpdrive.cache.jit as _module

sys.modules[__name__] = _module
