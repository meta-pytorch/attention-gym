"""Alias of :mod:`attn_gym.warpdrive.cache.compile`, kept until its callers import it there."""

import sys

import attn_gym.warpdrive.cache.compile as _module

sys.modules[__name__] = _module
