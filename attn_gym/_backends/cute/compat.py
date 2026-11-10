"""Alias of :mod:`attn_gym.warpdrive.compat`, kept until its callers import it there."""

import sys

import attn_gym.warpdrive.compat as _module

sys.modules[__name__] = _module
