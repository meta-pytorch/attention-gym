# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Named activations fused into the causal convolution epilogue.

``causal_conv1d(..., activation=<name>)`` selects an activation by string, so
compiled graphs specialize on it like any other static scalar. ``"silu"`` is
built in and ``None`` applies no activation. Register your own, including
closure factories in the style of FA4 score mods:

    from cutlass import cute

    from attn_gym.linear import register_activation


    def make_softcap(softcap):
        def softcap_activation(value):
            return softcap * cute.math.tanh(value / softcap, fastmath=True)

        def softcap_derivative(value):
            activated = cute.math.tanh(value / softcap, fastmath=True)
            return 1.0 - activated * activated

        return softcap_activation, softcap_derivative


    register_activation("softcap-20", *make_softcap(20.0))

Both functions run at kernel trace time on an FP32 register fragment and may
use only CuTeDSL expressions (arithmetic and ``cute.math``). ``forward``
returns the activated fragment; ``derivative`` evaluates
d(activation)/d(preactivation) from the same preactivation fragment and may
return a Python scalar. Compiled kernels are cached against the functions'
content — source text plus current closure and module-global values — so
captured constants participate in the cache identity and mutating one
recompiles instead of silently reusing a stale kernel. Two boundaries: imported
modules are treated as stable library references, so route mutable configuration
through closures or globals rather than module attributes; and mutating captured
state between an op's forward and backward is outside the contract, as with any
state consumed by saved autograd graphs. Closure and
``__main__``-defined activations cannot cross the parallel compiler-process
boundary; ``tune_causal_conv1d`` compiles those candidates in-process instead.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

from cutlass import cute

from attn_gym._backends.cute import function_cache_key


@cute.jit
def _silu(value):
    """Apply SiLU to an FP32 register tensor."""
    half = value * 0.5
    return half * cute.math.tanh(half, fastmath=True) + half


@cute.jit
def _silu_derivative(value):
    """Evaluate the SiLU derivative from its preactivation."""
    half = value * 0.5
    tanh_half = cute.math.tanh(half, fastmath=True)
    return (tanh_half + 1.0) * 0.5 + half * (1.0 - tanh_half * tanh_half) * 0.5


def _identity(value):
    """Pass the preactivation through unchanged."""
    return value


def _identity_derivative(value):
    """Evaluate the identity derivative; the scalar folds at trace time."""
    return 1.0


@dataclass(frozen=True)
class Activation:
    """One registered activation.

    Instances flow into the ``jit_cache``-keyed compile functions, whose cache
    canonicalizer encodes the two callables by content (source text plus
    current closure and global values), re-read on every key computation.
    """

    name: str | None
    """The registered name; ``None`` is the built-in identity (no activation)."""

    forward: Callable
    """Trace-time expression applied to the FP32 preactivation fragment."""

    derivative: Callable
    """Trace-time derivative, evaluated from the same preactivation fragment."""

    @property
    def crosses_process_boundary(self) -> bool:
        """Whether the callables can be pickled by reference into a fresh worker."""
        return all(
            fn.__module__ != "__main__"
            and inspect.unwrap(fn).__closure__ is None
            and "<locals>" not in inspect.unwrap(fn).__qualname__
            for fn in (self.forward, self.derivative)
        )


_ACTIVATIONS: dict[str | None, Activation] = {}


def _activation_identity(name: str | None, activation: Activation) -> tuple:
    """Compute the content identity, translating cache errors for registration.

    Runs at registration time; per-call resolution is a dictionary lookup.
    """
    try:
        return (
            function_cache_key(activation.forward),
            function_cache_key(activation.derivative),
        )
    except TypeError as error:
        raise ValueError(
            f"activation {name!r} functions have no stable cache identity: {error}"
        ) from error


def register_activation(name: str, forward: Callable, derivative: Callable) -> None:
    """Register a named activation usable as ``causal_conv1d(activation=name)``.

    See the module docstring for the function contract. Re-registering a name
    with an identical content identity is a no-op that keeps the original
    callables; changing the implementation under an existing name raises
    instead of silently serving stale cached kernels.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"activation name must be a nonempty string, got {name!r}")
    activation = Activation(name, forward, derivative)
    identity = _activation_identity(name, activation)
    existing = _ACTIVATIONS.get(name)
    if existing is not None:
        if _activation_identity(name, existing) != identity:
            raise ValueError(
                f"activation {name!r} is already registered with a different implementation; "
                "register the new implementation under a new name"
            )
        return
    _ACTIVATIONS[name] = activation


def resolve_activation(activation: str | None) -> Activation:
    """Return the registered activation for a public name argument."""
    registered = _ACTIVATIONS.get(activation)
    if registered is None:
        names = ", ".join(repr(name) for name in _ACTIVATIONS if name is not None)
        raise ValueError(f"unknown activation {activation!r}; expected None or one of {names}")
    return registered


register_activation("silu", _silu, _silu_derivative)
_ACTIVATIONS[None] = Activation(None, _identity, _identity_derivative)

__all__ = ["Activation", "register_activation", "resolve_activation"]
