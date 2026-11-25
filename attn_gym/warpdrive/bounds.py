"""Bounds checks for warpdrive's opt-in checked builds.

CuTe maps even out-of-range coordinates to offsets. During a checked trace, intercept
CuTe tensor indexing and constant negative view offsets. Report errors where the access
executes: raise in host launch code, emit a device assertion in kernel code (even a
statically invalid access can be in an untaken branch). See docs/warpdrive/dropping-down.md
for the checked-build contract and exclusions.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import reduce
from operator import mul

import cutlass
from cutlass import Int32, Int64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.ffi import ffi, mangle

from attn_gym.warpdrive.compat import TENSOR_CLASS, dsl_user_op
from attn_gym.warpdrive.protocol import active_opt_out, user_site


def _gpu_module():
    """The ``gpu.module`` being traced into, or None in a launch function's host code."""
    op = ir.InsertionPoint.current.block.owner
    while op is not None and op.name != "gpu.module":
        op = op.parent
    return op


def _string_constant(text: str) -> ir.Value:
    """Pointer to ``text`` as a NUL-terminated constant of the enclosing ``gpu.module``."""
    module = _gpu_module()
    size = len(text.encode()) + 1
    symbol = "wd_str_" + mangle(text)
    if symbol not in ir.SymbolTable(module.operation):
        with ir.InsertionPoint(module.regions[0].blocks[0]):
            llvm.GlobalOp(
                sym_name=symbol,
                global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
                linkage=ir.Attribute.parse("#llvm.linkage<internal>"),
                constant=True,
                value=ir.StringAttr.get(text + "\0"),
            )
    return llvm.AddressOfOp(llvm.PointerType.get(), symbol).result


def assert_fail(message: str, function: str, module: str = __name__) -> None:
    """CUDA's failed ``assert``: print ``message`` to stderr and abort the context with
    ``CUDA_ERROR_ASSERT``. CuTeDSL drops its own device asserts unless the whole module is
    compiled with assertions enabled, so this calls the driver's ``__assertfail`` directly."""
    ptr = llvm.PointerType.get()
    fail = ffi(name="__assertfail", params_types=[ptr, ptr, Int32, ptr, Int64], inline=False)
    fail(
        _string_constant(message),
        _string_constant(module),
        Int32(0),
        _string_constant(function),
        Int64(1),
    )


@cute.jit
def _compare(value, extent, message: cutlass.Constexpr[str]):
    if value < 0 or value >= extent:
        assert_fail(message, "index")


def _constant(value) -> int | None:
    """The Python value of an int or of a CuTeDSL integer built from one, else None."""
    if isinstance(value, int):
        return value
    held = getattr(value, "value", None)
    return held if isinstance(held, int) else None


def _extent(shape):
    """The number of coordinates of a mode (or of a whole shape): an int when every extent is
    known, else their product in 64 bits, which a 32-bit product can overflow."""
    leaves = cute.flatten_to_tuple(shape)
    known = [_constant(leaf) for leaf in leaves]
    if None not in known:
        return reduce(mul, known, 1)
    return reduce(mul, (Int64(leaf) for leaf in leaves))


def _fail(message: str) -> None:
    """A mistake known at trace time: the launch that runs it aborts, or, in a launch
    function's host code, the compile fails."""
    if _gpu_module() is None:
        raise IndexError(message)
    assert_fail(message, "index")


def _check_index(tensor, crd, site: str) -> None:
    shape = tensor.shape if isinstance(tensor.shape, tuple) else (tensor.shape,)
    if not isinstance(crd, tuple):  # one index into the whole tensor
        pairs = [(crd, shape, "linear index")]
    elif len(crd) == len(shape):
        pairs = [(c, shape[m], f"index of mode {m}") for m, c in enumerate(crd)]
    else:
        return
    for c, mode, what in pairs:
        if c is None or isinstance(c, (tuple, slice)):
            continue
        index, extent = _constant(c), _extent(mode)
        if index is not None and isinstance(extent, int):
            if not 0 <= index < extent:
                _fail(f"{site}: {what} {index} outside [0, {extent})")
        elif _gpu_module() is None:
            raise IndexError(
                f"{site}: {what} or its extent is not known at trace time, and a launch "
                f'function cannot check it; index in the kernel, or use unchecked("bounds", ...)'
            )
        else:  # in 64 bits, so an unsigned index past the signed range is outside too
            _compare(Int64(c), extent, f"{site}: {what} out of range")


def _check_offset(tensor, coord, site: str) -> None:
    for mode, c in enumerate(coord if isinstance(coord, tuple) else (coord,)):
        offset = _constant(c)
        if offset is not None and offset < 0:
            _fail(f"{site}: domain_offset {offset} of mode {mode} is before the tensor")


_checks_itself = False  # the checks' own ops are not checked
_originals = None  # CuTeDSL's functions, while the checks stand in for them


def _check(rule, what: str, tensor, crd, ip) -> None:
    """Apply ``rule`` where the checked op is inserted, unless an opt-out block is open."""
    global _checks_itself
    if _checks_itself:
        return
    _checks_itself = True
    try:
        site, opt = user_site(), active_opt_out("bounds")
        if opt is not None:
            opt.suppressed.append(f"{what} @ {site}")
        elif ip is None:
            rule(tensor, crd, site)
        else:
            with ip:
                rule(tensor, crd, site)
    finally:
        _checks_itself = False


def _install() -> None:
    """Route tensor indexing and ``domain_offset`` through the checks. Each stand-in is a
    ``dsl_user_op`` like the function it replaces, so that function's ops keep the caller's
    source line."""
    global _originals
    if _originals is not None:
        return
    get, put = TENSOR_CLASS.__getitem__, TENSOR_CLASS.__setitem__
    offset = cute.tensor.domain_offset
    _originals = (get, put, offset)

    @dsl_user_op
    def getitem(self, crd, *, loc=None, ip=None):
        _check(_check_index, "index", self, crd, ip)
        return get.__wrapped__(self, crd, loc=loc, ip=ip)

    @dsl_user_op
    def setitem(self, crd, data, *, loc=None, ip=None):
        _check(_check_index, "index", self, crd, ip)
        return put.__wrapped__(self, crd, data, loc=loc, ip=ip)

    @dsl_user_op
    def domain_offset(coord, tensor, *, loc=None, ip=None):
        _check(_check_offset, "domain_offset", tensor, coord, ip)
        return offset.__wrapped__(coord, tensor, loc=loc, ip=ip)

    TENSOR_CLASS.__getitem__, TENSOR_CLASS.__setitem__ = getitem, setitem
    cute.domain_offset = cute.tensor.domain_offset = domain_offset


def _uninstall() -> None:
    global _originals
    if _originals is not None:
        TENSOR_CLASS.__getitem__, TENSOR_CLASS.__setitem__, offset = _originals
        cute.domain_offset = cute.tensor.domain_offset = offset
        _originals = None


def set_checking(checked: bool) -> bool:
    """Trace what follows with (a checked kernel's code) or without the checks. Returns the
    setting it replaces, to restore when that code has been traced."""
    previous = _originals is not None
    (_install if checked else _uninstall)()
    return previous


@contextmanager
def indexing(checked: bool):
    """``set_checking`` for a block."""
    previous = set_checking(checked)
    try:
        yield
    finally:
        set_checking(previous)
