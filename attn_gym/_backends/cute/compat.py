"""Import-time compatibility across the supported CuTeDSL versions (4.6.2 through 4.8).

Use version predicates when selecting implementations on the host, not inside a
kernel or launch hot path. Keep renamed APIs here so kernels use stable names.
"""

from importlib.metadata import version

from cutlass import Int32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
from packaging.specifiers import SpecifierSet
from packaging.version import Version

CUTEDSL_VERSION = Version(version("nvidia-cutlass-dsl"))
# 4.6.0 and 4.6.1 elect inside a bulk copy, so one issued in elect_one() (as 4.8 requires)
# runs a nested elect.sync and hangs.
MIN_CUTEDSL_VERSION = "4.6.2"


def cutedsl_version_matches(specifier: str) -> bool:
    """Match a PEP 440 range or exact version, including qualifying prereleases.

    For example, ``>=4.8`` excludes ``4.8.0rc1`` but includes ``4.9.0.dev1``.
    Invalid specifiers raise ``packaging.specifiers.InvalidSpecifier``.
    """
    return SpecifierSet(specifier).contains(CUTEDSL_VERSION, prereleases=True)


if not cutedsl_version_matches(f">={MIN_CUTEDSL_VERSION}"):
    raise ImportError(
        f"attn_gym's CuTeDSL backends require nvidia-cutlass-dsl>={MIN_CUTEDSL_VERSION}, found "
        f"{CUTEDSL_VERSION}: "
        "earlier versions elect inside a cp.async.bulk copy, so a bulk copy issued in "
        "cute.arch.elect_one() runs a nested elect.sync and hangs"
    )


def _remote(ptr: cute.Pointer, rank: ir.Value, loc, ip) -> ir.Value:
    """``ptr``'s address in the SMEM of the cluster CTA ``rank`` (``mapa``)."""
    dsmem = llvm.PointerType.get(cute.AddressSpace.dsmem)
    return nvvm.mapa(dsmem, ptr.to_llvm_ptr(loc=loc, ip=ip), rank, loc=loc, ip=ip)


# CuTeDSL 4.8's PipelineState.advance(count) takes a count; older versions advance one epoch.
NATIVE_MULTI_ADVANCE = cutedsl_version_matches(">=4.8")

if cutedsl_version_matches(">=4.8"):
    from cutlass.memory import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols
    from cutlass.tensor_utils import LayoutEnum

    store_async_dsmem = cute.arch.store_async_dsmem
else:
    from cutlass.utils import LayoutEnum, SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols

    @dsl_user_op
    def store_async_dsmem(
        smem_ptr: cute.Pointer,
        value: int | Int32 | tuple[int | Int32, ...],
        mbar_ptr: cute.Pointer,
        peer_cta_rank: int | Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        """``cute.arch.store_async_dsmem`` with 4.8's 2- and 4-word ``value`` tuples (one
        ``st.async.v2``/``.v4``); 4.6 and 4.7 store one word. Inline PTX: their libNVVM cannot
        lower a vector ``st.async``."""
        words = value if isinstance(value, tuple) else (value,)
        n = len(words)
        if n not in (1, 2, 4):
            raise ValueError(f"store_async_dsmem stores 1, 2 or 4 words, got {n}")
        if smem_ptr.alignment % (4 * n):
            raise ValueError(
                f"store_async_dsmem: a {n}-word store needs a {4 * n}-byte aligned smem_ptr, got "
                f"alignment {smem_ptr.alignment}"
            )
        rank = Int32(peer_cta_rank).ir_value(loc=loc, ip=ip)
        addr, mbar = (
            llvm.ptrtoint(T.i32(), _remote(ptr, rank, loc, ip), loc=loc, ip=ip)
            for ptr in (smem_ptr, mbar_ptr)
        )
        words = [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
        shape = "" if n == 1 else f".v{n}"
        data = ", ".join(f"${i + 1}" for i in range(n))
        data = data if n == 1 else f"{{{data}}}"
        llvm.inline_asm(
            None,
            [addr, *words, mbar],
            f"st.async.shared::cluster.mbarrier::complete_tx::bytes{shape}.b32 [$0], {data}, "
            f"[${n + 1}];",
            ",".join(["r"] * (n + 2)),
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )


if cutedsl_version_matches(">=4.7"):
    from cutlass.base_dsl import Arch

    # The legacy FastDivmodDivisor warns from 4.7; the divmod arithmetic is the same.
    FastDivmodDivisor = cute.FastDivmodDivisorV2
    mbarrier_arrive = cute.arch.mbarrier_arrive
else:
    from cutlass.base_dsl.arch import Arch

    FastDivmodDivisor = cute.FastDivmodDivisor

    @dsl_user_op
    def mbarrier_arrive(
        mbar_ptr: cute.Pointer,
        peer_cta_rank_in_cluster: int | Int32 | None = None,
        arrive_count: int | Int32 = 1,
        *,
        scope: nvvm.MemScopeKind | None = None,
        loc=None,
        ip=None,
    ) -> None:
        """``cute.arch.mbarrier_arrive`` with the ``scope`` keyword 4.7 added (default CTA).

        4.6 always arrives at CTA scope; the NVVM op it lowers to already takes the scope.
        """
        if peer_cta_rank_in_cluster is None:
            addr = mbar_ptr.to_llvm_ptr(loc=loc, ip=ip)
        else:
            rank = Int32(peer_cta_rank_in_cluster).ir_value(loc=loc, ip=ip)
            addr = _remote(mbar_ptr, rank, loc, ip)
        nvvm.mbarrier_arrive(
            None,
            addr,
            count=Int32(arrive_count).ir_value(loc=loc, ip=ip),
            scope=nvvm.MemScopeKind.CTA if scope is None else scope,
            loc=loc,
            ip=ip,
        )


__all__ = [
    "CUTEDSL_VERSION",
    "MIN_CUTEDSL_VERSION",
    "NATIVE_MULTI_ADVANCE",
    "Arch",
    "FastDivmodDivisor",
    "LayoutEnum",
    "SmemAllocator",
    "TmemAllocator",
    "cutedsl_version_matches",
    "get_num_tmem_alloc_cols",
    "mbarrier_arrive",
    "store_async_dsmem",
]
