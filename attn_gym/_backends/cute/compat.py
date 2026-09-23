"""Import-time compatibility for optional CuTeDSL backends.

Use version predicates when selecting implementations on the host, not inside a
kernel or launch hot path. Keep renamed APIs here so kernels use stable names.
"""

from importlib.metadata import version

from packaging.specifiers import SpecifierSet
from packaging.version import Version

CUTEDSL_VERSION = Version(version("nvidia-cutlass-dsl"))


def cutedsl_version_matches(specifier: str) -> bool:
    """Match a PEP 440 range or exact version, including qualifying prereleases.

    For example, ``>=4.8`` excludes ``4.8.0rc1`` but includes ``4.9.0.dev1``.
    Invalid specifiers raise ``packaging.specifiers.InvalidSpecifier``.
    """
    return SpecifierSet(specifier).contains(CUTEDSL_VERSION, prereleases=True)


if cutedsl_version_matches(">=4.8"):
    from cutlass.memory import SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols
    from cutlass.tensor_utils import LayoutEnum
else:
    from cutlass.utils import LayoutEnum, SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols

__all__ = [
    "CUTEDSL_VERSION",
    "LayoutEnum",
    "SmemAllocator",
    "TmemAllocator",
    "cutedsl_version_matches",
    "get_num_tmem_alloc_cols",
]
