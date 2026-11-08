"""NVIDIA Intra-Kernel Profiling for CUTE DSL Kernels.

Record code regions *inside* GPU kernels into a profile buffer, decode them on the host, and
export native Perfetto TrackEvent traces (https://ui.perfetto.dev/). See docs/profiler.md.

Quick Start:
    from attn_gym.profiler import (
        compact_anchor_init, compact_profile_region, profile_session
    )

    @cute.kernel
    def my_kernel(output, prof_buf, max_events):
        bidx, _, _ = cute.arch.block_idx()
        compact_anchor_init(prof_buf, bidx, max_events)
        with compact_profile_region(prof_buf, max_events, TAG_COMPUTE, bidx, Int32(0)):
            compute_something()

    with profile_session(
        max_events_per_unit=1,
        num_units=(num_blocks, "Block"),
        tag_names=["compute"],
        trace_path="trace.pftrace",
        compact=True,
    ) as session:
        my_kernel(output, session.prof.tensor, session.prof.max_events_per_unit)
    session.ranges  # decoded after the block
"""

from importlib import import_module

# Device code loads eagerly; host code on first use, so a kernel that imports a recorder from
# this package keys only this file and ops.py (compile keys hash every module imported).
from attn_gym.profiler.ops import (
    compact_anchor_init,
    compact_event_stop,
    compact_event_stop_smem,
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    compact_profile_region,
    event_stop,
    read_globaltimer,
    read_globaltimer_lo32,
)

_HOST_EXPORTS = {
    "ProfileBuf": "host",
    "ProfileSession": "host",
    "TagTable": "host",
    "allocate_profile_buffer": "host",
    "decode_ranges": "host",
    "profile_session": "host",
    "parse_iket": "iket",
    "write_ranges": "perfetto",
    "OverlapSummary": "ranges",
    "Range": "ranges",
    "RangeSummary": "ranges",
    "overlap_between": "ranges",
    "summarize_by_tag": "ranges",
}


def __getattr__(name: str):
    if name not in _HOST_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_HOST_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [  # noqa: RUF022 (grouped by role)
    # Device
    "compact_anchor_init",
    "compact_event_stop",
    "compact_event_stop_smem",
    "compact_flush_smem_to_gmem",
    "compact_prepare_smem",
    "compact_profile_region",
    "event_stop",
    "read_globaltimer",
    "read_globaltimer_lo32",
    # Host capture
    "ProfileBuf",
    "ProfileSession",
    "TagTable",
    "allocate_profile_buffer",
    "decode_ranges",
    "parse_iket",
    "profile_session",
    # Ranges, statistics and traces
    "OverlapSummary",
    "Range",
    "RangeSummary",
    "overlap_between",
    "summarize_by_tag",
    "write_ranges",
]
