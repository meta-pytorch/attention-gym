"""Declarative CuTeDSL pipeline plans, measurements, analysis, and Perfetto export."""

from __future__ import annotations

from importlib import import_module

# The annotation markers run while kernels trace, so they load eagerly (and are hashed into
# the compile keys of kernels that import them through this package); the plan model,
# extractor, join, analysis and writers load on first use and stay out of those keys.
from attn_gym.profiler.pipeline.annotations import PipelineAnnotations

_LAZY_EXPORTS = {
    "Dependency": "plan",
    "PipelinePlan": "plan",
    "Region": "plan",
    "Resource": "plan",
    "Role": "plan",
    "SourceLocation": "plan",
    "Timeline": "plan",
    "extract_plan": "extract",
    "MeasuredCapture": "measure",
    "MeasuredRegion": "measure",
    "join": "measure",
    "PipelineAnalysis": "analysis",
    "analyze_pipeline": "analysis",
    "report": "summary",
    "write_pipeline_perfetto": "perfetto",
}


def __getattr__(name: str):
    """Load the plan model, measurement, analysis, and visualization helpers on first use."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Dependency",
    "MeasuredCapture",
    "MeasuredRegion",
    "PipelineAnalysis",
    "PipelineAnnotations",
    "PipelinePlan",
    "Region",
    "Resource",
    "Role",
    "SourceLocation",
    "Timeline",
    "analyze_pipeline",
    "extract_plan",
    "join",
    "report",
    "write_pipeline_perfetto",
]
