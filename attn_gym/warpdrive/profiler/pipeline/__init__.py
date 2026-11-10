"""Declarative CuTeDSL pipeline plans, measurements, analysis, and Perfetto export."""

from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "Dependency": "plan",
    "PipelinePlan": "plan",
    "Region": "plan",
    "Resource": "plan",
    "Role": "plan",
    "SourceLocation": "plan",
    "Timeline": "plan",
    "MeasuredCapture": "measure",
    "MeasuredRegion": "measure",
    "join": "measure",
    "PipelineAnalysis": "analysis",
    "analyze_pipeline": "analysis",
    "report": "summary",
    "write_pipeline_perfetto": "perfetto",
}


def __getattr__(name: str):
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
    "PipelinePlan",
    "Region",
    "Resource",
    "Role",
    "SourceLocation",
    "Timeline",
    "analyze_pipeline",
    "join",
    "report",
    "write_pipeline_perfetto",
]
