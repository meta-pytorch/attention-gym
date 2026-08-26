from pathlib import Path

from examples import cuda_graph_trace_comparison


def test_training_loop_trace_comparison_uses_raw_and_postprocessed_captures(
    monkeypatch,
    tmp_path: Path,
):
    calls = []

    def fake_training_loop(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        cuda_graph_trace_comparison,
        "hello_world_training_loop",
        fake_training_loop,
    )

    before = tmp_path / "before.json"
    after = tmp_path / "after.json.gz"
    cuda_graph_trace_comparison._capture_trace(False, False, before)
    cuda_graph_trace_comparison._capture_trace(True, True, after)

    assert calls == [
        {
            "enable_graph_annotations": False,
            "trace_path": before,
            "trace_format": "chrome_json",
            "fix_overlapping_events": False,
        },
        {
            "enable_graph_annotations": True,
            "trace_path": after,
            "trace_format": "chrome_json",
            "fix_overlapping_events": True,
        },
    ]
