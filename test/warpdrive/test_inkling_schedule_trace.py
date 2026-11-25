"""Offline schedule labeling of two visits to the same physical core layer."""

import pytest

from attn_gym.warpdrive.examples.inkling_megakernel.ref import InklingConfig
from attn_gym.warpdrive.examples.inkling_megakernel.schedule_trace import annotate_round

CONFIG = InklingConfig(num_layers=3, loop_start=1, loop_end=2, num_passes=2, full_layers=(1,))


def round_trace() -> dict:
    names = [
        "position",
        "draft_backbone_h2304",
        "draft_head_h2304_d1_s1",
        "cat",
        "t2_swa1024_h2304_s1",
        "t2_full_h2304_s1",
        "msg_h2304_t2_s1",
        "t2_full_msg_h2304_s1",
        "t2_swa1024_h2304_s1",
        "head_h2304_v200058_t2_s1",
        "copy_target_features",
        "at::native::ArgMaxOps<float>",
        "sum",
        "draft_ctx_h2304",
        "advance_position",
    ]
    events = []
    for i, name in enumerate(names):
        if name.startswith(("draft_", "t2_", "msg_", "head_")):
            name = "warpdrive_inkling_" + name
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": i * 10,
                "dur": 4,
                "name": name,
                "args": {"graph id": 2, "correlation": 9},
            }
        )
    # Same stream and graph, but a separate replay's activity is not part of the commit.
    events.append({**events[-1], "ts": 2000, "args": {"graph id": 2, "correlation": 10}})
    return {"traceEvents": list(reversed(events)), "schemaVersion": 1}


def test_annotate_round():
    trace = round_trace()
    annotated = annotate_round(trace, CONFIG)
    original = trace["traceEvents"]
    assert annotated["traceEvents"][: len(original)] == original
    assert annotated["schemaVersion"] == 1
    slices = {e["name"]: e for e in annotated["traceEvents"] if e.get("cat") == "Inkling schedule"}
    for label, start, end in [
        ("Draft 1 token", 0, 24),
        ("Verify 2 positions", 30, 104),
        ("Core pass 2/2", 60, 74),
        ("Visit 2: L1 full", 70, 74),
        ("Target vocabulary head", 90, 94),
        ("Accept prefix and commit", 110, 144),
    ]:
        assert (slices[label]["ts"], slices[label]["dur"]) == (start, end - start)
    assert "emitted_tokens" not in slices["Accept prefix and commit"]["args"]


def test_overlapping_gpu_timestamps_do_not_merge_phase_tracks():
    trace = round_trace()
    head = next(e for e in trace["traceEvents"] if "draft_head" in e["name"])
    head["dur"] = 11  # reported end 31, but verify starts at 30 on the same stream
    annotated = annotate_round(trace, CONFIG, emitted_tokens=1)
    spans = [e for e in annotated["traceEvents"] if e.get("cat") == "Inkling schedule"]
    tids = {e["tid"] for e in spans}
    assert len(tids) == 3 and min(tids) > 0 and head["tid"] not in tids
    draft = next(e for e in spans if e["name"] == "Draft 1 token")
    assert draft["ts"] + draft["dur"] == 30
    assert next(e for e in spans if "commit |" in e["name"])["args"]["emitted_tokens"] == 1
    assert head["dur"] == 11


@pytest.mark.parametrize("missing", ["_full_msg_", "ArgMaxOps", "draft_ctx_"])
def test_incomplete_schedule_is_not_labeled(missing):
    trace = round_trace()
    trace["traceEvents"] = [e for e in trace["traceEvents"] if missing not in e["name"]]
    with pytest.raises(ValueError, match="expected"):
        annotate_round(trace, CONFIG)
