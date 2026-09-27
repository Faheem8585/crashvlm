import json

from crashvlm.train.sft_data import make_examples, target_report


def rec(role="occupant", events=None, usable=True):
    return {
        "clip_id": f"1_{role}",
        "test_no": 1,
        "role": role,
        "complete": True,
        "events": events or {},
        "summary": {"view_usable": usable, "frontal_airbag_deployed": "yes"},
    }


def ev(t, visible=True):
    return {"t_ms": t, "attributes": {"onset_visible": visible}}


def test_target_times_relative_to_first_frame_and_window():
    r = rec(
        events={
            "airbag_deploy_start": ev(22.0),
            "head_contact": ev(400.0),
            "impact_onset": ev(-2.0, visible=False),
        }
    )
    t = target_report(r, "VTB", start_abs=-30.0, window_ms=184)
    assert t["events"]["airbag_deploy_start"] == 52  # 22 - (-30)
    assert t["events"]["impact_onset"] == 28  # hidden onset still has a first visible frame
    assert t["events"]["head_contact"] is None  # outside the shown window
    assert t["test_type"] == "frontal" and t["curtain_airbag_deployed"] == "not_visible"
    assert "airbag deployment at 52 ms" in t["description"]


def test_make_examples_skips_unusable_and_uses_crops():
    frames = [{"idx": i, "t_ms": -50.0 + 4 * i, "file": f"f{i}.jpg"} for i in range(90)]
    index = {"1_occupant": {"frames": frames}}
    exs = make_examples(
        [rec(events={"airbag_deploy_start": ev(22.0)})],
        index,
        {1: "VTB"},
        n_crops=3,
        jitter_ms=40,
        n_frames=10,
        step_ms=8,
        seed=0,
    )
    assert len(exs) == 3 and all(e["rel_ms"][0] == 0 for e in exs)
    assert len({json.loads(e["target"])["events"]["airbag_deploy_start"] for e in exs}) > 1
    assert make_examples([rec(usable=False)], index, {1: "VTB"}, 3, 40, 10, 8, 0) == []
