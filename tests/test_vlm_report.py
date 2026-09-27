import json

from crashvlm.models.vlm_report import build_messages, parse_report, select_frames

GOOD = {
    "test_type": "side pole",
    "events": {"impact_onset": 12, "airbag_deploy_start": "30 ms", "head_contact": None},
    "frontal_airbag_deployed": "no",
    "curtain_airbag_deployed": "Yes",
    "windshield_damage": "cracked",
    "door_opened": "no",
    "fire_or_fluid_leak": "no",
    "visual_severity": "high",
    "description": "The pole intrudes into the driver door.",
}


def test_parse_valid_report_in_code_fence_with_thinking():
    text = "<think>hmm</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    p = parse_report(text)
    assert p.ok and not p.issues
    r = p.report
    assert r["test_type"] == "side_pole" and r["curtain_airbag_deployed"] == "yes"
    assert r["events"]["impact_onset"] == 12 and r["events"]["airbag_deploy_start"] == 30
    assert r["events"]["head_contact"] is None and r["events"]["door_opens"] is None


def test_invalid_values_become_unknown_and_are_reported():
    bad = dict(GOOD, windshield_damage="a bit broken", events={"impact_onset": "early"})
    p = parse_report(json.dumps(bad))
    assert p.ok and p.report["windshield_damage"] is None
    assert p.report["events"]["impact_onset"] is None
    assert any("windshield_damage" in i for i in p.issues)
    assert any("impact_onset" in i for i in p.issues)


def test_unparseable_output():
    assert not parse_report("The car crashes into the wall.").ok
    assert not parse_report("{not json}").ok


def test_select_frames_and_messages():
    frames = [{"idx": i, "t_ms": -50.0 + i, "file": f"{i}.jpg"} for i in range(0, 350, 2)]
    sel = select_frames(frames, start_ms=-30.0, n=5, step_ms=8)
    assert [f["t_ms"] for f in sel] == [-30, -22, -14, -6, 2]
    msgs = build_messages(["img"] * 5, [0, 8, 16, 24, 32], "occupant")
    user = msgs[1]["content"]
    assert sum(c["type"] == "image" for c in user) == 5
    assert "t = 16 ms" in [c.get("text") for c in user]
