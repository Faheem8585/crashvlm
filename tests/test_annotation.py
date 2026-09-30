import json

import numpy as np
import pytest

from crashvlm.annotation.convert import convert
from crashvlm.annotation.cvat_io import (
    ImageEntry,
    Tag,
    frame_name,
    parse_frame_name,
    read_cvat_xml,
    write_cvat_xml,
)
from crashvlm.annotation.iaa import cohen_kappa, event_agreement, summary_agreement
from crashvlm.annotation.ontology import SUMMARY_LABEL, Ontology
from crashvlm.annotation.preannotate import build_entries
from crashvlm.models.zero_shot import curve_to_event, pos_neg_scores

ONTO = Ontology.load("configs/ontology.yaml")


def meta(clip_id="100_occupant", role="occupant", test_no=100, t0_frame=48):
    frames = [
        {"idx": i, "t_ms": float(i - t0_frame), "file": f"f_{i:04d}.jpg"} for i in range(0, 348, 4)
    ]
    return {"clip_id": clip_id, "test_no": test_no, "role": role, "frames": frames}


# --- ontology / CVAT labels ----------------------------------------------------------------


def test_cvat_labels_are_tags_with_valid_defaults():
    labels = ONTO.cvat_labels()
    names = [lab["name"] for lab in labels]
    assert SUMMARY_LABEL in names and "airbag_deploy_start" in names
    for lab in labels:
        assert lab["type"] == "tag"
        for a in lab["attributes"]:
            if a["input_type"] in ("select", "checkbox"):
                assert a["default_value"] in a["values"], (lab["name"], a["name"])
    json.dumps(labels)  # serialisable for the CVAT raw editor


def test_yes_no_values_stay_strings():
    door = next(
        a for lab in ONTO.cvat_labels() for a in lab["attributes"] if a["name"] == "door_opened"
    )
    assert door["values"] == ["yes", "no", "not_visible"]


def test_events_for_role_and_config():
    assert "max_deformation" in ONTO.events_for("exterior", "VTB")
    assert "max_deformation" not in ONTO.events_for("occupant", "VTB")
    assert "side_glass_break_onset" not in ONTO.events_for("exterior", "VTB")


# --- CVAT IO ------------------------------------------------------------------------------


def test_frame_name_roundtrip():
    assert parse_frame_name(frame_name("10186_exterior", 12)) == ("10186_exterior", 12)
    assert parse_frame_name("some/dir/10186_exterior__f_0012.jpg") == ("10186_exterior", 12)
    with pytest.raises(ValueError):
        parse_frame_name("f_0012.jpg")


def test_xml_roundtrip(tmp_path):
    ims = [
        ImageEntry(
            frame_name("1_exterior", 8),
            512,
            384,
            [Tag("impact_onset", "auto", {"onset_visible": "true"})],
        ),
        ImageEntry(frame_name("1_exterior", 0), 512, 384),
    ]
    write_cvat_xml(ims, tmp_path / "a.xml")
    back = read_cvat_xml(tmp_path / "a.xml")
    assert [b.name for b in back] == sorted(i.name for i in ims)
    assert back[1].tags[0].label == "impact_onset" and back[1].tags[0].source == "auto"
    assert back[1].tags[0].attributes == {"onset_visible": "true"}


# --- pre-labels -----------------------------------------------------------------------------


def test_t0_prelabel_only_for_verified_clips_and_on_nearest_frame():
    m = meta()
    ents = build_entries(m, (512, 384), ONTO, "VTB", 2, True, True)
    tagged = [(e.name, t.label) for e in ents for t in e.tags]
    assert tagged == [(frame_name(m["clip_id"], 48), "impact_onset")]
    assert all(not e.tags for e in build_entries(m, (512, 384), ONTO, "VTB", 2, False, True))
    assert all(not e.tags for e in build_entries(m, (512, 384), ONTO, "VTB", 2, True, False))
    assert len(ents) == len(m["frames"][::2])


def test_zero_shot_prelabels_respect_role():
    m = meta()
    zs = {"airbag_deploy_start": 20.0, "max_deformation": 90.0}  # max_deformation: exterior only
    ents = build_entries(m, (512, 384), ONTO, "VTB", 2, False, True, zs)
    labels = [t.label for e in ents for t in e.tags]
    assert labels == ["airbag_deploy_start"]


# --- conversion --------------------------------------------------------------------------------


def test_convert_records():
    m = meta()
    ims = [
        ImageEntry(
            frame_name(m["clip_id"], 0),
            512,
            384,
            [Tag(SUMMARY_LABEL, "manual", {"view_usable": "false", "door_opened": "no"})],
        ),
        ImageEntry(
            frame_name(m["clip_id"], 72),
            512,
            384,
            [Tag("airbag_deploy_start", "auto", {"onset_visible": "true"})],
        ),
        ImageEntry(
            frame_name(m["clip_id"], 80),
            512,
            384,
            [Tag("airbag_deploy_start", "manual"), Tag("bogus")],
        ),
        ImageEntry(frame_name("200_exterior", 0), 512, 384),
    ]
    metas = {m["clip_id"]: m, "200_exterior": meta("200_exterior", "exterior", 200)}
    recs = {
        r["clip_id"]: r for r in convert(ims, metas, ONTO, "pass_a", "A", {100: "VTB", 200: "ITV"})
    }
    r = recs[m["clip_id"]]
    assert (
        r["complete"]
        and r["summary"]["view_usable"] is False
        and r["summary"]["door_opened"] == "no"
    )
    ev = r["events"]["airbag_deploy_start"]
    assert ev["idx"] == 72 and ev["t_ms"] == 24.0 and ev["source"] == "auto"
    assert ev["attributes"]["onset_visible"] is True
    assert any("duplicate" in w for w in r["warnings"]) and any("bogus" in w for w in r["warnings"])
    assert recs["200_exterior"]["complete"] is False  # no summary: unfinished clip


# --- agreement ------------------------------------------------------------------------------


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert np.isnan(cohen_kappa([], []))


def rec(events, summary=None):
    return {"events": {k: {"t_ms": v} for k, v in events.items()}, "summary": summary or {}}


def test_event_and_summary_agreement():
    a = {"c1": rec({"head_contact": 40.0}), "c2": rec({"head_contact": 60.0}), "c3": rec({})}
    b = {"c1": rec({"head_contact": 48.0}), "c2": rec({"head_contact": 100.0}), "c3": rec({})}
    out = event_agreement(a, b, "head_contact")
    assert out["n_both"] == 2 and out["median_abs_diff_ms"] == pytest.approx(24.0)
    assert out["within_8ms"] == pytest.approx(0.5) and out["presence_agreement"] == 1.0
    s = summary_agreement(
        {"c1": rec({}, {"x": "yes"}), "c2": rec({}, {"x": "no"})},
        {"c1": rec({}, {"x": "yes"}), "c2": rec({}, {"x": "yes"})},
        "x",
    )
    assert s["agreement"] == 0.5


# --- zero-shot curve logic ----------------------------------------------------------------------


def test_curve_onset_and_peak():
    t = list(range(0, 100, 4))
    s = np.zeros(len(t))
    s[10:] = 0.9  # event from t=40
    s[3] = 0.95  # single-frame spike must not trigger an onset
    assert curve_to_event(s, t, "onset", 0.5, min_run=3, window=1) == 40
    peak = np.zeros(len(t))
    peak[15] = 0.8
    assert curve_to_event(peak, t, "peak", 0.5, window=1) == 60
    assert curve_to_event(np.zeros(len(t)), t, "onset") is None


def test_pos_neg_scores():
    logits = np.array([[5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]])
    p = pos_neg_scores(logits, n_pos=2)
    assert p[0] > 0.99 and p[1] < 0.01


def test_records_from_direct_labels():
    from crashvlm.annotation.direct import records_from_labels

    m = meta("300_exterior", "exterior", 300)
    labels = {
        "300_exterior": {
            "events": {
                "impact_onset": 1,  # snaps to the nearest 8 ms annotation frame
                "max_deformation": {"t_ms": 70, "onset_visible": False},
                "head_contact": 40,  # occupant-only event: kept but warned
            },
            "summary": {"windshield_damage": "cracked", "bogus": 1},
        }
    }
    (r,) = records_from_labels(
        labels, {"300_exterior": m}, ONTO, "model_a", "model", {300: "VTB"}, 2
    )
    assert r["complete"] and r["summary"]["windshield_damage"] == "cracked"
    assert (
        r["events"]["impact_onset"]["t_ms"] == 0.0
        and r["events"]["impact_onset"]["source"] == "model"
    )
    assert r["events"]["max_deformation"]["attributes"]["onset_visible"] is False
    assert r["events"]["max_deformation"]["t_ms"] in (64.0, 72.0)
    assert any("head_contact not defined" in w for w in r["warnings"])
    assert any("bogus" in w for w in r["warnings"])
