"""Instruction-tuning examples for M5, in exactly the M4 prompt format.

Target = the JSON report the model should write, built from:
- ``test_type``: NHTSA metadata (independent of the annotator),
- event times: annotation labels, converted to time since the first frame shown; events
  outside the shown window, or not visible at onset, become null,
- summary fields: annotation labels (tier C fields included so the schema stays complete;
  they are never used as evaluation targets),
- ``description``: a deterministic template of the labelled facts. It is not free text, so the
  model is not trained to invent narrative.

Each training clip is used several times with different random start crops (augmentation,
and the model must not learn a fixed position of the impact).
"""

from __future__ import annotations

import json

import numpy as np

from crashvlm.models.vlm_report import EVENTS, SUMMARY_FIELDS, TEST_TYPES, select_frames

CODE_TO_TYPE = {v: k for k, v in TEST_TYPES.items()}
NICE = {
    "impact_onset": "impact",
    "airbag_deploy_start": "airbag deployment",
    "head_contact": "head contact",
    "windshield_crack_onset": "windshield cracking",
    "side_glass_break_onset": "side window breaking",
    "door_opens": "a door opening",
}


def target_report(rec: dict, config_key: str, start_abs: float, window_ms: float) -> dict:
    events = {}
    for e in EVENTS:
        ev = rec["events"].get(e)
        rel = None
        if ev:  # first frame where visible, as the schema asks (also if onset was hidden)
            r = ev["t_ms"] - start_abs
            if 0 <= r <= window_ms:
                rel = round(float(r))
        events[e] = rel
    s = rec["summary"]
    rep = {"test_type": CODE_TO_TYPE[config_key], "events": events}
    for k in SUMMARY_FIELDS:
        rep[k] = s.get(k, "not_visible")
    seen = [
        f"{NICE[e]} at {t} ms"
        for e, t in sorted(events.items(), key=lambda x: (x[1] is None, x[1]))
        if t is not None
    ]
    kind = CODE_TO_TYPE[config_key].replace("_", " ").capitalize()
    view = "exterior" if rec["role"] == "exterior" else "driver"
    rep["description"] = f"{kind} crash test, {view} view. " + (
        "Visible: " + ", ".join(seen) + "." if seen else "No listed event is clearly visible."
    )
    return rep


def make_examples(
    recs: list[dict],
    index: dict,
    config_keys: dict[int, str],
    n_crops: int,
    jitter_ms: float,
    n_frames: int,
    step_ms: float,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    window = (n_frames - 1) * step_ms
    for r in recs:
        if not r["complete"] or not r["summary"].get("view_usable", True):
            continue
        info = index[r["clip_id"]]
        cfg = config_keys[int(r["test_no"])]
        for _ in range(n_crops):
            crop_start = info["frames"][0]["t_ms"] + float(rng.uniform(0, jitter_ms))
            frames = select_frames(info["frames"], crop_start, n_frames, step_ms)
            start = frames[0]["t_ms"]  # times are relative to the first frame shown
            out.append(
                {
                    "clip_id": r["clip_id"],
                    "role": r["role"],
                    "files": [f["file"] for f in frames],
                    "rel_ms": [f["t_ms"] - start for f in frames],
                    "target": json.dumps(target_report(r, cfg, start, window)),
                }
            )
    return out
