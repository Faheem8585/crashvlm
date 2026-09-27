"""Structured crash-assessment reports from a video VLM (M4 zero-shot, M5 fine-tuned).

The clip is shown as a sequence of time-stamped frames. Times are given relative to the first
frame shown, never relative to impact, so the model cannot copy the NCAP protocol clock (the
"t0 unknown" setting of M3). The model answers with one JSON object following ``REPORT_SCHEMA``;
``parse_report`` extracts and validates it and records every deviation instead of silently
repairing it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

EVENTS = [
    "impact_onset",
    "airbag_deploy_start",
    "head_contact",
    "windshield_crack_onset",
    "side_glass_break_onset",
    "door_opens",
]
TEST_TYPES = {"frontal": "VTB", "side_barrier": "ITV", "side_pole": "VTP"}
SUMMARY_FIELDS = {
    "frontal_airbag_deployed": ["yes", "no", "not_visible"],
    "curtain_airbag_deployed": ["yes", "no", "not_visible"],
    "windshield_damage": ["none", "cracked", "shattered", "not_visible"],
    "door_opened": ["yes", "no", "not_visible"],
    "fire_or_fluid_leak": ["yes", "no", "not_visible"],
    "visual_severity": ["low", "medium", "high"],
}

REPORT_SCHEMA = {
    "test_type": "one of: frontal, side_barrier, side_pole",
    "events": {e: "time in ms of the first frame where it is visible, or null" for e in EVENTS},
    **{k: "one of: " + ", ".join(v) for k, v in SUMMARY_FIELDS.items()},
    "description": "two sentences on what happens to the vehicle and the dummy",
}

SYSTEM = (
    "You are an engineer evaluating vehicle crash test videos. Report only what is visible in "
    "the frames. Never guess: use null for an event you cannot see and not_visible for a part "
    "that is out of view."
)


def instruction(view: str) -> str:
    what = {
        "exterior": "an exterior high-speed camera view of the vehicle",
        "occupant": "a high-speed camera view of the driver dummy",
    }[view]
    return (
        f"The frames below are consecutive stills from {what} in a crash test. Each frame is "
        "labelled with its time in milliseconds since the first frame shown.\n"
        "Answer with a single JSON object and nothing else, using exactly these keys:\n"
        + json.dumps(REPORT_SCHEMA, indent=1)
    )


def select_frames(frames: list[dict], start_ms: float, n: int, step_ms: float) -> list[dict]:
    """``n`` frames every ``step_ms`` from ``start_ms`` (nearest available frame, no repeats)."""
    out, used = [], set()
    for k in range(n):
        target = start_ms + k * step_ms
        f = min(frames, key=lambda f, t=target: abs(f["t_ms"] - t))
        if abs(f["t_ms"] - target) <= step_ms / 2 and f["idx"] not in used:
            out.append(f)
            used.add(f["idx"])
    return out


def build_messages(images: list, rel_times: list[float], view: str) -> list[dict]:
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction(view)}]
    for im, t in zip(images, rel_times, strict=True):
        content.append({"type": "text", "text": f"t = {t:.0f} ms"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": "JSON report:"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


@dataclass
class ParsedReport:
    ok: bool
    report: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_report(text: str) -> ParsedReport:
    """Extract the JSON object; normalise values to the schema and list every problem.

    ``ok`` is False only if no JSON object can be decoded at all. Invalid field values become
    None (unknown) and are listed in ``issues``; they are never mapped to a nearby valid value.
    """
    text = _THINK.sub("", text).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = _JSON_BLOCK.search(text)
    if not m:
        return ParsedReport(False, {}, ["no JSON object"])
    try:
        raw = json.loads(m.group())
    except json.JSONDecodeError as e:
        return ParsedReport(False, {}, [f"invalid JSON: {e.msg}"])

    issues: list[str] = []
    rep: dict[str, Any] = {"events": {}}
    tt = str(raw.get("test_type", "")).strip().lower().replace(" ", "_")
    rep["test_type"] = tt if tt in TEST_TYPES else None
    if rep["test_type"] is None:
        issues.append(f"test_type {raw.get('test_type')!r}")
    ev_raw = raw.get("events") if isinstance(raw.get("events"), dict) else {}
    if not ev_raw:
        issues.append("no events object")
    for e in EVENTS:
        v = ev_raw.get(e)
        if v is None or (isinstance(v, str) and v.strip().lower() in ("null", "none", "")):
            rep["events"][e] = None
            continue
        try:
            rep["events"][e] = float(str(v).lower().replace("ms", "").strip())
        except ValueError:
            rep["events"][e] = None
            issues.append(f"{e}={v!r}")
    for k, allowed in SUMMARY_FIELDS.items():
        v = str(raw.get(k, "")).strip().lower().replace(" ", "_")
        rep[k] = v if v in allowed else None
        if rep[k] is None:
            issues.append(f"{k}={raw.get(k)!r}")
    rep["description"] = str(raw.get("description", ""))[:1000]
    return ParsedReport(True, rep, issues)
