"""Annotation records from labels written directly (not via CVAT), e.g. by a model annotator.

Input format (JSON, one entry per clip)::

    {"8204_exterior": {
        "events": {"impact_onset": 0, "max_deformation": {"t_ms": 72, "onset_visible": true}},
        "summary": {"windshield_damage": "none", "visual_severity": "medium"}}}

An event value is a time in ms or a dict with ``t_ms`` plus event attributes. Times snap to the
nearest annotation frame. The output uses exactly the schema of ``convert.convert`` (CVAT
exports), with ``source`` set to the annotator name, so every downstream step treats the two
sources alike while keeping them distinguishable.
"""

from __future__ import annotations

from typing import Any

from crashvlm.annotation.ontology import Ontology
from crashvlm.annotation.preannotate import annotation_frames, nearest


def records_from_labels(
    labels: dict[str, dict],
    metas: dict[str, dict],
    onto: Ontology,
    pass_name: str,
    annotator: str,
    config_keys: dict[int, str],
    frame_stride: int,
) -> list[dict]:
    records = []
    for cid, lab in sorted(labels.items()):
        meta = metas[cid]
        frames = annotation_frames(meta, frame_stride)
        cfg_key = config_keys.get(meta["test_no"], "?")
        allowed = set(onto.events_for(meta["role"], cfg_key))
        rec: dict[str, Any] = {
            "clip_id": cid,
            "test_no": meta["test_no"],
            "role": meta["role"],
            "config_key": cfg_key,
            "pass": pass_name,
            "annotator": annotator,
            "complete": "summary" in lab,
            "events": {},
            "summary": onto.summary_defaults(),
            "warnings": [],
        }
        for k, v in (lab.get("summary") or {}).items():
            if k not in onto.clip_summary:
                rec["warnings"].append(f"unknown summary attribute {k!r}")
                continue
            rec["summary"][k] = v
        for name, v in (lab.get("events") or {}).items():
            if name not in onto.point_events:
                rec["warnings"].append(f"unknown label {name!r}")
                continue
            if name not in allowed:
                rec["warnings"].append(f"{name} not defined for {meta['role']}/{cfg_key}")
            spec = v if isinstance(v, dict) else {"t_ms": v}
            f = nearest(frames, float(spec["t_ms"]))
            attrs = {k: d.get("default") for k, d in onto.event_attributes(name).items()}
            attrs.update({k: val for k, val in spec.items() if k != "t_ms"})
            rec["events"][name] = {
                "idx": f["idx"],
                "t_ms": f["t_ms"],
                "source": annotator,
                "attributes": attrs,
            }
        if not rec["complete"]:
            rec["warnings"].append("no summary: clip not finished")
        records.append(rec)
    return records
