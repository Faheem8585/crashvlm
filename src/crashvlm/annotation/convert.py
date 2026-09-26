"""Convert CVAT exports into one annotation record per clip (JSONL).

Record schema::

    {"clip_id", "test_no", "role", "config_key", "pass", "annotator",
     "complete": bool,                       # the clip carries a clip_summary tag
     "events": {label: {"idx", "t_ms", "source", "attributes"}},
     "summary": {attribute: value},
     "warnings": [str]}

A point event that is absent from ``events`` means "did not happen / not visible". Clips
without a ``clip_summary`` tag are marked incomplete and excluded from training and evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from crashvlm.annotation.cvat_io import ImageEntry, parse_frame_name
from crashvlm.annotation.ontology import SUMMARY_LABEL, Ontology


def _cast(value: str, spec: dict[str, Any]) -> Any:
    if spec["type"] == "checkbox":
        return str(value).lower() == "true"
    return value


def convert(
    images: list[ImageEntry],
    metas: dict[str, dict],
    onto: Ontology,
    pass_name: str,
    annotator: str,
    config_keys: dict[int, str],
) -> list[dict]:
    """``metas``: clip meta.json dicts by clip_id; ``config_keys``: test_no -> VTB/ITV/VTP."""
    per_clip: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    seen_clips: set[str] = set()
    for im in images:
        clip, idx = parse_frame_name(im.name)
        seen_clips.add(clip)
        for tag in im.tags:
            per_clip[clip].append((idx, tag))

    records = []
    for clip in sorted(seen_clips):
        meta = metas[clip]
        t_of = {f["idx"]: f["t_ms"] for f in meta["frames"]}
        cfg_key = config_keys.get(meta["test_no"], "?")
        allowed = set(onto.events_for(meta["role"], cfg_key))
        rec: dict[str, Any] = {
            "clip_id": clip,
            "test_no": meta["test_no"],
            "role": meta["role"],
            "config_key": cfg_key,
            "pass": pass_name,
            "annotator": annotator,
            "complete": False,
            "events": {},
            "summary": onto.summary_defaults(),
            "warnings": [],
        }
        for idx, tag in sorted(per_clip[clip], key=lambda p: p[0]):
            if tag.label == SUMMARY_LABEL:
                if rec["complete"]:
                    rec["warnings"].append("duplicate clip_summary; first one kept")
                    continue
                for k, spec in onto.clip_summary.items():
                    if k in tag.attributes:
                        rec["summary"][k] = _cast(tag.attributes[k], spec)
                rec["complete"] = True
                continue
            if tag.label not in onto.point_events:
                rec["warnings"].append(f"unknown label {tag.label!r} at frame {idx}")
                continue
            if tag.label not in allowed:
                rec["warnings"].append(f"{tag.label} not defined for {meta['role']}/{cfg_key}")
            if tag.label in rec["events"]:
                rec["warnings"].append(f"duplicate {tag.label} at frame {idx}; earliest kept")
                continue
            specs = onto.event_attributes(tag.label)
            rec["events"][tag.label] = {
                "idx": idx,
                "t_ms": t_of.get(idx),
                "source": tag.source,
                "attributes": {
                    k: _cast(v, specs[k]) for k, v in tag.attributes.items() if k in specs
                },
            }
        if not rec["complete"]:
            rec["warnings"].append("no clip_summary tag: clip not finished")
        records.append(rec)
    return records
