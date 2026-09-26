"""Build CVAT image entries for an annotation pass, optionally with machine pre-labels.

Pre-label sources (``source="auto"`` in CVAT, so the export shows which ones were kept):

* ``t0``: impact onset at t = 0 ms, for clips whose overlay time is verified. By NCAP
  definition time zero is first contact, so this label comes from the measurement system.
* ``zero_shot``: onsets proposed by an image-text model (``crashvlm.models.zero_shot``),
  read from a predictions JSONL file.
"""

from __future__ import annotations

from crashvlm.annotation.cvat_io import ImageEntry, Tag, frame_name
from crashvlm.annotation.ontology import Ontology


def annotation_frames(meta: dict, frame_stride: int) -> list[dict]:
    """Every ``frame_stride``-th processed frame (processed frames are already strided)."""
    return meta["frames"][::frame_stride]


def nearest(frames: list[dict], t_ms: float) -> dict:
    return min(frames, key=lambda f: abs(f["t_ms"] - t_ms))


def build_entries(
    meta: dict,
    size: tuple[int, int],
    onto: Ontology,
    config_key: str,
    frame_stride: int,
    time_verified: bool,
    prelabels: bool,
    zero_shot: dict[str, float] | None = None,
) -> list[ImageEntry]:
    frames = annotation_frames(meta, frame_stride)
    w, h = size
    entries = {f["idx"]: ImageEntry(frame_name(meta["clip_id"], f["idx"]), w, h) for f in frames}
    if not prelabels:
        return list(entries.values())

    allowed = onto.events_for(meta["role"], config_key)
    proposals: dict[str, float] = {}
    if time_verified and "impact_onset" in allowed:
        t_min, t_max = frames[0]["t_ms"], frames[-1]["t_ms"]
        if t_min <= 0 <= t_max:
            proposals["impact_onset"] = 0.0
    for label, t in (zero_shot or {}).items():
        if label in allowed and label not in proposals and t is not None:
            proposals[label] = t
    for label, t in proposals.items():
        f = nearest(frames, t)
        defaults = {k: _str(v.get("default", "")) for k, v in onto.event_attributes(label).items()}
        entries[f["idx"]].tags.append(Tag(label, "auto", defaults))
    return list(entries.values())


def _str(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
