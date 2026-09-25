"""Collect per-clip ``meta.json`` sidecars into one table joined with test metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crashvlm.data.overlay import TimeModel


def load_clip_metas(clips_root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(clips_root).glob("*/meta.json"))]


def build_manifest(clips_root: Path, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in load_clip_metas(clips_root):
        tm = m["time_model"]
        t = [f["t_ms"] for f in m["frames"]]
        rows.append(
            {
                "clip_id": m["clip_id"],
                "test_no": m["test_no"],
                "role": m["role"],
                "codec": m["codec"],
                "src_width": m["width"],
                "src_height": m["height"],
                "n_frames": m["n_frames"],
                "n_saved": len(m["frames"]),
                "t_start_ms": min(t),
                "t_end_ms": max(t),
                "crop_frac": _crop_frac(m),
                "n_masks": len(m["masks"]),
                "time_source": tm["source"],
                "ms_per_frame": tm["ms_per_frame"],
                "time_agrees_convention": tm["agrees_with_convention"],
                # Only clips with a self-consistent OCR time fit are used for temporal-localisation
                # metrics; the rest still count for event classification.
                "time_verified": TimeModel(**tm).is_verified(),
            }
        )
    man = pd.DataFrame(rows)
    if man.empty:
        return man
    keep = [c for c in selection.columns if not c.endswith(("_url", "_name", "_bytes"))]
    return man.merge(selection[keep], on="test_no", how="left")


def _crop_frac(m: dict) -> float:
    x0, y0, x1, y1 = m["crop"]
    return round(1 - (x1 - x0) * (y1 - y0) / (m["width"] * m["height"]), 4)
