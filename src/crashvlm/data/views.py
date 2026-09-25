"""Normalise NHTSA camera-view file names and pick canonical views per test configuration.

Two roles per test: ``exterior`` (structural deformation) and ``occupant`` (a camera on the
driver dummy: airbag deployment, head contact). The chosen file name is kept in the catalog, so
the exact camera behind each role is always traceable.

Labs name cameras differently ("Camera No. 4 - Left Front Half.avi", "04 LEFT FRONT HALF.wmv",
"Onboard – Driver Front.mp4"). We only use high-speed views; real-time, pan, documentation and
"for web" edits are excluded because their time base is different or edited.
"""

from __future__ import annotations

import re
from typing import Any

# Priority-ordered regexes on the normalised name, per NHTSA testConfigurationKey.
CANONICAL_VIEWS: dict[str, dict[str, list[str]]] = {
    "VTB": {  # vehicle into barrier (frontal)
        "exterior": [r"^left front half$", r"^left overall$", r"^left angle$"],
        "occupant": [
            r"^driver (dummy )?on-?board( view)?$",
            r"^on-?board driver$",
            r"^on-?board - driver",
            r"^on-?board driver side$",  # older frontal tests (2012-2019)
            r"^driver close-?up$",  # last resort: exterior close-up of the driver
        ],
    },
    "ITV": {  # moving deformable barrier into vehicle (side)
        "exterior": [r"^overhead wide(-| )?view", r"^overhead close-?up"],
        "occupant": [
            r"^driver ?dummy front view",
            r"^on-?board - driver front",
            r"^driver ?dummy side view",
        ],
    },
    "VTP": {  # vehicle into pole (side)
        "exterior": [r"^overhead wide(-| )?view", r"^overhead close-?up"],
        "occupant": [
            r"^(on-?board - )?dummy front view",
            r"^(on-?board - )?dummy side view",
        ],
    },
}

EXCLUDE = re.compile(r"real[- ]?time|documenta|for web|pan view", re.I)


def normalize_view_name(original: str) -> str:
    s = original.replace("–", "-").replace("—", "-").replace("_", " ")
    s = re.sub(r"\.(wmv|avi|mp4|mov|mpg)$", "", s.strip(), flags=re.I)
    s = re.sub(r"^.*?camera\s*(no\.?)?\s*\d+\s*-\s*", "", s, flags=re.I)
    s = re.sub(r"^\d{1,2}\s+", "", s)  # "04 LEFT FRONT HALF"
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\(on-?board\)", "", s).strip()
    return s


def is_high_speed(original: str) -> bool:
    return EXCLUDE.search(original) is None


def select_views(
    config_key: str, videos: list[dict[str, Any]], max_bytes: float | None = None
) -> dict[str, dict[str, Any]]:
    """Return {role: video_record} for the canonical roles found in ``videos``.

    ``videos`` are NHTSA multimedia records with ``originalFileName``, ``media``, ``fileSize``.
    """
    rules = CANONICAL_VIEWS.get(config_key, {})
    chosen: dict[str, dict[str, Any]] = {}
    candidates = []
    for v in videos or []:
        name = v.get("originalFileName") or ""
        if not is_high_speed(name):
            continue
        size = float(v.get("fileSize") or 0)
        if max_bytes and size > max_bytes:
            continue
        candidates.append((normalize_view_name(name), v))
    for role, patterns in rules.items():
        for pat in patterns:
            hits = [v for norm, v in candidates if re.search(pat, norm)]
            if hits:
                chosen[role] = min(hits, key=lambda v: int(v.get("index") or 0))
                break
    return chosen
