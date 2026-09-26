"""Load the annotation ontology and derive the CVAT label configuration from it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUMMARY_LABEL = "clip_summary"

# CVAT label colours (cosmetic, stable per label)
_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#9a6324"]


@dataclass
class Ontology:
    point_events: dict[str, dict[str, Any]]
    common_event_attributes: dict[str, dict[str, Any]]
    clip_summary: dict[str, dict[str, Any]]
    annotation: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def load(cls, path: str | Path = "configs/ontology.yaml") -> Ontology:
        return cls(**yaml.safe_load(Path(path).read_text()))

    def events_for(self, role: str, config_key: str) -> list[str]:
        return [
            name
            for name, spec in self.point_events.items()
            if role in spec["roles"] and config_key in spec["configs"]
        ]

    def event_attributes(self, event: str) -> dict[str, dict[str, Any]]:
        return {**self.common_event_attributes, **self.point_events[event].get("attributes", {})}

    def summary_defaults(self) -> dict[str, Any]:
        return {k: _default(v) for k, v in self.clip_summary.items()}

    # --- CVAT -------------------------------------------------------------------------------

    def cvat_labels(self) -> list[dict[str, Any]]:
        """Label definitions for CVAT's "Raw" label constructor (all labels are tags)."""
        labels = []
        names = [*self.point_events, SUMMARY_LABEL]
        for i, name in enumerate(names):
            attrs = self.clip_summary if name == SUMMARY_LABEL else self.event_attributes(name)
            labels.append(
                {
                    "name": name,
                    "color": _PALETTE[i % len(_PALETTE)],
                    "type": "tag",
                    "attributes": [_cvat_attribute(k, v) for k, v in attrs.items()],
                }
            )
        return labels


def _default(spec: dict[str, Any]) -> Any:
    if "default" in spec:
        return spec["default"]
    if spec["type"] == "select":
        return spec["values"][0]
    return {"checkbox": False, "text": ""}[spec["type"]]


def _cvat_attribute(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["type"]
    default = _default(spec)
    if kind == "checkbox":
        values = ["true" if default else "false"]
        default = values[0]
    elif kind == "select":
        values = [str(v) for v in spec["values"]]
        default = str(default)
    else:  # text
        values = [str(default)]
        default = str(default)
    return {
        "name": name,
        "mutable": False,
        "input_type": kind,
        "default_value": default,
        "values": values,
    }
