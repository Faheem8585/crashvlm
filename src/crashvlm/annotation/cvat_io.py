"""Read and write CVAT "CVAT for images 1.1" XML with frame-level tags.

Frames are packaged flat as ``<clip_id>__f_<idx>.jpg``, so names sort by clip and then time, and
every CVAT image maps back to (clip, source frame index) without a lookup table.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

NAME_RE = re.compile(r"^(?P<clip>.+)__f_(?P<idx>\d+)\.jpg$")


def frame_name(clip_id: str, idx: int) -> str:
    return f"{clip_id}__f_{idx:04d}.jpg"


def parse_frame_name(name: str) -> tuple[str, int]:
    m = NAME_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a packaged frame name: {name}")
    return m["clip"], int(m["idx"])


@dataclass
class Tag:
    label: str
    source: str = "manual"  # CVAT: manual | auto | semi-auto
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class ImageEntry:
    name: str
    width: int
    height: int
    tags: list[Tag] = field(default_factory=list)


def write_cvat_xml(images: list[ImageEntry], path: Path) -> None:
    """Write annotations for import into a CVAT image task (image ids follow sorted names)."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for i, im in enumerate(sorted(images, key=lambda e: e.name)):
        el = ET.SubElement(
            root, "image", id=str(i), name=im.name, width=str(im.width), height=str(im.height)
        )
        for tag in im.tags:
            t = ET.SubElement(el, "tag", label=tag.label, source=tag.source)
            for k, v in tag.attributes.items():
                ET.SubElement(t, "attribute", name=k).text = str(v)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_cvat_xml(path: Path) -> list[ImageEntry]:
    """Parse a CVAT for images 1.1 export (annotations.xml)."""
    root = ET.parse(path).getroot()
    out = []
    for el in root.iter("image"):
        tags = [
            Tag(
                label=t.get("label", ""),
                source=t.get("source", "manual"),
                attributes={a.get("name", ""): (a.text or "") for a in t.iter("attribute")},
            )
            for t in el.findall("tag")
        ]
        out.append(
            ImageEntry(el.get("name", ""), int(el.get("width", 0)), int(el.get("height", 0)), tags)
        )
    return out
