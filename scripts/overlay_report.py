"""Stitch per-clip previews (original | processed) into QA sheets.

    uv run python scripts/overlay_report.py

Writes
  reports/m1/overlay_check.jpg   every clip
  reports/m1/overlay_suspect.jpg clips with a large crop or masked area, checked first
  reports/m1/overlay_suspect.txt the same list with numbers

The script also scans the processed frames of every clip for a surviving red->green T0 lamp and
exits with an error if it finds one. Inspect the sheets: no timestamps, frame counters or T0
dots may remain in the processed (right) images, and no crash-relevant content (car, dummy,
airbag) may be cropped or masked.
"""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from crashvlm.data.overlay import indicator_boxes

CROP_SUSPECT = 0.35  # KARCO side panels legitimately remove about a third of the frame
MASK_SUSPECT = 0.10


def sheet(paths: list[Path], out: Path) -> None:
    ims = [Image.open(p) for p in paths]
    w = max(i.width for i in ims)
    canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out, quality=85)


def lamp_left(clip_dir: Path, meta: dict, n: int = 15) -> bool:
    """Leak check on the *output*: does a red->green T0 lamp survive in the processed frames?"""
    frames = meta["frames"][:: max(1, len(meta["frames"]) // n)]
    imgs = [np.asarray(Image.open(clip_dir / f["file"]).convert("RGB")) for f in frames]
    return bool(indicator_boxes(imgs, min_area=25))  # processed frames are downscaled


def area_stats(meta: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = meta["crop"]
    kept = (x1 - x0) * (y1 - y0)
    masked = sum((b[2] - b[0]) * (b[3] - b[1]) for b in meta["masks"])
    return 1 - kept / (meta["width"] * meta["height"]), masked / max(kept, 1)


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    clips = sorted((Path(cfg.paths.processed) / "clips").glob("*/meta.json"))
    if not clips:
        raise SystemExit("no clips found; run scripts/prepare_clips.py first")
    out = Path(cfg.paths.reports) / "m1"
    out.mkdir(parents=True, exist_ok=True)
    sheet([p.parent / "preview.jpg" for p in clips], out / "overlay_check.jpg")

    suspects, lamps = [], []
    for p in clips:
        meta = json.loads(p.read_text())
        crop, masked = area_stats(meta)
        if crop > CROP_SUSPECT or masked > MASK_SUSPECT:
            suspects.append((p.parent, crop, masked))
        if lamp_left(p.parent, meta):
            lamps.append(p.parent)
    lines = [f"{d.name}\tcrop {c:.2f}\tmasked {m:.2f}" for d, c, m in suspects]
    lines += [f"{d.name}\tLEAK: red->green T0 lamp visible in processed frames" for d in lamps]
    (out / "overlay_suspect.txt").write_text("\n".join(lines) + "\n")
    if suspects:
        sheet([d / "preview.jpg" for d, _, _ in suspects], out / "overlay_suspect.jpg")
    print(f"{len(clips)} previews -> {out / 'overlay_check.jpg'}; {len(suspects)} suspects")
    if lamps:
        raise SystemExit(f"T0 LAMP LEAK in {len(lamps)} clips: {[d.name for d in lamps]}")
    print("lamp check: no red->green T0 lamp in any processed clip")


if __name__ == "__main__":
    main()
