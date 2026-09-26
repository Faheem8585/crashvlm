"""Stitch per-clip previews (original | processed) into QA sheets.

    uv run python scripts/overlay_report.py

Writes
  reports/m1/overlay_check.jpg   every clip
  reports/m1/overlay_suspect.jpg clips with a large crop or masked area, checked first
  reports/m1/overlay_suspect.txt the same list with numbers

Inspect them: no timestamps, frame counters or T0 dots may remain in the processed (right)
images, and no crash-relevant content (car, dummy, airbag) may be cropped or masked.
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image

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

    suspects = []
    for p in clips:
        crop, masked = area_stats(json.loads(p.read_text()))
        if crop > CROP_SUSPECT or masked > MASK_SUSPECT:
            suspects.append((p.parent, crop, masked))
    lines = [f"{d.name}\tcrop {c:.2f}\tmasked {m:.2f}" for d, c, m in suspects]
    (out / "overlay_suspect.txt").write_text("\n".join(lines) + "\n")
    if suspects:
        sheet([d / "preview.jpg" for d, _, _ in suspects], out / "overlay_suspect.jpg")
    print(f"{len(clips)} previews -> {out / 'overlay_check.jpg'}; {len(suspects)} suspects")


if __name__ == "__main__":
    main()
