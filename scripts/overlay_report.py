"""Stitch per-clip previews (original | processed) into one QA sheet.

    uv run python scripts/overlay_report.py

Writes reports/m1/overlay_check.jpg. Inspect it: no timestamps, frame counters or T0 dots
may remain in the right-hand (processed) images.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    previews = sorted((Path(cfg.paths.processed) / "clips").glob("*/preview.jpg"))
    if not previews:
        raise SystemExit("no previews found; run scripts/prepare_clips.py first")
    ims = [Image.open(p) for p in previews]
    w = max(i.width for i in ims)
    sheet = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    out = Path(cfg.paths.reports) / "m1"
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / "overlay_check.jpg", quality=85)
    print(f"{len(ims)} previews -> {out / 'overlay_check.jpg'}")


if __name__ == "__main__":
    main()
