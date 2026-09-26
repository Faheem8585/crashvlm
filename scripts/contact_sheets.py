"""Render time-stamped contact sheets for reviewing / labelling clips.

    uv run python scripts/contact_sheets.py                          # overview sheet per clip
    uv run python scripts/contact_sheets.py +clip=8204_exterior +t0=-8 +t1=40   # zoom window
    uv run python scripts/contact_sheets.py +label_sheets=true      # one labelling sheet per clip
    uv run python scripts/contact_sheets.py '+zooms="8204_exterior:-8:40,8204_occupant:6:70"'

Overview: every 16 ms from the start of the clip, 6 columns, frame time printed on each tile.
Zoom: every annotation frame (8 ms) between t0 and t1, larger tiles.
Sheets go to <data>/annotation/sheets/.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from crashvlm.annotation.ontology import Ontology
from crashvlm.annotation.preannotate import annotation_frames
from crashvlm.data.manifest import load_clip_metas


def render(clip_dir: Path, frames: list[dict], cols: int, tile_w: int, title: str) -> Image.Image:
    tiles = []
    font = ImageFont.load_default(size=max(12, tile_w // 14))
    for f in frames:
        im = Image.open(clip_dir / f["file"]).convert("RGB")
        im = im.resize((tile_w, round(im.height * tile_w / im.width)))
        d = ImageDraw.Draw(im)
        label = f"{f['t_ms']:+.0f} ms"
        box = d.textbbox((4, 4), label, font=font)
        d.rectangle(box, fill="black")
        d.text((4, 4), label, fill="yellow", font=font)
        tiles.append(im)
    th = max(t.height for t in tiles)
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * tile_w + (cols - 1) * 4, 28 + rows * (th + 4)), "white")
    ImageDraw.Draw(sheet).text((4, 4), title, fill="black", font=ImageFont.load_default(size=16))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (tile_w + 4), 28 + (i // cols) * (th + 4)))
    return sheet


@hydra.main(config_path="../configs", config_name="annotation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    stride = onto.annotation["frame_stride"]
    clips_root = Path(cfg.paths.processed) / "clips"
    out = Path(cfg.annotation.out) / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    metas = {m["clip_id"]: m for m in load_clip_metas(clips_root)}

    zooms = []
    if cfg.get("clip"):
        zooms.append((cfg.clip, float(cfg.t0), float(cfg.t1)))
    for z in str(cfg.get("zooms") or "").split(","):
        if z.strip():
            cid, t0, t1 = z.strip().split(":")
            zooms.append((cid, float(t0), float(t1)))
    if zooms:
        for cid, t0, t1 in zooms:
            frames = [f for f in annotation_frames(metas[cid], stride) if t0 <= f["t_ms"] <= t1]
            sheet = render(clips_root / cid, frames, 4, 360, f"{cid}  zoom")
            path = out / f"{cid}_zoom_{int(t0)}_{int(t1)}.jpg"
            sheet.save(path, quality=88)
            print(path)
        return

    if cfg.get("label_sheets"):
        # one sheet per clip for labelling: 8 ms steps over [-10, 110] ms at large size (impact,
        # airbag, head contact, max crush), 16 ms steps afterwards (door opening, rebound)
        for cid, m in sorted(metas.items()):
            fr = annotation_frames(m, stride)
            dense = [f for f in fr if -10 <= f["t_ms"] <= 110]
            late = [f for f in fr if f["t_ms"] > 110][1::2]
            top = render(clips_root / cid, dense, 4, 300, f"{cid}  ({m['source_name']})")
            bottom = render(clips_root / cid, late, 7, 170, "later, 16 ms steps")
            sheet = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height))
            sheet.paste(top, (0, 0))
            sheet.paste(bottom, (0, top.height))
            sheet.save(out / f"{cid}_label.jpg", quality=85)
        print(f"{len(metas)} labelling sheets -> {out}")
        return

    for cid, m in sorted(metas.items()):
        frames = annotation_frames(m, stride)[::2]  # 16 ms steps
        title = f"{cid}  ({m['source_name']})"
        render(clips_root / cid, frames, 6, 240, title).save(out / f"{cid}.jpg", quality=85)
    print(f"{len(metas)} overview sheets -> {out}")


if __name__ == "__main__":
    main()
