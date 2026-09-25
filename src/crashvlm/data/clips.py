"""Download a crash video, strip overlays, and write a time-stamped frame sequence.

Output per clip: ``<processed>/clips/<test_no>_<role>/`` with ``f_<idx>.jpg`` frames, a
``meta.json`` sidecar (source, crop/masks, time model, per-frame physical time) and a
``preview.jpg`` (original vs. processed) for visual QA. The source video is deleted afterwards
unless ``clips.keep_raw`` is set.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import requests
from PIL import Image, ImageDraw

from crashvlm.data.overlay import TextReader, analyze_overlay, apply_overlay


def clip_id(test_no: int, role: str) -> str:
    return f"{int(test_no)}_{role}"


def download(url: str, dest: Path, attempts: int = 5, timeout=(10, 60)) -> Path:
    """Stream ``url`` to ``dest``; on a stall or drop, resume with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    expected = 0
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                if r.status_code == 200:  # server ignored Range: start over
                    have = 0
                    expected = int(r.headers.get("content-length") or 0)
                elif r.status_code == 206:
                    expected = have + int(r.headers.get("content-length") or 0)
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if not expected or part.stat().st_size == expected:
                part.rename(dest)
                return dest
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2**attempt)
    raise OSError(f"download failed after {attempts} attempts: {url}")


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            yield i, frame.to_ndarray(format="rgb24")


def video_info(path: Path) -> dict:
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        return {
            "codec": s.codec_context.name,
            "width": s.codec_context.width,
            "height": s.codec_context.height,
            "container_fps": float(s.average_rate) if s.average_rate else None,
        }


def resize_max_side(img: np.ndarray, max_side: int) -> Image.Image:
    im = Image.fromarray(img)
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BICUBIC)
    return im


def _preview(orig: np.ndarray, proc: np.ndarray, label: str, height: int = 240) -> Image.Image:
    a = Image.fromarray(orig)
    b = Image.fromarray(proc)
    a = a.resize((round(a.width * height / a.height), height))
    b = b.resize((round(b.width * height / b.height), height))
    canvas = Image.new("RGB", (a.width + b.width + 8, height + 18), "white")
    canvas.paste(a, (0, 18))
    canvas.paste(b, (a.width + 8, 18))
    ImageDraw.Draw(canvas).text((4, 3), label, fill="black")
    return canvas


def prepare_clip(
    test_no: int,
    role: str,
    url: str,
    source_name: str,
    out_root: Path,
    tmp_dir: Path,
    cfg,
    reader: TextReader | None,
) -> dict:
    cid = clip_id(test_no, role)
    out_dir = out_root / cid
    raw = tmp_dir / f"{cid}{Path(url).suffix.lower()}"
    if not raw.exists():
        download(url, raw)
    try:
        info = video_info(raw)
        c = cfg.clips
        # pass 1: sparse full-resolution sample for overlay analysis
        sample: dict[int, np.ndarray] = {}
        n_frames = 0
        for i, img in iter_frames(raw):
            n_frames = i + 1
            if i % c.analysis_every == 0:
                sample[i] = img
        if n_frames == 0:
            raise ValueError("no decodable frames")
        ov = analyze_overlay(sample, reader, cfg)
        tm = ov.time_model

        # pass 2: cropped + masked, strided, resized frames
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        preview_idx = min(range(0, n_frames, c.stride), key=lambda i: abs(tm.t_ms(i) - 60.0))
        for i, img in iter_frames(raw):
            if i % c.stride:
                continue
            proc = apply_overlay(img, ov.crop, ov.masks, cfg.overlay.fill)
            fname = f"f_{i:04d}.jpg"
            resize_max_side(proc, c.max_side).save(out_dir / fname, quality=c.jpeg_quality)
            frames.append({"idx": i, "t_ms": round(tm.t_ms(i), 3), "file": fname})
            if i == preview_idx:
                label = f"{cid}  t={tm.t_ms(i):.0f} ms  time:{tm.source}"
                _preview(img, proc, label).save(out_dir / "preview.jpg", quality=85)
        meta = {
            "clip_id": cid,
            "test_no": int(test_no),
            "role": role,
            "source_url": url,
            "source_name": source_name,
            **info,
            "n_frames": n_frames,
            "stride": c.stride,
            "max_side": c.max_side,
            **ov.to_dict(),
            "frames": frames,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta
    finally:
        if not cfg.clips.keep_raw:
            raw.unlink(missing_ok=True)
