"""End-to-end inference for one uploaded crash video (used by the API and the demo).

video -> overlay removal (+ burned-in time if readable) -> frames every ``step_ms`` ->
backend -> report (same schema as the VLM reports) + per-frame event scores for a timeline.

Backends
- ``probe``: SigLIP embeddings + the exported linear probes. Runs on a laptop CPU/MPS.
- ``vlm``: a (fine-tuned) Qwen VLM via ``VLMRunner``. Needs a GPU.

Times are always relative to the first analysed frame: an arbitrary video has no known impact
time, which is the 't0 unknown' setting the models were evaluated in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from crashvlm.data.clips import iter_frames
from crashvlm.data.overlay import analyze_overlay, apply_overlay
from crashvlm.models.probe import decode, probe_curve
from crashvlm.models.vlm_report import EVENTS, build_messages, parse_report

DATA_CFG = Path(__file__).resolve().parents[3] / "configs" / "data.yaml"


@dataclass
class Frames:
    images: list[Image.Image]
    t_ms: list[float]  # relative to the first returned frame
    time_source: str  # "ocr" (read from the overlay) or "assumed_1000fps"
    meta: dict = field(default_factory=dict)


def load_video(path: str | Path, step_ms: float, max_frames: int, reader=None) -> Frames:
    """Decode, remove overlays, and resample to ``step_ms`` using the recovered time axis."""
    cfg = OmegaConf.load(DATA_CFG)
    raw = [img for _, img in iter_frames(Path(path))]
    if not raw:
        raise ValueError("no decodable video frames")
    sample = {i: raw[i] for i in range(0, len(raw), cfg.clips.analysis_every)}
    ov = analyze_overlay(sample, reader, cfg)
    tm = ov.time_model
    verified = tm.is_verified()
    ms_per_frame = tm.ms_per_frame if verified else 1.0
    t_all = np.arange(len(raw)) * ms_per_frame
    picked, t_rel = [], []
    target = 0.0
    for i, t in enumerate(t_all):
        if len(picked) >= max_frames:
            break
        if t + 1e-6 >= target:
            img = apply_overlay(raw[i], ov.crop, ov.masks, cfg.overlay.fill)
            im = Image.fromarray(img)
            s = cfg.clips.max_side / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)))
            picked.append(im)
            t_rel.append(float(t))
            target = t + step_ms
    return Frames(
        picked,
        t_rel,
        "ocr" if verified else "assumed_1000fps",
        {"n_decoded": len(raw), "crop": ov.crop.as_list(), "n_masks": len(ov.masks)},
    )


class ProbeBackend:
    def __init__(self, probes_path: str | Path, hub_model: str):
        import joblib

        from crashvlm.models.zero_shot import SiglipScorer

        self.bundle = joblib.load(probes_path)
        self.embedder = SiglipScorer(hub_model)

    def run(self, frames: Frames, view: str) -> dict:
        t0 = time.perf_counter()
        emb = self.embedder.image_features(frames.images).float().cpu().numpy()
        t = np.asarray(frames.t_ms)
        events, curves = {e: None for e in EVENTS}, {}
        for ev, p in self.bundle["events"].items():
            if view not in p["roles"]:
                continue
            c = probe_curve(p["clf"], emb)
            curves[ev] = [round(float(x), 4) for x in c]
            events[ev] = decode(c, t, p["threshold"], self.bundle["min_run"], self.bundle["smooth"])
        report = {"test_type": None, "events": events, "description": None}
        return {
            "report": report,
            "timeline": {"t_ms": frames.t_ms, "scores": curves},
            "latency_s": time.perf_counter() - t0,
        }


class VLMBackend:
    def __init__(self, model_id: str, adapter: str | None = None, load_in_4bit: bool = False):
        from crashvlm.models.vlm import VLMRunner

        self.runner = VLMRunner(model_id, load_in_4bit, adapter=adapter)

    def run(self, frames: Frames, view: str, n_frames: int = 24) -> dict:
        idx = np.linspace(0, len(frames.images) - 1, min(n_frames, len(frames.images))).round()
        idx = sorted({int(i) for i in idx})
        ims = [frames.images[i] for i in idx]
        ts = [frames.t_ms[i] for i in idx]
        res = self.runner.generate(build_messages(ims, ts, view))
        parsed = parse_report(res["text"])
        return {
            "report": parsed.report if parsed.ok else None,
            "parse_issues": parsed.issues,
            "raw_text": res["text"],
            "timeline": {"t_ms": ts, "scores": {}},
            "latency_s": res["latency_s"],
            "peak_vram_gb": res["peak_vram_gb"],
        }


def evaluate_video(path, view: str, backend, step_ms: float = 8, max_frames: int = 60, reader=None):
    if view not in ("exterior", "occupant"):
        raise ValueError("view must be 'exterior' or 'occupant'")
    frames = load_video(path, step_ms, max_frames, reader)
    out = backend.run(frames, view)
    out["meta"] = {"time_source": frames.time_source, "n_frames": len(frames.images), **frames.meta}
    return out, frames
