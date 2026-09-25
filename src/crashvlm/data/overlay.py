"""Remove burned-in overlays from NCAP high-speed clips and recover physical time from them.

Different test labs burn different overlays into the video: uniform side/top/bottom panels with
frame number, time and a red/green T0 indicator (KARCO, MGA), or small text boxes drawn over the
scene (TRC, Calspan). Left in place, these leak the answer to "when did the impact happen", so
models must never see them. We

1. crop uniform, temporally stable border bands (``detect_border_bands``),
2. mask text boxes that stay at a fixed position through the clip (``persistent_clusters``),
3. read the numbers in those boxes and fit ``t_ms = a * frame_idx + b`` (``fit_time_model``),
   which gives each frame its physical time and checks the NCAP convention (1000 fps, -50 ms).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersects(self, other: Box) -> bool:
        return (
            self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1
        )

    def union(self, other: Box) -> Box:
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def expand(self, m: int, w: int, h: int) -> Box:
        return Box(
            max(0, self.x0 - m), max(0, self.y0 - m), min(w, self.x1 + m), min(h, self.y1 + m)
        )

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


class TextReader(Protocol):
    def __call__(self, img: np.ndarray) -> list[tuple[Box, str, float]]: ...


class RapidOCRReader:
    """OCR via rapidocr-onnxruntime (CPU; models ship inside the wheel)."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def __call__(self, img: np.ndarray) -> list[tuple[Box, str, float]]:
        result, _ = self.engine(np.ascontiguousarray(img[..., ::-1]))  # RGB -> BGR
        out = []
        for pts, text, score in result or []:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append(
                (
                    Box(int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1),
                    text,
                    float(score),
                )
            )
        return out


# --- 1. border bands -------------------------------------------------------------------------


def _line_uniformity(lines: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """lines: (T, N, L, 3). Returns per-line (uniform fraction, temporal std of line colour)."""
    lines = lines.astype(np.int16)
    med = np.median(lines, axis=2, keepdims=True)
    close = np.abs(lines - med).max(axis=-1) <= tol
    frac = close.mean(axis=2).mean(axis=0)
    colour_std = med[:, :, 0, :].std(axis=0).max(axis=-1)
    return frac, colour_std


def _band_extent(is_band: np.ndarray, max_gap: int = 2) -> int:
    """Number of lines from the edge that belong to the band (tolerating short gaps)."""
    last, gap = -1, 0
    for i, b in enumerate(is_band):
        if b:
            last, gap = i, 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last + 1


def detect_border_bands(
    frames: list[np.ndarray], max_frac: float = 0.3, uniform_frac: float = 0.6, tol: int = 12
) -> Box:
    """Crop box (in full-frame pixels) that removes uniform, static border panels."""
    stack = np.stack(frames)  # (T, H, W, 3)
    _, h, w, _ = stack.shape
    mh, mw = int(h * max_frac), int(w * max_frac)

    def extent(lines: np.ndarray) -> int:
        frac, cstd = _line_uniformity(lines, tol)
        return _band_extent((frac >= uniform_frac) & (cstd <= tol))

    top = extent(stack[:, :mh])
    bottom = extent(stack[:, h - mh :][:, ::-1])
    left = extent(stack[:, :, :mw].transpose(0, 2, 1, 3))
    right = extent(stack[:, :, w - mw :].transpose(0, 2, 1, 3)[:, ::-1])
    return Box(left, top, w - right, h - bottom)


# --- 2. persistent text boxes ----------------------------------------------------------------


@dataclass
class TextCluster:
    box: Box  # union of all matched boxes (mask extent)
    frames: set[int] = field(default_factory=set)
    readings: list[tuple[int, str]] = field(default_factory=list)
    anchor: Box | None = None  # first sighting; matching is against this, never the union

    def __post_init__(self):
        if self.anchor is None:
            self.anchor = self.box

    def matches(self, box: Box, tol: float) -> bool:
        a = self.anchor
        aligned = abs(a.x0 - box.x0) <= tol or abs(a.x1 - box.x1) <= tol  # left/right aligned
        return a.intersects(box) and abs(a.cy - box.cy) <= tol and aligned


def persistent_clusters(
    detections: dict[int, list[tuple[Box, str, float]]],
    frame_shape: tuple[int, int],
    persist_frac: float = 0.8,
    max_drift: float = 0.01,
    edge_frac: float = 0.12,
    edge_persist_frac: float = 0.3,
) -> list[TextCluster]:
    """Group OCR boxes across frames; keep those present at the same place in most frames.

    Boxes of a changing number ("-50" -> "184") change width, so a box matches a cluster when it
    intersects the cluster's first box, its vertical centre moved less than ``max_drift`` of the
    diagonal, and its left or right edge stayed put (left- or right-aligned text).

    Burned-in overlays sit at the frame edges, and OCR loses them in over-exposed frames (flash,
    airbag), so clusters within ``edge_frac`` of an edge need only ``edge_persist_frac``.
    """
    h, w = frame_shape
    tol = max_drift * float(np.hypot(h, w))
    clusters: list[TextCluster] = []
    for idx in sorted(detections):
        for box, text, _ in detections[idx]:
            for c in clusters:
                if c.matches(box, tol):
                    c.box = c.box.union(box)
                    c.frames.add(idx)
                    c.readings.append((idx, text))
                    break
            else:
                clusters.append(TextCluster(box, {idx}, [(idx, text)]))

    def at_edge(b: Box) -> bool:
        m_h, m_w = edge_frac * h, edge_frac * w
        return b.y0 < m_h or b.y1 > h - m_h or b.x0 < m_w or b.x1 > w - m_w

    n = len(detections)
    return [
        c
        for c in clusters
        if len(c.frames) >= (edge_persist_frac if at_edge(c.anchor) else persist_frac) * n
    ]


# --- 3. time model ---------------------------------------------------------------------------

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
DECIMAL_COMMA = re.compile(r"(\d),(\d)")  # "-3,45 ms" (European formatting)
TIME_WORDS = re.compile(r"\bms\b|time|t\s*=", re.I)
OVERLAY_WORDS = re.compile(
    r"\bms\b|msec|fps|frame|time|camera|view|\bt0\b|\bt\s*=|\d{1,2}:\d{2}", re.I
)


def looks_like_overlay(c: TextCluster) -> bool:
    """Burned-in overlay text, as opposed to a label fixed to the vehicle.

    In onboard views the camera moves with the car, so vehicle placards ("M20180216") are just
    as static as overlays; masking them would hide the airbag behind them. Overlays either
    contain overlay vocabulary or show numbers that change from frame to frame.
    """
    texts = [t for _, t in c.readings]
    if any(OVERLAY_WORDS.search(t) for t in texts):
        return True
    numbers = {tuple(NUM.findall(t)) for t in texts}
    return len(numbers) >= max(2, len(texts) // 2)


@dataclass
class TimeModel:
    ms_per_frame: float
    t_ms_at_frame0: float
    source: str  # "ocr" or "convention"
    n_inliers: int = 0
    max_abs_resid_ms: float | None = None
    agrees_with_convention: bool | None = None
    n_points: int = 0  # sampled frames in which the overlay text was detected

    def t_ms(self, idx: int) -> float:
        return self.t_ms_at_frame0 + self.ms_per_frame * idx

    def is_verified(self, min_inlier_frac: float = 0.6, max_resid_ms: float = 1.0) -> bool:
        """OCR readings are self-consistent: most lie on one line within ``max_resid_ms``.

        Deliberately independent of the NCAP convention: some clips start at -5 ms, others are
        encoded with every frame repeated three times; OCR is right in both cases.
        """
        return (
            self.source == "ocr"
            and self.n_points > 0
            and self.n_inliers / self.n_points >= min_inlier_frac
            and (self.max_abs_resid_ms or 0.0) <= max_resid_ms
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return slope, float(np.median(y - slope * x))


def fit_time_model(
    clusters: list[TextCluster],
    convention_t0_frame: int = 50,
    convention_ms_per_frame: float = 1.0,
    min_inliers: int = 3,
) -> TimeModel:
    """Fit t_ms(frame) from numbers read in persistent overlay text; fall back to convention."""
    candidates = []
    for c in clusters:
        is_time_text = any(TIME_WORDS.search(t) for _, t in c.readings)
        series: dict[int, list[tuple[int, float, bool]]] = {}
        for idx, text in c.readings:
            for k, m in enumerate(NUM.findall(DECIMAL_COMMA.sub(r"\1.\2", text))):
                series.setdefault(k, []).append((idx, float(m), "." in m))
        for pts in series.values():
            if len({p[0] for p in pts}) < min_inliers:
                continue
            x = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts], dtype=float)
            slope, icpt = _theil_sen(x, y)
            if 0.05 <= slope <= 20:  # already milliseconds per frame
                scale = 1.0
            elif 5e-5 <= slope <= 0.02:  # seconds per frame
                scale = 1000.0
            else:
                continue
            resid = np.abs(y - (slope * x + icpt)) * scale
            inl = resid <= max(1.0, 0.5 * slope * scale)
            if inl.sum() < min_inliers:
                continue
            slope, icpt = _theil_sen(x[inl], y[inl])
            has_dec = any(p[2] for p in pts)
            score = (is_time_text, has_dec, int(inl.sum()))
            # count frames, not readings: OCR may split one overlay line into several boxes
            n_in, r_max = len(set(x[inl].tolist())), float(resid[inl].max())
            candidates.append((score, slope * scale, icpt * scale, n_in, r_max, len(c.frames)))
    conv = TimeModel(
        convention_ms_per_frame, -convention_t0_frame * convention_ms_per_frame, "convention"
    )
    if not candidates:
        return conv
    _, a, b, n, r, n_pts = max(candidates, key=lambda c: c[0])
    agrees = abs(a - conv.ms_per_frame) < 0.02 and abs(b - conv.t_ms_at_frame0) < 2.0
    return TimeModel(a, b, "ocr", n, r, agrees, n_pts)


# --- putting it together ---------------------------------------------------------------------


@dataclass
class OverlayResult:
    crop: Box
    masks: list[Box]  # in cropped coordinates
    time_model: TimeModel
    texts: list[dict]

    def to_dict(self) -> dict:
        return {
            "crop": self.crop.as_list(),
            "masks": [m.as_list() for m in self.masks],
            "time_model": self.time_model.to_dict(),
            "overlay_texts": self.texts,
        }


def analyze_overlay(frames: dict[int, np.ndarray], reader: TextReader | None, cfg) -> OverlayResult:
    """``frames``: full-resolution RGB frames keyed by frame index (a sparse sample)."""
    imgs = list(frames.values())
    h, w = imgs[0].shape[:2]
    ov, tm = cfg.overlay, cfg.timing
    crop = detect_border_bands(imgs, ov.band_max_frac, ov.band_uniform_frac, ov.band_tol)
    clusters: list[TextCluster] = []
    if reader is not None:
        dets = {i: reader(img) for i, img in frames.items()}
        clusters = persistent_clusters(
            dets,
            (h, w),
            ov.text_persist_frac,
            ov.text_max_drift,
            ov.text_edge_frac,
            ov.text_edge_persist_frac,
        )
    clusters = [c for c in clusters if looks_like_overlay(c)]
    masks = []
    for c in clusters:
        b = c.box.expand(ov.text_margin, w, h)
        if b.intersects(crop):
            masks.append(
                Box(
                    max(b.x0, crop.x0) - crop.x0,
                    max(b.y0, crop.y0) - crop.y0,
                    min(b.x1, crop.x1) - crop.x0,
                    min(b.y1, crop.y1) - crop.y0,
                )
            )
    time_model = fit_time_model(clusters, tm.convention_t0_frame, tm.convention_ms_per_frame)
    texts = [{"box": c.box.as_list(), "samples": [t for _, t in c.readings[:3]]} for c in clusters]
    return OverlayResult(crop, masks, time_model, texts)


def apply_overlay(img: np.ndarray, crop: Box, masks: list[Box], fill: int = 127) -> np.ndarray:
    out = img[crop.y0 : crop.y1, crop.x0 : crop.x1].copy()
    for m in masks:
        out[m.y0 : m.y1, m.x0 : m.x1] = fill
    return out
