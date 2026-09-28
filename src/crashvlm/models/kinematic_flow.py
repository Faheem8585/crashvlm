"""Video kinematics: time of maximum crush from optical flow on the vehicle body.

In the frontal exterior view (static camera, side-on), the rear part of the visible body (door,
B-pillar) moves almost rigidly. Its horizontal image velocity is proportional to the vehicle
velocity, which is zero at maximum dynamic crush. We track corner features with pyramidal
Lucas-Kanade and take the median horizontal displacement per frame.

Clock-free: the approach velocity comes from the first frames of the clip, the crash is detected
as the velocity dropping below ``drop_frac`` of it, and maximum crush is the first time after
that where the velocity towards the obstacle is <= ``stop_frac`` of the approach velocity.
No use is made of the lab's t0, so the method is valid in the 't0 unknown' setting. The region,
thresholds and smoothing were chosen on the training split only.
"""

from __future__ import annotations

import numpy as np


def body_velocity(
    frames: list[np.ndarray],
    t_ms: np.ndarray,
    region: tuple[float, float, float, float] = (0.45, 0.25, 1.0, 0.85),
) -> tuple[np.ndarray, np.ndarray]:
    """Median horizontal image velocity (px/ms) of tracked features in ``region``
    (x0, y0, x1, y1 as fractions of the frame). Frames: grayscale uint8 arrays."""
    import cv2

    h, w = frames[0].shape
    x0, y0, x1, y1 = region
    mask = np.zeros((h, w), np.uint8)
    mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 255
    v = []
    for i in range(1, len(frames)):
        p0 = cv2.goodFeaturesToTrack(frames[i - 1], 200, 0.01, 6, mask=mask)
        if p0 is None:
            v.append(np.nan)
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            frames[i - 1], frames[i], p0, None, winSize=(15, 15), maxLevel=2
        )
        d = (p1 - p0)[st[:, 0] == 1][:, 0, 0]
        v.append(float(np.median(d)) / (t_ms[i] - t_ms[i - 1]) if len(d) else np.nan)
    return np.asarray(t_ms[1:], float), np.asarray(v, float)


def max_crush_time(
    t: np.ndarray,
    v: np.ndarray,
    n_approach: int = 5,
    drop_frac: float = 0.8,
    stop_frac: float = 0.05,
    smooth: int = 3,
) -> float | None:
    """Clock-free time of maximum crush from a body velocity trace (see module docstring)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < n_approach + 3:
        return None
    v = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
    if smooth > 1:
        v = np.convolve(np.pad(v, smooth // 2, mode="edge"), np.ones(smooth) / smooth, "valid")
    approach = float(np.median(v[:n_approach]))
    if abs(approach) < 1e-6:
        return None
    toward = v * np.sign(approach)  # positive while moving towards the obstacle
    ref = abs(approach)
    crash = np.where(toward < drop_frac * ref)[0]
    if not len(crash):
        return None
    stop = np.where((np.arange(len(v)) > crash[0]) & (toward <= stop_frac * ref))[0]
    return float(t[stop[0]]) if len(stop) else None
