import numpy as np
import pytest

from crashvlm.models.kinematic_flow import body_velocity, max_crush_time


def textured_frames(shifts, h=120, w=200, seed=0):
    """Frames of a random texture translated horizontally by sub-pixel cumulative shifts."""
    import cv2

    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h, w + 200)) * 255).astype(np.uint8), (5, 5), 1.5)
    out = []
    for s in shifts:
        m = np.float32([[1, 0, s - 100], [0, 1, 0]])
        out.append(cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_CUBIC))
    return out


def test_flow_recovers_stop_time():
    # body approaches at 2 px/frame, decelerates linearly from frame 10, stops at frame 20
    t = np.arange(40) * 8.0
    v = np.where(np.arange(40) < 10, -2.0, np.clip(-2.0 + 0.2 * (np.arange(40) - 10), -2.0, 0.0))
    frames = textured_frames(np.concatenate([[0], np.cumsum(v[1:])]))
    tt, vel = body_velocity(frames, t, region=(0.0, 0.0, 1.0, 1.0))
    assert np.nanmedian(vel[:5]) == pytest.approx(-2.0 / 8.0, abs=0.03)  # px/ms
    tc = max_crush_time(tt, vel, smooth=1)
    assert tc == pytest.approx(20 * 8.0, abs=16)


def test_no_motion_gives_none():
    t = np.arange(20) * 8.0
    assert max_crush_time(t, np.zeros(20)) is None
