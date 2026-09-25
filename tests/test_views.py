import json
from pathlib import Path

import pytest

from crashvlm.data.views import is_high_speed, normalize_view_name, select_views

SAMPLES = json.loads((Path(__file__).parent / "fixtures" / "multimedia_samples.json").read_text())


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("Camera No. 4 - Left Front Half.avi", "left front half"),
        ("04 LEFT FRONT HALF.wmv", "left front half"),
        ("Camera No. 5 - Dummy Front View (Onboard).avi", "dummy front view"),
        ("Onboard – Driver Front.mp4", "onboard - driver front"),
        ("S170803-2 N6 Driver Door.wmv", "s170803-2 n6 driver door"),
    ],
)
def test_normalize(raw, norm):
    assert normalize_view_name(raw) == norm


def test_excludes_non_high_speed():
    assert not is_high_speed("Camera No. 1 - Real-Time Left Overall.wmv")
    assert not is_high_speed("2025 Volvo EX90 Side NCAP Pole - IMPACT FOR WEB.wmv")
    assert not is_high_speed("M20215200 - Documentation Video.avi")
    assert is_high_speed("Camera No. 2 - Left Overall.mp4")


def test_frontal_prefers_left_front_half_over_real_time():
    videos = [
        {"originalFileName": "Camera No. 1 - Real-Time Left Overall.wmv", "index": 1},
        {"originalFileName": "Camera No. 2 - Left Overall.mp4", "index": 2},
        {"originalFileName": "Camera No. 4 - Left Front Half.mp4", "index": 4},
        {"originalFileName": "Camera No. 16 - Driver Onboard.mp4", "index": 16},
    ]
    got = select_views("VTB", videos)
    assert got["exterior"]["index"] == 4
    assert got["occupant"]["index"] == 16


def test_size_cap():
    videos = [{"originalFileName": "Camera No. 4 - Left Front Half.mp4", "fileSize": "5e8"}]
    assert "exterior" not in select_views("VTB", videos, max_bytes=2e8)


def test_real_sample_coverage():
    """On 45 real NHTSA listings, tests with per-camera files must yield both views."""
    per_camera = [s for s in SAMPLES if len(s["videos"]) > 4]
    both = [s for s in per_camera if len(select_views(s["config_key"], s["videos"])) == 2]
    assert len(per_camera) >= 30
    assert len(both) / len(per_camera) >= 0.9
