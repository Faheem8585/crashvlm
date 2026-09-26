import numpy as np
import pytest
from omegaconf import OmegaConf

from crashvlm.data.overlay import (
    Box,
    TextCluster,
    analyze_overlay,
    apply_overlay,
    detect_border_bands,
    fit_time_model,
    indicator_boxes,
    looks_like_overlay,
    persistent_clusters,
)

CFG = OmegaConf.load("configs/data.yaml")


def scene_frames(n=6, h=120, w=200, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_no_bands_on_textured_scene():
    assert detect_border_bands(scene_frames()) == Box(0, 0, 200, 120)


def test_detects_bottom_and_left_panels_with_text():
    frames = scene_frames()
    for k, f in enumerate(frames):
        f[100:] = 128  # bottom grey band
        f[:, :40] = 90  # left panel
        f[108:112, 60 + k : 70 + k] = 255  # changing "text" inside the band
        f[20:24, 5:30] = 0  # static text in the panel
    text = np.zeros((120, 200), bool)  # OCR text boxes, as analyze_overlay passes them
    text[106:114, 58:82] = True
    text[18:26, 3:32] = True
    assert detect_border_bands(frames, text_mask=text) == Box(40, 0, 200, 100)


def test_dark_scene_region_with_lighting_change_is_not_a_band():
    # onboard view: a dark seat back on the left, lit up by the airbag in one frame
    frames = scene_frames(n=8)
    for k, f in enumerate(frames):
        f[:, :60] = 8 if k != 5 else 70
    assert detect_border_bands(frames).x0 == 0


def test_letterbox_bars_are_cropped():
    frames = scene_frames(n=6)
    for f in frames:
        f[:, :20] = 0
        f[:, -20:] = 0
    assert detect_border_bands(frames) == Box(20, 0, 180, 120)


def test_band_limited_to_max_frac():
    frames = scene_frames()
    for f in frames:
        f[:80] = 50  # a large uniform area is only cropped up to max_frac
    assert detect_border_bands(frames, max_frac=0.3).y0 == int(120 * 0.3)


def test_persistent_clusters_keep_fixed_text_drop_moving_sticker():
    dets = {}
    for i in range(10):
        dets[i * 25] = [
            (Box(10, 5, 40 + (i % 3), 15), f"{i * 25 - 50:.1f}", 0.9),  # timestamp, width varies
            (Box(60 + 12 * i, 50, 90 + 12 * i, 60), "NCAP FRONTAL", 0.9),  # moving sticker
        ]
    clusters = persistent_clusters(dets, (120, 200), persist_frac=0.8, max_drift=0.01)
    assert len(clusters) == 1
    assert clusters[0].box.x0 == 10 and len(clusters[0].frames) == 10


def cluster(texts):
    c = TextCluster(Box(0, 0, 10, 10))
    c.readings = texts
    c.frames = {i for i, _ in texts}
    return c


@pytest.mark.parametrize(
    "fmt",
    [
        lambda i: f"{i - 50:.1f}",  # TRC: "-50.0"
        lambda i: f"{i - 50} ms 1000 fps",  # Calspan
        lambda i: f"Frame # {i - 50} Time {(i - 50) / 1000:.3f}",  # KARCO, merged line
        lambda i: f"T= {i - 50 + 1.722:.3f} ms ImgN: {i} Rate: 1000 Durat: 0.360 s",  # MGA
    ],
)
def test_time_model_from_lab_formats(fmt):
    tm = fit_time_model([cluster([(i, fmt(i)) for i in range(0, 350, 25)])])
    assert tm.source == "ocr"
    assert tm.ms_per_frame == pytest.approx(1.0, abs=1e-6)
    assert abs(tm.t_ms_at_frame0 - (-50)) < 2
    assert tm.agrees_with_convention


def test_time_model_prefers_seconds_series_over_frame_counter_at_other_fps():
    # 2000 fps: frame counter slope 1/frame, time slope 0.5 ms/frame
    texts = [(i, f"Frame # {i - 100} Time {(i - 100) / 2000:.4f}") for i in range(0, 700, 50)]
    tm = fit_time_model([cluster(texts)])
    assert tm.ms_per_frame == pytest.approx(0.5)
    assert tm.t_ms_at_frame0 == pytest.approx(-50)
    assert tm.agrees_with_convention is False


def test_time_model_robust_to_ocr_errors():
    texts = [(i, f"{i - 50}") for i in range(0, 350, 25)]
    texts[3] = (75, "2S")  # misread
    texts[5] = (125, "75")  # sign/digit error
    tm = fit_time_model([cluster(texts)])
    assert tm.source == "ocr" and tm.ms_per_frame == pytest.approx(1.0)


def test_time_model_falls_back_to_convention():
    tm = fit_time_model([cluster([(i, "NHTSA") for i in range(0, 350, 25)])])
    assert tm.source == "convention"
    assert tm.t_ms(50) == 0 and tm.t_ms(150) == 100


def test_analyze_and_apply_overlay():
    frames = scene_frames(n=8)
    for f in frames:
        f[100:] = 128

    def reader(img):
        return [(Box(150, 20, 190, 30), "58 ms 1000 fps", 0.9)]  # overlay text over the scene

    sample = {i * 25: f for i, f in enumerate(frames)}
    res = analyze_overlay(sample, reader, CFG)
    assert res.crop == Box(0, 0, 200, 100)
    out = apply_overlay(frames[0], res.crop, res.masks, fill=127)
    assert out.shape == (100, 200, 3)
    assert (out[20:30, 150:190] == 127).all()


def test_decimal_comma_readings():
    texts = [(i, f"{i - 5:.2f} ms 01,000 fps".replace(".", ",")) for i in range(0, 300, 25)]
    tm = fit_time_model([cluster(texts)])
    assert tm.t_ms_at_frame0 == pytest.approx(-5) and tm.ms_per_frame == pytest.approx(1.0)


def test_verified_is_fit_quality_not_convention():
    # clip starting at -5 ms: disagrees with the convention but is a clean OCR fit
    tm = fit_time_model([cluster([(i, f"{i - 5}.0") for i in range(0, 300, 25)])])
    assert tm.agrees_with_convention is False and tm.is_verified()
    # mostly garbage readings with a few accidental collinear ones: not verified
    texts = [(i, f"{i - 50}") for i in range(0, 100, 25)] + [(i, "83") for i in range(100, 400, 25)]
    tm = fit_time_model([cluster(texts)])
    assert not tm.is_verified()
    assert not fit_time_model([cluster([(0, "x")])]).is_verified()


def test_edge_text_needs_lower_persistence():
    # detected in 4/10 frames (over-exposure): kept at the bottom edge, dropped mid-frame
    dets = {i * 25: [] for i in range(10)}
    for i in range(4):
        dets[i * 25] += [
            (Box(5, 110, 150, 119), f"{i * 25 - 50},00 ms Frame: {i * 25}", 0.9),
            (Box(60, 50, 120, 60), "MID TEXT", 0.9),
        ]
    kept = persistent_clusters(dets, (120, 200), persist_frac=0.8, max_drift=0.01)
    assert [c.anchor.y0 for c in kept] == [110]


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["-50 ms 1000", "-25 ms 1000", "0 ms 1000"], True),
        (["Camera View-03"] * 3, True),
        (["-50.0", "-25.0", "0.0", "25.0"], True),  # bare changing numbers (TRC)
        (["M20180216"] * 4, False),  # vehicle placard seen by an onboard camera
        (["NCAP 35 MPH FRONTAL"] * 4, False),
    ],
)
def test_looks_like_overlay(texts, expected):
    assert looks_like_overlay(cluster(list(enumerate(texts)))) is expected


def test_panel_with_changing_t0_dot_is_fully_cropped():
    # KARCO-style grey side panel whose red dot turns green at t0; the dot is not OCR text
    frames = scene_frames(n=10, h=240, w=400)
    for k, f in enumerate(frames):
        f[:, :120] = 128
        f[100:112, 50:62] = (255, 0, 0) if k < 3 else (0, 200, 0)  # ~5% of height, as in KARCO
    assert detect_border_bands(frames).x0 == 120


def test_indicator_lamp_detected_and_scene_colours_ignored():
    frames = scene_frames(n=6, h=240, w=400, seed=3)
    for k, f in enumerate(frames):
        f[100:130, 20:50] = (255, 0, 0) if k < 2 else (0, 215, 5)  # T0 lamp
        f[200:230, 300:340] = (220, 30, 30)  # a red object that stays red: not a lamp
        f[5:7, 5:7] = (255, 0, 0) if k < 2 else (0, 215, 5)  # 4 px speck: below min_area
    boxes = indicator_boxes(frames, min_area=100)
    assert boxes == [Box(20, 100, 50, 130)]


def test_indicator_lamp_with_mid_switch_frame():
    # the sampled frame at t0 catches the lamp mid-switch (neither red nor green)
    frames = scene_frames(n=8, h=240, w=400, seed=4)
    for k, f in enumerate(frames):
        colour = (255, 0, 0) if k < 2 else (120, 120, 0) if k == 2 else (0, 215, 5)
        f[100:130, 20:50] = colour
    assert indicator_boxes(frames, min_area=100) == [Box(20, 100, 50, 130)]
    assert indicator_boxes(frames, min_area=100, max_transition=0) == []


def test_time_model_prefers_ms_series_over_frame_counter_with_ocr_merge():
    # MGA 2014 overlay: "1,000 fps T0: 50 Frame: N  t,tt ms"; one reading merges frame and time
    texts = [
        (i, f"1,000 fps T0: 50 Frame: {i + 1} {i - 48.78:.2f} ms".replace(".", ","))
        for i in range(0, 350, 25)
    ]
    texts[6] = (150, "1,000fps T0: 50 Frame:151102,22")  # merged by OCR
    tm = fit_time_model([cluster(texts)])
    assert tm.t_ms_at_frame0 == pytest.approx(-48.78, abs=0.01)
