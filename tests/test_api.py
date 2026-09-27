"""API contract tests with a fake backend and a synthetic video (no model downloads)."""

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

import crashvlm.serve.api as api


class FakeBackend:
    def run(self, frames, view):
        t = frames.t_ms
        return {
            "report": {
                "test_type": None,
                "events": {"airbag_deploy_start": t[3]},
                "description": None,
            },
            "timeline": {"t_ms": t, "scores": {"airbag_deploy_start": [0.0] * len(t)}},
            "latency_s": 0.0,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "backend", lambda: FakeBackend())
    monkeypatch.setattr(api, "ocr_reader", lambda: None)
    return TestClient(api.app)


def make_video(path, n=40, w=160, h=120):
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(n):
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_evaluate_video_contract(client, tmp_path):
    make_video(tmp_path / "clip.mp4")
    with open(tmp_path / "clip.mp4", "rb") as f:
        r = client.post(
            "/evaluate_video", files={"file": ("clip.mp4", f)}, data={"view": "exterior"}
        )
    assert r.status_code == 200
    out = r.json()
    # no readable overlay -> 1000 fps assumed, 8 ms grid from the first frame
    assert out["meta"]["time_source"] == "assumed_1000fps"
    assert out["timeline"]["t_ms"][:3] == [0.0, 8.0, 16.0]
    assert out["report"]["events"]["airbag_deploy_start"] == 24.0


def test_rejects_bad_view_and_bad_file(client):
    r = client.post("/evaluate_video", files={"file": ("x.mp4", b"xx")}, data={"view": "top"})
    assert r.status_code == 422
    r = client.post(
        "/evaluate_video", files={"file": ("x.mp4", b"not a video")}, data={"view": "occupant"}
    )
    assert r.status_code == 422
