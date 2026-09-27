"""Gradio demo: upload a crash-test clip, see the event timeline, key frames and the report.

uv run python -m crashvlm.serve.demo          # http://127.0.0.1:7860
"""

from __future__ import annotations

import json

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashvlm.models.vlm_report import EVENTS  # noqa: E402
from crashvlm.serve.api import backend, ocr_reader  # noqa: E402
from crashvlm.serve.pipeline import evaluate_video  # noqa: E402

COLORS = ["#2a6fdb", "#e0602b", "#2f9e44", "#9c36b5", "#c92a2a", "#868e96"]


def timeline_plot(out: dict):
    tl = out["timeline"]
    color = {ev: COLORS[i % len(COLORS)] for i, ev in enumerate(EVENTS)}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for ev, scores in tl["scores"].items():
        ax.plot(tl["t_ms"], scores, color=color[ev], label=ev.replace("_", " "))
        t = out["report"]["events"].get(ev)
        if t is not None:
            ax.axvline(t, color=color[ev], ls="--", lw=1)
    ax.set_xlabel("ms since first analysed frame")
    ax.set_ylabel("P(event has happened)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run(video_path: str, view: str):
    if not video_path:
        raise gr.Error("Upload a video first.")
    out, frames = evaluate_video(video_path, view, backend(), reader=ocr_reader())
    events = out["report"]["events"] if out["report"] else {}
    gallery = []
    for ev, t in events.items():
        if t is None:
            continue
        i = min(range(len(frames.t_ms)), key=lambda k, t=t: abs(frames.t_ms[k] - t))
        gallery.append((frames.images[i], f"{ev.replace('_', ' ')} @ {t:.0f} ms"))
    fig = timeline_plot(out) if out["timeline"]["scores"] else None
    return fig, gallery, json.dumps({k: v for k, v in out.items() if k != "timeline"}, indent=1)


with gr.Blocks(title="CrashVLM") as demo:
    gr.Markdown(
        "# CrashVLM\nEvent detection in NCAP crash-test video. Times are relative to the first "
        "analysed frame. Research prototype: not a safety rating."
    )
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Crash test clip (high-speed camera)")
            view = gr.Radio(["occupant", "exterior"], value="occupant", label="Camera view")
            go = gr.Button("Evaluate", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="Event timeline")
            gallery = gr.Gallery(label="Detected events", columns=4, height=220)
            report = gr.Code(label="Report (JSON)", language="json")
    go.click(run, [video, view], [plot, gallery, report])

if __name__ == "__main__":
    demo.launch()
