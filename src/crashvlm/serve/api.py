"""FastAPI service.

    CRASHVLM_BACKEND=probe uv run uvicorn crashvlm.serve.api:app --port 8000
    curl -F file=@clip.wmv -F view=occupant localhost:8000/evaluate_video

Environment
- ``CRASHVLM_BACKEND``: ``probe`` (default, CPU) or ``vlm`` (GPU)
- ``CRASHVLM_PROBES``: path to the exported probes (probe backend)
- ``CRASHVLM_EMBEDDER``: SigLIP hub id matching the probes
- ``CRASHVLM_VLM`` / ``CRASHVLM_ADAPTER``: model and optional LoRA adapter (vlm backend)
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from crashvlm import __version__
from crashvlm.serve.pipeline import ProbeBackend, VLMBackend, evaluate_video

DATA = Path(os.environ.get("CRASHVLM_DATA", Path.home() / "crashvlm_data"))
app = FastAPI(title="CrashVLM", version=__version__)


@lru_cache(maxsize=1)
def backend():
    kind = os.environ.get("CRASHVLM_BACKEND", "probe")
    if kind == "probe":
        embedder = os.environ.get("CRASHVLM_EMBEDDER", "google/siglip2-base-patch16-224")
        default = DATA / "serve" / f"probes_{embedder.replace('/', '__')}.joblib"
        return ProbeBackend(os.environ.get("CRASHVLM_PROBES", default), embedder)
    if kind == "vlm":
        return VLMBackend(
            os.environ.get("CRASHVLM_VLM", "Qwen/Qwen3.5-4B"), os.environ.get("CRASHVLM_ADAPTER")
        )
    raise RuntimeError(f"unknown backend {kind!r}")


@lru_cache(maxsize=1)
def ocr_reader():
    from crashvlm.data.overlay import RapidOCRReader

    return RapidOCRReader()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "backend": os.environ.get("CRASHVLM_BACKEND", "probe"),
    }


@app.post("/evaluate_video")
async def evaluate(
    file: Annotated[UploadFile, File()], view: Annotated[str, Form()] = "occupant"
) -> dict:
    """Structured crash report + per-frame event scores for one camera view.

    Times are milliseconds since the first analysed frame (impact time is not assumed)."""
    if view not in ("exterior", "occupant"):
        raise HTTPException(422, "view must be 'exterior' or 'occupant'")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        try:
            out, _ = evaluate_video(tmp.name, view, backend(), reader=ocr_reader())
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    return out
