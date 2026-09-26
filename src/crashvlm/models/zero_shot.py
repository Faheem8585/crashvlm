"""Zero-shot frame scoring with a SigLIP-type image-text model, and event extraction from the
resulting score curves.

Model code needs the optional ``ml`` extra (torch, transformers). The curve logic is plain numpy,
so it can be tested and reused by the evaluation without a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.astype(float)
    kernel = np.ones(window) / window
    pad = window // 2
    return np.convolve(np.pad(x, (pad, window - 1 - pad), mode="edge"), kernel, mode="valid")


def curve_to_event(
    scores: np.ndarray,
    times: Sequence[float],
    mode: str,
    threshold: float = 0.5,
    min_run: int = 3,
    window: int = 3,
) -> float | None:
    """Time (ms) of the event implied by a per-frame score curve, or None if not detected."""
    s = smooth(np.asarray(scores, dtype=float), window)
    if mode == "peak":
        i = int(np.argmax(s))
        return float(times[i]) if s[i] > threshold else None
    if mode == "onset":
        above = s > threshold
        for i in range(len(s) - min_run + 1):
            if above[i : i + min_run].all():
                return float(times[i])
        return None
    raise ValueError(f"unknown mode {mode!r}")


def pos_neg_scores(logits: np.ndarray, n_pos: int) -> np.ndarray:
    """logits: (frames, n_pos + n_neg) -> P(positive) per frame via a 2-way softmax of means."""
    pos = logits[:, :n_pos].mean(axis=1)
    neg = logits[:, n_pos:].mean(axis=1)
    m = np.maximum(pos, neg)
    ep, en = np.exp(pos - m), np.exp(neg - m)
    return ep / (ep + en)


def _embedding(out):
    """``get_*_features`` returns a tensor in older transformers and a model output (with the
    projected embedding in ``pooler_output``) in newer ones."""
    return out if hasattr(out, "norm") else out.pooler_output


class SiglipScorer:
    """Image-text logits for SigLIP / SigLIP 2 checkpoints from the Hugging Face hub."""

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if dtype == "auto":
            dtype = "float16" if device == "cuda" else "float32"
        self.torch = torch
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        # resize to the vision tower's own input size (processor configs can disagree with it)
        side = self.model.config.vision_config.image_size
        self.image_size = {"height": side, "width": side}

    def text_features(self, texts: list[str]):
        inputs = self.processor(
            text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            f = _embedding(self.model.get_text_features(**inputs))
        return f / f.norm(dim=-1, keepdim=True)

    def image_features(self, images: list, batch_size: int = 32):
        feats = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(
                images=images[i : i + batch_size], size=self.image_size, return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
            with self.torch.no_grad():
                f = _embedding(self.model.get_image_features(pixel_values=pixel_values))
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return self.torch.cat(feats)

    def logits(self, image_feats, text_feats) -> np.ndarray:
        with self.torch.no_grad():
            scale = self.model.logit_scale.exp()
            bias = getattr(self.model, "logit_bias", None)
            out = image_feats.float() @ text_feats.float().T * scale.float()
            if bias is not None:
                out = out + bias.float()
        return out.cpu().numpy()
