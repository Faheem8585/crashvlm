"""Frame embeddings for the M3 baselines.

Backbones
- ``siglip:<hub id>``: SigLIP / SigLIP 2 image tower (also gives text embeddings for zero-shot).
- ``resnet50``: ImageNet ResNet-50 penultimate features: the non-VLM classic-CV reference, run
  through the same probe so the VLM gain is measured rather than assumed.

Embeddings are L2-normalised and stored as float16 per clip, together with frame times.
"""

from __future__ import annotations

import numpy as np

from crashvlm.models.zero_shot import SiglipScorer


class ResNetEmbedder:
    def __init__(self, device: str | None = None):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.torch = torch
        self.device = device
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.preprocess = weights.transforms()

    def image_features(self, images: list, batch_size: int = 64):
        feats = []
        for i in range(0, len(images), batch_size):
            x = self.torch.stack([self.preprocess(im) for im in images[i : i + batch_size]])
            with self.torch.no_grad():
                f = self.model(x.to(self.device))
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return self.torch.cat(feats)


def make_embedder(backbone: str):
    if backbone == "resnet50":
        return ResNetEmbedder()
    if backbone.startswith("siglip:"):
        return SiglipScorer(backbone.split(":", 1)[1])
    raise ValueError(f"unknown backbone {backbone!r}")


def backbone_slug(backbone: str) -> str:
    return backbone.replace("siglip:", "").replace("/", "__")


def to_f16(t) -> np.ndarray:
    return t.float().cpu().numpy().astype(np.float16)
