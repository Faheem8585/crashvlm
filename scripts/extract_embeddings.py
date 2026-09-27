"""M3: frame embeddings for every annotation frame (8 ms steps). GPU recommended (Colab).

    python scripts/extract_embeddings.py                                  # SigLIP 2 so400m
    python scripts/extract_embeddings.py embed.backbone=resnet50
    python scripts/extract_embeddings.py embed.backbone=siglip:google/siglip2-base-patch16-224

Reads <data>/annotation/pass_a/images.zip + clip_index.json (built by make_cvat_tasks.py).
Writes <data>/features/<backbone>.npz with, per clip, ``<clip>/emb`` (float16, L2-normalised)
and ``<clip>/t_ms``; SigLIP backbones also store the prompt embeddings under ``_text/<event>``.
"""

import io
import json
import zipfile
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from crashvlm.models.embed import backbone_slug, make_embedder, to_f16


@hydra.main(config_path="../configs", config_name="m3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ann = Path(cfg.annotation.out)
    index = json.loads((ann / "clip_index.json").read_text())
    model = make_embedder(cfg.embed.backbone)
    out = Path(cfg.paths.root) / "features"
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    if cfg.embed.backbone.startswith("siglip:"):
        prompts = yaml.safe_load(Path(cfg.annotation.prompts).read_text())
        for name, spec in prompts["events"].items():
            texts = spec["positive"] + spec["negative"]
            arrays[f"_text/{name}"] = to_f16(model.text_features(texts))
            arrays[f"_text/{name}/n_pos"] = np.array(len(spec["positive"]))
        arrays["_logit_scale"] = np.array(float(model.model.logit_scale.exp()))
        bias = getattr(model.model, "logit_bias", None)
        arrays["_logit_bias"] = np.array(float(bias) if bias is not None else 0.0)

    with zipfile.ZipFile(cfg.annotation.images_zip) as zf:
        names = set(zf.namelist())
        for cid, info in tqdm(sorted(index.items()), desc=cfg.embed.backbone):
            frames = [f for f in info["frames"] if f["file"] in names]
            if not frames:
                continue
            ims = [Image.open(io.BytesIO(zf.read(f["file"]))).convert("RGB") for f in frames]
            arrays[f"{cid}/emb"] = to_f16(model.image_features(ims, cfg.embed.batch_size))
            arrays[f"{cid}/t_ms"] = np.array([f["t_ms"] for f in frames], dtype=np.float32)

    path = out / f"{backbone_slug(cfg.embed.backbone)}.npz"
    np.savez_compressed(path, **arrays)
    print(f"{sum(k.endswith('/emb') for k in arrays)} clips -> {path}")


if __name__ == "__main__":
    main()
