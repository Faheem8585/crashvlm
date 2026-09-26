"""M2: zero-shot pre-labels with SigLIP 2 (GPU recommended; see docs/COLAB.md).

    python scripts/zero_shot_prelabels.py            # after `pip install -e ".[ml]"`

Reads <data>/annotation/pass_a/images.zip and clip_index.json. Writes
zero_shot_preds.jsonl (event onset per clip) and zero_shot_curves.npz (per-frame scores, reused in
M3), both under <data>/annotation/.
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

from crashvlm.annotation.ontology import Ontology
from crashvlm.models.zero_shot import SiglipScorer, curve_to_event, pos_neg_scores


@hydra.main(config_path="../configs", config_name="annotation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    prompts = yaml.safe_load(Path(cfg.annotation.prompts).read_text())
    out = Path(cfg.annotation.out)
    index = json.loads((out / "clip_index.json").read_text())
    model_id = cfg.annotation.model or prompts["model"]
    scorer = SiglipScorer(model_id)

    events = prompts["events"]
    texts, spans = [], {}
    for name, spec in events.items():
        spans[name] = (len(texts), len(spec["positive"]), len(spec["negative"]))
        texts += spec["positive"] + spec["negative"]
    text_feats = scorer.text_features(texts)

    curves, lines = {}, []
    with zipfile.ZipFile(cfg.annotation.images_zip) as zf:
        names = set(zf.namelist())
        for cid, info in tqdm(sorted(index.items()), desc="clips"):
            frames = [f for f in info["frames"] if f["file"] in names]
            if not frames:
                continue
            images = [Image.open(io.BytesIO(zf.read(f["file"]))).convert("RGB") for f in frames]
            logits = scorer.logits(
                scorer.image_features(images, cfg.annotation.batch_size), text_feats
            )
            times = [f["t_ms"] for f in frames]
            allowed = set(onto.events_for(info["role"], info["config_key"]))
            rec = {"clip_id": cid, "model": model_id, "events": {}}
            for name, (start, n_pos, n_neg) in spans.items():
                s = pos_neg_scores(logits[:, start : start + n_pos + n_neg], n_pos)
                curves[f"{cid}/{name}"] = s.astype(np.float32)
                if name not in allowed:
                    continue
                t = curve_to_event(
                    s,
                    times,
                    events[name]["mode"],
                    prompts["threshold"],
                    prompts["min_run"],
                    prompts["smooth"],
                )
                rec["events"][name] = {"t_ms": t, "max_score": float(s.max())}
            curves[f"{cid}/_t_ms"] = np.asarray(times, dtype=np.float32)
            lines.append(json.dumps(rec))

    (out / "zero_shot_preds.jsonl").write_text("\n".join(lines) + "\n")
    np.savez_compressed(out / "zero_shot_curves.npz", **curves)
    print(f"{len(lines)} clips -> {out / 'zero_shot_preds.jsonl'}")


if __name__ == "__main__":
    main()
