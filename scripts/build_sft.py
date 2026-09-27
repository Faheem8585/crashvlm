"""M5: build instruction-tuning data from the annotations (CPU).

    uv run python scripts/build_sft.py
    uv run python scripts/build_sft.py sft.train_fraction=0.25   # learning-curve ablation

Writes <data>/sft/train[_f<fraction>].jsonl and val.jsonl. Test clips are never used.
The training subset for a fraction < 1 is drawn by *test* (both views together), seeded.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas
from crashvlm.train.sft_data import make_examples


@hydra.main(config_path="../configs", config_name="m5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    s, v = cfg.sft, cfg.vlm
    onto = Ontology.load(cfg.annotation.ontology)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    index = json.loads((Path(cfg.annotation.out) / "clip_index.json").read_text())
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    split = dict(zip(sel.test_no, sel.split, strict=True))
    keys = dict(zip(sel.test_no, sel.config_key, strict=True))
    labels = json.loads(Path(cfg.m3.labels).read_text())
    recs = records_from_labels(
        labels, metas, onto, "a", "labels", keys, onto.annotation["frame_stride"]
    )
    out = Path(cfg.paths.root) / "sft"
    out.mkdir(parents=True, exist_ok=True)

    train_tests = sorted({int(r["test_no"]) for r in recs if split[int(r["test_no"])] == "train"})
    if s.train_fraction < 1:
        rng = np.random.default_rng(cfg.m3.seed)
        k = max(1, round(s.train_fraction * len(train_tests)))
        train_tests = sorted(rng.choice(train_tests, size=k, replace=False).tolist())
    parts = {
        "train": [r for r in recs if int(r["test_no"]) in set(train_tests)],
        "val": [r for r in recs if split[int(r["test_no"])] == "val"],
    }
    suffix = "" if s.train_fraction >= 1 else f"_f{s.train_fraction:g}"
    for name, rs in parts.items():
        ex = make_examples(
            rs,
            index,
            keys,
            s.n_crops if name == "train" else 1,
            cfg.m3.window_jitter_ms,
            v.n_frames,
            v.step_ms,
            cfg.m3.seed + (0 if name == "train" else 1),
        )
        path = out / (f"train{suffix}.jsonl" if name == "train" else "val.jsonl")
        path.write_text("".join(json.dumps(e) + "\n" for e in ex))
        print(f"{name}: {len({e['clip_id'] for e in ex})} clips, {len(ex)} examples -> {path}")


if __name__ == "__main__":
    main()
