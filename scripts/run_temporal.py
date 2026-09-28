"""Temporal CNN over frozen embeddings vs. per-frame probes (CPU).

    uv run python scripts/run_temporal.py

Same labels (tier A), splits, crops, decoder and metrics as M3. The decision threshold per event
is chosen on val; three seeds; the test split is scored once per seed.
Writes reports/m3/temporal.md.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas
from crashvlm.eval.baselines import decode_all, load_features, score
from crashvlm.eval.events import build_ground_truth, evaluate, window_crops
from crashvlm.models.probe import frame_targets
from crashvlm.models.temporal import predict_curves, train_tcn


@hydra.main(config_path="../configs", config_name="m3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mc = cfg.m3
    tols = tuple(mc.tolerances_ms)
    onto = Ontology.load(cfg.annotation.ontology)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    split = dict(zip(sel.test_no, sel.split, strict=True))
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet")
    verified = dict(zip(man.clip_id, man.time_verified, strict=True))
    recs = records_from_labels(
        json.loads(Path(mc.labels).read_text()),
        metas,
        onto,
        "a",
        "labels",
        dict(zip(sel.test_no, sel.config_key, strict=True)),
        onto.annotation["frame_stride"],
    )
    gts = build_ground_truth(
        recs,
        OmegaConf.to_container(mc.events_t0_unknown),
        lambda r, c: set(onto.events_for(r, c)),
        verified,
    )
    events = sorted({g.event for g in gts})
    by_clip: dict[str, dict] = {}
    for g in gts:
        by_clip.setdefault(g.clip_id, {})[g.event] = g
    rows = []
    for slug in mc.backbones:
        fpath = Path(cfg.paths.root) / "features" / f"{slug}.npz"
        if not fpath.exists():
            continue
        clips, *_ = load_features(fpath)

        def seq(cid, clips=clips):
            emb, t = clips[cid]
            y = np.zeros((len(t), len(events)))
            m = np.zeros(len(events))
            for k, ev in enumerate(events):
                g = by_clip[cid].get(ev)
                if g is not None:
                    m[k] = 1
                    y[:, k] = frame_targets(t, g.t_ms)
            return emb, y, m

        ids = {
            s: [
                c
                for c in by_clip
                if c in clips and split[by_clip[c][next(iter(by_clip[c]))].test_no] == s
            ]
            for s in ("train", "val", "test")
        }
        test_g = [g for g in gts if split[g.test_no] == "test"]
        val_g = [g for g in gts if split[g.test_no] == "val"]
        crop = window_crops([g.clip_id for g in test_g], mc.window_jitter_ms, mc.seed)
        for seed in range(3):
            model = train_tcn(
                [seq(c) for c in ids["train"]], [seq(c) for c in ids["val"]], len(events), seed=seed
            )
            curves = {}
            for c in ids["val"] + ids["test"]:
                pc = predict_curves(model, clips[c][0])
                for k, ev in enumerate(events):
                    if ev in by_clip[c]:
                        curves[(c, ev)] = (pc[:, k], clips[c][1])
            for ev in events:
                val = [g for g in val_g if g.event == ev]
                thr = max(
                    mc.thresholds,
                    key=lambda t, val=val, ev=ev: score(
                        evaluate(val, decode_all(curves, val, t, mc), tols, 0)[ev]
                    ),
                )
                test = [g for g in test_g if g.event == ev]
                m = evaluate(test, decode_all(curves, test, thr, mc, crop), tols, 0)[ev]
                rows.append(
                    {
                        "backbone": slug.split("__")[-1],
                        "seed": seed,
                        "event": ev,
                        "F1": m["f1"],
                        "hit@16": m["hit@16"],
                    }
                )
    df = pd.DataFrame(rows)
    agg = df.groupby(["event", "backbone"])[["F1", "hit@16"]].agg(["mean", "std"]).round(3)
    out = Path("reports/m3")
    (out / "temporal.md").write_text(
        "# Temporal CNN over frozen embeddings (test split, t0 unknown)\n\n"
        "Mean (std) over 3 seeds. Compare with the linear-probe rows in results.md.\n\n"
        + agg.to_markdown()
        + "\n"
    )
    print(agg.to_string())


if __name__ == "__main__":
    main()
