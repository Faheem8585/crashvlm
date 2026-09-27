"""M7: fit the per-event linear probes on train+val and save them for the API (CPU).

    uv run python scripts/export_probes.py
    uv run python scripts/export_probes.py serve.backbone=google__siglip2-so400m-patch14-384

C and threshold per event are taken from a train->val selection (as in M3), then the probe is
refit on train+val. Test clips are never used. Writes <data>/serve/probes_<backbone>.joblib.
"""

import json
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas
from crashvlm.eval.baselines import fit_probes, load_features
from crashvlm.eval.events import build_ground_truth
from crashvlm.models.probe import frame_targets, train_probe


@hydra.main(config_path="../configs", config_name="m7", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mc = cfg.m3
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
    sp = {s: [g for g in gts if split[g.test_no] == s] for s in ("train", "val")}
    clips, *_ = load_features(Path(cfg.paths.root) / "features" / f"{cfg.serve.backbone}.npz")
    events = sorted({g.event for g in gts})
    cs: dict = {}
    _, thr = fit_probes(clips, events, sp, gts, mc, tuple(mc.tolerances_ms), cs)
    out = {"backbone": cfg.serve.backbone, "min_run": mc.min_run, "smooth": mc.smooth, "events": {}}
    for ev in thr:
        fit = [g for g in sp["train"] + sp["val"] if g.event == ev and g.clip_id in clips]
        X = np.concatenate([clips[g.clip_id][0] for g in fit])
        y = np.concatenate([frame_targets(clips[g.clip_id][1], g.t_ms) for g in fit])
        clf = train_probe(X, y, cs[ev], mc.seed)
        roles = sorted({g.role for g in fit})
        out["events"][ev] = {"clf": clf, "C": cs[ev], "threshold": thr[ev], "roles": roles}
    path = Path(cfg.paths.root) / "serve" / f"probes_{cfg.serve.backbone}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    print(f"{len(out['events'])} probes -> {path}")


if __name__ == "__main__":
    main()
