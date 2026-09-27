"""M6 (CPU part): learning curve of the linear probes, and label severity vs. NHTSA measurements.

    uv run python scripts/run_m6_local.py

1. Learning curve: probes (SigLIP 2 base features) trained on 25 / 50 / 100 % of the training
   *tests* (both views together, 3 seeds each), selected on val, scored on the full test split
   in the 't0 unknown' setting.
2. Severity: Spearman correlation between the annotated ``visual_severity`` (tier C, exploratory)
   and measured crush and driver HIC per test type. Uses the exterior clip of each test.
Writes reports/m6/learning_curve.md|csv and severity.md.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas
from crashvlm.eval.baselines import decode_all, fit_probes, load_features
from crashvlm.eval.events import build_ground_truth, evaluate, window_crops


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
    out = Path("reports/m6")
    out.mkdir(parents=True, exist_ok=True)

    # 1. learning curve
    clips, *_ = load_features(Path(cfg.paths.root) / "features" / f"{cfg.m6.backbone}.npz")
    test = [g for g in gts if split[g.test_no] == "test"]
    crop = window_crops([g.clip_id for g in test], mc.window_jitter_ms, mc.seed)
    val = [g for g in gts if split[g.test_no] == "val"]
    train_tests = sorted({g.test_no for g in gts if split[g.test_no] == "train"})
    rows = []
    for frac in cfg.m6.fractions:
        for seed in range(cfg.m6.seeds if frac < 1 else 1):
            rng = np.random.default_rng(seed)
            k = max(2, round(frac * len(train_tests)))
            keep = set(rng.choice(train_tests, size=k, replace=False).tolist())
            train = [g for g in gts if g.test_no in keep]
            curves, thr = fit_probes(clips, events, {"train": train, "val": val}, gts, mc, tols)
            preds = {}
            for ev, t in thr.items():
                preds.update(decode_all(curves, [g for g in test if g.event == ev], t, mc, crop))
            res = evaluate([g for g in test if g.event in thr], preds, tols, 0)
            for ev, m in res.items():
                rows.append(
                    {
                        "fraction": frac,
                        "train_tests": k,
                        "seed": seed,
                        "event": ev,
                        "F1": m["f1"],
                        "hit@16": m["hit@16"],
                    }
                )
    lc = pd.DataFrame(rows)
    lc.to_csv(out / "learning_curve.csv", index=False)
    agg = lc.groupby(["event", "fraction", "train_tests"])[["F1", "hit@16"]].agg(["mean", "std"])
    (out / "learning_curve.md").write_text(
        "# Learning curve: SigLIP 2 base linear probes, test split, t0 unknown\n\n"
        f"Mean (std) over {cfg.m6.seeds} random subsets of training tests; 100 % is one run.\n\n"
        + agg.round(3).to_markdown()
        + "\n"
    )
    print(agg.round(3).to_string())

    # 2. severity vs measurements
    cat = pd.read_parquet(Path(cfg.paths.interim) / "catalog.parquet").set_index("test_no")
    sev_map = {"low": 0, "medium": 1, "high": 2}
    srows = []
    for r in recs:
        if r["role"] != "exterior" or not r["complete"]:
            continue
        c = cat.loc[int(r["test_no"])]
        srows.append(
            {
                "test_no": r["test_no"],
                "config": r["config_key"],
                "severity": sev_map.get(r["summary"].get("visual_severity")),
                "crush_mm": c.crush_mm,
                "driver_hic": c.get("driver_hic", np.nan),
            }
        )
    sv = pd.DataFrame(srows)
    lines = [
        "# Annotated visual severity vs NHTSA measurements (exploratory, tier-C label)\n",
        "Spearman rho (n); exterior view per test. Label distribution: "
        + str(sv.severity.value_counts().to_dict())
        + "\n",
    ]
    for cfgk, g in sv.groupby("config"):
        for meas in ("crush_mm", "driver_hic"):
            d = g[["severity", meas]].dropna()
            if len(d) > 5 and d.severity.nunique() > 1:
                rho, p = spearmanr(d.severity, d[meas])
                lines.append(f"- {cfgk} vs {meas}: rho = {rho:.2f} (p = {p:.2f}, n = {len(d)})")
            else:
                lines.append(
                    f"- {cfgk} vs {meas}: not computable (n = {len(d)}, "
                    f"{d.severity.nunique()} severity levels)"
                )
    (out / "severity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
