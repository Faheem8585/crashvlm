"""Physics-grounded event: can video models find the moment of maximum crush measured by the
vehicle's accelerometers? (Exterior views; labels from scripts/physics_targets.py.)

    uv run python scripts/run_physics_probe.py

Ground truth needs no annotator: t_max_crush from integrated sensor data, aligned with video
through the OCR-verified time axis (clips whose time axis is not verified are excluded from
timing). Same probes, decoder, splits, crops and metrics as M3.
Writes reports/physics/probe_results.md.
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from crashvlm.data.manifest import load_clip_metas
from crashvlm.eval.baselines import Blind, decode_all, fit_probes, load_features
from crashvlm.eval.events import GT, evaluate, window_crops

EVENT = "max_crush_physics"


@hydra.main(config_path="../configs", config_name="m3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mc = cfg.m3
    tols = tuple(mc.tolerances_ms)
    root = Path(cfg.paths.root)
    kin = pd.read_parquet(root / "physics" / "kinematics.parquet").dropna(subset=["t_max_crush_ms"])
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet")
    verified = dict(zip(man.clip_id, man.time_verified, strict=True))
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    clip_start = {c: m["frames"][0]["t_ms"] for c, m in metas.items()}

    gts = [
        GT(
            f"{r.test_no}_exterior",
            int(r.test_no),
            r.config,
            "exterior",
            EVENT,
            True,
            float(r.t_max_crush_ms),
            bool(verified.get(f"{r.test_no}_exterior", False)),
        )
        for r in kin.itertuples()
        if f"{r.test_no}_exterior" in metas
    ]
    split = dict(zip(kin.test_no, kin.split, strict=True))
    sp = {s: [g for g in gts if split[g.test_no] == s] for s in ("train", "val", "test")}
    crop = window_crops([g.clip_id for g in sp["test"]], mc.window_jitter_ms, mc.seed)

    # the blind prior sees the same kind of random start crops as the test clips
    train_crop = window_crops([g.clip_id for g in sp["train"]], mc.window_jitter_ms, mc.seed)
    blind = Blind(sp["train"], clip_start, train_crop)
    rows = []
    for setting, c in (("t0 known", None), ("t0 unknown", crop)):
        preds = {"blind (no video)": blind.predict(sp["test"], clip_start, c)}
        for slug in mc.backbones:
            fpath = root / "features" / f"{slug}.npz"
            if not fpath.exists():
                continue
            clips, *_ = load_features(fpath)
            curves, thr = fit_probes(clips, [EVENT], sp, gts, mc, tols)
            if EVENT in thr:
                preds[f"{slug.split('__')[-1]} linear probe"] = decode_all(
                    curves, sp["test"], thr[EVENT], mc, c
                )
        for name, p in preds.items():
            m = evaluate(sp["test"], p, tols, mc.bootstrap, mc.seed)[EVENT]
            ci = m.get("hit@16_ci")
            rows.append(
                {
                    "setting": setting,
                    "method": name,
                    "n_timing": m["n_timing"],
                    "hit@8": round(m["hit@8"], 2),
                    "hit@16": round(m["hit@16"], 2),
                    "hit@16 95% CI": f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "",
                    "median |err| ms": round(m["median_abs_err_ms"], 1),
                }
            )
    df = pd.DataFrame(rows)
    out = Path("reports/physics")
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# Physics-grounded event: time of maximum crush (exterior view, test split)\n",
        "Ground truth from integrated vehicle accelerations, not from an annotator. "
        f"Train/val/test tests with a result: {len({g.test_no for g in sp['train']})}/"
        f"{len({g.test_no for g in sp['val']})}/{len({g.test_no for g in sp['test']})}.\n",
        df.to_markdown(index=False),
    ]
    (out / "probe_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
