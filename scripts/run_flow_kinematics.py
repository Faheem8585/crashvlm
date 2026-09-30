"""Video-derived time of maximum crush (optical flow on the vehicle body) vs accelerometers.

    uv run python scripts/run_flow_kinematics.py                 # train split (design)
    uv run python scripts/run_flow_kinematics.py +eval_split=test  # touch test once

Frontal exterior views only (static side-on camera). The method is clock-free
(crashvlm.models.kinematic_flow); in the 'test' run the clip start is additionally cropped with
the M3 window crops. Compared against the blind constant (median training physics time,
relative to the clip start when t0 is unknown). Writes reports/physics/flow_<split>.md.
"""

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr

from crashvlm.eval.events import window_crops
from crashvlm.models.kinematic_flow import body_velocity, max_crush_time


@hydra.main(config_path="../configs", config_name="m3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    split_name = cfg.get("eval_split", "train")
    root = Path(cfg.paths.root)
    kin = pd.read_parquet(root / "physics" / "kinematics.parquet").dropna(subset=["t_max_crush_ms"])
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet").set_index("clip_id")
    kin = kin[kin.config == "VTB"]
    kin = kin[[man.loc[f"{t}_exterior", "time_verified"] for t in kin.test_no]]
    train = kin[kin.split == "train"]
    evalset = kin[kin.split == split_name]
    crop = (
        window_crops(
            [f"{t}_exterior" for t in evalset.test_no], cfg.m3.window_jitter_ms, cfg.m3.seed
        )
        if split_name == "test"
        else {}
    )

    def clip(test_no):
        cid = f"{test_no}_exterior"
        meta = json.loads((root / "processed" / "clips" / cid / "meta.json").read_text())
        t = np.array([f["t_ms"] for f in meta["frames"]])
        keep = t >= t[0] + crop.get(cid, 0.0)
        files = [f["file"] for f, k in zip(meta["frames"], keep, strict=True) if k]
        ims = [
            cv2.imread(str(root / "processed" / "clips" / cid / f), cv2.IMREAD_GRAYSCALE)
            for f in files
        ]
        return t[keep], ims

    # blind constant: median training physics time relative to the first frame shown. When the
    # evaluation clips are cropped, the training clips get the same kind of random crop.
    starts = {int(t): clip(t)[0][0] for t in train.test_no}
    train_crop = (
        window_crops([f"{t}_exterior" for t in train.test_no], cfg.m3.window_jitter_ms, cfg.m3.seed)
        if crop
        else {}
    )
    rel = float(
        np.median(
            [
                r.t_max_crush_ms
                - starts[int(r.test_no)]
                - train_crop.get(f"{r.test_no}_exterior", 0)
                for r in train.itertuples()
            ]
        )
    )

    def flow_time(test_no):
        t, ims = clip(int(test_no))
        return max_crush_time(*body_velocity(ims, t)), t

    # systematic lag of the stop criterion, estimated on the training split only
    train_err = []
    for r in train.itertuples():
        f, _ = flow_time(r.test_no)
        if f is not None:
            train_err.append(f - r.t_max_crush_ms)
    bias = float(np.median(train_err))
    rows = []
    for r in evalset.itertuples():
        f, t = flow_time(r.test_no)
        rows.append(
            {
                "test_no": int(r.test_no),
                "physics_ms": r.t_max_crush_ms,
                "flow_ms": None if f is None else f - bias,
                "blind_ms": t[0] + rel,
            }
        )
    d = pd.DataFrame(rows)
    lines = [
        f"# Time of maximum crush from video (optical flow), frontal, {split_name} split\n",
        f"n = {len(d)} frontal tests with a verified time axis and sensor result"
        + (
            f"; clip starts cropped at random (0-{cfg.m3.window_jitter_ms} ms)."
            if crop
            else (
                "; full clips (design split)."
                if split_name == "train"
                else "; full clips (held out: not used for design)."
            )
        )
        + f" Flow bias correction (median training error): {bias:+.1f} ms.\n",
    ]
    summary = {"split": split_name, "n": len(d), "bias_ms": bias, "cropped": bool(crop)}
    for name in ("flow_ms", "blind_ms"):
        e = (d[name] - d.physics_ms).dropna()
        summary[name[:-3]] = {
            "median_abs_err_ms": float(e.abs().median()),
            "within_8": float((e.abs() <= 8).sum() / len(d)),
            "within_16": float((e.abs() <= 16).sum() / len(d)),
            "spearman": float(spearmanr(d.dropna()[name], d.dropna().physics_ms)[0]),
        }
        det = d[name].notna().mean()
        rho, p = spearmanr(d.dropna()[name], d.dropna().physics_ms)
        lines.append(
            f"- {name[:-3]}: detected {det:.2f}, median error {e.median():+.1f} ms, "
            f"median |error| {e.abs().median():.1f} ms, within 8 ms "
            f"{(e.abs() <= 8).sum() / len(d):.2f}, "
            f"within 16 ms {(e.abs() <= 16).sum() / len(d):.2f}, "
            f"Spearman rho with physics {rho:.2f} (p = {p:.3f})"
        )
    lines += ["", d.round(1).to_markdown(index=False)]
    out = Path("reports/physics")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"flow_{split_name}.md").write_text("\n".join(lines) + "\n")
    (out / f"flow_{split_name}.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
