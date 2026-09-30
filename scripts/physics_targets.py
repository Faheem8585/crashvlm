"""Physics targets from NHTSA sensor data for all selected tests, and checks against labels.

    uv run python scripts/physics_targets.py

Per test: download the ISO-MME export, integrate structural accelerations
(crashvlm.data.sensors), and delete the archive (only derived values are kept).
Writes <data>/physics/kinematics.parquet and reports/physics/{kinematics.csv, checks.md}:
- dynamic vs residual (NHTSA) crush per test type: a physical sanity check,
- the visual max_deformation labels (primary annotator and blind cross-check) against the
  measured time of maximum crush.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr

from crashvlm.data.clips import download
from crashvlm.data.nhtsa import NHTSAClient
from crashvlm.data.sensors import kinematics, open_iso


def label_times(path: Path, role: str = "exterior") -> dict[int, float]:
    if not path.exists():
        return {}
    out = {}
    for cid, lab in json.loads(path.read_text()).items():
        t, r = cid.split("_")
        ev = (lab.get("events") or {}).get("max_deformation")
        if r == role and ev is not None:
            out[int(t)] = float(ev["t_ms"] if isinstance(ev, dict) else ev)
    return out


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.root)
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    client = NHTSAClient(cfg.paths.api_cache)
    tmp = root / "tmp_sensors"
    tmp.mkdir(parents=True, exist_ok=True)

    def one(row):
        t, config = int(row.test_no), row.config_key
        try:
            docs = client.get_json(f"vehicle-documents/test-no/{t}")["results"][0]
            path = download(docs["isoFiles"], tmp / f"{t}.zip")
            try:
                k = kinematics(open_iso(path, t), config)
            finally:
                path.unlink(missing_ok=True)
            return k.as_row()
        except Exception as e:  # noqa: BLE001 - recorded per test, never silently dropped
            return {"test_no": t, "config": config, "note": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, [r for _, r in sel.iterrows()]))
    kin = pd.DataFrame(rows).merge(sel[["test_no", "split", "crush_mm"]], on="test_no")
    (root / "physics").mkdir(exist_ok=True)
    kin.to_parquet(root / "physics" / "kinematics.parquet", index=False)
    out = Path("reports/physics")
    out.mkdir(parents=True, exist_ok=True)
    kin.to_csv(out / "kinematics.csv", index=False)

    ok = kin.dropna(subset=["t_max_crush_ms"])
    lines = [
        "# Physics targets from NHTSA sensor data\n",
        f"Tests with a result: {len(ok)} / {len(kin)}. Failures: "
        + (
            ", ".join(
                f"{r.test_no} ({r.note})" for r in kin[kin.t_max_crush_ms.isna()].itertuples()
            )
            or "none"
        ),
        "\n## Time of maximum dynamic crush (ms after t0)\n",
        ok.groupby("config")["t_max_crush_ms"]
        .describe()[["count", "min", "25%", "50%", "75%", "max"]]
        .round(1)
        .to_markdown(),
        "\n## Dynamic crush vs NHTSA residual crush\n",
        "ITV: dynamic value = barrier-to-vehicle approach, which includes the deformable barrier "
        "face, so it is larger than vehicle crush by construction.\n",
    ]
    for cfgk, g in ok.groupby("config"):
        d = g.dropna(subset=["dynamic_crush_mm", "crush_mm"])
        if len(d) > 5:
            rho, p = spearmanr(d.dynamic_crush_mm, d.crush_mm)
            ratio = (d.dynamic_crush_mm / d.crush_mm).median()
            lines.append(
                f"- {cfgk}: Spearman rho = {rho:.2f} (p = {p:.3f}, n = {len(d)}); "
                f"median dynamic/residual = {ratio:.2f}"
            )
    lines += [
        "\n## How each time was obtained\n",
        ok.groupby(["config", "note"]).size().to_markdown(),
    ]
    lines.append("\n## Visual max_deformation labels vs measured time of maximum crush\n")
    lines.append(
        "Per test type: median |label - physics| and share within 16 ms, next to a blind "
        "constant (leave-one-out median physics time of the same test type).\n"
    )
    rows = []
    for name, path in [
        ("primary annotator", Path("annotations/labels_v1.json")),
        ("blind cross-check", Path("annotations/check_crossannotator.json")),
    ]:
        lab = label_times(path)
        d = ok[ok.test_no.isin(lab)].copy()
        d["label"] = d.test_no.map(lab)
        for cfgk, g in d.groupby("config"):
            ref = ok[ok.config == cfgk]
            loo = np.array([ref[ref.test_no != t].t_max_crush_ms.median() for t in g.test_no])
            err = (g.label - g.t_max_crush_ms).abs()
            err0 = np.abs(loo - g.t_max_crush_ms.to_numpy())
            rows.append(
                {
                    "labels": name,
                    "config": cfgk,
                    "n": len(g),
                    "label median |err| ms": round(err.median(), 1),
                    "label within 16 ms": round((err <= 16).mean(), 2),
                    "constant median |err| ms": round(float(np.median(err0)), 1),
                    "constant within 16 ms": round(float((err0 <= 16).mean()), 2),
                }
            )
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    (out / "checks.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
