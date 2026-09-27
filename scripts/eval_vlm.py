"""M4/M5: evaluate VLM crash reports (CPU).

    uv run python scripts/eval_vlm.py

For every run in ``m4.runs`` (``<data>/vlm/<run>.jsonl``):
- event detection/timing with the M3 metrics, 't0 unknown' setting (same crops as M3),
  directly comparable with the M3 rows,
- ``test_type`` accuracy against NHTSA metadata (ground truth independent of the annotator),
- summary-field accuracy against the labels (tier B fields; tier C fields reported only as
  distributions),
- hallucination: (a) ``fire_or_fluid_leak = yes`` (no clip has one), (b) events reported as
  present where the label says absent, (c) events that cannot occur for the view/test type,
- parse rate, latency and peak VRAM.
Writes reports/m4/results.md and results.json.
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
from crashvlm.eval.events import build_ground_truth, evaluate
from crashvlm.models.vlm_report import EVENTS, TEST_TYPES

TIER_B = ["frontal_airbag_deployed", "curtain_airbag_deployed"]


def load_run(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@hydra.main(config_path="../configs", config_name="m4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    data = Path(cfg.paths.root)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    cfg_key = dict(zip(sel.test_no, sel.config_key, strict=True))
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet")
    verified = dict(zip(man.clip_id, man.time_verified, strict=True))
    labels = json.loads(Path(cfg.m3.labels).read_text())
    recs = {
        r["clip_id"]: r
        for r in records_from_labels(
            labels, metas, onto, "a", "labels", cfg_key, onto.annotation["frame_stride"]
        )
    }

    def allowed(role, config):
        return set(onto.events_for(role, config))

    gts_all = build_ground_truth(
        list(recs.values()), OmegaConf.to_container(cfg.m3.events_t0_unknown), allowed, verified
    )
    tols = tuple(cfg.m3.tolerances_ms)
    results, rows, summary_rows = {}, [], []
    for run in cfg.m4.runs:
        path = data / "vlm" / f"{run}.jsonl"
        if not path.exists():
            print(f"skip {run}: {path} not found")
            continue
        outs = {o["clip_id"]: o for o in load_run(path)}
        gts = [g for g in gts_all if g.clip_id in outs]
        preds = {}
        for g in gts:
            o = outs[g.clip_id]
            rel = (o["report"].get("events") or {}).get(g.event) if o["parse_ok"] else None
            preds[(g.clip_id, g.event)] = None if rel is None else o["start_abs_ms"] + rel
        ev_res = evaluate(gts, preds, tols, cfg.m3.bootstrap, cfg.m3.seed)

        n = len(outs)
        parsed = [o for o in outs.values() if o["parse_ok"]]
        tt_ok = [
            TEST_TYPES.get(o["report"].get("test_type")) == cfg_key[metas[o["clip_id"]]["test_no"]]
            for o in parsed
        ]
        fire = [o["report"].get("fire_or_fluid_leak") == "yes" for o in parsed]
        claims = false_claims = impossible = 0
        for o in parsed:
            r = recs.get(o["clip_id"])
            if r is None:
                continue
            ok_ev = allowed(r["role"], r["config_key"])
            for e in EVENTS:
                if (o["report"].get("events") or {}).get(e) is None:
                    continue
                claims += 1
                if e not in ok_ev:
                    impossible += 1
                elif e not in r["events"] and r["summary"].get("view_usable", True):
                    false_claims += 1
        field_acc = {}
        for k in TIER_B:
            pairs = [
                (o["report"].get(k), recs[o["clip_id"]]["summary"].get(k))
                for o in parsed
                if o["clip_id"] in recs
            ]
            field_acc[k] = float(np.mean([a == b for a, b in pairs])) if pairs else float("nan")
        lat = [o["latency_s"] for o in outs.values()]
        vram = [o["peak_vram_gb"] for o in outs.values() if o.get("peak_vram_gb") is not None]
        summ = {
            "clips": n,
            "parse_rate": len(parsed) / n if n else float("nan"),
            "schema_issues_per_clip": float(np.mean([len(o["issues"]) for o in parsed]))
            if parsed
            else float("nan"),
            "test_type_acc (NHTSA)": float(np.mean(tt_ok)) if tt_ok else float("nan"),
            **{f"{k}_acc": v for k, v in field_acc.items()},
            "fire_hallucination_rate": float(np.mean(fire)) if fire else float("nan"),
            "event_claims": claims,
            "claims_contradicting_labels": false_claims / claims if claims else float("nan"),
            "claims_impossible_for_view": impossible / claims if claims else float("nan"),
            "latency_s_median": float(np.median(lat)) if lat else float("nan"),
            "peak_vram_gb_max": float(max(vram)) if vram else float("nan"),
        }
        results[run] = {"events": ev_res, "summary": summ}
        summary_rows.append(
            {"run": run, **{k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}}
        )
        for ev, m in ev_res.items():
            ci = m.get("hit@16_ci")
            rows.append(
                {
                    "event": ev,
                    "run": run,
                    "n": m["n"],
                    "pos": m["n_pos"],
                    "F1": round(m["f1"], 2),
                    "hit@8": round(m["hit@8"], 2),
                    "hit@16": round(m["hit@16"], 2),
                    "hit@16 95% CI": f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "",
                    "median |err| ms": round(m["median_abs_err_ms"], 1),
                }
            )

    out = Path(cfg.m4.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    md = ["# M4 results: VLM crash reports (test split, t0 unknown)\n"]
    if summary_rows:
        md += ["## Report quality\n", pd.DataFrame(summary_rows).T.to_markdown(), ""]
        md += [
            "## Events (compare with the M3 't0 unknown' table)\n",
            pd.DataFrame(rows).sort_values(["event", "run"]).to_markdown(index=False),
        ]
    else:
        md.append("No runs found.")
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
