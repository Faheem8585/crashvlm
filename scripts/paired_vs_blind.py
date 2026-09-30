"""Paired comparison of every VLM run (and the best probes) with the blind baseline (CPU).

    uv run python scripts/paired_vs_blind.py

t0 unknown, test split, the same crops for every method. For each timed event and tolerance,
the difference in hit@k (method minus blind) is computed on the same clips, with a bootstrap
over tests. A difference whose interval excludes 0 is evidence that the method reads the
video rather than a timing prior. Writes reports/m4/paired.md.
"""

import json
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas
from crashvlm.eval.events import build_ground_truth

PROBES = [
    "resnet50 linear probe",
    "siglip2-so400m-patch14-384 linear probe",
]


def hits(gts, preds, tol) -> dict:
    """(clip, event) -> 1/0 for every present event with a trusted time; a miss counts as 0."""
    out = {}
    for g in gts:
        if not (g.present and g.timing_ok and g.t_ms is not None):
            continue
        p = preds.get((g.clip_id, g.event))
        out[(g.clip_id, g.event)] = int(p is not None and abs(p - g.t_ms) <= tol)
    return out


@hydra.main(config_path="../configs", config_name="m4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    data = Path(cfg.paths.root)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    cfg_key = dict(zip(sel.test_no, sel.config_key, strict=True))
    split = dict(zip(sel.test_no, sel.split, strict=True))
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet")
    verified = dict(zip(man.clip_id, man.time_verified, strict=True))
    recs = records_from_labels(
        json.loads(Path(cfg.m3.labels).read_text()),
        metas,
        onto,
        "a",
        "labels",
        cfg_key,
        onto.annotation["frame_stride"],
    )
    gts = [
        g
        for g in build_ground_truth(
            recs,
            OmegaConf.to_container(cfg.m3.events_t0_unknown),
            lambda r, c: set(onto.events_for(r, c)),
            verified,
        )
        if split[g.test_no] == "test"
    ]
    test_of = {g.clip_id: g.test_no for g in gts}

    m3 = json.loads(Path("reports/m3/predictions.json").read_text())

    def m3_preds(name):
        raw = m3[f"t0 unknown | {name}"]
        return {tuple(k.split("|")): v for k, v in raw.items()}

    methods = {name: m3_preds(name) for name in PROBES if f"t0 unknown | {name}" in m3}
    for run in cfg.m4.runs:
        path = data / "vlm" / f"{run}.jsonl"
        if not path.exists():
            continue
        preds = {}
        for line in path.read_text().splitlines():
            o = json.loads(line)
            for ev, rel in ((o["report"] or {}).get("events") or {}).items():
                if o["parse_ok"] and rel is not None:
                    preds[(o["clip_id"], ev)] = o["start_abs_ms"] + rel
        # score VLMs only on clips they were run on
        methods[run] = (preds, {o for o in (json.loads(x)["clip_id"] for x in path.open())})
    blind = m3_preds("blind (no video)")

    rng = np.random.default_rng(cfg.m3.seed)
    rows = []
    for name, spec in methods.items():
        preds, clips = spec if isinstance(spec, tuple) else (spec, None)
        for tol in (8, 16):
            hm = hits([g for g in gts if clips is None or g.clip_id in clips], preds, tol)
            hb = hits([g for g in gts if clips is None or g.clip_id in clips], blind, tol)
            by_event = defaultdict(list)
            for key in hm:
                by_event[key[1]].append(key)
            for ev, keys in sorted(by_event.items()):
                by_test = defaultdict(list)
                for k in keys:
                    by_test[test_of[k[0]]].append(hm[k] - hb[k])
                tests = list(by_test)
                diff = float(np.mean([d for t in tests for d in by_test[t]]))
                boots = []
                for _ in range(cfg.m3.bootstrap):
                    pick = rng.choice(len(tests), len(tests))
                    ds = [d for i in pick for d in by_test[tests[i]]]
                    boots.append(np.mean(ds))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append(
                    {
                        "event": ev,
                        "tol_ms": tol,
                        "method": name,
                        "n": len(keys),
                        "method_hit": round(float(np.mean([hm[k] for k in keys])), 2),
                        "blind_hit": round(float(np.mean([hb[k] for k in keys])), 2),
                        "diff": round(diff, 2),
                        "diff 95% CI": f"{lo:+.2f} to {hi:+.2f}",
                        "CI excludes 0": bool(lo > 0 or hi < 0),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["event", "tol_ms", "method"])
    out = Path("reports/m4")
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired.md").write_text(
        "# Paired comparison with the blind baseline (test split, t0 unknown)\n\n"
        "hit@k on the same clips; difference = method minus blind; bootstrap over tests.\n\n"
        + df.to_markdown(index=False)
        + "\n"
    )
    (out / "paired.json").write_text(df.to_json(orient="records", indent=1))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
