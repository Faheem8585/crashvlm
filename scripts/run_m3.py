"""M3: blind constant baseline vs. SigLIP zero-shot vs. linear probes (SigLIP, ResNet-50),
evaluated on the test split in two settings. Runs on CPU from precomputed embeddings.

    uv run python scripts/run_m3.py

Settings
- ``t0 known``: the clip's time axis is anchored at impact (lab setting). The blind
  baseline, which predicts the median training onset per event/test type, is very strong here
  because NCAP protocols are highly standardised.
- ``t0 unknown``: each test clip is cropped at a random start (0 to ``window_jitter_ms``, seeded)
  and predictions may only use time relative to the first remaining frame, as with an arbitrary
  crash video. Video models are unaffected (they read frames, not the clock). The blind baseline
  must place the event relative to the clip start. ``impact_onset`` is also evaluated here,
  because finding the impact is then a real task.

Hyper-parameters (probe C, decision threshold) are chosen per event on the validation split
(t0-known setting, score = mean(F1, hit@16)). The test split is used once. Writes
reports/m3/results.md, results.json, predictions.json.
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
from crashvlm.eval.events import build_ground_truth, evaluate, window_crops
from crashvlm.models.probe import decode, frame_targets, probe_curve, train_probe, zero_shot_curve


def load_features(path: Path):
    z = np.load(path)
    clips = {}
    for k in z.files:
        if k.endswith("/emb"):
            cid = k[: -len("/emb")]
            clips[cid] = (z[k], z[f"{cid}/t_ms"])
    text = {
        k[len("_text/") :]: z[k] for k in z.files if k.startswith("_text/") and "/n_pos" not in k
    }
    npos = {k[len("_text/") : -len("/n_pos")]: int(z[k]) for k in z.files if k.endswith("/n_pos")}
    scale = float(z["_logit_scale"]) if "_logit_scale" in z.files else None
    bias = float(z["_logit_bias"]) if "_logit_bias" in z.files else 0.0
    return clips, text, npos, scale, bias


def score(metrics: dict) -> float:
    return float(np.nanmean([metrics["f1"], metrics.get("hit@16", np.nan)]))


def decode_all(curves, gts, thr, mc, crop=None) -> dict:
    out = {}
    for g in gts:
        key = (g.clip_id, g.event)
        if key not in curves:
            continue
        curve, t = curves[key]
        if crop is not None:
            keep = t >= t[0] + crop[g.clip_id]
            curve, t = curve[keep], t[keep]
        out[key] = decode(curve, t, thr, mc.min_run, mc.smooth)
    return out


class Blind:
    """Median training onset per (event, config, role), both as absolute time (t0 known) and
    as time since the clip's first frame (t0 unknown); predicts 'absent' if the event occurs in
    fewer than half of the training clips."""

    def __init__(self, train, clip_start: dict[str, float]):
        self.abs, self.rel = {}, {}
        groups: dict = {}
        for g in train:
            groups.setdefault((g.event, g.config, g.role), []).append(g)
        for key, gs in groups.items():
            pos = [g for g in gs if g.present and g.t_ms is not None]
            if len(pos) < len(gs) / 2:
                self.abs[key] = self.rel[key] = None
                continue
            timed = [g for g in pos if g.timing_ok] or pos
            self.abs[key] = float(np.median([g.t_ms for g in timed]))
            self.rel[key] = float(np.median([g.t_ms - clip_start[g.clip_id] for g in timed]))

    def predict(self, gts, clip_start, crop=None) -> dict:
        out = {}
        for g in gts:
            key = (g.event, g.config, g.role)
            if crop is None:
                out[(g.clip_id, g.event)] = self.abs.get(key)
            else:
                rel = self.rel.get(key)
                start = clip_start[g.clip_id] + crop[g.clip_id]
                out[(g.clip_id, g.event)] = None if rel is None else start + rel
        return out


def tune_thr(curves, val_all, events, mc, tols) -> dict:
    thr = {}
    for ev in events:
        val = [g for g in val_all if g.event == ev and (g.clip_id, ev) in curves]
        if not val:
            continue
        thr[ev] = max(
            mc.thresholds,
            key=lambda t, val=val, ev=ev: score(
                evaluate(val, decode_all(curves, val, t, mc), tols, 0)[ev]
            ),
        )
    return thr


def fit_probes(clips, events, sp, all_gts, mc, tols):
    curves, thr = {}, {}
    for ev in events:
        tr = [g for g in sp["train"] if g.event == ev and g.clip_id in clips]
        if not tr:
            continue
        X = np.concatenate([clips[g.clip_id][0] for g in tr])
        y = np.concatenate([frame_targets(clips[g.clip_id][1], g.t_ms) for g in tr])
        if y.min() == y.max():
            continue
        best = None
        val = [g for g in sp["val"] if g.event == ev]
        for C in mc.probe_C:
            clf = train_probe(X, y, C, mc.seed)
            cv = {
                (g.clip_id, ev): (probe_curve(clf, clips[g.clip_id][0]), clips[g.clip_id][1])
                for g in all_gts
                if g.event == ev and g.clip_id in clips
            }
            for t in mc.thresholds:
                s = score(evaluate(val, decode_all(cv, val, t, mc), tols, 0)[ev])
                if best is None or s > best[0]:
                    best = (s, cv, t)
        curves.update(best[1])
        thr[ev] = best[2]
    return curves, thr


@hydra.main(config_path="../configs", config_name="m3", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mc = cfg.m3
    tols = tuple(mc.tolerances_ms)
    onto = Ontology.load(cfg.annotation.ontology)
    data = Path(cfg.paths.root)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    clip_start = {c: m["frames"][0]["t_ms"] for c, m in metas.items()}
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    split = dict(zip(sel.test_no, sel.split, strict=True))
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet")
    verified = dict(zip(man.clip_id, man.time_verified, strict=True))
    labels = json.loads(Path(mc.labels).read_text())
    recs = records_from_labels(
        labels,
        metas,
        onto,
        "a",
        "labels",
        dict(zip(sel.test_no, sel.config_key, strict=True)),
        onto.annotation["frame_stride"],
    )

    def allowed(role, config):
        return set(onto.events_for(role, config))

    settings = {
        "t0 known": build_ground_truth(recs, OmegaConf.to_container(mc.events), allowed, verified),
        "t0 unknown": build_ground_truth(
            recs, OmegaConf.to_container(mc.events_t0_unknown), allowed, verified
        ),
    }
    events = sorted({g.event for gts in settings.values() for g in gts})
    all_gts = list({(g.clip_id, g.event): g for gts in settings.values() for g in gts}.values())
    sp = {s: [g for g in all_gts if split[g.test_no] == s] for s in ("train", "val", "test")}
    test_sets = {n: [g for g in gts if split[g.test_no] == "test"] for n, gts in settings.items()}
    crop = window_crops([g.clip_id for g in test_sets["t0 unknown"]], mc.window_jitter_ms, mc.seed)

    methods: dict[str, tuple[dict, dict]] = {}
    for slug in mc.backbones:
        fpath = data / "features" / f"{slug}.npz"
        if not fpath.exists():
            print(f"skip {slug}: {fpath} not found")
            continue
        clips, text, npos, scale, bias = load_features(fpath)
        name = slug.split("__")[-1]
        if text:
            curves = {
                (g.clip_id, g.event): (
                    zero_shot_curve(clips[g.clip_id][0], text[g.event], npos[g.event], scale, bias),
                    clips[g.clip_id][1],
                )
                for g in all_gts
                if g.clip_id in clips and g.event in text
            }
            methods[f"{name} zero-shot"] = (curves, tune_thr(curves, sp["val"], events, mc, tols))
        methods[f"{name} linear probe"] = fit_probes(clips, events, sp, all_gts, mc, tols)

    blind = Blind(sp["train"], clip_start)
    results, preds_out, rows = {}, {}, []
    for setting, test in test_sets.items():
        c = crop if setting == "t0 unknown" else None
        preds = {"blind (no video)": blind.predict(test, clip_start, c)}
        for mname, (curves, thr) in methods.items():
            preds[mname] = {}
            for ev, t in thr.items():
                preds[mname].update(
                    decode_all(curves, [g for g in test if g.event == ev], t, mc, c)
                )
        for mname, p in preds.items():
            # score a method only on the events it covers (zero-shot has no impact prompt)
            covered = None if mname not in methods else set(methods[mname][1])
            tg = [g for g in test if covered is None or g.event in covered]
            res = evaluate(tg, p, tols, mc.bootstrap, mc.seed)
            results[f"{setting} | {mname}"] = res
            preds_out[f"{setting} | {mname}"] = {f"{k[0]}|{k[1]}": v for k, v in p.items()}
            for ev, m in res.items():
                ci = m.get("hit@16_ci")
                rows.append(
                    {
                        "setting": setting,
                        "event": ev,
                        "method": mname,
                        "n": m["n"],
                        "pos": m["n_pos"],
                        "F1": round(m["f1"], 2),
                        "hit@8": round(m["hit@8"], 2),
                        "hit@16": round(m["hit@16"], 2),
                        "hit@16 95% CI": f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "",
                        "median |err| ms": round(m["median_abs_err_ms"], 1),
                    }
                )

    out = Path(mc.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "predictions.json").write_text(json.dumps(preds_out))
    df = pd.DataFrame(rows)
    n_test = len({g.test_no for g in test_sets["t0 known"]})
    md = [
        "# M3 results (test split)\n",
        f"Test tests: {n_test}; labels: `{mc.labels}` (model-produced, see "
        "docs/ANNOTATION_QA.md). Tier-A events; hit@k counts a missed detection as a miss; "
        "CIs: bootstrap over tests. Backbones: " + ", ".join(mc.backbones) + ".",
        f"t0 unknown: each test clip cropped at a random start in [0, {mc.window_jitter_ms}] ms "
        f"(seed {mc.seed}).\n",
    ]
    for setting in test_sets:
        tab = df[df.setting == setting].drop(columns="setting").sort_values(["event", "method"])
        md += [f"## {setting}\n", tab.to_markdown(index=False), ""]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
