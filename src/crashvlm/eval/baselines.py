"""Baseline machinery shared by M3 (evaluation) and M7 (serving): feature loading, probe
fitting with validation-based selection, curve decoding and the blind no-video baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crashvlm.eval.events import evaluate
from crashvlm.models.probe import decode, frame_targets, probe_curve, train_probe


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
    as time since the first frame shown (t0 unknown); predicts 'absent' if the event occurs in
    fewer than half of the training clips.

    For t0 unknown, pass ``train_crop`` (the same random start crops, drawn for the training
    clips). The prior is then measured from the cropped start, exactly like a model trained on
    cropped clips. Without it, the prior is measured from the uncropped start and is late by
    the mean crop (about 20 ms for a 0-40 ms jitter), which understates the baseline."""

    def __init__(self, train, clip_start: dict[str, float], train_crop: dict | None = None):
        train_crop = train_crop or {}
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
            self.rel[key] = float(
                np.median(
                    [g.t_ms - clip_start[g.clip_id] - train_crop.get(g.clip_id, 0.0) for g in timed]
                )
            )

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


def fit_probes(clips, events, sp, all_gts, mc, tols, chosen_C=None):
    """Per event: pick C and threshold on val. ``chosen_C`` (dict) receives the winning C."""
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
                    best = (s, cv, t, C)
        curves.update(best[1])
        thr[ev] = best[2]
        if chosen_C is not None:
            chosen_C[ev] = best[3]
    return curves, thr
