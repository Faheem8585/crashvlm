"""Event detection and temporal localisation metrics, shared by every method (M3-M6).

Ground truth per (clip, event), for clips that are complete and ``view_usable``, where the
event is defined for the clip's role and test type:

- ``present``: the annotator tagged the event.
- ``t_ms``: onset time. It counts for timing only if the onset was directly visible
  (``onset_visible``) and the clip's time axis is OCR-verified (``timing_ok``).

A method returns ``{(clip_id, event): t_ms or None}`` (None = predicts absent).

Metrics per event:
- presence precision / recall / F1,
- ``hit@k``: share of timing-eligible positives predicted within k ms (a missed detection counts
  as a miss), plus the median absolute error over detected positives,
- 95 % bootstrap intervals over *tests* (both views of a test are resampled together).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GT:
    clip_id: str
    test_no: int
    config: str
    role: str
    event: str
    present: bool
    t_ms: float | None
    timing_ok: bool


def build_ground_truth(
    records: list[dict], events: dict[str, list[str]], allowed, time_verified: dict[str, bool]
) -> list[GT]:
    """``allowed(role, config) -> set of events``; ``events``: event -> roles to evaluate."""
    out = []
    for r in records:
        if not r["complete"] or not r["summary"].get("view_usable", True):
            continue
        ok_events = allowed(r["role"], r["config_key"])
        for ev, roles in events.items():
            if r["role"] not in roles or ev not in ok_events:
                continue
            e = r["events"].get(ev)
            visible = bool(e and e["attributes"].get("onset_visible", True))
            out.append(
                GT(
                    r["clip_id"],
                    int(r["test_no"]),
                    r["config_key"],
                    r["role"],
                    ev,
                    e is not None,
                    e["t_ms"] if e else None,
                    visible and bool(time_verified.get(r["clip_id"], False)),
                )
            )
    return out


def _event_metrics(gts: list[GT], preds: dict, tols) -> dict:
    tp = fp = fn = 0
    errs, hits = [], defaultdict(int)
    n_timing = 0
    for g in gts:
        p = preds.get((g.clip_id, g.event))
        if g.present and p is not None:
            tp += 1
        elif g.present:
            fn += 1
        elif p is not None:
            fp += 1
        if g.present and g.timing_ok:
            n_timing += 1
            if p is not None:
                err = abs(p - g.t_ms)
                errs.append(err)
                for k in tols:
                    hits[k] += err <= k
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    m = {
        "n": len(gts),
        "n_pos": tp + fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_timing": n_timing,
        "median_abs_err_ms": float(np.median(errs)) if errs else float("nan"),
    }
    for k in tols:
        m[f"hit@{k}"] = hits[k] / n_timing if n_timing else float("nan")
    return m


def evaluate(
    gts: list[GT], preds: dict, tols=(8, 16, 32), n_boot: int = 1000, seed: int = 0
) -> dict[str, dict]:
    """Per-event metrics with bootstrap CIs (resampling tests) for f1 and hit@16."""
    rng = np.random.default_rng(seed)
    out = {}
    for ev in sorted({g.event for g in gts}):
        eg = [g for g in gts if g.event == ev]
        m = _event_metrics(eg, preds, tols)
        by_test = defaultdict(list)
        for g in eg:
            by_test[g.test_no].append(g)
        tests = list(by_test)
        boots = defaultdict(list)
        for _ in range(n_boot):
            sample = [g for t in rng.choice(tests, size=len(tests)) for g in by_test[t]]
            bm = _event_metrics(sample, preds, tols)
            for key in ("f1", "hit@16"):
                if not np.isnan(bm.get(key, np.nan)):
                    boots[key].append(bm[key])
        for key, vals in boots.items():
            if vals:
                m[f"{key}_ci"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        out[ev] = m
    return out


def constant_baseline(train: list[GT]) -> dict[tuple[str, str, str], float | None]:
    """ "Blind" predictor fit on training labels: for each (event, config, role) predict present
    if the event is present in at least half of training clips, at the median training onset.
    Uses no video at all, so it measures how much the NCAP protocol alone gives away."""
    groups = defaultdict(list)
    for g in train:
        groups[(g.event, g.config, g.role)].append(g)
    table = {}
    for key, gs in groups.items():
        pos = [g for g in gs if g.present]
        if len(pos) >= len(gs) / 2:
            ts = [g.t_ms for g in pos if g.timing_ok] or [g.t_ms for g in pos]
            table[key] = float(np.median(ts))
        else:
            table[key] = None
    return table


def apply_constant(table, gts: list[GT]) -> dict:
    return {(g.clip_id, g.event): table.get((g.event, g.config, g.role)) for g in gts}
