"""Agreement between two annotation passes over the same clips.

With a single annotator labelling twice this is *intra*-rater reliability. Pass A was
pre-labelled and pass B was not, so disagreement also includes any anchoring effect of the
pre-labels. Both effects are reported together, not separated.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters over the same items (nan if undefined)."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return 1.0 if po == 1 else float("nan")
    return (po - pe) / (1 - pe)


def event_agreement(
    recs_a: dict[str, dict], recs_b: dict[str, dict], event: str, tolerances_ms=(8, 16, 32)
) -> dict:
    """Presence agreement (kappa) and timing difference where both passes marked the event."""
    clips = sorted(set(recs_a) & set(recs_b))
    pres_a = [event in recs_a[c]["events"] for c in clips]
    pres_b = [event in recs_b[c]["events"] for c in clips]
    diffs = []
    for c in clips:
        ea, eb = recs_a[c]["events"].get(event), recs_b[c]["events"].get(event)
        if ea and eb and ea["t_ms"] is not None and eb["t_ms"] is not None:
            diffs.append(abs(ea["t_ms"] - eb["t_ms"]))
    out = {
        "event": event,
        "n_clips": len(clips),
        "n_a": int(sum(pres_a)),
        "n_b": int(sum(pres_b)),
        "presence_agreement": float(np.mean([x == y for x, y in zip(pres_a, pres_b, strict=True)]))
        if clips
        else float("nan"),
        "presence_kappa": cohen_kappa(pres_a, pres_b),
        "n_both": len(diffs),
        "median_abs_diff_ms": float(np.median(diffs)) if diffs else float("nan"),
    }
    for tol in tolerances_ms:
        out[f"within_{tol}ms"] = (
            float(np.mean([d <= tol for d in diffs])) if diffs else float("nan")
        )
    return out


def summary_agreement(recs_a: dict[str, dict], recs_b: dict[str, dict], attribute: str) -> dict:
    clips = sorted(set(recs_a) & set(recs_b))
    a = [recs_a[c]["summary"].get(attribute) for c in clips]
    b = [recs_b[c]["summary"].get(attribute) for c in clips]
    return {
        "attribute": attribute,
        "n_clips": len(clips),
        "agreement": float(np.mean([x == y for x, y in zip(a, b, strict=True)]))
        if clips
        else float("nan"),
        "kappa": cohen_kappa(a, b),
    }
