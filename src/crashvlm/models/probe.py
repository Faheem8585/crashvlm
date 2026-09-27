"""Frame-level "has the event happened yet?" classifiers on frozen embeddings.

Training frames get label 1 from the annotated onset onwards (clips where the event is present)
and 0 everywhere else, including every frame of clips where the event never happens. The
predicted onset is the start of the first sustained run above a threshold
(``zero_shot.curve_to_event``), so zero-shot curves and probe curves are decoded identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from crashvlm.models.zero_shot import curve_to_event, pos_neg_scores


def frame_targets(t_ms: np.ndarray, onset: float | None) -> np.ndarray:
    if onset is None:
        return np.zeros(len(t_ms), dtype=int)
    return (t_ms >= onset - 1e-6).astype(int)


def train_probe(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0) -> LogisticRegression:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X.astype(np.float32), y)
    return clf


def probe_curve(clf: LogisticRegression, emb: np.ndarray) -> np.ndarray:
    return clf.predict_proba(emb.astype(np.float32))[:, 1]


def zero_shot_curve(
    emb: np.ndarray, text: np.ndarray, n_pos: int, scale: float, bias: float
) -> np.ndarray:
    logits = emb.astype(np.float32) @ text.astype(np.float32).T * scale + bias
    return pos_neg_scores(logits, n_pos)


def decode(curve: np.ndarray, t_ms: np.ndarray, thr: float, min_run: int, smooth: int):
    return curve_to_event(curve, t_ms, "onset", thr, min_run, smooth)
