"""Seeded test selection and vehicle-grouped train/val/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vehicle_group(df: pd.DataFrame) -> pd.Series:
    # The same make/model can appear in frontal, side and pole tests (and across model years
    # sharing a platform). Grouping by make+model keeps one vehicle's footage in one split.
    return df["make"].fillna("?").str.upper() + "|" + df["model"].fillna("?").str.upper()


def select_tests(cat: pd.DataFrame, per_config: dict[str, int], seed: int) -> pd.DataFrame:
    usable = cat[cat["has_both_views"]]
    parts = []
    for cfg_key, n in per_config.items():
        pool = usable[usable["config_key"] == cfg_key]
        parts.append(pool.sample(n=min(n, len(pool)), random_state=seed))
    return pd.concat(parts).sort_values("test_no").reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    fractions: dict[str, float],
    seed: int,
    iaa_tests: int = 0,
    strata: str = "config_key",
) -> pd.DataFrame:
    """Assign whole vehicle groups to splits, keeping each stratum (test type) near its targets.

    Groups are visited in seeded random order; each goes to the split whose fill ratio, over the
    strata the group touches, would stay lowest after adding it.
    """
    df = df.copy()
    df["group"] = vehicle_group(df)
    rng = np.random.default_rng(seed)
    groups = rng.permutation(sorted(df["group"].unique()))
    counts = df.groupby(["group", strata]).size().unstack(fill_value=0)
    names = list(fractions)
    per_stratum = df[strata].value_counts()
    targets = {s: {c: fractions[s] * n for c, n in per_stratum.items()} for s in names}
    filled = {s: dict.fromkeys(per_stratum.index, 0) for s in names}
    assign = {}
    for g in groups:
        cnt = counts.loc[g]
        cnt = cnt[cnt > 0]

        def ratio(s, cnt=cnt):
            if not fractions[s]:
                return np.inf
            return max((filled[s][c] + n) / targets[s][c] for c, n in cnt.items())

        k = min(names, key=ratio)
        assign[g] = k
        for c, n in cnt.items():
            filled[k][c] += int(n)
    df["split"] = df["group"].map(assign)
    df["iaa"] = False
    test_idx = df.index[df["split"] == "test"].to_numpy()
    if iaa_tests and len(test_idx):
        pick = rng.choice(test_idx, size=min(iaa_tests, len(test_idx)), replace=False)
        df.loc[pick, "iaa"] = True
    return df
