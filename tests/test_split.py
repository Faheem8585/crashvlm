import pandas as pd

from crashvlm.data.split import group_split, select_tests


def make_catalog(n=120):
    models = [f"M{i % 50}" for i in range(n)]  # repeated models across configs
    return pd.DataFrame(
        {
            "test_no": range(n),
            "config_key": ["VTB", "ITV", "VTP"] * (n // 3),
            "make": ["MAKE"] * n,
            "model": models,
            "has_both_views": [True] * n,
        }
    )


def test_select_is_seeded_and_stratified():
    cat = make_catalog()
    a = select_tests(cat, {"VTB": 10, "ITV": 5, "VTP": 5}, seed=1)
    b = select_tests(cat, {"VTB": 10, "ITV": 5, "VTP": 5}, seed=1)
    assert a["test_no"].tolist() == b["test_no"].tolist()
    assert a["config_key"].value_counts().to_dict() == {"VTB": 10, "ITV": 5, "VTP": 5}


def test_group_split_no_vehicle_leakage_and_iaa_in_test():
    df = group_split(make_catalog(), {"train": 0.6, "val": 0.15, "test": 0.25}, seed=0, iaa_tests=5)
    assert df.groupby("group")["split"].nunique().max() == 1
    assert df.loc[df["iaa"], "split"].eq("test").all() and df["iaa"].sum() == 5
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.6) < 0.1 and abs(frac["test"] - 0.25) < 0.1
    per_cfg = pd.crosstab(df["config_key"], df["split"], normalize="index")
    assert (per_cfg["test"] - 0.25).abs().max() < 0.12
