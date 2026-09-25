"""M1 step 2: seeded, stratified test selection and vehicle-grouped split.

    uv run python scripts/select_tests.py

Writes <data>/interim/selection.parquet. The first run freezes the chosen test IDs, splits and
IAA flags to configs/splits/selection_v1.csv (committed); later runs reuse that file.
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from crashvlm.data.split import group_split, select_tests, vehicle_group


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    interim = Path(cfg.paths.interim)
    cat = pd.read_parquet(interim / "catalog.parquet")
    per_config = OmegaConf.to_container(cfg.selection.per_config)
    fractions = OmegaConf.to_container(cfg.selection.split)
    frozen = Path(cfg.selection.frozen)
    if frozen.exists():
        # The live API keeps adding tests, so the selection is frozen after the first run.
        keep = pd.read_csv(frozen)[["test_no", "split", "iaa"]]
        sel = cat.merge(keep, on="test_no", how="inner")
        sel["group"] = vehicle_group(sel)
        missing = set(keep["test_no"]) - set(sel["test_no"])
        assert not missing, f"frozen tests missing from catalog: {sorted(missing)}"
        print(f"using frozen selection {frozen}")
    else:
        sel = select_tests(cat, per_config, cfg.seed)
        sel = group_split(sel, fractions, cfg.seed, cfg.selection.iaa_tests)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        sel[["test_no", "config_key", "make", "model", "split", "iaa"]].to_csv(frozen, index=False)
        print(f"froze selection -> {frozen} (commit this file)")
    sel.to_parquet(interim / "selection.parquet", index=False)
    cols = ["test_no", "config_key", "year", "make", "model", "model_year", "split", "iaa"]
    sel[cols].to_csv(interim / "selection.csv", index=False)

    print(pd.crosstab(sel["config_key"], sel["split"], margins=True).to_string())
    print(f"IAA subset: {int(sel['iaa'].sum())} tests; vehicle groups: {sel['group'].nunique()}")
    leak = sel.groupby("group")["split"].nunique().max()
    assert leak == 1, "a vehicle group spans multiple splits"


if __name__ == "__main__":
    main()
