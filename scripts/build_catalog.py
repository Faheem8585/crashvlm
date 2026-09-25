"""M1 step 1: crawl the NHTSA API and build the NCAP test catalog.

    uv run python scripts/build_catalog.py

Writes <data>/interim/catalog.parquet (one row per test) and occupants.parquet.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from crashvlm.data.catalog import build_catalog
from crashvlm.data.nhtsa import NHTSAClient


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    client = NHTSAClient(cfg.paths.api_cache)
    cat, occ = build_catalog(client, cfg)
    out = Path(cfg.paths.interim)
    out.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(out / "catalog.parquet", index=False)
    occ.to_parquet(out / "occupants.parquet", index=False)

    print(f"catalog: {len(cat)} tests, {int(cat['has_both_views'].sum())} with both views")
    summary = cat.groupby("config_key").agg(
        tests=("test_no", "size"),
        both_views=("has_both_views", "sum"),
        years=("year", lambda s: f"{s.min()}-{s.max()}"),
        crush_known=("crush_mm", "count"),
        driver_hic_known=("driver_hic", "count"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
