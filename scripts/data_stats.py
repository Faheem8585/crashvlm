"""Write docs/generated/data_stats.md from the catalog, selection and clip manifest.

    uv run python scripts/data_stats.py

All dataset numbers quoted in docs come from this file; nothing is typed in by hand.
"""

from datetime import date
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    interim, processed = Path(cfg.paths.interim), Path(cfg.paths.processed)
    cat = pd.read_parquet(interim / "catalog.parquet")
    sel = pd.read_parquet(interim / "selection.parquet")
    parts = [f"# Dataset statistics (generated {date.today().isoformat()})\n"]

    parts.append("## Catalog: NCAP tests 2012+ in the NHTSA database\n")
    parts.append(
        md(
            cat.groupby("config_key").agg(
                tests=("test_no", "size"),
                both_views=("has_both_views", "sum"),
                first_year=("year", "min"),
                last_year=("year", "max"),
                crush_known=("crush_mm", "count"),
                driver_hic_known=("driver_hic", "count"),
            )
        )
    )
    parts.append(f"\n## Selection and split (seed {cfg.seed})\n")
    parts.append(md(pd.crosstab(sel["config_key"], sel["split"], margins=True)))
    parts.append(
        f"\nVehicle groups: {sel['group'].nunique()}; IAA tests: {int(sel['iaa'].sum())}\n"
    )
    parts.append("\n### Test labs in the selection\n")
    parts.append(md(pd.crosstab(sel["performer"], sel["config_key"])))

    man_path = processed / "manifest.parquet"
    if man_path.exists():
        man = pd.read_parquet(man_path)
        parts.append(f"\n## Processed clips ({len(man)})\n")
        parts.append(
            md(
                man.groupby(["config_key", "role"])
                .agg(
                    clips=("clip_id", "size"),
                    time_from_ocr=("time_source", lambda s: int((s == "ocr").sum())),
                    time_verified=("time_verified", "sum"),
                    median_crop_frac=("crop_frac", "median"),
                    median_masks=("n_masks", "median"),
                )
                .round(3)
            )
        )
        parts.append("\n### Source codecs\n")
        parts.append(md(man.groupby(["codec", "performer"]).size().rename("clips").to_frame()))
    out = Path("docs/generated")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_stats.md").write_text("\n".join(parts) + "\n")
    print(f"wrote {out / 'data_stats.md'}")


if __name__ == "__main__":
    main()
