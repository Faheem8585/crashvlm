"""M1 step 3: download canonical views, strip overlays, extract time-stamped frames.

    uv run python scripts/prepare_clips.py clips.limit=4          # smoke run
    uv run python scripts/prepare_clips.py                        # full selection (Colab)

Resumable: clips with an existing meta.json are skipped. Failures go to errors.json.
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from crashvlm.data.clips import clip_id, prepare_clip
from crashvlm.data.manifest import build_manifest
from crashvlm.data.overlay import RapidOCRReader


@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    sel = sel[sel["split"].isin(list(cfg.clips.splits))]
    clips_root = Path(cfg.paths.processed) / "clips"
    tmp = Path(cfg.paths.tmp)
    # Round-robin over (test type, lab, role) so a small `clips.limit` still covers every
    # overlay layout instead of the first few tests of one lab.
    buckets: dict[tuple, list] = {}
    for _, r in sel.sort_values("test_no").iterrows():
        for role in cfg.clips.roles:
            if isinstance(r[f"{role}_url"], str):
                key = (r.config_key, r.performer, role)
                buckets.setdefault(key, []).append(
                    (int(r.test_no), role, r[f"{role}_url"], r[f"{role}_name"])
                )
    queues = [buckets[k] for k in sorted(buckets)]
    jobs = []
    while any(queues):
        for q in queues:
            if q:
                jobs.append(q.pop(0))
    if cfg.clips.limit:
        jobs = jobs[: cfg.clips.limit]

    reader = RapidOCRReader()
    clips_root.mkdir(parents=True, exist_ok=True)
    err_path = Path(cfg.paths.processed) / "errors.json"
    errors = {}
    for test_no, role, url, name in tqdm(jobs, desc="clips"):
        if (clips_root / clip_id(test_no, role) / "meta.json").exists():
            continue
        try:
            prepare_clip(test_no, role, url, name, clips_root, tmp, cfg, reader)
        except Exception as e:  # keep going; failures are reported, never silently dropped
            errors[clip_id(test_no, role)] = f"{type(e).__name__}: {e}"
            err_path.write_text(json.dumps(errors, indent=1))

    err_path.write_text(json.dumps(errors, indent=1))
    man = build_manifest(clips_root, sel)
    man.to_parquet(Path(cfg.paths.processed) / "manifest.parquet", index=False)
    print(f"clips ready: {len(man)} / {len(jobs)} requested, errors: {len(errors)} -> {err_path}")
    if len(man):
        print(man.groupby(["config_key", "role"])["time_source"].value_counts().to_string())
        agree = man["time_agrees_convention"].dropna()
        if len(agree):
            print(f"OCR time agrees with NCAP convention: {int(agree.sum())}/{len(agree)}")


if __name__ == "__main__":
    main()
