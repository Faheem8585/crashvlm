"""M2: package annotation passes as CVAT image tasks, with pre-labels for pass A.

    uv run python scripts/make_cvat_tasks.py
    uv run python scripts/make_cvat_tasks.py \
        annotation.zero_shot_preds=data/annotation/zero_shot_preds.jsonl

Writes to <data>/annotation/:
  cvat_labels.json          paste into CVAT > Project > Labels > Raw
  clip_index.json           clip -> role, test, frame times (used by the zero-shot scorer)
  pass_a/images.zip         all clips, every 2nd processed frame (8 ms)
  pass_a/prelabels.xml      "CVAT 1.1" annotations to upload into the pass_a task
  pass_b/images.zip         IAA clips only, no pre-labels
"""

import json
import zipfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from crashvlm.annotation.cvat_io import frame_name, write_cvat_xml
from crashvlm.annotation.ontology import Ontology
from crashvlm.annotation.preannotate import annotation_frames, build_entries
from crashvlm.data.manifest import load_clip_metas


def load_zero_shot(path) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    preds = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        preds[r["clip_id"]] = {k: v["t_ms"] for k, v in r["events"].items()}
    return preds


@hydra.main(config_path="../configs", config_name="annotation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    stride = onto.annotation["frame_stride"]
    out = Path(cfg.annotation.out)
    clips_root = Path(cfg.paths.processed) / "clips"
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet").set_index("test_no")
    man = pd.read_parquet(Path(cfg.paths.processed) / "manifest.parquet").set_index("clip_id")
    metas = {m["clip_id"]: m for m in load_clip_metas(clips_root)}
    zero_shot = load_zero_shot(cfg.annotation.zero_shot_preds)

    missing = [f"{t}_{r}" for t in sel.index for r in cfg.clips.roles if f"{t}_{r}" not in metas]
    if missing:
        print(f"WARNING: {len(missing)} selected clips not processed: {missing[:10]}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "cvat_labels.json").write_text(json.dumps(onto.cvat_labels(), indent=1))
    index = {
        cid: {
            "test_no": m["test_no"],
            "role": m["role"],
            "config_key": sel.loc[m["test_no"], "config_key"],
            "frames": [
                {"idx": f["idx"], "t_ms": f["t_ms"], "file": frame_name(cid, f["idx"])}
                for f in annotation_frames(m, stride)
            ],
        }
        for cid, m in metas.items()
    }
    (out / "clip_index.json").write_text(json.dumps(index))

    for pass_name, spec in onto.annotation["passes"].items():
        cids = sorted(metas)
        if spec["clips"] == "iaa":
            iaa_tests = set(sel.index[sel["iaa"]])
            cids = [c for c in cids if metas[c]["test_no"] in iaa_tests]
        pdir = out / pass_name
        pdir.mkdir(parents=True, exist_ok=True)
        entries = []
        n_auto = 0
        with zipfile.ZipFile(pdir / "images.zip", "w", zipfile.ZIP_STORED) as zf:
            for cid in cids:
                m = metas[cid]
                frames = annotation_frames(m, stride)
                size = Image.open(clips_root / cid / frames[0]["file"]).size
                for f in frames:
                    zf.write(clips_root / cid / f["file"], frame_name(cid, f["idx"]))
                ents = build_entries(
                    m,
                    size,
                    onto,
                    sel.loc[m["test_no"], "config_key"],
                    stride,
                    bool(man.loc[cid, "time_verified"]),
                    spec["prelabels"],
                    zero_shot.get(cid),
                )
                n_auto += sum(len(e.tags) for e in ents)
                entries.extend(ents)
        if spec["prelabels"]:
            write_cvat_xml(entries, pdir / "prelabels.xml")
        mb = (pdir / "images.zip").stat().st_size / 1e6
        print(
            f"{pass_name}: {len(cids)} clips, {len(entries)} images, {mb:.0f} MB, "
            f"{n_auto} pre-labels"
        )


if __name__ == "__main__":
    main()
