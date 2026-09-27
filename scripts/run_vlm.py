"""M4/M5: run a video VLM on test clips and store raw + parsed crash reports (GPU; Colab).

    python scripts/run_vlm.py                                               # Qwen3.5-4B
    python scripts/run_vlm.py vlm.model=Qwen/Qwen3-VL-8B-Instruct vlm.load_in_4bit=true \
        vlm.run_name=qwen3-vl-8b-4bit_zs

Each clip is cropped with the same seeded random start as the M3 't0 unknown' setting, and the
model sees ``n_frames`` frames every ``step_ms`` with times relative to the first frame shown.
Resumable: clips already present in the output JSONL are skipped.
"""

import io
import json
import zipfile
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from crashvlm.eval.events import window_crops
from crashvlm.models.vlm import VLMRunner
from crashvlm.models.vlm_report import build_messages, parse_report, select_frames


@hydra.main(config_path="../configs", config_name="m4", version_base="1.3")
def main(cfg: DictConfig) -> None:
    v = cfg.vlm
    ann = Path(cfg.annotation.out)
    index = json.loads((ann / "clip_index.json").read_text())
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    split = dict(zip(sel.test_no, sel.split, strict=True))
    clips = sorted(c for c, info in index.items() if split[info["test_no"]] in v.splits)
    if v.limit:
        clips = clips[: v.limit]
    crop = window_crops(clips, cfg.m3.window_jitter_ms, cfg.m3.seed)

    out = Path(cfg.paths.root) / "vlm" / f"{v.run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["clip_id"] for line in out.read_text().splitlines() if line}
    runner = VLMRunner(v.model, v.load_in_4bit, v.max_side, v.max_new_tokens, v.adapter)

    with zipfile.ZipFile(cfg.annotation.images_zip) as zf, out.open("a") as fh:
        for cid in tqdm([c for c in clips if c not in done], desc=v.run_name):
            info = index[cid]
            start = info["frames"][0]["t_ms"] + crop[cid]
            frames = select_frames(info["frames"], start, v.n_frames, v.step_ms)
            ims = [Image.open(io.BytesIO(zf.read(f["file"]))).convert("RGB") for f in frames]
            rel = [f["t_ms"] - start for f in frames]
            res = runner.generate(build_messages(ims, rel, info["role"]))
            parsed = parse_report(res["text"])
            rec = {
                "clip_id": cid,
                "model": v.model,
                "adapter": v.adapter,
                "start_abs_ms": start,
                "frame_rel_ms": rel,
                **res,
                "parse_ok": parsed.ok,
                "report": parsed.report,
                "issues": parsed.issues,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
