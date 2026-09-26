"""M2: convert a CVAT export into annotation records and report pre-label acceptance.

    uv run python scripts/convert_cvat.py annotation.cvat_export=~/Downloads/annotations.xml \
        annotation.pass_name=pass_a annotation.annotator=A

Writes <data>/annotation/<pass>.jsonl and reports/m2/<pass>_summary.md.
"""

import json
from collections import Counter
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from crashvlm.annotation.convert import convert
from crashvlm.annotation.cvat_io import parse_frame_name, read_cvat_xml
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas


def prelabel_outcomes(records: list[dict], prelabel_xml: Path) -> pd.DataFrame:
    """Per proposed pre-label: kept as proposed, moved (re-drawn elsewhere), or rejected."""
    proposed = {}
    for im in read_cvat_xml(prelabel_xml):
        clip, idx = parse_frame_name(im.name)
        for t in im.tags:
            proposed[(clip, t.label)] = idx
    by_clip = {r["clip_id"]: r for r in records if r["complete"]}
    rows = []
    for (clip, label), idx in proposed.items():
        if clip not in by_clip:
            continue
        ev = by_clip[clip]["events"].get(label)
        if ev is None:
            outcome, shift = "rejected", None
        elif ev["source"] == "auto" and ev["idx"] == idx:
            outcome, shift = "kept", 0
        else:
            outcome, shift = "moved", ev["idx"] - idx
        rows.append({"clip_id": clip, "label": label, "outcome": outcome, "shift_frames": shift})
    return pd.DataFrame(rows)


@hydra.main(config_path="../configs", config_name="annotation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    a = cfg.annotation
    if not a.cvat_export:
        raise SystemExit("set annotation.cvat_export=<path to annotations.xml>")
    onto = Ontology.load(a.ontology)
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(cfg.paths.processed) / "clips")}
    sel = pd.read_parquet(Path(cfg.paths.interim) / "selection.parquet")
    records = convert(
        read_cvat_xml(Path(a.cvat_export).expanduser()),
        metas,
        onto,
        a.pass_name,
        a.annotator,
        dict(zip(sel["test_no"], sel["config_key"], strict=True)),
    )
    out = Path(a.out) / f"{a.pass_name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))

    done = [r for r in records if r["complete"]]
    lines = [
        f"# {a.pass_name} (annotator {a.annotator})\n",
        f"Clips in export: {len(records)}; complete (have clip_summary): {len(done)}\n",
    ]
    counts = Counter(e for r in done for e in r["events"])
    lines.append("## Events marked\n")
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items())]
    warnings = [(r["clip_id"], w) for r in records for w in r["warnings"]]
    lines.append(f"\n## Warnings ({len(warnings)})\n")
    lines += [f"- {c}: {w}" for c, w in warnings[:200]]

    pre = Path(a.out) / a.pass_name / "prelabels.xml"
    if pre.exists():
        oc = prelabel_outcomes(records, pre)
        if len(oc):
            tab = pd.crosstab(oc["label"], oc["outcome"], margins=True)
            lines.append("\n## Pre-label outcomes (complete clips)\n")
            lines.append(tab.to_string())
            moved = oc.loc[oc["outcome"] == "moved", "shift_frames"].abs()
            if len(moved):
                lines.append(f"\nMedian |shift| of moved pre-labels: {moved.median():.0f} frames")
    rep = Path(cfg.paths.reports) / "m2" / f"{a.pass_name}_summary.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines) + "\n")
    print(f"{len(done)}/{len(records)} complete -> {out}; report {rep}")


if __name__ == "__main__":
    main()
