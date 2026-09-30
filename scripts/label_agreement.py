"""Agreement between two sets of directly written labels (e.g. two model annotators).

    uv run python scripts/label_agreement.py annotations/labels_v1.json \
        annotations/check_crossannotator.json --name crossannotator

Writes reports/annotation/<name>.md and <name>_disagreements.csv (clips to adjudicate: presence
disagreements and timing differences above --tol ms). Only clips labelled in both files count.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.iaa import event_agreement, summary_agreement
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tol", type=float, default=16.0)
    ap.add_argument("--data", default=str(Path.home() / "crashvlm_data"))
    args = ap.parse_args()

    onto = Ontology.load("configs/ontology.yaml")
    stride = onto.annotation["frame_stride"]
    metas = {m["clip_id"]: m for m in load_clip_metas(Path(args.data) / "processed" / "clips")}
    sel = pd.read_parquet(Path(args.data) / "interim" / "selection.parquet")
    cfg_keys = dict(zip(sel["test_no"], sel["config_key"], strict=True))
    la, lb = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    common = sorted(set(la) & set(lb))

    def recs(labels):
        rs = records_from_labels(
            {c: labels[c] for c in common}, metas, onto, "x", "x", cfg_keys, stride
        )
        return {r["clip_id"]: r for r in rs}

    ra, rb = recs(la), recs(lb)
    events = pd.DataFrame([event_agreement(ra, rb, e) for e in onto.point_events])
    summ = pd.DataFrame([summary_agreement(ra, rb, k) for k in onto.clip_summary if k != "notes"])

    rows = []
    for c in common:
        for e in onto.point_events:
            ea, eb = ra[c]["events"].get(e), rb[c]["events"].get(e)
            if (ea is None) != (eb is None):
                rows.append(
                    {
                        "clip_id": c,
                        "item": e,
                        "a": ea and ea["t_ms"],
                        "b": eb and eb["t_ms"],
                        "kind": "presence",
                    }
                )
            elif ea and eb and abs(ea["t_ms"] - eb["t_ms"]) > args.tol:
                rows.append(
                    {"clip_id": c, "item": e, "a": ea["t_ms"], "b": eb["t_ms"], "kind": "timing"}
                )
        for k in onto.clip_summary:
            if k != "notes" and ra[c]["summary"].get(k) != rb[c]["summary"].get(k):
                rows.append(
                    {
                        "clip_id": c,
                        "item": k,
                        "a": ra[c]["summary"].get(k),
                        "b": rb[c]["summary"].get(k),
                        "kind": "summary",
                    }
                )
    dis = pd.DataFrame(rows, columns=["clip_id", "item", "a", "b", "kind"])

    out = Path("reports/annotation")
    out.mkdir(parents=True, exist_ok=True)
    dis.to_csv(out / f"{args.name}_disagreements.csv", index=False)
    md = [
        f"# Label agreement: {args.name}",
        f"\nA = `{args.a}`, B = `{args.b}`; clips labelled by both: {len(common)}. "
        f"Timing tolerance for adjudication: {args.tol:.0f} ms.\n",
        "## Point events\n",
        events.round(3).to_string(index=False),
        "\n## Clip summary\n",
        summ.round(3).to_string(index=False),
        f"\n## Items to adjudicate: {len(dis)}\n",
        dis["kind"].value_counts().to_string() if len(dis) else "none",
    ]
    (out / f"{args.name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
