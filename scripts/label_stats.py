"""Summary of an annotation file: events per type/role, timing spread, summary attributes.

    uv run python scripts/label_stats.py annotations/claude_pass_a.json

Writes reports/annotation/label_stats.md.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from crashvlm.annotation.direct import records_from_labels
from crashvlm.annotation.ontology import Ontology
from crashvlm.data.manifest import load_clip_metas

data = Path.home() / "crashvlm_data"
onto = Ontology.load("configs/ontology.yaml")
metas = {m["clip_id"]: m for m in load_clip_metas(data / "processed" / "clips")}
sel = pd.read_parquet(data / "interim" / "selection.parquet")
man = pd.read_parquet(data / "processed" / "manifest.parquet").set_index("clip_id")
labels = json.loads(Path(sys.argv[1]).read_text())
recs = records_from_labels(
    labels,
    metas,
    onto,
    "a",
    "claude",
    dict(zip(sel.test_no, sel.config_key, strict=True)),
    onto.annotation["frame_stride"],
)
rows = []
for r in recs:
    for e, v in r["events"].items():
        rows.append(
            {
                "config": r["config_key"],
                "role": r["role"],
                "event": e,
                "t_ms": v["t_ms"],
                "visible": v["attributes"].get("onset_visible", True),
                "time_verified": bool(man.loc[r["clip_id"], "time_verified"]),
            }
        )
ev = pd.DataFrame(rows)
ok = ev[ev.visible & ev.time_verified & (ev.event != "impact_onset")]
timing = ok.groupby(["event", "config", "role"])["t_ms"].describe()[["count", "25%", "50%", "75%"]]
summ = pd.DataFrame([r["summary"] for r in recs])
md = [
    f"# Label statistics: `{sys.argv[1]}`\n",
    f"Clips: {len(recs)}; complete: {sum(r['complete'] for r in recs)}; "
    f"view_usable: {int(summ.view_usable.sum())}; "
    f"warnings: {sum(len(r['warnings']) for r in recs)}\n",
    "## Events per test type and role (all tags)\n",
    pd.crosstab([ev.event], [ev.config, ev.role]).to_string(),
    "\n## Onset timing, ms (visible onsets on time-verified clips)\n",
    timing.round(1).to_string(),
    "\n## Clip summary attributes\n",
]
for k in [
    "frontal_airbag_deployed",
    "curtain_airbag_deployed",
    "door_opened",
    "fire_or_fluid_leak",
    "visual_severity",
]:
    md.append(f"- {k}: {summ[k].value_counts().to_dict()}")
out = Path("reports/annotation/label_stats.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(md) + "\n")
print("\n".join(md))
