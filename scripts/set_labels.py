"""Add or replace clip labels in an annotation JSON file (used for model-assisted labels).

    uv run python scripts/set_labels.py annotations/labels_v1.json '{"<clip_id>": {...}}'

The label format is documented in crashvlm.annotation.direct. Entries are validated against
the ontology before they are written.
"""

import json
import sys
from pathlib import Path

from crashvlm.annotation.ontology import Ontology

onto = Ontology.load("configs/ontology.yaml")
path, new = Path(sys.argv[1]), json.loads(sys.argv[2])
labels = json.loads(path.read_text()) if path.exists() else {}
for cid, lab in new.items():
    for ev in lab.get("events", {}):
        assert ev in onto.point_events, f"{cid}: unknown event {ev}"
    for k, v in lab.get("summary", {}).items():
        spec = onto.clip_summary[k]
        if spec["type"] == "select":
            assert v in spec["values"], f"{cid}: {k}={v!r} not in {spec['values']}"
    labels[cid] = lab
path.write_text(json.dumps(dict(sorted(labels.items())), indent=1) + "\n")
print(f"{len(labels)} clips labelled in {path}")
