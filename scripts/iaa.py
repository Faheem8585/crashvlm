"""M2: agreement between pass A (pre-labelled) and pass B (repeat, unassisted) on IAA clips.

    uv run python scripts/iaa.py

Reads <data>/annotation/pass_a.jsonl and pass_b.jsonl; writes reports/m2/iaa.md and iaa.json.
"""

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from crashvlm.annotation.iaa import event_agreement, summary_agreement
from crashvlm.annotation.ontology import Ontology


def load(path: Path) -> dict[str, dict]:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["clip_id"]: r for r in recs if r["complete"]}


@hydra.main(config_path="../configs", config_name="annotation", version_base="1.3")
def main(cfg: DictConfig) -> None:
    onto = Ontology.load(cfg.annotation.ontology)
    root = Path(cfg.annotation.out)
    a, b = load(root / "pass_a.jsonl"), load(root / "pass_b.jsonl")
    common = set(a) & set(b)
    a = {k: v for k, v in a.items() if k in common}
    b = {k: v for k, v in b.items() if k in common}
    events = pd.DataFrame([event_agreement(a, b, e) for e in onto.point_events])
    summ = pd.DataFrame([summary_agreement(a, b, k) for k in onto.clip_summary if k != "notes"])
    out = Path(cfg.paths.reports) / "m2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "iaa.json").write_text(
        json.dumps(
            {"events": events.to_dict("records"), "summary": summ.to_dict("records")}, indent=1
        )
    )
    md = [
        "# Intra-rater agreement: pass A (pre-labelled) vs pass B (unassisted), "
        f"{len(common)} clips\n",
        "## Point events\n",
        events.round(3).to_string(index=False),
        "\n## Clip summary\n",
        summ.round(3).to_string(index=False),
    ]
    (out / "iaa.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
