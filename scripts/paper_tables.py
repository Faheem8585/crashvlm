"""Generate the paper's LaTeX tables from the result JSON files (no hand-typed numbers).

    uv run python scripts/paper_tables.py

Reads reports/m3/results.json, reports/m6/learning_curve.csv and (if present)
reports/m4/results.json; writes paper/tables/*.tex. Missing results become \\pending cells.
"""

import json
from pathlib import Path

import pandas as pd

OUT = Path("paper/tables")
SIG_PROBE = "siglip2-base-patch16-224 linear probe"
EV = {
    "impact_onset": "Impact",
    "airbag_deploy_start": "Airbag",
    "head_contact": "Head contact",
    "windshield_crack_onset": "Windshield crack",
    "side_glass_break_onset": "Side glass",
}
METHODS = [
    ("blind (no video)", "Blind (no video)"),
    ("resnet50 linear probe", "ResNet-50 + probe"),
    ("siglip2-base-patch16-224 zero-shot", "SigLIP 2-B zero-shot"),
    ("siglip2-base-patch16-224 linear probe", "SigLIP 2-B + probe"),
    ("siglip2-so400m-patch14-384 zero-shot", "SigLIP 2-So400m zero-shot"),
    ("siglip2-so400m-patch14-384 linear probe", "SigLIP 2-So400m + probe"),
]


def cell(m: dict | None, key: str) -> str:
    if m is None or key not in m or m[key] != m[key]:  # missing or NaN
        return r"\pending"
    return f"{m[key]:.2f}"


def m3_table(res: dict, setting: str, events: list[str]) -> str:
    cols = "l" + "c" * len(events)
    lines = [
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "Method & " + " & ".join(EV[e] for e in events) + r" \\",
        r"\midrule",
    ]
    for key, name in METHODS:
        per_ev = res.get(f"{setting} | {key}")
        vals = [
            (
                cell(per_ev.get(e) if per_ev else None, "hit@16")
                if not per_ev or e in per_ev
                else "--"
            )
            for e in events
        ]
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path("reports/m3/results.json").read_text())
    known = [
        "airbag_deploy_start",
        "head_contact",
        "windshield_crack_onset",
        "side_glass_break_onset",
    ]
    (OUT / "m3_t0_known.tex").write_text(m3_table(res, "t0 known", known))
    (OUT / "m3_t0_unknown.tex").write_text(m3_table(res, "t0 unknown", ["impact_onset", *known]))

    lc = pd.read_csv("reports/m6/learning_curve.csv")
    agg = lc.groupby(["event", "fraction"])["hit@16"].agg(["mean", "std"]).reset_index()
    fr = sorted(agg.fraction.unique())
    lines = [
        rf"\begin{{tabular}}{{l{'c' * len(fr)}}}",
        r"\toprule",
        "Event & " + " & ".join(f"{f:.0%}".replace("%", r"\%") for f in fr) + r" \\",
        r"\midrule",
    ]
    for ev, g in agg.groupby("event"):
        vals = []
        for f in fr:
            r = g[g.fraction == f].iloc[0]
            vals.append(
                f"{r['mean']:.2f}" + (f" $\\pm$ {r['std']:.2f}" if r["std"] == r["std"] else "")
            )
        lines.append(f"{EV[ev]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "m6_learning_curve.tex").write_text("\n".join(lines) + "\n")

    m4 = Path("reports/m4/results.json")
    rows = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Parse & Test type & Fire halluc. & Contradicting claims & Latency (s) \\",
        r"\midrule",
    ]
    runs = json.loads(m4.read_text()) if m4.exists() else {}
    for run in ["qwen3.5-4b_zs", "qwen3-vl-8b-4bit_zs", "qwen3.5-4b_lora"]:
        s = runs.get(run, {}).get("summary")
        keys = [
            "parse_rate",
            "test_type_acc (NHTSA)",
            "fire_hallucination_rate",
            "claims_contradicting_labels",
            "latency_s_median",
        ]
        rows.append(run.replace("_", r"\_") + " & " + " & ".join(cell(s, k) for k in keys) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "m4_reports.tex").write_text("\n".join(rows) + "\n")

    paired = Path("reports/m4/paired.json")
    if paired.exists():
        (OUT / "m4_paired.tex").write_text(paired_table(json.loads(paired.read_text())))
    print(f"tables -> {OUT}")


PAIRED_METHODS = [
    ("resnet50 linear probe", "ResNet-50 probe"),
    ("siglip2-so400m-patch14-384 linear probe", "SigLIP 2-So400m probe"),
    ("qwen3.5-4b_zs", "Qwen3.5-4B"),
    ("qwen3-vl-8b-4bit_zs", "Qwen3-VL-8B"),
    ("qwen3.5-4b_lora", "Qwen3.5-4B LoRA"),
]


def paired_table(rows: list[dict]) -> str:
    """hit@8 and hit@16 of the blind baseline and every video method on the same clips;
    arrows mark paired differences whose bootstrap interval excludes zero."""
    idx = {(r["event"], r["tol_ms"], r["method"]): r for r in rows}
    lines = [
        r"\begin{tabular}{ll" + "c" * (1 + len(PAIRED_METHODS)) + "}",
        r"\toprule",
        "Event & $k$ & Blind & " + " & ".join(n for _, n in PAIRED_METHODS) + r" \\",
        r"\midrule",
    ]
    for ev in ["impact_onset", "airbag_deploy_start", "head_contact"]:
        for tol in (8, 16):
            blind = next((r for (e, t, _), r in idx.items() if e == ev and t == tol), None)
            if blind is None:
                continue
            vals = []
            for key, _ in PAIRED_METHODS:
                r = idx.get((ev, tol, key))
                if r is None:
                    vals.append(r"\pending")
                    continue
                mark = ""
                if r["CI excludes 0"]:
                    mark = r"$^{\uparrow}$" if r["diff"] > 0 else r"$^{\downarrow}$"
                vals.append(f"{r['method_hit']:.2f}{mark}")
            lines.append(
                f"{EV[ev]} & {tol} & {blind['blind_hit']:.2f} & " + " & ".join(vals) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def number_macros() -> None:
    """paper/numbers.tex: every number quoted in the paper text, as LaTeX macros."""
    import os

    data = Path(os.environ.get("CRASHVLM_DATA", Path.home() / "crashvlm_data"))
    cat = pd.read_parquet(data / "interim" / "catalog.parquet")
    sel = pd.read_parquet(data / "interim" / "selection.parquet")
    man = pd.read_parquet(data / "processed" / "manifest.parquet")
    res = json.loads(Path("reports/m3/results.json").read_text())
    lab = json.loads(Path("annotations/labels_v1.json").read_text())

    def hit(setting, method, ev):
        return res[f"{setting} | {method}"][ev]["hit@16"]

    m = {
        "NCatalog": len(cat),
        "NTests": len(sel),
        "NClips": len(man),
        "NTestTests": int((sel.split == "test").sum()),
        "NVerified": int(man.time_verified.sum()),
        "NLabelled": len(lab),
        "BlindAirbagKnown": f"{hit('t0 known', 'blind (no video)', 'airbag_deploy_start'):.2f}",
        "BlindHeadKnown": f"{hit('t0 known', 'blind (no video)', 'head_contact'):.2f}",
        "BlindAirbagUnknown": f"{hit('t0 unknown', 'blind (no video)', 'airbag_deploy_start'):.2f}",
        "ProbeAirbagUnknown": f"{hit('t0 unknown', SIG_PROBE, 'airbag_deploy_start'):.2f}",
        "BlindImpactUnknown": f"{hit('t0 unknown', 'blind (no video)', 'impact_onset'):.2f}",
        "ResnetImpactUnknown": f"{hit('t0 unknown', 'resnet50 linear probe', 'impact_onset'):.2f}",
    }
    big = "siglip2-so400m-patch14-384 linear probe"
    if f"t0 unknown | {big}" in res:
        m["LargeProbeAirbagUnknown"] = f"{hit('t0 unknown', big, 'airbag_deploy_start'):.2f}"
    m4 = Path("reports/m4/results.json")
    zs = json.loads(m4.read_text()).get("qwen3.5-4b_zs") if m4.exists() else None
    if zs:
        ev, sm = zs["events"], zs["summary"]
        m["VlmAirbagUnknown"] = f"{ev['airbag_deploy_start']['hit@16']:.2f}"
        m["VlmHeadUnknown"] = f"{ev['head_contact']['hit@16']:.2f}"
        m["VlmImpactUnknown"] = f"{ev['impact_onset']['hit@16']:.2f}"
        m["VlmParse"] = f"{sm['parse_rate']:.2f}"
        m["VlmTestType"] = f"{sm['test_type_acc (NHTSA)']:.2f}"
        m["VlmContradict"] = f"{sm['claims_contradicting_labels']:.2f}"
        m["VlmLatency"] = f"{sm['latency_s_median']:.0f}"
    pj = Path("reports/m4/paired.json")
    if pj.exists():
        pr = {(r["event"], r["tol_ms"], r["method"]): r for r in json.loads(pj.read_text())}
        for tag, ev in (
            ("Impact", "impact_onset"),
            ("Airbag", "airbag_deploy_start"),
            ("Head", "head_contact"),
        ):
            for tol, word in ((8, "Eight"), (16, "Sixteen")):
                any_row = next((r for (e, t, _), r in pr.items() if e == ev and t == tol), None)
                if any_row:
                    m[f"Blind{tag}{word}"] = f"{any_row['blind_hit']:.2f}"
                for run, name in (
                    ("qwen3.5-4b_zs", "Zs"),
                    ("qwen3-vl-8b-4bit_zs", "Vl"),
                    ("qwen3.5-4b_lora", "Lora"),
                ):
                    r = pr.get((ev, tol, run))
                    if r:
                        m[f"{name}{tag}{word}"] = f"{r['method_hit']:.2f}"
                        m[f"{name}{tag}{word}Diff"] = f"{r['diff']:+.2f}"
    tl = data / "lora" / "qwen3.5-4b_lora" / "train_log.json"
    if tl.exists():
        log = json.loads(tl.read_text())
        m["LoraSteps"] = len(log["steps"])
        c = log["config"]["sft"]
        n_train = sum(1 for _ in (data / "sft" / c["train_file"]).open())
        m["LoraPlannedSteps"] = -(-n_train // c["grad_accum"]) * c["epochs"]
        m["LoraTrainHours"] = f"{log['train_minutes'] / 60:.1f}"
    if m4.exists():
        for run, name in (("qwen3-vl-8b-4bit_zs", "Vl"), ("qwen3.5-4b_lora", "Lora")):
            sm = json.loads(m4.read_text()).get(run, {}).get("summary")
            if sm:
                m[f"{name}TestType"] = f"{sm['test_type_acc (NHTSA)']:.2f}"
                m[f"{name}Contradict"] = f"{sm['claims_contradicting_labels']:.2f}"
                m[f"{name}Fire"] = f"{sm['fire_hallucination_rate']:.2f}"
    kin = pd.read_parquet(data / "physics" / "kinematics.parquet")
    m["NPhysics"] = int(kin.t_max_crush_ms.notna().sum())
    fl = {
        s: json.loads(Path(f"reports/physics/flow_{s}.json").read_text())
        for s in ("train", "val", "test")
        if Path(f"reports/physics/flow_{s}.json").exists()
    }
    if {"val", "test"} <= set(fl):
        n_ho = fl["val"]["n"] + fl["test"]["n"]
        hit8 = sum(fl[s]["flow"]["within_8"] * fl[s]["n"] for s in ("val", "test"))
        m["NFlowHeldOut"] = n_ho
        m["FlowHeldOutWithinEight"] = int(round(hit8))
        m["FlowTestMedErr"] = f"{fl['test']['flow']['median_abs_err_ms']:.1f}"
        m["FlowTrainRho"] = f"{fl['train']['flow']['spearman']:.2f}"
        m["BlindTrainRho"] = f"{fl['train']['blind']['spearman']:.2f}"
        m["FlowBias"] = f"{fl['train']['bias_ms']:+.1f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    Path("paper/numbers.tex").write_text(
        "% generated by scripts/paper_tables.py\n" + "\n".join(lines) + "\n"
    )


def readme_results() -> None:
    """Fill the README results block from reports/m3 and reports/m4 (markdown)."""
    res = json.loads(Path("reports/m3/results.json").read_text())
    events = [
        "impact_onset",
        "airbag_deploy_start",
        "head_contact",
        "windshield_crack_onset",
        "side_glass_break_onset",
    ]
    lines = [
        "Hit@16 (share of test events timed within 16 ms; misses count) on the 23-test split.",
        "**t0 unknown**: clips cropped at a random start, as for any real video.\n",
        "| Method | " + " | ".join(EV[e] for e in events) + " |",
        "|---|" + "---|" * len(events),
    ]
    for key, name in METHODS:
        per = res.get(f"t0 unknown | {key}")
        if per is None:
            vals = ["pending (GPU)"] * len(events)
        else:
            vals = [f"{per[e]['hit@16']:.2f}" if e in per else "–" for e in events]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    blind_known = res["t0 known | blind (no video)"]
    lines.append(
        "\nWith the lab clock (**t0 known**), the blind no-video baseline scores "
        f"{blind_known['airbag_deploy_start']['hit@16']:.2f} (airbag) and "
        f"{blind_known['head_contact']['hit@16']:.2f} (head contact).\n"
    )
    paired = Path("reports/m4/paired.json")
    if paired.exists():
        idx = {(r["event"], r["tol_ms"], r["method"]): r for r in json.loads(paired.read_text())}
        lines += [
            "**Video VLMs vs. the blind baseline** (same clips, t0 unknown; ↑/↓ = paired "
            "bootstrap interval excludes zero):\n",
            "| Event | k (ms) | Blind | " + " | ".join(n for _, n in PAIRED_METHODS) + " |",
            "|---|---|---|" + "---|" * len(PAIRED_METHODS),
        ]
        for ev in ["impact_onset", "airbag_deploy_start", "head_contact"]:
            for tol in (8, 16):
                rs = [idx.get((ev, tol, k)) for k, _ in PAIRED_METHODS]
                base = next((r for r in rs if r), None)
                if base is None:
                    continue
                cells = []
                for r in rs:
                    if r is None:
                        cells.append("–")
                        continue
                    arrow = ("↑" if r["diff"] > 0 else "↓") if r["CI excludes 0"] else ""
                    cells.append(f"{r['method_hit']:.2f}{arrow}")
                lines.append(
                    f"| {EV[ev]} | {tol} | {base['blind_hit']:.2f} | " + " | ".join(cells) + " |"
                )
        lines.append(
            "\nReport quality, hallucination and latency: `reports/m4/results.md`; "
            "intervals: `reports/m4/paired.md`."
        )
    block = "\n".join(lines)
    readme = Path("README.md").read_text()
    a, b = "<!-- results:start -->", "<!-- results:end -->"
    start, end = readme.index(a) + len(a), readme.index(b)
    Path("README.md").write_text(readme[:start] + "\n" + block + "\n" + readme[end:])


if __name__ == "__main__":
    main()
    number_macros()
    readme_results()
