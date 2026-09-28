# CrashVLM

**Can open vision-language models read a crash test video, and how much of what they "see"
is just the clock?**

CrashVLM is a research prototype for evaluating vehicle crash test videos with vision-language
models. It is built on the public NHTSA NCAP database: 100 frontal, side-barrier and side-pole
tests, with two high-speed camera views each. It contains:
- an open data pipeline, including overlay removal with an automated leak check and OCR time
  recovery;
- a frame-level event ontology and 200 annotated views, with documented label-quality checks;
- a leakage-aware evaluation protocol;
- baselines from a blind no-video predictor through SigLIP 2 probes to Qwen VLMs that write
  structured JSON crash reports;
- a FastAPI service, a Gradio demo and a workshop-paper draft.

> **Status.** Data, annotation, CPU baselines, API/demo and paper draft are done. The GPU runs
> (SigLIP 2 so400m, Qwen zero-shot, LoRA) are fully scripted and smoke-tested, and waiting for a
> Colab session ([docs/COLAB.md](docs/COLAB.md)). Results below are generated from `reports/`
> by `scripts/paper_tables.py`; nothing is typed in by hand.

```mermaid
flowchart LR
  A[NHTSA API<br/>1,583 NCAP tests] --> B[Selection<br/>100 tests, vehicle-grouped split]
  B --> C[Download + overlay removal<br/>crop panels, mask text and T0 lamp<br/>OCR time axis]
  C --> D[(200 clips<br/>8 ms frames)]
  D --> E[Annotation<br/>ontology, contact sheets,<br/>blind cross-model QA]
  D --> F[Frozen features<br/>SigLIP 2 / ResNet-50]
  F --> G[Blind baseline, zero-shot,<br/>linear probes]
  D --> H[Qwen VLM<br/>zero-shot / LoRA<br/>JSON crash report]
  E --> G
  E --> H
  G --> I[Evaluation<br/>t0 known vs t0 unknown<br/>hit@k, F1, bootstrap CIs]
  H --> I
  I --> J[Paper + API + demo]
```

> **Demo GIF: placeholder.** To be recorded from the Gradio app (`python -m crashvlm.serve.demo`).
> NHTSA frames are not redistributed in this repository.

## Results

<!-- results:start -->
Hit@16 (share of test events timed within 16 ms; misses count) on the 23-test split.
**t0 unknown**: clips cropped at a random start, as for any real video.

| Method | Impact | Airbag | Head contact | Windshield crack | Side glass |
|---|---|---|---|---|---|
| Blind (no video) | 0.45 | 0.35 | 0.33 | 0.50 | 0.09 |
| ResNet-50 + probe | 0.75 | 0.45 | 0.44 | 0.50 | 0.27 |
| SigLIP 2-B zero-shot | – | 0.45 | 0.33 | 0.25 | 0.00 |
| SigLIP 2-B + probe | 0.55 | 0.60 | 0.50 | 0.62 | 0.27 |
| SigLIP 2-So400m zero-shot | pending (GPU) | pending (GPU) | pending (GPU) | pending (GPU) | pending (GPU) |
| SigLIP 2-So400m + probe | pending (GPU) | pending (GPU) | pending (GPU) | pending (GPU) | pending (GPU) |

With the lab clock (**t0 known**), the blind no-video baseline scores 1.00 (airbag) and 0.94 (head contact).

VLM crash reports (Qwen3.5-4B / Qwen3-VL-8B, zero-shot and LoRA): pending (GPU).
<!-- results:end -->

## Findings (honest version)

1. **The clock gives the answer away.** NCAP tests are so standardised that a predictor which
   never looks at the video times airbag deployment within 16 ms in almost every test case, once
   the lab's impact time is known. Timing benchmarks on standardised crash tests are
   meaningless without this baseline.
2. **Without the clock, video features beat the blind guess.** On randomly cropped clips (the
   realistic case), linear probes on frozen SigLIP 2 or ResNet-50 features do better for every
   event, most clearly for finding the impact. With 23 test tests, most intervals still overlap.
3. **Zero-shot prompting detects but cannot time.** Text prompts find airbags but place them
   poorly and fail on glass breakage. Probes on the same features are consistently better.
4. **More labels barely move frozen-feature probes.** The bottleneck is the frozen features
   and the decoder, which is why LoRA adaptation is the next step.
5. **Video can be checked against physics.** The car's own accelerometers give the measured
   moment of maximum crush for 94 tests. Visual labels and appearance-based models could not
   find it. Optical flow on the car body does: within 8 ms in all 12 held-out frontal tests,
   without using the lab clock ([reports/physics](reports/physics)).
6. **Data work beat model work.** Burned-in timestamps and a red→green impact lamp would have
   leaked the labels. Two OCR time-parsing bugs shifted 8 clips by about 50 ms. Same-model label
   agreement overstated reliability. Each was caught by an explicit check.

Details: [docs/FINDINGS.md](docs/FINDINGS.md) · label quality:
[docs/ANNOTATION_QA.md](docs/ANNOTATION_QA.md) · paper draft:
[paper/crashvlm_draft.pdf](paper/crashvlm_draft.pdf)

## Important limitation: who labelled the data

All 200 views were annotated by a language model (Claude), not by a human, from time-stamped
contact sheets under a written guideline. Quality was checked with blind re-annotation by
other model instances, and labels are used in reliability tiers. A human-verified subset is
the most important missing piece. See [docs/ANNOTATION_QA.md](docs/ANNOTATION_QA.md).

## Quick start

On a Mac with iCloud "Desktop & Documents" enabled, keep data and the virtual environment out of
the synced folders (iCloud otherwise throttles I/O and hides the environment's `.pth` files):

```bash
export CRASHVLM_DATA=~/crashvlm_data UV_PROJECT_ENVIRONMENT=~/.venvs/crashvlm
```

```bash
uv sync                                          # core (data pipeline, evaluation)
uv run python scripts/build_catalog.py           # NHTSA API -> catalog
uv run python scripts/select_tests.py            # frozen selection + split
uv run python scripts/prepare_clips.py           # download, overlay removal, frames
uv run python scripts/overlay_report.py          # QA sheets + T0-lamp leak check
uv sync --extra ml
uv run python scripts/extract_embeddings.py embed.backbone=siglip:google/siglip2-base-patch16-224
uv run python scripts/run_m3.py 'm3.backbones=[google__siglip2-base-patch16-224,resnet50]'
```

API and demo (CPU, probe backend):

```bash
uv sync --extra ml --extra serve
uv run python scripts/export_probes.py
uv run uvicorn crashvlm.serve.api:app --port 8000   # POST /evaluate_video
uv run python -m crashvlm.serve.demo                # Gradio UI
```

GPU steps (SigLIP 2 so400m, Qwen zero-shot, LoRA): [docs/COLAB.md](docs/COLAB.md).

## Repository

| Path | Content |
|---|---|
| `src/crashvlm/data` | NHTSA client, view selection, overlay removal, OCR time axis, clip extraction |
| `src/crashvlm/annotation` | ontology, CVAT I/O, pre-labels, direct labels, agreement metrics |
| `src/crashvlm/models` | SigLIP scorer, embeddings, probes, VLM runner and report parser |
| `src/crashvlm/train` | LoRA instruction data |
| `src/crashvlm/eval` | event metrics, bootstrap CIs, blind baseline, shared probe selection |
| `src/crashvlm/serve` | inference pipeline, FastAPI, Gradio |
| `configs/` | Hydra configs per milestone; `configs/splits/selection_v1.csv` is the frozen split |
| `annotations/` | labels and QA passes (model-produced) |
| `docs/` | data card, design decisions, annotation guideline and QA, findings, Colab cells |
| `paper/` | LaTeX draft, generated tables, verified bibliography |

## Data and licence

NHTSA footage is publicly accessible but its licence is listed as "unknown" on data.gov, and it
is produced by contract laboratories. This repository therefore contains no video or frames,
only code, test IDs, annotations and derived metrics. See [docs/DATA_CARD.md](docs/DATA_CARD.md).
