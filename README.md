# CrashVLM

Vision-language evaluation of vehicle crash test videos (NHTSA NCAP), with a planned link to
simulation geometry.

> **Status: M1 (data pipeline).** No models have been trained or evaluated yet; this README
> will carry results only once they are produced by the evaluation scripts.

## Research question

Can open vision-language models, adapted with a small annotated dataset, reliably detect and
describe crash-relevant events and structural deformation in crash test videos, and does
adding geometric / simulation information improve the assessment?

## Quick start (M1)

On a Mac with iCloud "Desktop & Documents" enabled, keep the data and the virtual environment
out of the synced folders. iCloud otherwise throttles the pipeline's I/O and hides the
environment's `.pth` files, which breaks imports:

```bash
export CRASHVLM_DATA=~/crashvlm_data UV_PROJECT_ENVIRONMENT=~/.venvs/crashvlm
```

```bash
uv sync
uv run python scripts/build_catalog.py      # crawl NHTSA API -> data/interim/catalog.parquet
uv run python scripts/select_tests.py       # seeded selection + vehicle-grouped split
uv run python scripts/prepare_clips.py clips.limit=4   # smoke run: 4 clips
uv run python scripts/overlay_report.py     # QA sheet: reports/m1/overlay_check.jpg
```

Full run on Colab or Kaggle: [docs/COLAB.md](docs/COLAB.md).

- [docs/DATA_CARD.md](docs/DATA_CARD.md): sources, licence status, processing, known biases
- [docs/DESIGN.md](docs/DESIGN.md): pipeline diagram, decisions, leakage controls
- [docs/generated/data_stats.md](docs/generated/data_stats.md): dataset numbers (script-generated)
