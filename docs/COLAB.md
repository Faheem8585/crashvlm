# Running on Colab / Kaggle

M1 needs no GPU. It runs locally (~10 s per clip on an M-series Mac, ~6 MB disk per clip) or on
Colab when local disk is short. Frames go to Google Drive so later GPU milestones can reuse them.

Replace `<github-user>` once the repo is pushed.

```python
# 1. environment
!pip -q install uv
!git clone https://github.com/<github-user>/crashvlm.git
%cd crashvlm
!uv sync --locked
```

```python
# 2. keep data on Drive
import os

from google.colab import drive

drive.mount("/content/drive")
os.environ["CRASHVLM_DATA"] = "/content/drive/MyDrive/crashvlm/data"
```

```python
# 3. M1 pipeline (the selection is read from configs/splits/selection_v1.csv)
!uv run python scripts/build_catalog.py
!uv run python scripts/select_tests.py
!uv run python scripts/prepare_clips.py          # resumable: re-run after a disconnect
!uv run python scripts/data_stats.py
!uv run python scripts/overlay_report.py
```

Expected output of `prepare_clips.py`: `clips ready: 200 / 200 requested, errors: 0`, plus
per-type time-source counts. Any failures are listed in `$CRASHVLM_DATA/processed/errors.json`.
Before M2, open `reports/m1/overlay_check.jpg` and check that no processed frame shows a
timestamp, frame counter or T0 dot.

On Kaggle, drop step 2 and set `CRASHVLM_DATA=/kaggle/working/data`, then save the output as
a Kaggle dataset.

## M2: zero-shot pre-labels (GPU, ~5 min on a T4)

Upload `data/annotation/pass_a/images.zip` and `data/annotation/clip_index.json` from your Mac
to `MyDrive/crashvlm/data/annotation/` (same relative layout). Then, in a GPU runtime:

```python
!git clone https://github.com/<github-user>/crashvlm.git
%cd crashvlm
!pip -q install -e ".[ml]"   # reuses Colab's preinstalled torch
```

```python
import os

from google.colab import drive

drive.mount("/content/drive")
os.environ["CRASHVLM_DATA"] = "/content/drive/MyDrive/crashvlm/data"
```

```python
!python scripts/zero_shot_prelabels.py
```

Expected output: `200 clips -> .../annotation/zero_shot_preds.jsonl`. Download
`zero_shot_preds.jsonl` (and `zero_shot_curves.npz`, which M3 reuses) into `data/annotation/`
on the Mac, then rebuild the pass A pre-labels:

```bash
uv run python scripts/make_cvat_tasks.py annotation.zero_shot_preds=data/annotation/zero_shot_preds.jsonl
```

## M3: SigLIP 2 so400m embeddings (GPU, ~5 min on a T4)

The large model (4.5 GB) does not fit on the Mac's free disk, so only this step runs on Colab.
Everything else in M3 runs locally on CPU from the resulting `.npz`.

Upload `~/crashvlm_data/annotation/pass_a/images.zip` (469 MB) and
`~/crashvlm_data/annotation/clip_index.json` to `MyDrive/crashvlm/data/annotation/` (keep the
`pass_a/` subfolder). Then, in a **GPU runtime**:

```python
!git clone https://github.com/<github-user>/crashvlm.git
%cd crashvlm
!pip -q install -e ".[ml]"
```

```python
import os

from google.colab import drive

drive.mount("/content/drive")
os.environ["CRASHVLM_DATA"] = "/content/drive/MyDrive/crashvlm/data"
```

```python
!python scripts/extract_embeddings.py
```

Expected output: `200 clips -> /content/drive/MyDrive/crashvlm/data/features/google__siglip2-so400m-patch14-384.npz`
(about 20 MB). Download it into `~/crashvlm_data/features/` on the Mac, then:

```bash
uv run python scripts/run_m3.py
```

## M4: zero-shot VLM crash reports (GPU; T4 16 GB or L4 24 GB)

Same Drive layout as M3 (`images.zip` + `clip_index.json`). The selection is read from the repo.
bitsandbytes is only needed for the 4-bit run.

```python
!pip -q install -e ".[ml]" bitsandbytes accelerate
```

```python
# Qwen3.5-4B in bf16 (~9 GB VRAM): 46 test clips
!python scripts/run_vlm.py vlm.model=Qwen/Qwen3.5-4B vlm.run_name=qwen3.5-4b_zs
```

```python
# Qwen3-VL-8B in 4-bit NF4 (~7 GB VRAM)
!python scripts/run_vlm.py vlm.model=Qwen/Qwen3-VL-8B-Instruct vlm.load_in_4bit=true vlm.run_name=qwen3-vl-8b-4bit_zs
```

Each run appends to `$CRASHVLM_DATA/vlm/<run_name>.jsonl`, so it can simply be re-run after a
disconnect. Copy the two JSONL files into `~/crashvlm_data/vlm/` on the Mac, then:

```bash
uv run python scripts/eval_vlm.py
```

A T4 has no native bf16. If Qwen3.5-4B fails to load or is very slow there, use an L4 runtime,
or Kaggle's P100/T4.

## M5: LoRA fine-tuning (GPU; L4 24 GB recommended)

On the Mac, build the instruction data first and upload `~/crashvlm_data/sft/` to
`MyDrive/crashvlm/data/sft/`:

```bash
uv run python scripts/build_sft.py
uv run python scripts/build_sft.py sft.train_fraction=0.25
uv run python scripts/build_sft.py sft.train_fraction=0.5
```

On Colab (same setup cells as M4):

```python
# full training set: 122 clips x 4 crops = 488 examples, 3 epochs
!python scripts/train_lora.py vlm.model=Qwen/Qwen3.5-4B vlm.run_name=qwen3.5-4b_lora
```

```python
# evaluate the adapter on the test split (same crops as M3/M4)
!python scripts/run_vlm.py vlm.model=Qwen/Qwen3.5-4B vlm.adapter=$CRASHVLM_DATA/lora/qwen3.5-4b_lora/adapter vlm.run_name=qwen3.5-4b_lora
```

```python
# learning curve (optional): 25 % and 50 % of the training tests
for f in ["0.25", "0.5"]:
    !python scripts/train_lora.py sft.train_file=train_f{f}.jsonl vlm.run_name=qwen3.5-4b_lora_f{f}
    !python scripts/run_vlm.py vlm.adapter=$CRASHVLM_DATA/lora/qwen3.5-4b_lora_f{f}/adapter vlm.run_name=qwen3.5-4b_lora_f{f}
```

Copy `vlm/*.jsonl` and `lora/*/train_log.json` back to `~/crashvlm_data/`, add the run names
to `m4.runs` and run `uv run python scripts/eval_vlm.py`. To push an adapter to the Hugging
Face Hub: `huggingface-cli login`, then
`huggingface-cli upload <user>/crashvlm-qwen3.5-4b-lora $CRASHVLM_DATA/lora/qwen3.5-4b_lora/adapter`.
