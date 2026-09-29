# Running the remaining GPU steps on Kaggle

Kaggle gives 30 free GPU hours per week and runs a committed notebook in the background for up
to 12 h, which suits LoRA training better than Colab. The notebook is `notebooks/kaggle_gpu.ipynb`.
It runs step D (LoRA training + evaluation) and then, if time is left, step C (Qwen3-VL-8B).
Steps A and B are already done.

## 1. Account (once)

1. Sign in at kaggle.com.
2. Settings → **Phone verification**. Without it, notebooks get no GPU and no internet.

## 2. Upload the data as a private dataset (once, ~10 min)

1. kaggle.com → **Create** → **New Dataset**.
2. Drag in the **7 files** from `~/Downloads/CRASHVLM_UPLOAD/data`:
   - `annotation/pass_a/images.zip`
   - `annotation/clip_index.json`
   - `interim/selection.parquet`
   - `sft/train.jsonl`, `sft/val.jsonl`, `sft/train_f0.25.jsonl`, `sft/train_f0.5.jsonl`

   Dragging the whole `data` folder also works; the folder layout does not matter. If Kaggle
   unpacks `images.zip`, that is fine too: the notebook rebuilds it.
3. Title: `crashvlm-data`. Visibility: **Private** (the default). Do not make it public, because
   NHTSA frames must not be re-hosted.
4. Click **Create** and wait until processing has finished.

## 3. Create the notebook (once)

1. kaggle.com → **Create** → **New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_gpu.ipynb` (a copy is in
   `~/Downloads/CRASHVLM_UPLOAD/`).
3. Right panel, **Session options**:
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
4. Right panel → **Add Input** → **Your Work / Datasets** → `crashvlm-data` → **Add**.
5. Menu **Add-ons → Secrets** → **Add Secret**:
   - Label: `GITHUB_TOKEN`
   - Value: the same GitHub token as in Colab.

   Then tick the checkbox next to it so this notebook can use it.

## 4. Smoke test (interactive, ~15 min)

Run **cell 1** (setup), then **cell 2** (smoke test).

- Cell 1 must end with `setup OK`.
- Cell 2 must end with `smoke test OK`.

Cell 2 trains 2 steps and writes 1 report. It shows that training, saving and loading the
adapter work before you spend hours. If either cell fails, copy the last ~30 lines and send
them to Claude.

## 5. The long run (background, up to ~11 h)

1. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**.
2. You can close the browser. Kaggle runs it on its own servers and emails you when it is done.
3. To watch progress: open the notebook → **Versions** (clock icon) → the running version →
   **Logs**.

What the run does:

| Step | Time on T4 |
|---|---|
| Setup | ~5 min |
| D: LoRA training, 488 examples × 3 epochs | stops at 5.5 h at the latest; the best adapter is kept |
| D: evaluation of the adapter on the 46 test clips | ~1.5 h |
| C: Qwen3-VL-8B 4-bit zero-shot | ~1–2 h, only if more than 2 h are left |
| Pack results | 1 min |

The time cap keeps the whole run inside Kaggle's 12 h limit. If training is cut short, it is
recorded in `train_log.json` (`stopped_by_time_budget`), and the paper reports it.

## 6. Get the results back

1. Open the finished version → **Output** tab.
2. Download **`crashvlm_gpu_results.zip`**. It contains `vlm/*.jsonl`,
   `lora/qwen3.5-4b_lora/train_log.json` and the adapter.
3. Put it in `~/Downloads` and tell Claude "Kaggle results are in". Claude then runs
   `eval_vlm.py` and `paper_tables.py` and updates the paper.

## If something goes wrong

- **No GPU or internet option:** the phone is not verified (step 1).
- **`MISSING ...` in cell 1:** the dataset is not attached (step 3.4), or it is still processing.
- **`git clone failed`:** the secret is missing, not ticked for this notebook, or the token has
  expired.
- **CUDA out of memory in training:** make sure the accelerator is **T4 x2**, not a single T4 or a
  P100.
- **Weekly quota:** a full run uses about 11 of the 30 free GPU hours.
