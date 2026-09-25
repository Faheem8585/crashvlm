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
