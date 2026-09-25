# Design notes

Living document: each milestone adds its decisions. Rationale first, then the rule it creates.

## M1: data pipeline

```mermaid
flowchart LR
  A[NHTSA API<br/>test list, vehicle,<br/>occupant, media] -->|build_catalog.py| B[(catalog.parquet<br/>occupants.parquet)]
  B -->|select_tests.py<br/>seed, vehicle-grouped split| C[(selection_v1.csv<br/>frozen, committed)]
  C -->|prepare_clips.py| D[download → overlay analysis<br/>crop + mask + OCR time fit<br/>→ strided JPEG frames]
  D --> E[(clips/&lt;test&gt;_&lt;role&gt;/<br/>frames + meta.json)]
  E -->|build_manifest| F[(manifest.parquet)]
  E -->|overlay_report.py| G[QA sheet]
  F -->|data_stats.py| H[docs/generated/data_stats.md]
```

### Decisions

| Decision | Why |
|---|---|
| NHTSA NCAP only; frontal barrier, side MDB and side pole | Public, documented access with per-test measurements. Small-overlap (IIHS) and YouTube footage have unclear licensing. |
| Two views per test (`exterior`, `occupant`) | Deformation is visible only from outside; airbag and head contact only from the occupant camera. Two views keep annotation at ~200 clips. |
| High-speed views only | Real-time and edited views use other time bases. |
| Split by vehicle group, balanced within test type | The same model appears in several test types; splitting by test would leak appearance. |
| Selection frozen after first run | The live API keeps growing, so a re-run could otherwise draw a different set. |
| Raw video deleted after extraction | ~2 GB free locally; the full run can also run on Colab. |
| Stride 4, longest side 512 px | 4 ms resolution is finer than the events of interest (airbag deployment takes tens of ms). ~6 MB per clip. |

### Leakage controls (binding for M3–M6)

1. **Burned-in overlays are removed** before any model sees a frame. They contain frame number,
   time and a red→green T0 dot, i.e. the answer to "when did the impact happen".
   `reports/m1/overlay_check.jpg` must be checked again whenever the overlay code changes.
2. **Positional prior.** Most clips place t = 0 near frame 50 of ~350. Temporal-localisation
   evaluation must (a) crop each test clip to a randomised window (seeded) so onset is not at a
   fixed relative position, and (b) report a *constant-position baseline*, which predicts the
   training-set mean relative position. A model is only credited with what it gains over that baseline.
3. **Time ground truth only from verified clips.** `manifest.time_verified` requires a
   self-consistent OCR fit. It is not the same as agreeing with the NCAP convention: some clips
   start at −5 ms, and some are encoded with frames tripled, so the convention would be wrong there.
4. **Vehicle placards stay visible.** Masking every static text also masked car-fixed labels in
   onboard views and hid the side airbag. Test IDs never repeat across splits, so they cannot
   leak labels.
5. **Lab confounder.** Lab identity is visible (camera setup, lighting). Results will be
   reported per lab where n allows it.

### Tests that guard these rules

- `tests/test_overlay.py`: band cropping, moving vs. fixed text, overlay vs. placard text,
  time fits for every lab's format (incl. decimal commas, 2000 fps, OCR errors), and edge
  overlays lost in over-exposed frames.
- `tests/test_views.py`: view selection on 45 real NHTSA file listings.
- `tests/test_split.py`: no vehicle group in two splits; IAA tests only in the test split.
