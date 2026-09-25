# Data card: CrashVLM NCAP clip set

All counts referenced here live in [generated/data_stats.md](generated/data_stats.md), which
`scripts/data_stats.py` writes from the pipeline outputs. No number in this card is typed by hand.

## Source

| | |
|---|---|
| Publisher | U.S. National Highway Traffic Safety Administration (NHTSA) |
| Database | Vehicle Crash Test Database, <https://www.nhtsa.gov/research-data/research-testing-databases/> |
| Access | Public JSON API `https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/`, media on `https://nrd-static.nhtsa.dot.gov/`; no key, no login |
| Retrieved | Catalog crawled 2026-09-25; raw API responses cached under `data/raw/api/` |
| Test types used | NCAP and optional NCAP tests from 2012 on: frontal rigid barrier (`VTB`), side moving deformable barrier (`ITV`), side pole (`VTP`) |

The API ignores server-side filters and page sizes (20 rows per page), so `build_catalog.py`
crawls every page once and filters locally.

## Licence and redistribution

- data.gov lists this dataset as **public access** with licence **"unknown"**
  (<https://catalog.data.gov/dataset/vehicle-safety-research-and-development-database-vehicle-browse-tests>).
- The footage was filmed by NHTSA contract labs (Calspan, KARCO, MGA, TRC, MCW). We therefore
  do **not** assume that it is automatically public domain as work of the U.S. government.
- NHTSA's website-policy page returned HTTP 403 to automated fetches during this check, so its
  wording is not quoted here.

**Policy of this repository:** raw videos and extracted frames are never committed or re-hosted.
The repo publishes code, test numbers (`configs/splits/selection_v1.csv`), annotations and derived
metrics; anyone can rebuild the frames from NHTSA with the scripts. The IIHS small-overlap test
and YouTube footage are excluded because their licensing is unclear.

## Composition

- **Catalog:** every NCAP test of the three types from 2012 on, one row per test, joined with the
  vehicle record and dummy measurements (`data/interim/catalog.parquet`, `occupants.parquet`).
- **Selection:** 40 frontal + 30 side-MDB + 30 side-pole tests, seeded (seed 42), frozen in
  `configs/splits/selection_v1.csv`.
- **Split:** train / val / test ≈ 60 / 15 / 25 %, assigned by **vehicle group (make + model)**,
  so one vehicle's footage never appears in two splits. The split is balanced within each test type.
- **IAA subset:** 15 test-split tests are annotated twice for inter-annotator agreement (M2).
- **Views per test (two roles):**
  - `exterior`: structural deformation. Frontal: *Left Front Half* (fallbacks *Left Overall*,
    *Left Angle*). Side and pole: *Overhead Wide View*.
  - `occupant`: a camera on the driver dummy (airbag, head contact). Onboard views where they
    exist; older frontal tests fall back to *Onboard Driver Side* or *Driver Close-Up*. The exact
    source camera of every clip is stored in `meta.json` (`source_name`).
  - Only high-speed views are used. Real-time, pan, documentation and "for web" edits are
    excluded because their time base differs or they are edited.

### Metadata fields used as ground truth

| Field | Source (API) | Notes |
|---|---|---|
| test type, closing speed, impact angle, lab, year | test list | month is often `00`, so only the year is used |
| make, model, model year, test weight, crush (`vaxCrushDistance`) | vehicle info | for side-MDB tests the rated car is vehicle 2 (vehicle 1 is the barrier) |
| HIC (+ window), chest 3 ms clip, femur loads, belt load | occupant info | per seat; in the side tests inspected, ES-2re / SID-IIs chest-clip and femur fields are 0 (stored as missing) |

**Zeros mean "not measured".** The API stores missing values as `0`, so weight, crush and injury
fields equal to 0 are stored as NaN (`catalog.ZERO_IS_MISSING`). Intrusion measurements are not
part of the API fields used here; they may be in the test-report PDFs (not parsed yet).

## Processing (`scripts/prepare_clips.py`)

1. Download the canonical view, with resume and retry. The raw file is deleted after extraction
   (`clips.keep_raw=false`) to fit a small disk.
2. **Overlay removal** (`crashvlm.data.overlay`), which is needed because the burned-in overlays
   leak the labels:
   - Each lab burns in its own overlay: side or bottom panels with frame number, time and a
     red→green **T0 indicator dot** (KARCO), text strips (MGA, Calspan), or a timestamp box over
     the scene (TRC).
   - Uniform, static border bands are cropped.
   - Text boxes found by OCR (RapidOCR) are masked when they stay in the same place across the clip
     **and** look like an overlay: overlay words (ms, fps, frame, time, camera…) or numbers that
     change between frames.
   - Placards fixed to the vehicle (e.g. the test ID seen by an onboard camera) are deliberately
     *not* masked: masking them hid the side airbag in one QA case.
3. **Physical time:**
   - The numbers read from the masked overlay are fitted as `t_ms = a·frame + b` with a robust
     Theil–Sen fit.
   - A clip is **time-verified** when OCR readings in at least 60 % of sampled frames fall on one
     line within 1 ms.
   - Otherwise the NCAP convention (1000 fps, clip starts at −50 ms) is recorded, flagged as
     unverified.
   - Temporal-localisation metrics use verified clips only.
4. Every 4th decoded frame is saved as a JPEG, longest side 512 px, with its physical time in
   `meta.json`. Each clip also gets a `preview.jpg` (original | processed).
5. `scripts/overlay_report.py` stitches all previews into `reports/m1/overlay_check.jpg` for a
   manual check. The check must find no timestamps, frame counters or T0 dots in the processed
   images.

## Known issues and biases

- **Legacy video (2012–2013):** 720-pixel-wide WMV/MPEG-2 files with barely legible overlays.
  They cause almost all unverified time fits; see the generated table.
- **Not every clip follows the convention:**
  - some clips start at −5 ms instead of −50 ms;
  - one 2012 onboard camera reads as ~800 fps;
  - some 2012 files repeat every frame three times (1,082 decoded frames for ~360 ms).
  - The per-clip OCR fit handles these cases. Assuming a fixed convention would not.
- **Positional prior:** in most clips impact (t = 0) sits at the same relative position (frame
  ~50 of ~350). A model can exploit this without looking at the video. Evaluation must use
  randomised clip windows and report a constant-position baseline (see `DESIGN.md`).
- **Narrow operating range:** NCAP runs fixed protocols (frontal 56 km/h, side MDB 62 km/h, pole
  32 km/h), so impact speed barely varies. Severity differences come from the vehicle, not the test.
- **Population:** US-market vehicles and five US labs. Lab identity is visible in the camera
  setup and could act as a confounder; lab is recorded per test for stratified analysis.
- **Frame rate:** 1000 fps capture subsampled to 250 Hz (stride 4) gives a 4 ms timing resolution.

## Intended use

Research on automated evaluation of crash-test video. Not for vehicle safety ratings, which
remain NHTSA's official star ratings based on instrumented measurements.
