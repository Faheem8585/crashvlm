# Dataset statistics (generated 2026-09-26)

## Catalog: NCAP tests 2012+ in the NHTSA database

| config_key | tests | both_views | first_year | last_year | crush_known | driver_hic_known |
|---|---|---|---|---|---|---|
| ITV | 541 | 529 | 2012 | 2026 | 540 | 521 |
| VTB | 536 | 514 | 2012 | 2026 | 535 | 519 |
| VTP | 506 | 478 | 2012 | 2026 | 506 | 497 |

## Selection and split (seed 42)

| config_key | test | train | val | All |
|---|---|---|---|---|
| ITV | 7 | 19 | 4 | 30 |
| VTB | 9 | 25 | 6 | 40 |
| VTP | 7 | 19 | 4 | 30 |
| All | 23 | 63 | 14 | 100 |

Vehicle groups: 83; IAA tests: 15


### Test labs in the selection

| performer | ITV | VTB | VTP |
|---|---|---|---|
| CALSPAN | 4 | 8 | 3 |
| KARCO ENGINEERING | 10 | 12 | 7 |
| MEDICAL COLLEGE OF WISCONSIN | 1 | 0 | 0 |
| MGA RESEARCH | 12 | 13 | 13 |
| TRC OF OHIO | 3 | 7 | 7 |

## Processed clips (200)

| config_key | role | clips | time_from_ocr | time_verified | median_crop_frac | median_masks |
|---|---|---|---|---|---|---|
| ITV | exterior | 30 | 30 | 30 | 0.038 | 1.0 |
| ITV | occupant | 30 | 30 | 30 | 0.001 | 1.0 |
| VTB | exterior | 40 | 38 | 36 | 0.004 | 1.0 |
| VTB | occupant | 40 | 40 | 39 | 0.004 | 1.0 |
| VTP | exterior | 30 | 30 | 29 | 0.029 | 1.0 |
| VTP | occupant | 30 | 30 | 29 | 0.012 | 1.0 |

### Source codecs

| codec | performer | clips |
|---|---|---|
| h264 | CALSPAN | 23 |
| h264 | KARCO ENGINEERING | 26 |
| h264 | MGA RESEARCH | 34 |
| h264 | TRC OF OHIO | 6 |
| mpeg2video | CALSPAN | 4 |
| vc1 | TRC OF OHIO | 22 |
| wmv2 | MGA RESEARCH | 42 |
| wmv3 | CALSPAN | 3 |
| wmv3 | KARCO ENGINEERING | 32 |
| wmv3 | MEDICAL COLLEGE OF WISCONSIN | 2 |
| wmv3 | TRC OF OHIO | 6 |
