# Annotation QA log

All labels in `annotations/labels_v1.json` were produced with model-assisted annotation: a
large language model (the primary annotator) labelled the time-stamped labelling sheets
(`scripts/contact_sheets.py +label_sheets=true`) under the written guideline. No human has
labelled these clips yet. This file records how label quality is checked; the numbers come from
`scripts/label_agreement.py` (`reports/annotation/`), never typed in by hand.

## Check 1: blind cross-annotator, first 39 clips

A second, different language model labelled the same 39 clips from the same sheets and
guideline, without access to the primary labels. Both annotators are models from the same
family, so errors can be correlated: this measures consistency between two annotators, not
correctness against a human.
Report: `reports/annotation/crossannotator_batch1.md`.

### What the disagreements showed

| Area | Finding | Resolution |
|---|---|---|
| Events with a sharp visual cue (impact onset, airbag deployment, head contact) | Close agreement | Kept |
| `max_deformation` | The second annotator is almost always exactly +24 ms later. "Maximum visible intrusion" cannot be pinned from side or overhead video at 8 ms. | Label marked **low reliability**. Planned replacement: time of maximum dynamic crush from the vehicle accelerometer (velocity zero crossing) in the NHTSA sensor data, i.e. a physical rather than visual ground truth. Until then, excluded from headline timing metrics. |
| Summary attributes `curtain_airbag_deployed`, `windshield_damage`, `a_pillar_deformation`, `door_opened` | Most disagreements are "not_visible" vs "yes/no/none": the guideline did not say when a part counts as visible | Guideline clarified (below); the primary labels already follow the clarified rule |
| Through-windshield onboard cameras in pole tests | The second annotator read the crack web as side glass in 4 clips, and as windshield in 1 | Kept as windshield: a coherent web of cracks is laminated windshield glass, while side glass shatters into loose fragments |
| 10039 exterior A-pillar | Re-examined on the sheet: the A-pillar base buckles from +46 ms | Primary label changed none -> minor |

### Guideline clarification (v1.1)

`not_visible` is used only when the image region where the item would appear is out of frame
or fully occluded for the whole clip: the side-window line for `curtain_airbag_deployed`, the
windshield for `windshield_damage`, the A-pillar for `a_pillar_deformation`, a door edge for
`door_opened`. If that region is visible, answer yes/no (or none/minor/…) even when the answer
is "nothing happened".

## Check 2: blind cross-annotator, next 36 clips (after guideline v1.1)

Report: `reports/annotation/crossannotator_batch2.md`. Same set-up as check 1; the second
annotator received the v1.1 visibility rule.

### Findings

- **Timed events stay reliable.** Impact onset, windshield crack and side glass break agree when
  both annotators mark them, and airbag onset timing is within 16 ms whenever both mark it.
- **Airbag presence in exterior frontal views is the main presence disagreement.** The primary
  annotator marks the frontal airbag when it is seen *through the side window*
  (`onset_visible: false`); the second annotator usually leaves it out. Both readings follow the
  guideline. **Rule for evaluation:** airbag onset *timing* is scored on occupant views only;
  exterior-view airbag tags count only for presence.
- **The v1.1 visibility rule did not fix three summary attributes.** `windshield_damage` and
  `a_pillar_deformation` disagreements are still mostly "none" vs "not_visible". From a
  side-on exterior camera the windshield and A-pillar are seen at a grazing angle, and the two
  annotators draw that line differently. `visual_severity` (low/medium/high) is subjective
  (kappa about 0.2 in both checks).
- **Adjudicated on the sheets:** 14468 occupant (side airbag clearly visible from +22 ms,
  primary kept) and 14394 exterior (the "door opening" is the side mirror folding at ~+70 ms,
  so no door event; primary kept).

### Consequence: label reliability tiers used from M3 on

| Tier | Labels | Use |
|---|---|---|
| A (headline metrics) | `impact_onset`, `airbag_deploy_start` (occupant views), `head_contact`, `windshield_crack_onset`, `side_glass_break_onset`, `fire_or_fluid_leak`, `door_opened` | Training targets and headline evaluation |
| B (secondary) | `frontal_airbag_deployed`, `curtain_airbag_deployed`, `view_usable`, airbag presence in exterior views | Reported separately, with the agreement numbers next to them |
| C (not used as ground truth) | `max_deformation` (to be replaced by accelerometer data), `windshield_damage`, `a_pillar_deformation`, `visual_severity` | Descriptive only; `visual_severity` is compared with measured crush/HIC as an exploratory analysis, never as a target |

## Time-axis corrections found while labelling

Labelling test 8660 showed the impact at "+57 ms". The cause was a bug in reading the burned-in
time: a frame counter was taken as the time series (fix `b6418a6`). Some Calspan clips carry
no time at all, only "T0: <frame>" (fix `aa0074a`). Six clips were reprocessed; one existing
label (7639 exterior) was moved to the corrected axis. All 200 clips now carry
`code_version` ≥ `b6418a6`.

## Pass A complete: known limitations of the labels

Generated summary: `reports/annotation/label_stats.md`.

- **Timing resolution is 8 ms, and labels cluster on the same tiles.** Median onsets fall exactly
  on sheet tiles (e.g. frontal airbag +22 ms, side airbag +14 ms, frontal head contact +46 ms),
  with interquartile ranges at or near zero. Part of this is real: NCAP protocols fix speed and
  sensor timing. Part is probably annotator prior: the same tile gets picked when the true onset
  lies between tiles. **Consequence:** timing metrics use tolerances of at least 8 ms, and no
  claim is made about differences between vehicles below 8 ms.
- **`max_deformation` clusters at 70 ms (frontal) and 62/78 ms (side, pole).** This confirms the
  tier-C decision; the accelerometer-based replacement is required before the label is used.
- **`visual_severity` never uses "low"** (125 medium, 75 high): uninformative as a target, as
  already decided.
- **`fire_or_fluid_leak` is "no" for all 200 clips.** Airbag smoke and red dummy-marker powder
  appear in several clips and are noted as *not* fire, so this is the hallucination probe the
  ontology intended.

## Check 3: blind repeat by an independent run of the primary annotator (30 IAA clips)

An independent run of the primary annotator's model labelled the 30 IAA clips
blind, in shuffled order. Report: `reports/annotation/repeat_iaa.md`.

- **Timed events are highly consistent:** impact onset 30/30 identical; airbag onset presence
  kappa 0.93, 18/18 within 8 ms; head contact kappa 0.86, 10/11 within 8 ms.
- **Glass events are the weak spot:** windshield-crack presence kappa 0.38, side-glass kappa
  0.53. Most disagreements are the same judgement call on side-view cameras: is the web of cracks
  in the upper corner windshield or front side window? The second annotator flagged exactly these
  clips as uncertain.
- **Same-model agreement is inflated where the task is subjective.** `max_deformation` agrees to
  0 ms on all 15 clips here, but was systematically 24 ms later for the cross-annotator.
  `visual_severity` gets kappa 0.93 here versus about 0.2 against the cross-annotator. Two runs of the same
  model share its priors, so this check measures *reproducibility*, not *correctness*. The
  tier-C decisions stand, and cross-model agreement (checks 1-2) is the more honest quality
  number.
