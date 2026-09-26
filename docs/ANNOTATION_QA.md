# Annotation QA log

All labels in `annotations/claude_pass_a.json` were produced by Claude (Opus 5.5) from the
time-stamped labelling sheets (`scripts/contact_sheets.py +label_sheets=true`). No human has
labelled these clips. This file records how label quality is checked; the numbers come from
`scripts/label_agreement.py` (`reports/annotation/`), never typed in by hand.

## Check 1: blind second model annotator (Claude Sonnet), first 39 clips

A Sonnet sub-agent labelled the same 39 clips from the same sheets and guideline, without
access to the primary labels. Both annotators are Claude models, so errors can be correlated:
this measures consistency between two models, not correctness against a human.
Report: `reports/annotation/claude_vs_sonnet_batch1.md`.

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
