# Findings so far (living document)

Every number here is copied from a script-generated report under `reports/`; the report is
named next to it. Labels are model-produced (docs/ANNOTATION_QA.md), and the test split has 23
crash tests, so confidence intervals are wide. Pending GPU runs are listed at the end.

1. **The NCAP clock gives the answer away** (`reports/m3/results.md`, "t0 known"). A blind
   predictor that never looks at the video (median training onset per event and test type)
   gets airbag onset 100 % and head contact 94 % within 16 ms, beating every video model.
   Crash-video timing benchmarks built on standardised tests are only meaningful against such
   a baseline, or without the lab clock.
2. **Without the clock, video features beat the blind baseline** (`reports/m3/results.md`,
   "t0 unknown": random clip start crops). Hit@16 for the blind baseline vs. the best linear
   probe: impact 0.45 vs 0.75 (ResNet-50), airbag 0.35 vs 0.60, head contact 0.33 vs 0.50,
   windshield crack 0.50 vs 0.62 (SigLIP 2 base). Intervals overlap for most events.
3. **Zero-shot SigLIP is weak at timing.** Text-prompt curves detect airbags (F1 0.92) but
   place onsets poorly (hit@16 0.45, and 0 for side-glass breakage); a linear probe on the same
   features is consistently better.
4. **More labels barely help frozen-feature probes** (`reports/m6/learning_curve.md`). From
   25 % to 100 % of the training tests, hit@16 moves within seed noise for airbag, head contact
   and impact. Only side-glass breakage improves clearly (0.03 → 0.27). With a linear probe,
   the frozen features and onset decoding, not label quantity, are the bottleneck. This
   motivates the LoRA (M5) and temporal-model comparisons.
5. **The annotated "visual severity" does not track measured severity**
   (`reports/m6/severity.md`). Spearman rho against crush and driver HIC ranges from -0.34 to
   0.42 over six comparisons. Only side-MDB vs HIC reaches p = 0.02, which does not survive a
   Bonferroni correction for six tests. The label also never uses "low". Reading severity off a
   single camera view is not supported by these labels; the VLM severity analysis (M6) will be
   judged against the measurements directly, not against this label.
6. **Pipeline lessons that mattered more than modelling** (docs/DESIGN.md, docs/ANNOTATION_QA.md):
   - burned-in overlays (timestamps, a red→green T0 lamp) had to be removed, and removal had to
     be validated on the full set;
   - OCR time parsing had two bugs that shifted 8 clips by about 50 ms, both found while labelling;
   - same-model label agreement overstates reliability on subjective labels.
7. **Demo failure case** (docs/DESIGN.md, M7): on test clip 8212 (driver close-up), the probes
   fire on the first two frames and report airbag and windshield crack at 0 ms. This is a real
   model error, identical on the training-pipeline frames.

## Pending (needs a GPU; cells in docs/COLAB.md)

- SigLIP 2 so400m embeddings (the numbers above use the base model)
- Qwen3.5-4B and Qwen3-VL-8B zero-shot reports (M4), LoRA fine-tuning and learning curve (M5)
- VLM ablations (frames, resolution, prompt) and VLM severity vs crush/HIC (M6)
