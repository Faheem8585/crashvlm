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

## Physics ground truth from vehicle sensors (`reports/physics/`)

8. **Measured time of maximum crush for 94 / 100 tests** (`reports/physics/checks.md`), from
   integrated structural accelerations in NHTSA's ISO-MME exports:
   - frontal and pole: the velocity towards the obstacle reaches zero;
   - side MDB: barrier and vehicle velocities meet.

   Frontal maximum crush is very consistent (median 70.8 ms). Dynamic crush exceeds NHTSA's
   residual crush by a median factor of 1.33 (frontal) and 1.29 (pole), as expected from elastic
   rebound. The rank correlation with residual crush is 0.31 (frontal) and 0.64 (pole). Unit
   tests check the integration on synthetic signals with a known answer.
9. **Visual "maximum deformation" labels carry no information beyond the protocol.** Against
   the measured times, labels are no closer than a constant per test type (frontal: 3.5 vs
   3.4 ms median error; pole: worse than the constant). Dropping this label from the headline
   metrics (tier C) is confirmed, and the physics time replaces it.
10. **Frame-appearance features cannot find the kinematic turning point**
    (`reports/physics/probe_results.md`). Linear probes trained on the measured time reach
    hit@16 of 0.15-0.20 on test, below the blind baseline even without the clock (0.35).
    Whole-image motion energy (tried on the training split only) fails too: glass fragments,
    airbag smoke and dummy motion swamp the vehicle's own motion. Maximum crush is where the
    vehicle's velocity is zero. Measuring it from video needs tracking of the body (e.g. the
    photogrammetry targets on NCAP vehicles), which is a concrete PhD-level direction:
    video-derived kinematics validated against accelerometers.
11. **Video kinematics recover the measured time of maximum crush in frontal tests**
    (`reports/physics/flow_{train,val,test}.md`). Pyramidal Lucas-Kanade optical flow on the
    rear of the visible body gives the vehicle's image velocity, and maximum crush is where it
    stops. The method is clock-free and has one bias correction estimated on train (+3.1 ms).
    - Designed on 23 training tests: rank correlation with the accelerometer time 0.71, versus
      -0.12 for the blind constant.
    - On the 12 held-out frontal tests (6 val, 6 test; the test clips were additionally cropped
      at a random start), every prediction is within 8 ms of the accelerometer time. The test
      median error is 0.9 ms. The method also catches the outlier (test 10039: 49 ms measured,
      51 ms from video, 95 ms blind).
    - This is the first result in the project where a vision method is validated against a
      physical measurement rather than a visual label.
    - Scope: frontal side-on views only, n = 12 held out; side and pole views need a different
      camera geometry.
12. **Temporal context helps one event, inconsistently** (`reports/m3/temporal.md`). A small
    dilated temporal CNN over frozen frame embeddings (receptive field about 120 ms, random-crop
    training, 3 seeds) raises airbag hit@16 with ResNet-50 features from 0.45 (probe) to
    0.75 ± 0.05. It does not improve impact or head contact, is worse for glass events, and is
    unstable with SigLIP features (airbag 0.42 ± 0.39). With 63 training tests, temporal models
    over-fit. The learning curve (finding 4) and this result both say the data are too small for
    learned temporal models, which is exactly where physics-based kinematics (finding 11) help.
