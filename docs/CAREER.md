# Career material for the BMW PhD application

All numbers come from `reports/` (generated). Items marked **[after GPU runs]** must be filled
from `reports/m4/results.md` once the Colab runs are done. Do not claim them before.

A note on honesty: the annotations were produced by an AI model (Claude) under your
ontology, guideline and QA protocol. You designed and built the annotation pipeline and quality
checks; you did not hand-label 200 clips. Say exactly that. It is a stronger story ("I built a
QA protocol for model-produced labels and found where it fails") and it survives any
follow-up question.

## CV bullets

- **Built an open crash-video pipeline on the NHTSA database.** It covers 1,583 NCAP tests,
  from which I selected 100 tests with 200 high-speed camera views. It includes overlay removal
  with an automated leak check and OCR recovery of each clip's time axis (193/200 verified).
  Python, PyAV, RapidOCR, Hydra, pytest, CI.
- **Designed a leakage-aware benchmark for crash-event timing.** It showed that a blind,
  no-video baseline times airbag deployment within 16 ms in 100 % of test cases when the lab
  clock is known, and still in 75 % without it. Paired tests against that baseline showed that
  frozen-encoder probes don't beat it, while video VLMs do at 8 ms (airbag 0.90 vs 0.25 hit@8,
  23-test split, bootstrap CIs). I found and corrected a bias in my own baseline that had
  overstated the probes.
- **Built the annotation and label-QA workflow.** It covers a 7-event ontology, CVAT
  integration, model-produced labels, and blind cross-model agreement checks (airbag onset
  kappa 0.93, 18/18 within 8 ms on repeat). The checks exposed that same-model agreement
  overstates reliability on subjective labels.
- **Grounded video analysis in physics.** I derived the measured time of maximum crush for 94
  NHTSA tests by integrating the vehicles' accelerometer data (ISO-MME), and showed that
  clock-free optical flow on the car body recovers it within 8 ms in 12/12 held-out frontal
  tests (median test error 0.9 ms), where visual labels and appearance-based probes failed.
- **Implemented VLM crash reporting.** Qwen3.5-4B and Qwen3-VL-8B (4-bit) write structured JSON
  reports, with LoRA fine-tuning (PEFT), a hallucination probe and a FastAPI + Gradio service
  with a Docker image. All models produced schema-valid reports for 100 % of test clips.
  LoRA fine-tuning on a free GPU, although cut short at 20 steps, improved impact timing from
  0.20 to 0.65 hit@8 (blind 0.35) and reduced contradicting claims.

## Research proposal paragraph (cover letter)

My prototype CrashVLM showed me where the difficulty in AI-based crash video evaluation lies. It
is not model capacity. Standardised test protocols let a clock-only baseline match any video
model, and burned-in overlays silently leak labels. In the PhD, I would build on this in three
steps:
1. Ground video understanding in physics, using the instrumented measurements (acceleration
   traces, intrusions, injury criteria) as supervision and evaluation targets instead of visual
   labels alone.
2. Close the loop to simulation. Rendering explicit crash simulations of parametric structures
   gives unlimited, exactly labelled video. With it, I would study the sim-to-real gap for VLMs
   and train geometric models (mesh GNNs, point networks) that connect deformation fields to
   what cameras see.
3. Turn this into engineering tools: structured, verifiable crash reports that flag
   discrepancies between simulation and test for the engineer, with calibrated uncertainty
   rather than fluent text.

The common thread is evaluation validity: every claim a model makes about a crash should be
checkable against a measurement.

## LinkedIn post (draft)

> Can a vision-language model read a crash test video?
>
> Over the past weeks I built CrashVLM, an open research prototype on the public NHTSA crash
> test database: 100 NCAP tests, 200 high-speed camera views, SigLIP 2 probes, and Qwen VLMs
> that write structured crash reports.
>
> The most interesting result was not a model. NCAP tests are so standardised that a predictor
> that never looks at the video times airbag deployment almost perfectly, as long as you know
> the lab's impact time. Take the clock away (random clip crops, like any real video) and that
> shortcut shrinks, but a well-built prior still gets 75 % within 16 ms. Only at 8 ms do video
> models clearly beat it.
>
> Other lessons: burned-in timestamps and a red→green impact lamp can leak your labels; two
> runs of the same model agreeing with each other doesn't mean either is right; and your
> baseline deserves as much care as your model. A bug in mine had made my probes look better
> than they were.
>
> Code, data card, label-QA protocol and paper draft: [link]
> #ComputerVision #VisionLanguageModels #VehicleSafety #CrashTest #MachineLearning

## Interview questions (with concise answers)

1. **Why Qwen3.5-4B / Qwen3-VL-8B and not a bigger model?**
   - Both take interleaved image sequences natively and fit a 16–24 GB GPU (bf16, or 4-bit
     NF4 for 8B).
   - Qwen3.5-4B has an official LoRA path that fits a free-tier T4/L4.
   - With 23 test tests, a larger model's gain would not be distinguishable from noise. Better
     evaluation mattered more than scale.
2. **CLIP-type encoders vs generative VLMs: when would you use which?**
   - Encoders give per-frame scores cheaply, and are easy to probe, calibrate and time. That is
     ideal for event onsets and for a CPU service.
   - Generative VLMs give structured, multi-field reports and reasoning across frames, but they
     are slower, need a strict parser, and can hallucinate. So I measure fire claims and
     contradicting events explicitly.
   - In practice I'd combine them: encoder curves for timing, the VLM for the report.
3. **How do you model time?**
   - Frames on a fixed 8 ms grid with explicit timestamps in the prompt.
   - Probes predict a monotone "has happened" state, decoded as the first sustained crossing.
   - Crucially, times are given relative to the first frame shown, never to impact, so the
     model can't copy the protocol clock.
   - Next step: a temporal head (e.g. R(2+1)D or a small transformer on frame embeddings)
     instead of per-frame probes.
4. **How do you know your labels are any good?**
   - I checked them three ways: cross-model blind re-annotation, a same-model blind repeat,
     and adjudication of disagreements on the frames.
   - I report agreement per event, and tier the labels: only reliable events enter headline
     metrics.
   - I found that same-model agreement is inflated on subjective labels. The honest next step
     is a human-verified subset.
5. **Point clouds or meshes for crash simulation data?**
   - Crash FE models are meshes with connectivity, thickness and material. Mesh GNNs
     (MeshGraphNet-style) keep that topology, which matters for buckling and contact.
   - Point clouds (PointNet++) are simpler and topology-free, and fit scanned or deformed
     geometry.
   - For surrogates on FE output I'd start with mesh GNNs; for sim-to-real matching with
     measured deformation, point clouds.
6. **How would you tackle the sim-to-real gap for video?**
   - Render simulations with the physical camera setups and paint schemes.
   - Measure the gap with the same leakage-aware protocol: t0 unknown, blind baselines.
   - Adapt on features that exist in both domains (deformation, airbag geometry), not on
     appearance.
   - Use physics as the shared label space: simulated and measured accelerations align in
     time, visual labels do not.
7. **Where does physics enter?**
   - As ground truth: accelerometer-derived timing of maximum crush instead of visual guesses.
   - As constraints: monotone event states, and physically plausible ordering (impact before
     airbag before head contact).
   - As evaluation: do VLM severity ratings track crush and HIC? Here my annotated severity
     label did not, which is itself a finding.
8. **Why should anyone trust your evaluation?**
   - The test split is by vehicle model and touched once.
   - Hyper-parameters are tuned on validation.
   - Bootstrap intervals are over tests.
   - Blind baselines are reported in two settings.
   - All numbers are generated by scripts.
   - Overlay leakage is checked automatically on every rebuild.
   - Limitations (model-produced labels, 23 test tests, US NCAP only) are stated up front.
9. **Tell us about a mistake in your own evaluation.**
   - My blind baseline learned "time since the first frame" on uncropped clips and was applied
     to randomly cropped test clips, so it was about 20 ms late. It made my probes look better
     than the prior.
   - A fine-tuned VLM beat it with near-constant predictions, which cannot happen if the
     baseline is fair. That told me to look at the baseline, not celebrate the model.
   - After the fix, the probes' advantage disappeared; the VLMs' advantage at 8 ms survived a
     paired test. I documented the correction in the findings rather than quietly updating
     the numbers.
