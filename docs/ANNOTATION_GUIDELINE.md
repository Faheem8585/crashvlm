# Annotation guideline (ontology v1)

The source of truth for labels and attributes is [`configs/ontology.yaml`](../configs/ontology.yaml);
this document explains how to apply them. If the two disagree, fix the YAML and this document
together, and bump `version`.

## What you annotate

Each clip is one camera view of one crash test, **after overlay removal**, sampled every 8 ms
(every 2nd processed frame). There are two roles:

- **exterior**: the car from outside (frontal: left front half; side and pole: overhead).
- **occupant**: a camera on the driver dummy (onboard, or a driver close-up for older frontal tests).

You mark two kinds of things with CVAT **tags** (no boxes):

1. **Point events**: a tag on the **first frame where the event is visible**.
2. **One `clip_summary` tag per clip, on the clip's first frame.** This is required, and it is
   also the "done" marker: a clip without it counts as unfinished and is excluded.

## Point events

| Label | Roles | Tests | Put the tag on… |
|---|---|---|---|
| `impact_onset` | exterior, occupant | all | the first frame where the car touches the barrier, MDB or pole. **Usually pre-filled**: check it and move it if it is wrong. |
| `airbag_deploy_start` | occupant, exterior | all | the first frame where any airbag (frontal, curtain, side torso) visibly leaves its cover. The cover opening counts; the airbag being fully inflated is too late. |
| `max_deformation` | exterior | all | the frame of maximum visible intrusion, just before the structure springs back or the car separates from the barrier/MDB. |
| `head_contact` | occupant | all | the first frame where the dummy head touches an airbag or an interior part. Set `surface` = airbag / structure / other. |
| `windshield_crack_onset` | exterior, occupant | all | the first frame with a visible **new** crack. Pre-existing marks do not count. |
| `side_glass_break_onset` | exterior, occupant | side, pole | the first frame where a side window visibly breaks. |
| `door_opens` | exterior | all | the first frame where any door visibly unlatches and opens. |

Rules:
- **At most one tag per event label per clip.** If an event happens twice (e.g. two airbags),
  tag the **earliest**.
- **Absent tag = the event is not visible in this clip.** Do not add a tag "just in case".
- If the event is **already under way** in the first frame where you can see it (glare,
  occlusion, off-screen), tag that frame and **untick `onset_visible`**.
- **Pre-filled tags** (source `auto`) are machine proposals: `impact_onset` from the measured
  time zero, and other events from a zero-shot image-text model. Treat them as suggestions:
  - correct → leave it;
  - wrong frame → delete it and add a new tag on the right frame;
  - event not visible → delete it.

  Never keep a pre-label just because it is close. The export records which pre-labels you
  kept, and that acceptance rate is itself a result.
- Step with `F`/`D` (next/previous frame). Zoom in for glass and airbag covers.

## Clip summary (`clip_summary`, on the first frame)

| Attribute | Values | Guidance |
|---|---|---|
| `view_usable` | ☑ / ☐ | Untick if glare, blur or occlusion hide most of what this role should show. Still fill in the rest as far as you can. |
| `frontal_airbag_deployed` | yes / no / not_visible | Driver or passenger frontal airbag. |
| `curtain_airbag_deployed` | yes / no / not_visible | Roof-rail curtain airbag. |
| `a_pillar_deformation` | none / minor / severe / not_visible | minor = visible bend or buckle; severe = clear fold or rearward displacement of the pillar. |
| `windshield_damage` | none / cracked / shattered / not_visible | State at the end of the clip. |
| `door_opened` | yes / no / not_visible | Any door, at any time in the clip. |
| `fire_or_fluid_leak` | yes / no / not_visible | Default `no`. Mark `yes` only for visible flame, smoke from a fire, or a fluid stream. This label exists to catch model hallucinations, so be strict. |
| `visual_severity` | low / medium / high | Your overall impression of structural severity **in this view**, compared with the other clips of the same test type. |
| `notes` | text | Anything odd: camera shake, dummy obscured, lighting change, suspected overlay remnants. |

`not_visible` means the view does not show that part. It does **not** mean "no".

## Procedure

1. **Pass A** (task `pass_a`, all clips, pre-labelled). Work in blocks of ≤ 20 clips and note
   start and end times; the report uses annotation time per clip.
2. **Pass B** (task `pass_b`, the 30 IAA clips, *not* pre-labelled). Start it **at least 7 days
   after** finishing those clips in pass A, and do not look at pass A while doing it.
3. Export each task as **CVAT for images 1.1** and convert it (see below).

With one annotator, pass A vs. pass B measures **intra-rater** reliability. Pass A was
pre-labelled and pass B was not, so their difference also includes any anchoring by the pre-labels.
The report states this explicitly and does not claim inter-annotator agreement.

## CVAT set-up (free cloud plan: 1 project, 3 tasks, 1 GB)

1. `uv run python scripts/make_cvat_tasks.py` produces, under `data/annotation/`:
   `cvat_labels.json`, `pass_a/images.zip`, `pass_a/prelabels.xml`, `pass_b/images.zip`.
2. In CVAT: **Projects → +** → name `crashvlm`. Open **Labels → Raw**, paste the contents of
   `cvat_labels.json`, then **Done**.
3. **Create task** `pass_a` in the project → upload `pass_a/images.zip`. Keep **sorting method:
   lexicographical** so the frames of each clip stay together and in order. Submit.
4. Open the task → **Actions → Upload annotations** → format **CVAT 1.1** → `pass_a/prelabels.xml`.
5. Later, create task `pass_b` from `pass_b/images.zip` in the same way, without uploading annotations.
6. Export: **Actions → Export task dataset → CVAT for images 1.1** → unzip → `annotations.xml`.

```bash
uv run python scripts/convert_cvat.py annotation.cvat_export=~/Downloads/pass_a/annotations.xml annotation.pass_name=pass_a
```

```bash
uv run python scripts/convert_cvat.py annotation.cvat_export=~/Downloads/pass_b/annotations.xml annotation.pass_name=pass_b
```

```bash
uv run python scripts/iaa.py
```

The converter lists every clip without a summary, duplicate tags, and labels used in the wrong
role (`reports/m2/<pass>_summary.md`). Fix these in CVAT and re-export rather than editing the JSONL.
