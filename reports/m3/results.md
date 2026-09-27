# M3 results (test split)

Test tests: 23; labels: `annotations/claude_pass_a.json` (model-produced, see docs/ANNOTATION_QA.md). Tier-A events; hit@k counts a missed detection as a miss; CIs: bootstrap over tests. Backbones: google__siglip2-base-patch16-224, resnet50.
t0 unknown: each test clip cropped at a random start in [0, 40] ms (seed 42).

## t0 known

| event                  | method                                |   n |   pos |   F1 |   hit@8 |   hit@16 | hit@16 95% CI   |   median |err| ms |
|:-----------------------|:--------------------------------------|----:|------:|-----:|--------:|---------:|:----------------|------------------:|
| airbag_deploy_start    | blind (no video)                      |  22 |    21 | 0.98 |    1    |     1    | 1.00-1.00       |               0   |
| airbag_deploy_start    | resnet50 linear probe                 |  22 |    21 | 0.98 |    0.3  |     0.4  | 0.19-0.63       |              28   |
| airbag_deploy_start    | siglip2-base-patch16-224 linear probe |  22 |    21 | 0.98 |    0.55 |     0.55 | 0.32-0.76       |               8   |
| airbag_deploy_start    | siglip2-base-patch16-224 zero-shot    |  22 |    21 | 0.92 |    0.4  |     0.4  | 0.19-0.62       |              23.2 |
| head_contact           | blind (no video)                      |  22 |    18 | 0.9  |    0.72 |     0.94 | 0.82-1.00       |               0.5 |
| head_contact           | resnet50 linear probe                 |  22 |    18 | 0.76 |    0.22 |     0.44 | 0.21-0.68       |              16   |
| head_contact           | siglip2-base-patch16-224 linear probe |  22 |    18 | 0.89 |    0.44 |     0.5  | 0.28-0.74       |              16   |
| head_contact           | siglip2-base-patch16-224 zero-shot    |  22 |    18 | 0.76 |    0.06 |     0.28 | 0.07-0.47       |              48   |
| side_glass_break_onset | blind (no video)                      |  27 |    11 | 0.48 |    0.27 |     0.36 | 0.11-0.62       |              10.5 |
| side_glass_break_onset | resnet50 linear probe                 |  27 |    11 | 0.62 |    0.27 |     0.27 | 0.00-0.43       |              20   |
| side_glass_break_onset | siglip2-base-patch16-224 linear probe |  27 |    11 | 0.58 |    0.09 |     0.27 | 0.00-0.50       |              40   |
| side_glass_break_onset | siglip2-base-patch16-224 zero-shot    |  27 |    11 | 0.22 |    0    |     0    | 0.00-0.00       |             204   |
| windshield_crack_onset | blind (no video)                      |  45 |     8 | 0.76 |    1    |     1    | 1.00-1.00       |               0.8 |
| windshield_crack_onset | resnet50 linear probe                 |  45 |     8 | 0.74 |    0.25 |     0.5  | 0.50-0.50       |              16   |
| windshield_crack_onset | siglip2-base-patch16-224 linear probe |  45 |     8 | 0.67 |    0.62 |     0.62 | 0.00-1.00       |               8   |
| windshield_crack_onset | siglip2-base-patch16-224 zero-shot    |  45 |     8 | 0.55 |    0.25 |     0.25 | 0.00-0.50       |               8   |

## t0 unknown

| event                  | method                                |   n |   pos |   F1 |   hit@8 |   hit@16 | hit@16 95% CI   |   median |err| ms |
|:-----------------------|:--------------------------------------|----:|------:|-----:|--------:|---------:|:----------------|------------------:|
| airbag_deploy_start    | blind (no video)                      |  22 |    21 | 0.98 |    0.15 |     0.4  | 0.20-0.62       |              18.3 |
| airbag_deploy_start    | resnet50 linear probe                 |  22 |    21 | 0.98 |    0.3  |     0.45 | 0.23-0.67       |              20   |
| airbag_deploy_start    | siglip2-base-patch16-224 linear probe |  22 |    21 | 0.98 |    0.55 |     0.55 | 0.32-0.76       |               8   |
| airbag_deploy_start    | siglip2-base-patch16-224 zero-shot    |  22 |    21 | 0.92 |    0.4  |     0.45 | 0.25-0.67       |              15.2 |
| head_contact           | blind (no video)                      |  22 |    18 | 0.9  |    0.17 |     0.28 | 0.08-0.50       |              25   |
| head_contact           | resnet50 linear probe                 |  22 |    18 | 0.76 |    0.22 |     0.44 | 0.21-0.68       |              16   |
| head_contact           | siglip2-base-patch16-224 linear probe |  22 |    18 | 0.89 |    0.44 |     0.5  | 0.28-0.74       |              16   |
| head_contact           | siglip2-base-patch16-224 zero-shot    |  22 |    18 | 0.76 |    0.06 |     0.28 | 0.07-0.50       |              40   |
| impact_onset           | blind (no video)                      |  23 |    23 | 1    |    0.2  |     0.4  | 0.19-0.63       |              26   |
| impact_onset           | resnet50 linear probe                 |  23 |    23 | 1    |    0.55 |     0.8  | 0.63-0.95       |               8   |
| impact_onset           | siglip2-base-patch16-224 linear probe |  23 |    23 | 1    |    0.35 |     0.5  | 0.29-0.72       |              20   |
| side_glass_break_onset | blind (no video)                      |  27 |    11 | 0.48 |    0    |     0.09 | 0.00-0.33       |              31.1 |
| side_glass_break_onset | resnet50 linear probe                 |  27 |    11 | 0.64 |    0.27 |     0.27 | 0.00-0.42       |              20   |
| side_glass_break_onset | siglip2-base-patch16-224 linear probe |  27 |    11 | 0.58 |    0.09 |     0.27 | 0.00-0.50       |              40   |
| side_glass_break_onset | siglip2-base-patch16-224 zero-shot    |  27 |    11 | 0.22 |    0    |     0    | 0.00-0.00       |             204   |
| windshield_crack_onset | blind (no video)                      |  45 |     8 | 0.76 |    0.12 |     0.25 | 0.00-0.50       |              18.2 |
| windshield_crack_onset | resnet50 linear probe                 |  45 |     8 | 0.74 |    0.25 |     0.5  | 0.50-0.50       |              16   |
| windshield_crack_onset | siglip2-base-patch16-224 linear probe |  45 |     8 | 0.67 |    0.62 |     0.62 | 0.00-1.00       |               8   |
| windshield_crack_onset | siglip2-base-patch16-224 zero-shot    |  45 |     8 | 0.55 |    0.25 |     0.25 | 0.00-0.50       |               8   |

