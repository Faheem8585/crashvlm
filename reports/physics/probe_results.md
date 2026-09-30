# Physics-grounded event: time of maximum crush (exterior view, test split)

Ground truth from integrated vehicle accelerations, not from an annotator. Train/val/test tests with a result: 59/13/22.

| setting    | method                                  |   n_timing |   hit@8 |   hit@16 | hit@16 95% CI   |   median |err| ms |
|:-----------|:----------------------------------------|-----------:|--------:|---------:|:----------------|------------------:|
| t0 known   | blind (no video)                        |         20 |    0.4  |     0.65 | 0.41-0.84       |              13   |
| t0 known   | siglip2-so400m-patch14-384 linear probe |         20 |    0.1  |     0.1  | 0.00-0.25       |              59.9 |
| t0 known   | siglip2-base-patch16-224 linear probe   |         20 |    0.05 |     0.15 | 0.00-0.33       |              47.9 |
| t0 known   | resnet50 linear probe                   |         20 |    0.1  |     0.2  | 0.05-0.39       |              63.7 |
| t0 unknown | blind (no video)                        |         20 |    0.2  |     0.5  | 0.29-0.72       |              15.5 |
| t0 unknown | siglip2-so400m-patch14-384 linear probe |         20 |    0.1  |     0.1  | 0.00-0.25       |              59.9 |
| t0 unknown | siglip2-base-patch16-224 linear probe   |         20 |    0.05 |     0.15 | 0.00-0.33       |              47.9 |
| t0 unknown | resnet50 linear probe                   |         20 |    0.1  |     0.2  | 0.05-0.39       |              63.7 |
