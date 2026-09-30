# Paired comparison with the blind baseline (test split, t0 unknown)

hit@k on the same clips; difference = method minus blind; bootstrap over tests.

| event                  |   tol_ms | method                                  |   n |   method_hit |   blind_hit |   diff | diff 95% CI    | CI excludes 0   |
|:-----------------------|---------:|:----------------------------------------|----:|-------------:|------------:|-------:|:---------------|:----------------|
| airbag_deploy_start    |        8 | qwen3-vl-8b-4bit_zs                     |  20 |         0.9  |        0.25 |   0.65 | +0.35 to +0.85 | True            |
| airbag_deploy_start    |        8 | qwen3.5-4b_lora                         |  20 |         0.8  |        0.25 |   0.55 | +0.25 to +0.80 | True            |
| airbag_deploy_start    |        8 | qwen3.5-4b_zs                           |  20 |         0.7  |        0.25 |   0.45 | +0.15 to +0.70 | True            |
| airbag_deploy_start    |        8 | resnet50 linear probe                   |  20 |         0.3  |        0.25 |   0.05 | -0.25 to +0.35 | False           |
| airbag_deploy_start    |        8 | siglip2-so400m-patch14-384 linear probe |  20 |         0.55 |        0.25 |   0.3  | +0.00 to +0.60 | False           |
| airbag_deploy_start    |       16 | qwen3-vl-8b-4bit_zs                     |  20 |         0.9  |        0.75 |   0.15 | -0.10 to +0.40 | False           |
| airbag_deploy_start    |       16 | qwen3.5-4b_lora                         |  20 |         0.9  |        0.75 |   0.15 | -0.10 to +0.40 | False           |
| airbag_deploy_start    |       16 | qwen3.5-4b_zs                           |  20 |         0.75 |        0.75 |   0    | -0.25 to +0.30 | False           |
| airbag_deploy_start    |       16 | resnet50 linear probe                   |  20 |         0.45 |        0.75 |  -0.3  | -0.60 to +0.00 | False           |
| airbag_deploy_start    |       16 | siglip2-so400m-patch14-384 linear probe |  20 |         0.7  |        0.75 |  -0.05 | -0.35 to +0.25 | False           |
| head_contact           |        8 | qwen3-vl-8b-4bit_zs                     |  18 |         0.11 |        0.22 |  -0.11 | -0.33 to +0.17 | False           |
| head_contact           |        8 | qwen3.5-4b_lora                         |  18 |         0.56 |        0.22 |   0.33 | +0.00 to +0.67 | False           |
| head_contact           |        8 | qwen3.5-4b_zs                           |  18 |         0.33 |        0.22 |   0.11 | -0.17 to +0.39 | False           |
| head_contact           |        8 | resnet50 linear probe                   |  18 |         0.22 |        0.22 |   0    | -0.28 to +0.22 | False           |
| head_contact           |        8 | siglip2-so400m-patch14-384 linear probe |  18 |         0.17 |        0.22 |  -0.06 | -0.33 to +0.22 | False           |
| head_contact           |       16 | qwen3-vl-8b-4bit_zs                     |  18 |         0.17 |        0.67 |  -0.5  | -0.78 to -0.22 | True            |
| head_contact           |       16 | qwen3.5-4b_lora                         |  18 |         0.83 |        0.67 |   0.17 | -0.17 to +0.44 | False           |
| head_contact           |       16 | qwen3.5-4b_zs                           |  18 |         0.39 |        0.67 |  -0.28 | -0.61 to +0.06 | False           |
| head_contact           |       16 | resnet50 linear probe                   |  18 |         0.44 |        0.67 |  -0.22 | -0.56 to +0.11 | False           |
| head_contact           |       16 | siglip2-so400m-patch14-384 linear probe |  18 |         0.5  |        0.67 |  -0.17 | -0.56 to +0.22 | False           |
| impact_onset           |        8 | qwen3-vl-8b-4bit_zs                     |  20 |         0.2  |        0.35 |  -0.15 | -0.40 to +0.15 | False           |
| impact_onset           |        8 | qwen3.5-4b_lora                         |  20 |         0.65 |        0.35 |   0.3  | +0.10 to +0.50 | True            |
| impact_onset           |        8 | qwen3.5-4b_zs                           |  20 |         0.2  |        0.35 |  -0.15 | -0.40 to +0.10 | False           |
| impact_onset           |        8 | resnet50 linear probe                   |  20 |         0.45 |        0.35 |   0.1  | -0.20 to +0.45 | False           |
| impact_onset           |        8 | siglip2-so400m-patch14-384 linear probe |  20 |         0.2  |        0.35 |  -0.15 | -0.40 to +0.10 | False           |
| impact_onset           |       16 | qwen3-vl-8b-4bit_zs                     |  20 |         0.4  |        0.75 |  -0.35 | -0.60 to -0.05 | True            |
| impact_onset           |       16 | qwen3.5-4b_lora                         |  20 |         0.9  |        0.75 |   0.15 | -0.10 to +0.35 | False           |
| impact_onset           |       16 | qwen3.5-4b_zs                           |  20 |         0.2  |        0.75 |  -0.55 | -0.80 to -0.30 | True            |
| impact_onset           |       16 | resnet50 linear probe                   |  20 |         0.75 |        0.75 |   0    | -0.25 to +0.25 | False           |
| impact_onset           |       16 | siglip2-so400m-patch14-384 linear probe |  20 |         0.45 |        0.75 |  -0.3  | -0.50 to -0.10 | True            |
| side_glass_break_onset |        8 | qwen3-vl-8b-4bit_zs                     |  11 |         0.09 |        0.18 |  -0.09 | -0.25 to +0.00 | False           |
| side_glass_break_onset |        8 | qwen3.5-4b_lora                         |  11 |         0.27 |        0.18 |   0.09 | +0.00 to +0.25 | False           |
| side_glass_break_onset |        8 | qwen3.5-4b_zs                           |  11 |         0.18 |        0.18 |   0    | -0.23 to +0.25 | False           |
| side_glass_break_onset |        8 | resnet50 linear probe                   |  11 |         0.27 |        0.18 |   0.09 | -0.27 to +0.33 | False           |
| side_glass_break_onset |        8 | siglip2-so400m-patch14-384 linear probe |  11 |         0.27 |        0.18 |   0.09 | -0.18 to +0.40 | False           |
| side_glass_break_onset |       16 | qwen3-vl-8b-4bit_zs                     |  11 |         0.09 |        0.45 |  -0.36 | -0.60 to -0.11 | True            |
| side_glass_break_onset |       16 | qwen3.5-4b_lora                         |  11 |         0.36 |        0.45 |  -0.09 | -0.50 to +0.18 | False           |
| side_glass_break_onset |       16 | qwen3.5-4b_zs                           |  11 |         0.27 |        0.45 |  -0.18 | -0.44 to +0.00 | False           |
| side_glass_break_onset |       16 | resnet50 linear probe                   |  11 |         0.27 |        0.45 |  -0.18 | -0.62 to +0.15 | False           |
| side_glass_break_onset |       16 | siglip2-so400m-patch14-384 linear probe |  11 |         0.45 |        0.45 |   0    | -0.40 to +0.42 | False           |
| windshield_crack_onset |        8 | qwen3-vl-8b-4bit_zs                     |   8 |         0.88 |        0.5  |   0.38 | -0.25 to +0.88 | False           |
| windshield_crack_onset |        8 | qwen3.5-4b_lora                         |   8 |         0.75 |        0.5  |   0.25 | -0.25 to +0.50 | False           |
| windshield_crack_onset |        8 | qwen3.5-4b_zs                           |   8 |         0.88 |        0.5  |   0.38 | +0.12 to +0.50 | True            |
| windshield_crack_onset |        8 | resnet50 linear probe                   |   8 |         0.25 |        0.5  |  -0.25 | -0.50 to +0.00 | False           |
| windshield_crack_onset |        8 | siglip2-so400m-patch14-384 linear probe |   8 |         0.38 |        0.5  |  -0.12 | -0.75 to +0.50 | False           |
| windshield_crack_onset |       16 | qwen3-vl-8b-4bit_zs                     |   8 |         0.88 |        0.75 |   0.12 | -0.38 to +0.75 | False           |
| windshield_crack_onset |       16 | qwen3.5-4b_lora                         |   8 |         0.75 |        0.75 |   0    | -0.38 to +0.38 | False           |
| windshield_crack_onset |       16 | qwen3.5-4b_zs                           |   8 |         0.88 |        0.75 |   0.12 | +0.00 to +0.38 | False           |
| windshield_crack_onset |       16 | resnet50 linear probe                   |   8 |         0.5  |        0.75 |  -0.25 | -0.50 to +0.25 | False           |
| windshield_crack_onset |       16 | siglip2-so400m-patch14-384 linear probe |   8 |         0.62 |        0.75 |  -0.12 | -0.75 to +0.62 | False           |
