# M4 results: VLM crash reports (test split, t0 unknown)

## Report quality

|                             | 0             | 1                   | 2               |
|:----------------------------|:--------------|:--------------------|:----------------|
| run                         | qwen3.5-4b_zs | qwen3-vl-8b-4bit_zs | qwen3.5-4b_lora |
| clips                       | 46            | 46                  | 46              |
| parse_rate                  | 1.0           | 1.0                 | 1.0             |
| schema_issues_per_clip      | 0.0           | 0.0                 | 0.0             |
| test_type_acc (NHTSA)       | 0.37          | 0.478               | 0.457           |
| frontal_airbag_deployed_acc | 0.783         | 0.717               | 0.587           |
| curtain_airbag_deployed_acc | 0.696         | 0.391               | 0.696           |
| fire_hallucination_rate     | 0.0           | 0.065               | 0.0             |
| event_claims                | 94            | 100                 | 117             |
| claims_contradicting_labels | 0.106         | 0.25                | 0.085           |
| claims_impossible_for_view  | 0.053         | 0.06                | 0.0             |
| latency_s_median            | 104.495       | 141.274             | 104.72          |
| peak_vram_gb_max            | 11.468        | 6.676               | 6.986           |

## Events (compare with the M3 't0 unknown' table)

| event                  | run                 |   n |   pos |   F1 |   hit@8 |   hit@16 | hit@16 95% CI   |   median |err| ms |
|:-----------------------|:--------------------|----:|------:|-----:|--------:|---------:|:----------------|------------------:|
| airbag_deploy_start    | qwen3-vl-8b-4bit_zs |  22 |    21 | 0.98 |    0.9  |     0.9  | 0.75-1.00       |                 0 |
| airbag_deploy_start    | qwen3.5-4b_lora     |  22 |    21 | 0.98 |    0.8  |     0.9  | 0.76-1.00       |                 8 |
| airbag_deploy_start    | qwen3.5-4b_zs       |  22 |    21 | 0.89 |    0.7  |     0.75 | 0.55-0.90       |                 0 |
| head_contact           | qwen3-vl-8b-4bit_zs |  22 |    18 | 0.48 |    0.11 |     0.17 | 0.00-0.36       |                20 |
| head_contact           | qwen3.5-4b_lora     |  22 |    18 | 0.9  |    0.56 |     0.83 | 0.63-1.00       |                 8 |
| head_contact           | qwen3.5-4b_zs       |  22 |    18 | 0.67 |    0.33 |     0.39 | 0.17-0.60       |                 8 |
| impact_onset           | qwen3-vl-8b-4bit_zs |  23 |    23 | 1    |    0.2  |     0.4  | 0.19-0.62       |                24 |
| impact_onset           | qwen3.5-4b_lora     |  23 |    23 | 1    |    0.65 |     0.9  | 0.75-1.00       |                 8 |
| impact_onset           | qwen3.5-4b_zs       |  23 |    23 | 0.98 |    0.2  |     0.2  | 0.05-0.38       |                40 |
| side_glass_break_onset | qwen3-vl-8b-4bit_zs |  27 |    11 | 0.17 |    0.09 |     0.09 | 0.00-0.33       |                 8 |
| side_glass_break_onset | qwen3.5-4b_lora     |  27 |    11 | 0.67 |    0.27 |     0.36 | 0.00-0.67       |                12 |
| side_glass_break_onset | qwen3.5-4b_zs       |  27 |    11 | 0.43 |    0.18 |     0.27 | 0.00-0.50       |                 8 |
| windshield_crack_onset | qwen3-vl-8b-4bit_zs |  45 |     8 | 0.41 |    0.88 |     0.88 | 0.50-1.00       |                 0 |
| windshield_crack_onset | qwen3.5-4b_lora     |  45 |     8 | 0.71 |    0.75 |     0.75 | 0.50-1.00       |                 0 |
| windshield_crack_onset | qwen3.5-4b_zs       |  45 |     8 | 0.7  |    0.88 |     0.88 | 0.60-1.00       |                 0 |
