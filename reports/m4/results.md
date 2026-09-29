# M4 results: VLM crash reports (test split, t0 unknown)

## Report quality

|                             | 0             |
|:----------------------------|:--------------|
| run                         | qwen3.5-4b_zs |
| clips                       | 46            |
| parse_rate                  | 1.0           |
| schema_issues_per_clip      | 0.0           |
| test_type_acc (NHTSA)       | 0.37          |
| frontal_airbag_deployed_acc | 0.783         |
| curtain_airbag_deployed_acc | 0.696         |
| fire_hallucination_rate     | 0.0           |
| event_claims                | 94            |
| claims_contradicting_labels | 0.106         |
| claims_impossible_for_view  | 0.053         |
| latency_s_median            | 104.495       |
| peak_vram_gb_max            | 11.468        |

## Events (compare with the M3 't0 unknown' table)

| event                  | run           |   n |   pos |   F1 |   hit@8 |   hit@16 | hit@16 95% CI   |   median |err| ms |
|:-----------------------|:--------------|----:|------:|-----:|--------:|---------:|:----------------|------------------:|
| airbag_deploy_start    | qwen3.5-4b_zs |  22 |    21 | 0.89 |    0.7  |     0.75 | 0.55-0.90       |                 0 |
| head_contact           | qwen3.5-4b_zs |  22 |    18 | 0.67 |    0.33 |     0.39 | 0.17-0.60       |                 8 |
| impact_onset           | qwen3.5-4b_zs |  23 |    23 | 0.98 |    0.2  |     0.2  | 0.05-0.38       |                40 |
| side_glass_break_onset | qwen3.5-4b_zs |  27 |    11 | 0.43 |    0.18 |     0.27 | 0.00-0.50       |                 8 |
| windshield_crack_onset | qwen3.5-4b_zs |  45 |     8 | 0.7  |    0.88 |     0.88 | 0.60-1.00       |                 0 |
