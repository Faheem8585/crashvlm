# Label agreement: claude_vs_sonnet_batch1

A = `annotations/claude_pass_a.json`, B = `annotations/check_sonnet.json`; clips labelled by both: 39. Timing tolerance for adjudication: 16 ms.

## Point events

                 event  n_clips  n_a  n_b  presence_agreement  presence_kappa  n_both  median_abs_diff_ms  within_8ms  within_16ms  within_32ms
          impact_onset       39   39   39               1.000           1.000      39                 0.0       1.000        1.000        1.000
   airbag_deploy_start       39   29   28               0.923           0.805      27                 0.0       0.926        1.000        1.000
       max_deformation       39   20   19               0.974           0.949      19                24.0       0.053        0.263        0.947
          head_contact       39   17   19               0.949           0.897      17                 8.0       0.706        0.882        0.941
windshield_crack_onset       39    7    1               0.846           0.215       1                 0.0       1.000        1.000        1.000
side_glass_break_onset       39    6   10               0.846           0.536       5                 8.0       0.600        1.000        1.000
            door_opens       39    0    1               0.974           0.000       0                 NaN         NaN          NaN          NaN

## Clip summary

              attribute  n_clips  agreement  kappa
            view_usable       39      0.872  0.253
frontal_airbag_deployed       39      0.974  0.953
curtain_airbag_deployed       39      0.436  0.037
   a_pillar_deformation       39      0.718  0.470
      windshield_damage       39      0.308 -0.022
            door_opened       39      0.744  0.506
     fire_or_fluid_leak       39      1.000  1.000
        visual_severity       39      0.718  0.476

## Items to adjudicate: 122

kind
summary     87
presence    19
timing      16
