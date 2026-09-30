# Label agreement: repeat_iaa

A = `annotations/labels_v1.json`, B = `annotations/check_repeat.json`; clips labelled by both: 30. Timing tolerance for adjudication: 16 ms.

## Point events

                 event  n_clips  n_a  n_b  presence_agreement  presence_kappa  n_both  median_abs_diff_ms  within_8ms  within_16ms  within_32ms
          impact_onset       30   30   30               1.000           1.000      30                 0.0       1.000        1.000          1.0
   airbag_deploy_start       30   18   19               0.967           0.930      18                 0.0       1.000        1.000          1.0
       max_deformation       30   15   15               1.000           1.000      15                 0.0       1.000        1.000          1.0
          head_contact       30   11   13               0.933           0.862      11                 8.0       0.909        0.909          1.0
windshield_crack_onset       30    7   11               0.733           0.378       5                 0.0       1.000        1.000          1.0
side_glass_break_onset       30    8   10               0.800           0.526       6                 0.0       0.833        0.833          1.0
            door_opens       30    0    0               1.000           1.000       0                 NaN         NaN          NaN          NaN

## Clip summary

              attribute  n_clips  agreement  kappa
            view_usable       30      1.000  1.000
frontal_airbag_deployed       30      0.867  0.793
curtain_airbag_deployed       30      0.867  0.800
   a_pillar_deformation       30      0.533  0.195
      windshield_damage       30      0.733  0.580
            door_opened       30      0.767  0.500
     fire_or_fluid_leak       30      1.000  1.000
        visual_severity       30      0.967  0.932

## Items to adjudicate: 57

kind
summary     38
presence    17
timing       2
