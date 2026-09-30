# Label agreement: crossannotator_batch2

A = `annotations/labels_v1.json`, B = `annotations/check_crossannotator.json` (batch 2); clips labelled by both: 36. Timing tolerance for adjudication: 16 ms.

## Point events

                 event  n_clips  n_a  n_b  presence_agreement  presence_kappa  n_both  median_abs_diff_ms  within_8ms  within_16ms  within_32ms
          impact_onset       36   36   36               1.000           1.000      36                 0.0       0.972        0.972        0.972
   airbag_deploy_start       36   26   16               0.722           0.471      16                 0.0       0.875        1.000        1.000
       max_deformation       36   18   18               1.000           1.000      18                16.0       0.167        0.556        1.000
          head_contact       36   16   16               0.889           0.775      14                16.0       0.429        0.857        1.000
windshield_crack_onset       36    7    6               0.917           0.719       5                 0.0       1.000        1.000        1.000
side_glass_break_onset       36    8    3               0.861           0.483       3                 8.0       1.000        1.000        1.000
            door_opens       36    0    1               0.972           0.000       0                 NaN         NaN          NaN          NaN

## Clip summary

              attribute  n_clips  agreement  kappa
            view_usable       36      0.806  0.000
frontal_airbag_deployed       36      0.667  0.509
curtain_airbag_deployed       36      0.611  0.406
   a_pillar_deformation       36      0.444  0.151
      windshield_damage       36      0.194  0.013
            door_opened       36      0.917  0.838
     fire_or_fluid_leak       36      1.000  1.000
        visual_severity       36      0.472  0.170

## Items to adjudicate: 138

kind
summary     104
presence     23
timing       11
