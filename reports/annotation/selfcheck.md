# Label agreement: selfcheck

A = `annotations/claude_pass_a.json`, B = `annotations/claude_pass_a.json`; clips labelled by both: 40. Timing tolerance for adjudication: 16 ms.

## Point events

                 event  n_clips  n_a  n_b  presence_agreement  presence_kappa  n_both  median_abs_diff_ms  within_8ms  within_16ms  within_32ms
          impact_onset       40   40   40                 1.0             1.0      40                 0.0         1.0          1.0          1.0
   airbag_deploy_start       40   30   30                 1.0             1.0      30                 0.0         1.0          1.0          1.0
       max_deformation       40   20   20                 1.0             1.0      20                 0.0         1.0          1.0          1.0
          head_contact       40   18   18                 1.0             1.0      18                 0.0         1.0          1.0          1.0
windshield_crack_onset       40    7    7                 1.0             1.0       7                 0.0         1.0          1.0          1.0
side_glass_break_onset       40    6    6                 1.0             1.0       6                 0.0         1.0          1.0          1.0
            door_opens       40    0    0                 1.0             1.0       0                 NaN         NaN          NaN          NaN

## Clip summary

              attribute  n_clips  agreement  kappa
            view_usable       40        1.0    1.0
frontal_airbag_deployed       40        1.0    1.0
curtain_airbag_deployed       40        1.0    1.0
   a_pillar_deformation       40        1.0    1.0
      windshield_damage       40        1.0    1.0
            door_opened       40        1.0    1.0
     fire_or_fluid_leak       40        1.0    1.0
        visual_severity       40        1.0    1.0

## Items to adjudicate: 0

none
