# Temporal CNN over frozen embeddings (test split, t0 unknown)

Mean (std) over 3 seeds. Compare with the linear-probe rows in results.md.

|                                                        |   ('F1', 'mean') |   ('F1', 'std') |   ('hit@16', 'mean') |   ('hit@16', 'std') |
|:-------------------------------------------------------|-----------------:|----------------:|---------------------:|--------------------:|
| ('airbag_deploy_start', 'resnet50')                    |            0.977 |           0     |                0.75  |               0.05  |
| ('airbag_deploy_start', 'siglip2-base-patch16-224')    |            0.977 |           0     |                0.417 |               0.388 |
| ('head_contact', 'resnet50')                           |            0.854 |           0.035 |                0.426 |               0.064 |
| ('head_contact', 'siglip2-base-patch16-224')           |            0.891 |           0.016 |                0.315 |               0.14  |
| ('impact_onset', 'resnet50')                           |            1     |           0     |                0.75  |               0.18  |
| ('impact_onset', 'siglip2-base-patch16-224')           |            1     |           0     |                0.633 |               0.257 |
| ('side_glass_break_onset', 'resnet50')                 |            0.572 |           0.04  |                0.091 |               0.091 |
| ('side_glass_break_onset', 'siglip2-base-patch16-224') |            0.56  |           0.077 |                0.152 |               0.052 |
| ('windshield_crack_onset', 'resnet50')                 |            0.752 |           0.047 |                0.333 |               0.072 |
| ('windshield_crack_onset', 'siglip2-base-patch16-224') |            0.824 |           0     |                0.375 |               0.217 |
