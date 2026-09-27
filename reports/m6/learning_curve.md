# Learning curve: SigLIP 2 base linear probes, test split, t0 unknown

Mean (std) over 3 random subsets of training tests; 100 % is one run.

|                                      |   ('F1', 'mean') |   ('F1', 'std') |   ('hit@16', 'mean') |   ('hit@16', 'std') |
|:-------------------------------------|-----------------:|----------------:|---------------------:|--------------------:|
| ('airbag_deploy_start', 0.25, 16)    |            0.977 |           0     |                0.533 |               0.076 |
| ('airbag_deploy_start', 0.5, 31)     |            0.977 |           0     |                0.433 |               0.029 |
| ('airbag_deploy_start', 1.0, 62)     |            0.977 |         nan     |                0.6   |             nan     |
| ('head_contact', 0.25, 16)           |            0.891 |           0.016 |                0.444 |               0.111 |
| ('head_contact', 0.5, 31)            |            0.898 |           0.003 |                0.481 |               0.179 |
| ('head_contact', 1.0, 62)            |            0.895 |         nan     |                0.5   |             nan     |
| ('impact_onset', 0.25, 16)           |            1     |           0     |                0.417 |               0.029 |
| ('impact_onset', 0.5, 31)            |            1     |           0     |                0.55  |               0.05  |
| ('impact_onset', 1.0, 62)            |            1     |         nan     |                0.55  |             nan     |
| ('side_glass_break_onset', 0.25, 16) |            0.448 |           0.131 |                0.03  |               0.052 |
| ('side_glass_break_onset', 0.5, 31)  |            0.532 |           0.017 |                0.182 |               0     |
| ('side_glass_break_onset', 1.0, 62)  |            0.583 |         nan     |                0.273 |             nan     |
| ('windshield_crack_onset', 0.25, 16) |            0.827 |           0.029 |                0.5   |               0     |
| ('windshield_crack_onset', 0.5, 31)  |            0.839 |           0.104 |                0.5   |               0.125 |
| ('windshield_crack_onset', 1.0, 62)  |            0.737 |         nan     |                0.625 |             nan     |
