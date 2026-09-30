# Physics targets from NHTSA sensor data

Tests with a result: 94 / 100. Failures: 7639 (closing velocity stays above 10 % of v0 (min 2.18 m/s)), 8357 (closing velocity stays above 10 % of v0 (min 1.91 m/s)), 10977 (closing velocity stays above 10 % of v0 (min 7.88 m/s)), 15425 (no usable channel or v0), 15542 (MissingSchema: Invalid URL 'None': No scheme supplied. Perhaps you meant https://None?), 15628 (MissingSchema: Invalid URL 'None': No scheme supplied. Perhaps you meant https://None?)

## Time of maximum dynamic crush (ms after t0)

| config   |   count |   min |   25% |   50% |   75% |   max |
|:---------|--------:|------:|------:|------:|------:|------:|
| ITV      |      27 |  56.4 |  61.7 |  71.2 |  83.4 | 139.4 |
| VTB      |      38 |  49   |  66.4 |  70.8 |  72.8 |  82.2 |
| VTP      |      29 |  32.9 |  77.9 |  94.9 | 110.2 | 200   |

## Dynamic crush vs NHTSA residual crush

ITV: dynamic value = barrier-to-vehicle approach, which includes the deformable barrier face, so it is larger than vehicle crush by construction.

- ITV: Spearman rho = 0.21 (p = 0.283, n = 27); median dynamic/residual = 2.56
- VTB: Spearman rho = 0.31 (p = 0.062, n = 38); median dynamic/residual = 1.33
- VTP: Spearman rho = 0.64 (p = 0.000, n = 29); median dynamic/residual = 1.29

## How each time was obtained

|                             |   0 |
|:----------------------------|----:|
| ('ITV', 'velocity-minimum') |  10 |
| ('ITV', 'zero-crossing')    |  17 |
| ('VTB', 'zero-crossing')    |  38 |
| ('VTP', 'velocity-minimum') |   1 |
| ('VTP', 'zero-crossing')    |  28 |

## Visual max_deformation labels vs measured time of maximum crush

Per test type: median |label - physics| and share within 16 ms, next to a blind constant (leave-one-out median physics time of the same test type).

| labels               | config   |   n |   label median |err| ms |   label within 16 ms |   constant median |err| ms |   constant within 16 ms |
|:---------------------|:---------|----:|------------------------:|---------------------:|---------------------------:|------------------------:|
| primary annotator    | ITV      |  26 |                     8.7 |                 0.69 |                       12.7 |                    0.77 |
| primary annotator    | VTB      |  37 |                     3.5 |                 1    |                        3.4 |                    0.97 |
| primary annotator    | VTP      |  29 |                    22.4 |                 0.31 |                       16.3 |                    0.48 |
| blind cross-check    | ITV      |   8 |                    12.6 |                 0.62 |                       10.9 |                    0.75 |
| blind cross-check    | VTB      |  18 |                    23.1 |                 0.17 |                        4.9 |                    0.94 |
| blind cross-check    | VTP      |  10 |                     9.2 |                 0.6  |                       11   |                    0.6  |
