# Time of maximum crush from video (optical flow), frontal, train split

n = 23 frontal tests with a verified time axis and sensor result; full clips (design split). Flow bias correction (median training error): +3.1 ms.

- flow: detected 1.00, median error +0.0 ms, median |error| 2.3 ms, within 8 ms 0.96, within 16 ms 1.00, Spearman rho with physics 0.71 (p = 0.000)
- blind: detected 1.00, median error +0.0 ms, median |error| 2.9 ms, within 8 ms 0.78, within 16 ms 0.96, Spearman rho with physics -0.12 (p = 0.591)

|   test_no |   physics_ms |   flow_ms |   blind_ms |
|----------:|-------------:|----------:|-----------:|
|      7863 |         67.4 |      59.9 |       70.3 |
|      8156 |         70.6 |      70.9 |       70.6 |
|      8189 |         64   |      65.5 |      117.2 |
|      8297 |         71.2 |      70.9 |       70.6 |
|      9012 |         71.9 |      70.9 |       70.6 |
|      9077 |         63.1 |      64.7 |       72.3 |
|      9494 |         68.6 |      70.9 |       70.6 |
|      9557 |         60.6 |      70.9 |       70.6 |
|      9761 |         70.9 |      70.9 |       70.6 |
|      9961 |         70.7 |      74.9 |       70.6 |
|      9982 |         70.9 |      70.9 |       70.6 |
|     10168 |         70   |      66.9 |       70.6 |
|     10175 |         73.6 |      66.9 |       70.6 |
|     10671 |         75.5 |      75.7 |       71.4 |
|     10690 |         79.8 |      83.3 |       71   |
|     10768 |         75.6 |      74.9 |       70.6 |
|     10962 |         67.4 |      66.9 |       70.6 |
|     10963 |         69.7 |      67.4 |       71.1 |
|     11264 |         69.5 |      66.9 |       70.6 |
|     11350 |         82.2 |      78.9 |       70.6 |
|     11583 |         64.8 |      67.7 |       71.3 |
|     14394 |         63.5 |      66.9 |       70.6 |
|     15319 |         72.1 |      70.9 |       70.6 |
