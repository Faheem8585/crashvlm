# Time of maximum crush from video (optical flow), frontal, test split

n = 6 frontal tests with a verified time axis and sensor result; clip starts cropped at random (0-40 ms). Flow bias correction (median training error): +3.1 ms.

- flow: detected 1.00, median error +0.9 ms, median |error| 0.9 ms, within 8 ms 1.00, within 16 ms 1.00, Spearman rho with physics 0.99 (p = 0.000)
- blind: detected 1.00, median error +31.7 ms, median |error| 31.7 ms, within 8 ms 0.00, within 16 ms 0.17, Spearman rho with physics 0.55 (p = 0.257)

|   test_no |   physics_ms |   flow_ms |   blind_ms |
|----------:|-------------:|----------:|-----------:|
|      8212 |         66.5 |      66.9 |       90.6 |
|      8949 |         71.8 |      70.9 |      106.6 |
|     10028 |         64.8 |      66.9 |       78.6 |
|     10039 |         49   |      50.9 |       94.6 |
|     10185 |         62.1 |      62.9 |       90.6 |
|     15519 |         74.1 |      74.9 |      110.6 |
