# Time of maximum crush from video (optical flow), frontal, val split

n = 6 frontal tests with a verified time axis and sensor result; full clips (held out: not used for design). Flow bias correction (median training error): +3.1 ms.

- flow: detected 1.00, median error -2.3 ms, median |error| 3.2 ms, within 8 ms 1.00, within 16 ms 1.00, Spearman rho with physics 0.78 (p = 0.066)
- blind: detected 1.00, median error +1.2 ms, median |error| 3.8 ms, within 8 ms 0.67, within 16 ms 0.83, Spearman rho with physics 0.85 (p = 0.034)

|   test_no |   physics_ms |   flow_ms |   blind_ms |
|----------:|-------------:|----------:|-----------:|
|      7620 |         56.7 |      61.9 |       69.6 |
|      8204 |         72.5 |      74.9 |       70.6 |
|     10961 |         77.1 |      71.9 |      107.6 |
|     11066 |         66.4 |      62.9 |       70.6 |
|     14065 |         74   |      70.9 |       70.6 |
|     15410 |         72.5 |      70.9 |       70.6 |
