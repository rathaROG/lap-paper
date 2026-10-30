| Method | Replay float32 | Replay float64 | Centroid update | SORT update |
| --- | --- | --- | --- | --- |
| lapjv | 0.759--1.852 | 0.722--1.846 | 0.936--1.076 | 0.995--1.018 |
| lapjvx | 0.787--1.925 | 0.728--1.887 | 0.943--1.081 | 0.996--1.015 |
| lapjvxa | 0.872--2.053 | 0.830--1.966 | 0.973--1.072 | 0.990--1.042 |
| lapjvh | 2.041--3.145 | 1.878--2.795 | 1.063--1.147 | 1.008--1.034 |
| laprust | 2.032--3.259 | 1.918--2.739 | 1.051--1.142 | 0.984--1.049 |
| lapcpp | 2.002--3.065 | 1.858--2.626 | 1.042--1.132 | 0.982--1.037 |
| scipy | 1.000--1.000 | 1.000--1.000 | 1.000--1.000 | 1.000--1.000 |
| lapjvs | 0.773--2.057 | 0.733--1.898 | 0.926--1.081 | 0.986--1.033 |
| lapjvc | 0.758--0.995 | 0.713--0.927 | 0.969--1.014 | 0.992--1.029 |

**Table 12.** Real-sequence speedup ranges relative to SciPy, computed within each pipeline and originating session. Values are minima--maxima across pipelines of SciPy's nine-round median time divided by the method's nine-round median time; replay covers eight pipelines per precision, and tracker updates cover four per tracker. Ranges are descriptive extrema, not interquartile ranges.

| Pipeline | GT boxes | Det. boxes | Det./frame | LAP calls | Rows min--max | Cols min--max | Square % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOT16-04 / GT / Centroid | 51545 | 51545 | 51.60 | 998 | 47--56 | 47--56 | 94.0 |
| MOT16-04 / GT / SORT | 51545 | 51545 | 51.60 | 998 | 47--56 | 47--56 | 91.7 |
| MOT16-04 / RF768 / Centroid | 51545 | 30621 | 30.65 | 998 | 22--39 | 23--39 | 18.6 |
| MOT16-04 / RF768 / SORT | 51545 | 30621 | 30.65 | 976 | 23--39 | 24--42 | 7.2 |
| MOT20-03 / GT / Centroid | 106637 | 106637 | 106.74 | 998 | 84--134 | 84--134 | 74.9 |
| MOT20-03 / GT / SORT | 106637 | 106637 | 106.74 | 998 | 84--134 | 84--134 | 66.1 |
| MOT20-03 / RF768 / Centroid | 106637 | 45990 | 46.04 | 998 | 34--62 | 34--62 | 14.6 |
| MOT20-03 / RF768 / SORT | 106637 | 45990 | 46.04 | 575 | 36--62 | 39--66 | 8.5 |

**Table C1.** Real-sequence workload over 999 decoded frames per clip. GT and detector box totals count observations, not unique people. LAP dimensions and square percentages describe actual SciPy-captured assignment calls; SORT fast paths are retained. RF768 denotes Nano at 768 pixels and confidence 0.2, with person class 1, max_det 300 and minimum box width 15 pixels.

| Pipeline | scipy | lapjvh | laprust | lapcpp |
| --- | --- | --- | --- | --- |
| MOT16-04 / GT / Centroid | 0.121 [0.118, 0.124] | 0.106 [0.106, 0.107] | 0.112 [0.110, 0.114] | 0.107 [0.107, 0.108] |
| MOT16-04 / GT / SORT | 1.518 [1.489, 1.524] | 1.490 [1.467, 1.498] | 1.447 [1.440, 1.484] | 1.464 [1.451, 1.503] |
| MOT16-04 / RF768 / Centroid | 0.071 [0.071, 0.076] | 0.067 [0.066, 0.069] | 0.068 [0.066, 0.072] | 0.068 [0.066, 0.070] |
| MOT16-04 / RF768 / SORT | 0.992 [0.986, 1.016] | 0.985 [0.974, 1.021] | 1.009 [0.974, 1.019] | 1.011 [0.968, 1.019] |
| MOT20-03 / GT / Centroid | 0.308 [0.300, 0.325] | 0.269 [0.269, 0.274] | 0.270 [0.268, 0.272] | 0.272 [0.271, 0.274] |
| MOT20-03 / GT / SORT | 3.079 [3.040, 3.127] | 2.977 [2.968, 3.036] | 3.005 [2.990, 3.041] | 2.992 [2.989, 3.039] |
| MOT20-03 / RF768 / Centroid | 0.109 [0.106, 0.114] | 0.101 [0.097, 0.104] | 0.097 [0.096, 0.097] | 0.100 [0.097, 0.103] |
| MOT20-03 / RF768 / SORT | 1.432 [1.405, 1.464] | 1.412 [1.391, 1.471] | 1.410 [1.402, 1.439] | 1.414 [1.403, 1.442] |

**Table C2.** Tracker update time in milliseconds per frame: median [Q1, Q3] across nine whole-clip means. Fresh Persons, decoding, detector inference, result copying and evaluation are excluded. The frozen artifact includes all nine methods; these absolute times show SciPy and the three native implementations.

| Pipeline / precision | scipy | lapjvh | laprust | lapcpp |
| --- | --- | --- | --- | --- |
| MOT16-04 / GT / Centroid / float32 | 14.37 [14.22, 15.10] | 5.35 [5.34, 5.70] | 5.41 [5.37, 5.73] | 5.73 [5.62, 6.01] |
| MOT16-04 / GT / Centroid / float64 | 14.26 [13.54, 14.82] | 5.61 [5.56, 6.07] | 5.60 [5.59, 5.74] | 6.15 [5.96, 6.50] |
| MOT16-04 / GT / SORT / float32 | 14.40 [14.34, 14.58] | 5.56 [5.55, 5.85] | 5.76 [5.61, 5.96] | 5.75 [5.70, 5.96] |
| MOT16-04 / GT / SORT / float64 | 13.83 [13.76, 14.10] | 5.76 [5.73, 5.80] | 5.98 [5.77, 6.08] | 6.08 [6.05, 6.10] |
| MOT16-04 / RF768 / Centroid / float32 | 8.68 [8.65, 8.69] | 4.25 [4.24, 4.39] | 4.27 [4.26, 4.30] | 4.33 [4.30, 4.37] |
| MOT16-04 / RF768 / Centroid / float64 | 8.26 [8.22, 8.62] | 4.40 [4.26, 4.46] | 4.30 [4.29, 4.47] | 4.44 [4.29, 4.46] |
| MOT16-04 / RF768 / SORT / float32 | 9.55 [9.45, 10.01] | 4.05 [4.02, 4.17] | 4.10 [4.05, 4.29] | 4.20 [4.18, 4.35] |
| MOT16-04 / RF768 / SORT / float64 | 9.24 [9.07, 9.47] | 4.05 [4.04, 4.26] | 4.05 [4.04, 4.06] | 4.20 [4.13, 4.29] |
| MOT20-03 / GT / Centroid / float32 | 45.87 [45.43, 47.25] | 15.81 [15.64, 16.89] | 15.61 [15.51, 16.05] | 16.93 [16.48, 17.55] |
| MOT20-03 / GT / Centroid / float64 | 45.22 [44.60, 50.03] | 18.25 [16.81, 18.73] | 16.99 [16.68, 17.31] | 18.39 [17.91, 18.91] |
| MOT20-03 / GT / SORT / float32 | 48.55 [46.34, 48.97] | 15.44 [14.98, 15.86] | 14.90 [14.77, 15.61] | 15.84 [15.58, 16.52] |
| MOT20-03 / GT / SORT / float64 | 45.72 [45.27, 47.11] | 16.36 [16.16, 16.93] | 16.69 [16.08, 16.92] | 17.41 [16.90, 18.06] |
| MOT20-03 / RF768 / Centroid / float32 | 15.32 [14.88, 16.23] | 6.37 [6.17, 7.05] | 6.37 [6.23, 7.28] | 6.67 [6.58, 7.36] |
| MOT20-03 / RF768 / Centroid / float64 | 14.02 [13.64, 14.33] | 6.42 [6.33, 6.80] | 6.29 [6.26, 6.86] | 6.53 [6.22, 6.71] |
| MOT20-03 / RF768 / SORT / float32 | 15.02 [14.98, 15.25] | 5.41 [5.40, 5.41] | 5.47 [5.45, 5.54] | 5.70 [5.67, 5.72] |
| MOT20-03 / RF768 / SORT / float64 | 14.57 [14.49, 14.68] | 5.54 [5.53, 5.54] | 5.55 [5.52, 5.57] | 5.59 [5.56, 5.60] |

**Table C3.** Common-cost replay time in microseconds per assignment: median [Q1, Q3] over nine corpus means. Every method receives the same captured pipeline corpus. The float64 view widens the captured float32 costs; cost construction is unchanged. Adapter conversion, finite checks and pair normalization remain inside replay timing.

| Pipeline | IDF1 % min--max | MOTA % min--max | IDSW min--max | FP min--max | FN min--max |
| --- | --- | --- | --- | --- | --- |
| MOT16-04 / GT / Centroid | 99.48--100.00 | 100.00--100.00 | 0--1 | 0--0 | 0--0 |
| MOT16-04 / GT / SORT | 99.12--99.12 | 99.78--99.78 | 0--0 | 0--0 | 115--115 |
| MOT16-04 / RF768 / Centroid | 28.07--28.44 | 40.41--40.42 | 1568--1576 | 4108--4109 | 25032--25033 |
| MOT16-04 / RF768 / SORT | 34.99--34.99 | 40.60--40.60 | 535--535 | 2273--2273 | 27811--27811 |
| MOT20-03 / GT / Centroid | 100.00--100.00 | 100.00--100.00 | 0--0 | 0--0 | 0--0 |
| MOT20-03 / GT / SORT | 99.76--99.76 | 99.51--99.51 | 0--0 | 0--0 | 518--518 |
| MOT20-03 / RF768 / Centroid | 28.50--28.90 | 38.13--38.13 | 2990--2993 | 1168--1168 | 61815--61815 |
| MOT20-03 / RF768 / SORT | 36.75--36.75 | 35.58--35.58 | 729--729 | 619--619 | 67348--67348 |

**Table C4.** Supplied-box accuracy ranges across all nine assignment methods. IDF1 and MOTA are percentages; IDSW, FP and FN are counts. Evaluation uses motmetrics 1.4.0 and IoU at least 0.5 outside all timing. Converted annotations lack official MOTChallenge class/ignore metadata, so these are not leaderboard scores. Equal-cost assignment choices can propagate to different identities; no method-independent tracking-accuracy gain is claimed.
