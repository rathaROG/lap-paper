# Generated result tables

Single-call values are median microseconds per matrix.

| Case | JVRUST | JVHYBRID | JVCPP | JVX | JVS | JVC | SciPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square128 | 69.83 | 72.06 | 78.05 | 126.40 | 123.79 | 312.63 | 313.69 |
| square512 | 1206.09 | 1222.33 | 1278.50 | 3691.12 | 3695.19 | 7300.23 | 5896.75 |
| wide64x512 | 13.65 | 13.57 | 14.90 | 1008.63 | 903.39 | 2115.31 | 92.38 |
| tall512x64 | 29.34 | 28.52 | 89.81 | 1040.55 | 965.23 | 2252.74 | 174.70 |
| near256x320 | 84.21 | 85.35 | 87.37 | 203.03 | 201.27 | 5002.55 | 661.64 |
| blocks256x320 | 10261.50 | 10521.58 | 11485.38 | 11060.58 | 10955.17 | 6920.70 | 1697.41 |
| ties128x160 | 61.20 | 64.26 | 66.00 | 71.56 | 68.73 | 107.97 | 91.13 |
| optional128x160 | 15.65 | 16.60 | 24.89 | 550.32 | 456.33 | 2574.64 | 522.59 |
| optional256x320 | 85.29 | 83.61 | 92.78 | 4301.48 | 3368.38 | 18277.62 | 2709.11 |
| tight128x160 | 10.77 | 11.47 | 18.53 | 503.39 | 412.44 | 3145.36 | 1239.18 |
| negative128x160 | 17.20 | 17.99 | 25.79 | 1116.55 | 941.58 | 3708.12 | 517.90 |
| optional_square128 | 101.53 | 103.68 | 120.93 | 224.59 | 215.80 | 702.62 | 619.33 |

Batch values are median milliseconds per whole batch.

| Case | Method | Median ms |
| --- | --- | ---: |
| batch1024x8x10 | jvcpp_batch_t1 | 0.671 |
| batch1024x8x10 | jvcpp_batch_t4 | 0.404 |
| batch1024x8x10 | jvhybrid_batch_t1 | 0.589 |
| batch1024x8x10 | jvhybrid_batch_t4 | 0.294 |
| batch1024x8x10 | jvrust_batch_t1 | 0.584 |
| batch1024x8x10 | jvrust_batch_t4 | 0.276 |
| batch1024x8x10 | jvrust_loop | 1.310 |
| batch1024x8x10 | jvx_batch_t1 | 6.680 |
| batch1024x8x10 | jvx_batch_t4 | 20.275 |
| batch1024x8x10 | scipy_loop | 3.637 |
| batch16x128x160_optional | jvcpp_batch_t1 | 0.519 |
| batch16x128x160_optional | jvcpp_batch_t4 | 0.249 |
| batch16x128x160_optional | jvhybrid_batch_t1 | 0.450 |
| batch16x128x160_optional | jvhybrid_batch_t4 | 0.238 |
| batch16x128x160_optional | jvrust_batch_t1 | 0.436 |
| batch16x128x160_optional | jvrust_batch_t4 | 0.248 |
| batch16x128x160_optional | jvrust_loop | 0.463 |
| batch16x128x160_optional | jvx_batch_t1 | 9.362 |
| batch16x128x160_optional | jvx_batch_t4 | 5.681 |
| batch16x128x160_optional | scipy_loop | 8.748 |
| batch64x64x80 | jvcpp_batch_t1 | 0.793 |
| batch64x64x80 | jvcpp_batch_t4 | 0.389 |
| batch64x64x80 | jvhybrid_batch_t1 | 0.745 |
| batch64x64x80 | jvhybrid_batch_t4 | 0.339 |
| batch64x64x80 | jvrust_batch_t1 | 0.754 |
| batch64x64x80 | jvrust_batch_t4 | 0.332 |
| batch64x64x80 | jvrust_loop | 0.810 |
| batch64x64x80 | jvx_batch_t1 | 1.783 |
| batch64x64x80 | jvx_batch_t4 | 3.302 |
| batch64x64x80 | scipy_loop | 3.094 |

Memory values are MiB; each row uses one fresh subprocess.

| Case | Method | Peak before | Peak after | Additional peak |
| --- | --- | ---: | ---: | ---: |
| required | jvrust | 81.699 | 81.863 | 0.164 |
| required | jvhybrid | 81.785 | 81.953 | 0.168 |
| required | jvcpp | 81.453 | 81.617 | 0.164 |
| required | jvx | 81.363 | 209.543 | 128.180 |
| required | scipy | 81.418 | 81.559 | 0.141 |
| optional | jvrust | 81.762 | 81.930 | 0.168 |
| optional | jvhybrid | 82.020 | 82.188 | 0.168 |
| optional | jvcpp | 81.574 | 81.738 | 0.164 |
| optional | jvx | 81.574 | 226.266 | 144.691 |
| optional | scipy | 81.484 | 226.289 | 144.805 |
