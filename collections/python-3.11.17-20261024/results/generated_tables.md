# Generated result tables

Single-call values are median microseconds per matrix.

| Case | JVRUST | JVHYBRID | JVCPP | JVX | JVS | JVC | SciPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square128 | 59.71 | 61.22 | 71.42 | 111.37 | 109.65 | 260.29 | 268.04 |
| square512 | 1033.70 | 1047.44 | 1111.29 | 3265.40 | 3276.91 | 6031.73 | 5066.82 |
| wide64x512 | 11.26 | 11.31 | 13.80 | 866.98 | 815.35 | 1803.09 | 76.49 |
| tall512x64 | 23.76 | 23.28 | 78.40 | 887.86 | 832.86 | 1945.91 | 148.59 |
| near256x320 | 72.11 | 71.52 | 81.86 | 171.94 | 173.29 | 4014.82 | 559.79 |
| blocks256x320 | 8991.83 | 8744.50 | 9801.58 | 9417.83 | 9580.05 | 5601.48 | 1443.51 |
| ties128x160 | 49.30 | 53.29 | 52.25 | 59.57 | 60.04 | 93.52 | 77.25 |
| optional128x160 | 13.45 | 14.20 | 19.29 | 517.50 | 437.54 | 2178.79 | 454.00 |
| optional256x320 | 68.50 | 69.23 | 75.29 | 3652.79 | 3047.66 | 14849.92 | 2284.79 |
| tight128x160 | 9.17 | 9.43 | 14.28 | 451.08 | 378.58 | 2540.20 | 1083.24 |
| negative128x160 | 13.61 | 14.33 | 19.23 | 1021.79 | 848.29 | 3028.26 | 435.84 |
| optional_square128 | 86.70 | 88.23 | 96.25 | 194.29 | 192.23 | 590.52 | 516.45 |

Batch values are median milliseconds per whole batch.

| Case | Method | Median ms |
| --- | --- | ---: |
| batch1024x8x10 | jvcpp_batch_t1 | 0.508 |
| batch1024x8x10 | jvcpp_batch_t4 | 0.282 |
| batch1024x8x10 | jvhybrid_batch_t1 | 0.508 |
| batch1024x8x10 | jvhybrid_batch_t4 | 0.244 |
| batch1024x8x10 | jvrust_batch_t1 | 0.504 |
| batch1024x8x10 | jvrust_batch_t4 | 0.245 |
| batch1024x8x10 | jvrust_loop | 1.158 |
| batch1024x8x10 | jvx_batch_t1 | 5.873 |
| batch1024x8x10 | jvx_batch_t4 | 19.281 |
| batch1024x8x10 | scipy_loop | 2.803 |
| batch16x128x160_optional | jvcpp_batch_t1 | 0.439 |
| batch16x128x160_optional | jvcpp_batch_t4 | 0.241 |
| batch16x128x160_optional | jvhybrid_batch_t1 | 0.396 |
| batch16x128x160_optional | jvhybrid_batch_t4 | 0.219 |
| batch16x128x160_optional | jvrust_batch_t1 | 0.390 |
| batch16x128x160_optional | jvrust_batch_t4 | 0.214 |
| batch16x128x160_optional | jvrust_loop | 0.406 |
| batch16x128x160_optional | jvx_batch_t1 | 8.430 |
| batch16x128x160_optional | jvx_batch_t4 | 5.372 |
| batch16x128x160_optional | scipy_loop | 7.677 |
| batch64x64x80 | jvcpp_batch_t1 | 0.710 |
| batch64x64x80 | jvcpp_batch_t4 | 0.286 |
| batch64x64x80 | jvhybrid_batch_t1 | 0.658 |
| batch64x64x80 | jvhybrid_batch_t4 | 0.298 |
| batch64x64x80 | jvrust_batch_t1 | 0.655 |
| batch64x64x80 | jvrust_batch_t4 | 0.293 |
| batch64x64x80 | jvrust_loop | 0.728 |
| batch64x64x80 | jvx_batch_t1 | 1.586 |
| batch64x64x80 | jvx_batch_t4 | 2.931 |
| batch64x64x80 | scipy_loop | 2.707 |

Memory values are MiB; each row uses one fresh subprocess.

| Case | Method | Peak before | Peak after | Additional peak |
| --- | --- | ---: | ---: | ---: |
| required | jvrust | 78.793 | 78.957 | 0.164 |
| required | jvhybrid | 78.844 | 79.008 | 0.164 |
| required | jvcpp | 78.750 | 78.914 | 0.164 |
| required | jvx | 78.688 | 206.852 | 128.164 |
| required | scipy | 78.305 | 78.461 | 0.156 |
| optional | jvrust | 78.570 | 78.734 | 0.164 |
| optional | jvhybrid | 78.938 | 79.105 | 0.168 |
| optional | jvcpp | 78.730 | 78.891 | 0.160 |
| optional | jvx | 78.629 | 223.316 | 144.688 |
| optional | scipy | 78.707 | 223.520 | 144.812 |
