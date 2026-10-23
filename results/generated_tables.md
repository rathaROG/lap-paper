# Generated result tables

Single-call values are median microseconds per matrix.

| Case | JVRUST | JVHYBRID | JVCPP | JVX | JVS | JVC | SciPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square128 | 66.00 | 67.99 | 75.03 | 117.85 | 113.44 | 289.92 | 286.09 |
| square512 | 1154.27 | 1155.28 | 1199.35 | 3469.28 | 3401.31 | 6812.50 | 5562.75 |
| wide64x512 | 12.99 | 12.88 | 14.59 | 958.65 | 873.42 | 1939.21 | 89.05 |
| tall512x64 | 26.50 | 26.24 | 84.45 | 967.06 | 887.39 | 1911.43 | 172.49 |
| near256x320 | 80.50 | 80.28 | 85.75 | 190.42 | 185.61 | 4698.29 | 596.42 |
| blocks256x320 | 9704.65 | 9653.83 | 10871.62 | 10164.90 | 10145.95 | 6484.25 | 1540.40 |
| ties128x160 | 50.83 | 58.11 | 58.91 | 60.32 | 57.43 | 95.20 | 78.02 |
| optional128x160 | 14.95 | 15.33 | 25.16 | 526.17 | 431.37 | 2442.30 | 464.23 |
| optional256x320 | 72.35 | 69.87 | 86.90 | 3615.47 | 2895.25 | 16191.33 | 2263.69 |
| tight128x160 | 10.12 | 10.32 | 18.39 | 456.01 | 377.27 | 2820.47 | 1170.98 |
| negative128x160 | 14.42 | 15.14 | 24.40 | 991.37 | 806.95 | 3261.84 | 432.10 |
| optional_square128 | 89.97 | 86.59 | 112.39 | 185.62 | 182.69 | 597.88 | 495.40 |

Batch values are median milliseconds per whole batch.

| Case | Method | Median ms |
| --- | --- | ---: |
| batch1024x8x10 | jvcpp_batch_t1 | 0.664 |
| batch1024x8x10 | jvcpp_batch_t4 | 0.344 |
| batch1024x8x10 | jvhybrid_batch_t1 | 0.543 |
| batch1024x8x10 | jvhybrid_batch_t4 | 0.269 |
| batch1024x8x10 | jvrust_batch_t1 | 0.549 |
| batch1024x8x10 | jvrust_batch_t4 | 0.259 |
| batch1024x8x10 | jvrust_loop | 1.349 |
| batch1024x8x10 | jvx_batch_t1 | 7.108 |
| batch1024x8x10 | jvx_batch_t4 | 17.696 |
| batch1024x8x10 | scipy_loop | 3.253 |
| batch16x128x160_optional | jvcpp_batch_t1 | 0.534 |
| batch16x128x160_optional | jvcpp_batch_t4 | 0.258 |
| batch16x128x160_optional | jvhybrid_batch_t1 | 0.429 |
| batch16x128x160_optional | jvhybrid_batch_t4 | 0.229 |
| batch16x128x160_optional | jvrust_batch_t1 | 0.429 |
| batch16x128x160_optional | jvrust_batch_t4 | 0.226 |
| batch16x128x160_optional | jvrust_loop | 0.441 |
| batch16x128x160_optional | jvx_batch_t1 | 8.534 |
| batch16x128x160_optional | jvx_batch_t4 | 5.121 |
| batch16x128x160_optional | scipy_loop | 7.780 |
| batch64x64x80 | jvcpp_batch_t1 | 0.788 |
| batch64x64x80 | jvcpp_batch_t4 | 0.328 |
| batch64x64x80 | jvhybrid_batch_t1 | 0.715 |
| batch64x64x80 | jvhybrid_batch_t4 | 0.317 |
| batch64x64x80 | jvrust_batch_t1 | 0.716 |
| batch64x64x80 | jvrust_batch_t4 | 0.334 |
| batch64x64x80 | jvrust_loop | 0.774 |
| batch64x64x80 | jvx_batch_t1 | 1.710 |
| batch64x64x80 | jvx_batch_t4 | 3.208 |
| batch64x64x80 | scipy_loop | 2.762 |

Memory values are MiB; each row uses one fresh subprocess.

| Case | Method | Peak before | Peak after | Additional peak |
| --- | --- | ---: | ---: | ---: |
| required | jvrust | 83.301 | 83.453 | 0.152 |
| required | jvhybrid | 83.242 | 83.359 | 0.117 |
| required | jvcpp | 83.039 | 83.160 | 0.121 |
| required | jvx | 83.074 | 211.285 | 128.211 |
| required | scipy | 82.961 | 83.113 | 0.152 |
| optional | jvrust | 83.500 | 83.621 | 0.121 |
| optional | jvhybrid | 83.359 | 83.508 | 0.148 |
| optional | jvcpp | 83.254 | 83.359 | 0.105 |
| optional | jvx | 83.285 | 227.898 | 144.613 |
| optional | scipy | 82.957 | 227.703 | 144.746 |
