# Generated result tables

Single-call values are median microseconds per matrix.

| Case | JVRUST | JVHYBRID | JVCPP | JVX | JVS | JVC | SciPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square128 | 68.47 | 69.82 | 78.82 | 127.52 | 120.93 | 296.40 | 294.05 |
| square512 | 1108.85 | 1123.00 | 1190.53 | 3385.24 | 3504.47 | 6294.77 | 5247.77 |
| wide64x512 | 13.01 | 12.56 | 15.02 | 957.28 | 890.37 | 1954.62 | 76.29 |
| tall512x64 | 26.01 | 26.20 | 86.26 | 953.45 | 874.88 | 1872.11 | 158.69 |
| near256x320 | 76.54 | 77.29 | 87.25 | 188.29 | 187.12 | 4379.26 | 588.21 |
| blocks256x320 | 9707.80 | 9776.38 | 10714.25 | 10442.83 | 10583.38 | 6280.60 | 1565.43 |
| ties128x160 | 56.20 | 61.41 | 59.67 | 68.46 | 65.97 | 104.75 | 84.46 |
| optional128x160 | 14.89 | 15.39 | 21.17 | 551.83 | 483.97 | 2292.61 | 447.76 |
| optional256x320 | 76.52 | 77.20 | 84.92 | 4053.53 | 3381.00 | 15973.10 | 2315.12 |
| tight128x160 | 9.96 | 10.43 | 15.53 | 487.20 | 417.84 | 2656.32 | 1169.17 |
| negative128x160 | 14.83 | 16.03 | 21.45 | 1095.88 | 957.44 | 3292.79 | 448.89 |
| optional_square128 | 91.49 | 95.30 | 107.96 | 213.91 | 207.66 | 656.27 | 563.53 |

Batch values are median milliseconds per whole batch.

| Case | Method | Median ms |
| --- | --- | ---: |
| batch1024x8x10 | jvcpp_batch_t1 | 0.565 |
| batch1024x8x10 | jvcpp_batch_t4 | 0.346 |
| batch1024x8x10 | jvhybrid_batch_t1 | 0.556 |
| batch1024x8x10 | jvhybrid_batch_t4 | 0.263 |
| batch1024x8x10 | jvrust_batch_t1 | 0.563 |
| batch1024x8x10 | jvrust_batch_t4 | 0.264 |
| batch1024x8x10 | jvrust_loop | 1.343 |
| batch1024x8x10 | jvx_batch_t1 | 7.413 |
| batch1024x8x10 | jvx_batch_t4 | 17.429 |
| batch1024x8x10 | scipy_loop | 3.181 |
| batch16x128x160_optional | jvcpp_batch_t1 | 0.486 |
| batch16x128x160_optional | jvcpp_batch_t4 | 0.242 |
| batch16x128x160_optional | jvhybrid_batch_t1 | 0.442 |
| batch16x128x160_optional | jvhybrid_batch_t4 | 0.225 |
| batch16x128x160_optional | jvrust_batch_t1 | 0.429 |
| batch16x128x160_optional | jvrust_batch_t4 | 0.223 |
| batch16x128x160_optional | jvrust_loop | 0.444 |
| batch16x128x160_optional | jvx_batch_t1 | 9.220 |
| batch16x128x160_optional | jvx_batch_t4 | 5.580 |
| batch16x128x160_optional | scipy_loop | 7.823 |
| batch64x64x80 | jvcpp_batch_t1 | 0.779 |
| batch64x64x80 | jvcpp_batch_t4 | 0.430 |
| batch64x64x80 | jvhybrid_batch_t1 | 0.718 |
| batch64x64x80 | jvhybrid_batch_t4 | 0.316 |
| batch64x64x80 | jvrust_batch_t1 | 0.725 |
| batch64x64x80 | jvrust_batch_t4 | 0.313 |
| batch64x64x80 | jvrust_loop | 0.789 |
| batch64x64x80 | jvx_batch_t1 | 1.795 |
| batch64x64x80 | jvx_batch_t4 | 3.213 |
| batch64x64x80 | scipy_loop | 2.824 |

Memory values are MiB; each row uses one fresh subprocess.

| Case | Method | Peak before | Peak after | Additional peak |
| --- | --- | ---: | ---: | ---: |
| required | jvrust | 83.211 | 83.336 | 0.125 |
| required | jvhybrid | 83.477 | 83.594 | 0.117 |
| required | jvcpp | 83.133 | 83.273 | 0.141 |
| required | jvx | 83.625 | 211.770 | 128.145 |
| required | scipy | 83.082 | 83.297 | 0.215 |
| optional | jvrust | 83.348 | 83.516 | 0.168 |
| optional | jvhybrid | 83.340 | 83.492 | 0.152 |
| optional | jvcpp | 83.156 | 83.289 | 0.133 |
| optional | jvx | 83.082 | 227.742 | 144.660 |
| optional | scipy | 83.129 | 227.875 | 144.746 |
