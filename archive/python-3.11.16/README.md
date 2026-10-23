# Python 3.11.16 results (superseded)

Complete results of the tests and runs made on 2026-10-22 with Python 3.11.16, NumPy 2.4.6 and SciPy 1.17.1 on Windows (Intel Core i9-13900KS, MSVC 14.44, Rust 1.99.1). They were archived before the rerun on Python 3.14.9. The main text reports the Python 3.14.9 run; Appendix A of the manuscript summarizes this one, and `check_artifact.py` verifies the appendix tables and statements against the files here.

| Folder | Contents |
| --- | --- |
| paper-results/ | The paper's complete `results/` folder from that run: timing samples and summaries, enumeration validation, package test validation and logs, formulation and scaling runs, run manifest, and per-CPU load |
| paper-snapshot/ | The manuscript, README and figures as built from those results |
| benchmark-results/ | Runs of the shared package benchmarks from the same day, comparing lapcpp, laprust, LAPX and SciPy (local only; git-ignored) |
| benchmarks/log.txt | Console log of a shared benchmark run (local only; git-ignored) |

The packages measured in `paper-results/` are recorded in `paper-results/runtime.json` (LAPX reported version 0.11.0rc1, lapcpp 1.0.0, laprust 1.0.0). The benchmark runs used the builds installed at the time of each run, as recorded in their JSON metadata.
