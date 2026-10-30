# Paper results

The manuscript uses four current measurement archives, all collected on
29 October 2026:

| Archive | Evidence used in the paper |
| --- | --- |
| [`controlled-windows-20261029.zip`](controlled-windows-20261029.zip) | Windows Python 3.11.18 and 3.14.10 with LAPX 0.13.0 and laprust/lapcpp 1.2.0; Tables 4-8 and A1-A4 |
| [`controlled-linux-20261029.zip`](controlled-linux-20261029.zip) | Linux Python 3.11.18 and 3.14.10 with the same solver versions; Tables B1-B6 and the current four-environment Table B7 |
| [`application-windows-20261029.zip`](application-windows-20261029.zip) | Cached tracking/replay with Python 3.11.18 and current solvers; Tables 12 and C1-C4 |
| [`lsape-windows-20261029.zip`](lsape-windows-20261029.zip) | Fresh Windows Python 3.11.18 build, strict verification and three-session LSAPE campaign; Tables 10-11 |

Each controlled archive contains 17,892 timing samples and 830 memory
observations across two complete 204-case environments. Each environment has
994 measured method/case settings, 8,946 timing samples and 415 memory
observations. All 3,925 comparable objectives and recorded cardinalities agree
exactly in each of the six environment pairs. Frozen input identities and
seeded measurement schedules also agree. The archives retain raw observations,
summaries, environments, native binary hashes, source identities and checksum
manifests. Calibration and smoke records are not publication measurements.

The Linux runs record CPU 0 for single-problem timing and logical CPUs
0, 2, 4 and 6 for batches, with one thread per numerical-library pool.
They retain one-second per-CPU utilization snapshots and the supplied terminal
transcript. CPU model, core-sibling mapping, governor and thermal/frequency
telemetry were not captured by this rerun. Utilization snapshots do not prove
isolation, and the four-environment comparison does not isolate interpreter,
compiler, wheel or operating-system effects. Nine-round IQRs are descriptive.

The application archive contains 648 tracker-update round means, 1,296 replay
round means and 7,539 captured matrices across eight pipelines. All 72
method/pipeline settings were interleaved within each of nine seeded-shuffled
rounds in one campaign. Its 150,780 backend/matrix checks have zero recorded
objective gap, including the untimed Rust alias. Saved-data and independent
direct-API audits are preserved with the raw observations. These are finite
corpus checks, not a general numerical proof. Frozen detector/GT caches were
reused; detector inference was not repeated.

The LSAPE archive retains 9,045 timing-round averages across three 67-case
sessions, all 804 input identities, 4,020 recorded timing certificates, fresh
build/source identities and the 10,190-case strict verification. Independent
saved-data checks reconstruct paired-log estimates, three-session ranges,
48 descriptive block-count rows and paper table values. Exploratory bootstrap
intervals are preserved, not resampled by this audit. This campaign remeasures
the unchanged five-arm implementation independently of the JV package updates.

Verify file integrity, sample coverage, summary arithmetic, manuscript table
values and publication figure identities without running a solver:

```sh
python benchmark/verify_results.py
```

Verification requires all four current ZIPs. It extracts temporary copies and
removes them on completion; no expanded results directory or superseded
`results.zip` is used. Override a current source explicitly with
`--windows-results`, `--linux-results`, `--application-results` or
`--lsape-results`. The verifier checks the 23 measured numeric tables and the
source/input/export hashes of 18 regenerated figures. It also checks that all
four controlled campaigns identify the current measurement scripts.

For new measurements, follow [the benchmark instructions](README.md). Data
attribution and terms are listed in [the real-world attribution](real_world/ATTRIBUTION.md).

## Separate numerical development evidence

[`numerical-regression-evidence-20261028.json`](numerical-regression-evidence-20261028.json) preserves a saved 28 October development-build probe, selected-pair records, source/binary identities and an exact-rational reconstruction of its extreme-scale rectangular example. It is separate from the four performance campaigns and was not rerun on their installed wheels. Its positive exact cost gap is concealed by equal rounded binary64 totals; it is not a new timing observation.
