# Independent LSAPE benchmark

This experiment compares five formulations and implementations within the
LSAPE v4 Hungarian code. Its kernels, frozen 67-case input grid and numeric
contract are separate from the dense JV campaign.

| Arm | Implementation and layout |
| --- | --- |
| O | Unchanged LSAPE v4, full buffer, equal endpoint penalties |
| U | Unchanged LSAPE v4, full buffer, one-sided penalty |
| S | Specialized code, compact buffer, ordered rejection scan |
| T | Specialized code, compact buffer, guarded rejection cache |
| P | Specialized code, full buffer matching U's leading dimension |

All five arms use one binary and the same complete-call boundary: validation,
float64 conversion, column-major copying, allocation, solve and int64 output.
O/U compares formulations, U/P compares specialization with matched layout,
P/T compares layouts, and S/T compares rejection tracking.

Requires Windows x64, MSVC C++ tools, NumPy, SciPy, psutil and threadpoolctl.
Use `VCVARS64` only if Visual Studio discovery cannot locate `vcvars64.bat`.
Run from the repository root with an explicit output location:

```powershell
$LsapeOut = Join-Path $env:TEMP "lap-paper-lsape-py31118"
python -B -m benchmark.lsape build --output-dir "$LsapeOut/build"
python -B -m benchmark.lsape verify --build-dir "$LsapeOut/build" --output-dir "$LsapeOut/verification"
python -B -m benchmark.lsape smoke --build-dir "$LsapeOut/build" --output "$LsapeOut/smoke"
```

The build retains compiler flags, source and binary hashes and a source snapshot.
Verification and runs check the archived builder that produced the binary,
alongside the current compiled inputs and the recorded binary hash.
It uses `/O2 /EHsc /std:c++17 /LD /MD /DNDEBUG /fp:precise`; primary calls have
no diagnostic counters. Verification checks 10,190 matrices using enumeration,
primal/dual certificates, final-state identity, and two SciPy objective references.
The bounded smoke checks all five arms across the six grid families, using one
input, two rounds and 1 ms target blocks. Its records cannot enter full summaries.

Full measurements require explicit opt-in and a fresh process/output folder for
each session. Inputs and the existing LSAPE protocol remain frozen: four inputs
per case, nine shuffled rounds and 30 ms target blocks. Session indices describe
three fresh processes, without claiming independent-day replication.

```powershell
python -B -m benchmark.lsape run --allow-full-campaign --build-dir "$LsapeOut/build" --session-index 1 --output "$LsapeOut/session-1"
# Repeat in fresh processes for session indices 2 and 3.
python -B -m benchmark.lsape summarize "$LsapeOut/session-1/report.json" "$LsapeOut/session-2/report.json" "$LsapeOut/session-3/report.json" --output "$LsapeOut/summary"
```

The manifest is [data/rectangular-grid-v1.json](data/rectangular-grid-v1.json).
All generated builds, source copies, verification records and timing results stay
inside the chosen output directories. Without `--output`/`--output-dir`, they go to
`benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/lsape-<command>`, and the
summary goes beside the session folders. Existing output is never overwritten.
Builds must match the running Python patch version. The shared measurement lock
prevents overlap with dense/application runs on the same host.

Costs and limits must be finite, nonnegative binary64 values bounded by
`DBL_MAX/[16(n+m+1)]`. For `h=float64(L/2)` and `p=h+h`, the equal-endpoint
objective equals the one-sided objective plus `h*(m-n)`. Validation is finite
sampled evidence, not a guarantee of universal floating-point exactness.

[src/lsape-v4.zip](src/lsape-v4.zip) is the supplied July 2017 upstream release;
the extracted source remains byte-identical and retains its CeCILL-C licences.
[src/upstream_download_check.json](src/upstream_download_check.json) records its
author-hosted download identity. The specialized header retains attribution and
describes its changes. No JV performance or mechanism claim follows from these
within-LSAPE contrasts.
