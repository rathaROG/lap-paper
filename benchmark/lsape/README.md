# Controlled LSAPE study (Tables 9–11)

This study measures a one-sided, uniform-penalty adaptation **inside the LSAPE v4 Hungarian implementation**. It compares LSAPE with itself, not with the paper's JV solvers. The unchanged upstream function is **LSAPE-v4**; the adaptation is **LSAPE-v4-1S**. The adaptation keeps Hungarian tree growth, FIFO traversal, dual updates and path reversal, and adds no JV reductions, JV search, orientation or batching.

This folder is self-contained. It holds the upstream LSAPE v4 source, the adapted header, the C++ bridge and every script needed to build, verify, measure and summarize.

## The five arms

| Name | Entry point | Cost buffer, leading dimension | Implementation |
| --- | --- | --- | --- |
| LSAPE-v4 (O) | `solve` | full (n+1)(m+1), ld=n+1 | unmodified upstream `hungarianLSAPE`, `init_type=1`, `forb_assign=false`, endpoint prices h,h |
| LSAPE-v4 [p,0] (U) | `solve_one_sided` | full (n+1)(m+1), ld=n+1 | same upstream function, deletion price p and insertion price 0 |
| LSAPE-v4-1S/scan (S) | `solve_implicit_scan` | compact n·m, ld=n | specialized preprocessing and implicit border; ordered rejection scan |
| LSAPE-v4-1S/tracked (T) | `solve_specialized` | compact n·m, ld=n | same, with cached rejection and scan fallback |
| LSAPE-v4-1S/tracked/padded (P) | `solve_specialized_padded` | full one-sided (n+1)(m+1), ld=n+1 | tracked adaptation on U's buffer geometry |

All five share one translation unit, compiler configuration, binary, argument validation, float64 conversion, column-major copy and pair output. The ratios isolate different things:
- **O/U** measures the formulation.
- **U/P** measures specialization with a matched layout.
- **O/P** is the total of the two.
- **P/T** measures layout.

A ratio above 1 favors the denominator.

## Workflow

Requires Windows x64, MSVC (Visual Studio 2022 C++ tools; set `VCVARS64` if they are not found automatically), Python with NumPy and SciPy. The paper used Python 3.11.17, NumPy 2.4.6 and SciPy 1.17.1.

```powershell
python -B build_bridge.py
python -B verify_implementation.py --output-dir runs/verification-new
python -B rectangular_campaign.py --manifest results/rectangular-grid-v1.json --session-index 1 --output results/session-1
python -B rectangular_campaign.py --manifest results/rectangular-grid-v1.json --session-index 2 --output results/session-2
python -B rectangular_campaign.py --manifest results/rectangular-grid-v1.json --session-index 3 --output results/session-3
python -B summarize_rectangular.py results/session-1/report.json results/session-2/report.json results/session-3/report.json --output results/summary
```

1. **Build:** `build_bridge.py` compiles `_lsape_native.pyd` into `build/` with `/O2 /EHsc /std:c++17 /LD /MD /DNDEBUG /fp:precise`, without `/fp:fast` or `/fp:contract`. It also records the toolchain, the flags and the hashes of every input.
2. **Verify:** `verify_implementation.py` runs the strict verification. That covers 190 enumerated small cases and 10,000 randomized stress cases, full primal–dual certificates, exact final-state identity between the one-sided arms and upstream, and capped- and square-SciPy objective checks. It refuses a build whose recorded source hashes differ from the current files.
3. **Measure:** `rectangular_campaign.py` runs one session per process over the frozen 67-case grid in `results/rectangular-grid-v1.json` (manifest SHA-256 `875745fc…`). `--validate-only` checks every input without timing. The paper ran three sessions in fresh processes on logical CPU 0, with nine shuffled rounds and blocks calibrated to 30 ms.
4. **Summarize:** `summarize_rectangular.py` pairs rounds within each session, takes the median and range over the three sessions, and writes `summary.json`, `all_cases.csv`, `all_case_methods.csv`, `paper_table.csv` (Table 10's twelve preselected rows) and `block_counts.csv` (Table 11).

Use new output directories every time; existing runs are never overwritten. Saved records name locations by role rather than by local path ([`redaction.py`](redaction.py)). `controlled_comparison.py` provides the shared campaign harness and a smaller validation run.

## Frozen grid

| Block | Shapes/distributions | Limits | Cases |
| --- | --- | --- | ---: |
| A | Uniform squares, N=128,256,512,1024 | 0.5 and 2/N | 8 |
| B | Uniform N×5N/4 and transposes | 0.5 and 2/max | 16 |
| C | Congested N×5N/4 and transposes | 0.5 | 8 |
| D | Uniform N×4N and transposes, N=64..512 | 0.5 and 2/max | 16 |
| E | Uniform 64×M and transposes, M=1024,4096,16384 | 0.5 and 2/M | 12 |
| F | Threshold ties: 256²; 256×320, 256×1024, 64×4096 with transposes | L=2h, h=2^(−⌈log2 max⌉) | 7 |

## Objective and numeric contract

For K real pairs, h = float64(L/2) and p = h + h:

```
F = sum(selected costs) + h(n+m−2K),   P = sum(selected costs) + p(n−K),   F = P + h(m−n)
```

Costs and L must be finite, nonnegative binary64 values no larger than DBL_MAX/[16(n+m+1)]. The bridge requires round-to-nearest with FTZ/DAZ disabled, and rejects calls otherwise.

## Paper results (27 October 2026, one Windows host)

For the primary comparison O/P over all 67 cases:
- 39 cases are above 1.03 (favoring P);
- 16 are below 0.97 (favoring O);
- 12 fall in the practical band [0.97, 1.03].

The matched-layout U/P ratio exceeds 1.03 in 66 cases. The cache-only S/T ratio shows no consistent benefit. These counts describe one host, compiler, grid and timing contract; they are not significance tests.

## Source provenance and licence

`src/lsape-v4.zip` is the July 2017 LSAPE v4 release by Sébastien Bougleux. It matched a fresh download from the author-hosted URL on 27 October 2026 (`src/upstream_download_check.json`, SHA-256 `2861b207…`). `src/upstream-v4/` is that archive extracted unchanged, under the CeCILL-C licence it ships with (`Licence_CeCILL-C_V1-*.txt`). `src/specialized/hungarian-lsape-uniform.hh` is the attributed adaptation, and its header documents its derivation and changes.
