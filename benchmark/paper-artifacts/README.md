# Reproducible paper artifacts

From the repository root, install `benchmark/requirements-artifacts.txt`, run
`python benchmark/paper_evidence.py extract`, then run
`python benchmark/generate_paper_artifacts.py`.
The generator requires Python, NumPy and Matplotlib; it reads saved results
and never imports benchmark runners, generates matrices, or invokes solvers.
The source scripts, shared identity map, generated TeX fragments, figures and
these backing exports form one reviewable change. No commit is made by the
generator.

The four primary input roots are `benchmark/runs/` plus:

- `linux-py3.11.18-20261031-0455/paper-run-light`
- `linux-py3.14.10-20261031-0630/paper-run-light`
- `windows-py3.11.18-20261030-2322/paper-run-light`
- `windows-py3.14.10-20261031-0322/paper-run-light`

S1 round-truncation estimates use the supplementary full-protocol session
`linux-py3.11.18-20261030-1434/paper-run`; its times never enter primary figures.
S1 coverage, protocol and numerical-boundary audits also read the four primary
manifests and validation records, without combining their timing estimates.

## Output map

- `latex/images/current/`: PDF figures and PNG previews. Names describe the question and
  chart form; manuscript numbering is consecutive. F1–F9 cover the main results,
  F_structure restores the ordered sweeps, A2 is the sole heatmap, F11 shows
  rejection, and A4 retains memory-counter visibility.
- `latex/generated/`: T1–T4, application appendix tables, native and LSAPE
  classification tables, per-API application values, the supplementary
  line/strip-panel atlas fragment and the round-sensitivity table. The atlas entry point is
  `latex/supplementary_figures.tex`.
- `F*_*.csv`, `A*_*.csv`, `T3_counter_anchors.csv`: plotted points and table values.
- `S2_primary_conditions.csv.gz`: every reported primary timing condition, its
  saved estimator and round spread, comparator, input/attempt/session identity,
  case metadata and reference to the saved source.
- `S2_planned_exclusions.csv`: absent canonical arms, explicitly not measured.
- `S2_build_source_index.csv`: saved environment/configuration/build/manifest
  file identities. It points to exact archived metadata rather than silently
  treating nominal package versions as complete build provenance.
- `S2_rejection_validation.csv`: stored per-input validation outcomes, not
  outcomes from a new solve.
- `S3_*`: controlled policy timings, complete layout timings and eligibility,
  native event/capacity counters, workspace counters, deduplicated diagnostic
  build records, and construction-only diagnostics.
  `S3_native_counters.csv.gz` and `S3_workspace_counters.csv.gz` are lossless
  CSVs; the latter retains the large per-matrix output-validation arrays.
- `S4_*`, `S5_*`, `S6_*`: full application, batch/memory and LSAPE evidence.
- `campaign_summary.json`, `support_summary.json`, `s1_sensitivity.json`: derived
  claims with environment identities and source fields.
- `reference_sensitivity.*` and `reference_sensitivity_*.csv`: the post-hoc
  same-OS reference diagnostic, including seven flagged events, three observed
  CSGRAPH class disagreements and two reference-substitution changes. These
  sensitivity checks do not replace primary observations or identify causality.
- `artifact_index.json`: generated-file hashes, rebuilt by the single entry point.
- `generator_input_files.json`: exact paths, sizes and hashes of files actually
  read from the restored evidence during a complete generation.

Generated text uses UTF-8, LF and repository-relative POSIX source links. CSVs
larger than 10 MB use deterministic gzip (fixed timestamp, no source filename).
Generated copies replace external compiler/interpreter locations with an
explicit provenance marker and a pointer to the archived source field. Raw
archived input bytes are never normalized. The five evidence archives and their per-file hashes
are documented in `benchmark/evidence/README.md`.

Each environment remains separate. The practical ratio classification is loss
below 0.97, tie from 0.97 through 1.03, win above 1.03. Round spread and process
ranges are descriptive, not confidence intervals. The driver can regenerate
one component using `--only campaign`, `--only support` or `--only sensitivity`.

The style map is `benchmark/paper_artifact_panels/style_mapping.json`. Panel
scripts adapt the agreed mockup designs but do not depend on files outside this
repository. Counters, memory, applications and LSAPE retain their distinct
measurement boundaries and estimators.
