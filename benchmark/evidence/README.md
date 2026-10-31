# Frozen evidence for the paper

The five archives contain saved observations, not executable benchmark builds.
Four archives supply the primary light sessions; the older Linux/Python 3.11
archive supplies only the separate protocol-sensitivity supplement.

From a checkout containing these archives, restore the checked files and generate
the paper artifacts without running any solver or benchmark:

```sh
python -m pip install -r benchmark/requirements-artifacts.txt
python benchmark/paper_evidence.py extract
python benchmark/generate_paper_artifacts.py
```

The extractor verifies archive and member SHA-256 digests, allows only the
manifest's `benchmark/runs/` paths, and refuses to overwrite a differing file.
An identical existing file is retained. `python benchmark/paper_evidence.py verify`
checks archive hashes and any already-restored files without extracting them.

`manifest.json` records every member's relative POSIX path, byte count, digest,
archive and role. `generator_input` files are the actual files read during a
complete generator run, including summaries, protocols, build identities,
diagnostics, application observations, LSAPE observations and saved rejection
validation. Additional `primary_raw_audit` files retain all primary campaign
attempt measurement/validation logs so estimates can be independently checked.
Unused source snapshots and compiled binaries are excluded. This is sufficient
for result and figure regeneration; it is not a packaged execution environment
for rerunning measurements.

Archives have sorted members, fixed timestamps, fixed permissions and maximum
DEFLATE compression. They preserve original evidence bytes and line endings.
To rebuild the archives after a complete saved-data generation:

```sh
python benchmark/paper_evidence.py pack
```

That command takes its required-file list from
`benchmark/paper-artifacts/generator_input_files.json`, produced by tracing files
actually opened during generation. Generated CSVs larger than 10 MB use gzip
with a fixed timestamp and no embedded filename. Text exports use UTF-8, LF and
relative source links. Exact rendering hashes additionally depend on Python,
Matplotlib and its font/rendering dependencies; the pinned package versions match
the publication artifact environment. Numeric evidence and source checksums remain
independently inspectable.

These files are prepared for version control as part of the paper update. The
generation and packaging commands never create a Git commit or publish data.
