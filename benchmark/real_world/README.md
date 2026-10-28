# Real-video assignment and tracking benchmark

`run.py` reproduces stages from the paper's two supplied MOT16-04/MOT20-03 video excerpts. Four frozen detection/reference caches are included, so tracking and assignment replay do not require videos, model weights, detector inference, or a GPU. The selected RF-DETR Nano condition is **768 px, confidence 0.2**, person class 1, at most 300 detections and a minimum width of 15 source-image pixels. GT identities are retained only in the reference field, never in tracker detections.

## Files and immutable measurements

- `data/inputs/*.json.gz`: exact frozen JSON bytes, compressed losslessly; each includes 999 frames of detections and reference IDs/boxes.
- `data/preview/`: two unchanged native frame 500 PNGs, annotated comparisons, captions and provenance. Clean frames contain no overlays or AI changes.
- `provenance/data_manifest.json`: compressed and original JSON SHA256 identities; `provenance/core_semantics.json` pins the measured solver/timing function ASTs.
- `patches/pyppbox-imgsz.patch`: optional RF input-regeneration patch; unnecessary for frozen-input tracker/replay stages.
- `data/matrices/*.npz`: the eight captured assignment corpora (7,539 float32 matrices, 9 MiB) replayed for Table C3, listed with their SHA256 in `provenance/data_manifest.json`. Replay reads them directly. Tests check the solver, tracker-update and replay measurement boundaries.

The paper's timing rounds, predictions, profiles, metrics and summaries are included in [`../results.zip`](../results.zip), at `real_world/observations.zip`. The archive manifest records file checksums. Ground-truth and detector configurations use separate measurement sessions and their timing samples are not pooled. See [`../RESULTS.md`](../RESULTS.md) for verification. New runs produce separate observations.

Recorded locations use role placeholders such as `<dataset-root>`, `<pyppbox-root>` or `<python>`. Files are identified by SHA-256. Programs resolve bundled data relative to this directory; external datasets, pyppbox checkouts, checkpoints, collections and output locations are explicit options.

## Verify the bundled inputs

From this directory, using Python 3.11:

```sh
python -B run.py --verify-inputs
python -B -m unittest tests.test_portable -v
python -B tests/portability_smoke.py
```

These checks need only Python's standard library and do not contact the network. Gzip inputs are expanded into a content-addressed system temporary cache when a tracker stage needs ordinary JSON files. Every expansion is checked against the original JSON SHA256.

Command examples use POSIX shell line continuation. In PowerShell, put each command on one line or use PowerShell's backtick continuation instead.

## CPU-only assignment replay

Install `requirements-replay.txt` in your own environment. SciPy alone is a usable baseline; install `lapx==0.12.0`, `laprust==1.1.0`, or `lapcpp==1.1.0` for the solver families you select. The recorded environment is not installed or changed automatically.

```sh
python -B run.py --replay-only --output /path/to/new-replay \
  --solvers scipy lapjvh laprust --cases mot16_04_gt_centroid
```

`lapjvh` is LAPX's JVHYBRID (hybrid Rust/C++) interface; `laprust` is the Rust family also exposed as `lap.lapjvr`. The published nine-method panel uses `lapjv lapjvx lapjvxa lapjvh laprust lapcpp scipy lapjvs lapjvc`. The original full campaign additionally validates the Rust alias outside timing. Replay never silently substitutes SciPy for a missing/failed backend.

Defaults are nine shuffled rounds, seed 20261027, CPU 0, one visible library thread, calibrated 30 ms target blocks, full selected corpora, and separate float32/float64 panels. Float64 widens the same captured float32 application costs; it does not rebuild tracking costs in double precision. Use `--cpu -1` only when affinity is unavailable and report the difference.

A deliberately small path/dependency smoke, **not publication timings**:

```sh
python -B run.py --replay-only --output /path/to/new-replay-smoke \
  --solvers scipy lapjvh --cases mot16_04_gt_centroid \
  --precisions float32 --rounds 1 --block-seconds 0.001 --limit-matrices 2
```

## Cached tracker updates

Use the dependencies in `requirements-tracking.txt` and pyppbox 4.0.0 from commit `f3f1e8e671d7cad7e0df2f2469ae0f5ee6c27258` at [the upstream repository](https://github.com/rathaumons/pyppbox). Either an installed package or `--pyppbox-root /path/to/checkout` works. Exact Centroid/SORT/Person source hashes are checked; a different 4.0.0 tree is rejected rather than silently changing the tracker.

Cached tracking still needs PyYAML and OpenCV bindings: importing `Person` reaches `UnifiedStrings` and `commontools`, which import `yaml` and `cv2`. The CPU recipe pins `PyYAML==6.0.3` and `opencv-python-headless==4.13.0.92`; the latter's wheel metadata requires NumPy >= 2 on Python >= 3.9, which admits the frozen NumPy 2.4.6. It needs no GUI, videos, decoding or GPU. Create a fresh environment and install only one OpenCV flavor; avoid adding the headless wheel alongside an existing desktop or custom `cv2` installation. These CPU import dependencies do not change the archived measurement environment.

The stock SORT module imports `laprust`, so that dependency is required for tracker runs even when SciPy is the selected hook. It is optional for assignment-only replay when another family is selected. Source checks allow CRLF/LF line-ending differences while preserving the original measured hashes.

Use the pinned solver versions. If your package index lacks a matching wheel, provide it through your index or `--find-links`:

```sh
python -m pip install --find-links /path/to/measured-solver-wheels -r requirements-tracking.txt
```

Assignment-only replay uses its smaller requirements and does not need PyYAML, OpenCV or pyppbox. The import check below verifies cached-tracking dependencies without processing frames or calling assignment solvers.

After installing the fresh CPU environment, check that import closure without running tracking:

```sh
python -B tests/check_tracking_imports.py --pyppbox-root /path/to/pyppbox --require-headless
```

```sh
python -B run.py --tracker-only --output /path/to/new-tracking \
  --pyppbox-root /path/to/pyppbox --solvers scipy lapjvh laprust
```

This stage repeats the original `run_tracking` primary boundary: only sequential `tracker.update` calls are timed. Person construction, video decoding, detector inference, prediction copies and accuracy evaluation are outside timing. Warmup is followed by fresh tracker state; SORT's identity counter is reset and original gates/unique-feasible bypass remain. It creates primary update timings/predictions/metrics; assignment profiling is left to the full campaign.

Two-frame wiring smoke for both actual tracker implementations:

```sh
python -B run.py --tracker-only --output /path/to/new-tracker-smoke \
  --pyppbox-root /path/to/pyppbox --solvers scipy lapjvh laprust \
  --sources gt --sequences mot16_04 --frames 2 --rounds 1 --warmup-frames 1
```

## Full campaign and report stages

`--campaign` invokes the portable derivative of the original complete driver: all nine solvers, both sequences, selected sources, both trackers, separate primary/profile/capture passes, common-matrix replay, then evaluation. It needs all comparison solver packages and the tracker dependencies. It is intentionally not launched automatically.

```sh
python -B run.py --campaign --output /path/to/new-campaign --pyppbox-root /path/to/pyppbox
python -B run.py --report-only --collection /path/to/new-campaign --output /path/to/new-report
```

Report generation reads saved summary data and uses `requirements-report.txt`; it does not rerun measurements. `--collection` is required and must point at a tracker, replay or campaign output from this package. Use a fresh output directory for every stage; existing artifacts are rejected.

## Optional video/detector regeneration

This stage is unnecessary for cached benchmarking. Supply your own video/converted-GT directory with the filenames recorded in the manifests and the canonical RF-DETR Nano checkpoint. For the pinned pyppbox checkout, apply `patches/pyppbox-imgsz.patch` before this stage if `DCFGRFDETR.imgsz` is unavailable. The patch passes input resolution to the model constructor **before** optimization. Cached tracker runs do not require this patch.

```sh
python -B run.py --regenerate-inputs --output /path/to/new-inputs \
  --dataset-root /path/to/mix_v1.0 --pyppbox-root /path/to/pyppbox \
  --checkpoint /path/to/rf-detr-nano.pth --rf-imgsz 768 --rf-conf 0.2
```

Regeneration requires pyppbox's optional RF-DETR 1.10.1, PyTorch CUDA and OpenCV dependencies, GPU 0 and an existing checkpoint. No videos or weights are distributed or downloaded by this package. Canonical checkpoint SHA256 is `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`; published MD5 is `fb6504cce7fbdc783f7a46991f07639f`. Actual optimized input tensors are checked during excluded warmups and recorded alongside requested/config/model/optimized resolutions.

## Interpretation and attribution

The supplied excerpts contain 999 frames each at 24 fps, with dimensions 1920 × 1080 and 1174 × 880. They are recoded excerpts rather than the original full MOTChallenge sequences: official MOT16-04 has 1,050 frames at 30 fps; MOT20-03 has 2,405 frames at 25 fps with a published width of 1173 pixels. The supplied GT exceeds each video's coverage; only frames 1–999 are used. Initial, middle and final alignment checks are recorded, but original-image-level pixel equivalence is not claimed.

Converted GT lacks original class/visibility/ignore fields. IDF1/MOTA and detector overlap scores are **supplied-box diagnostics**, not official MOTChallenge leaderboard scores. RF settings were selected using overlapping pilot frames; this is exploratory calibration, not held-out detector validation. Tracker-update FPS excludes inference/decode and is not end-to-end throughput. IQRs describe repeated timing variability on these clips, not population confidence intervals. See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset citations, CC BY-NC-SA 3.0 data terms, and upstream AGPL/GPL notices; the repository's MIT code license does not replace those terms.
