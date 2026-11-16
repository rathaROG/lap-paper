# Microscopy provenance and attribution

The excerpt selection and detection-preparation recipe follow **Raphael Reme**,
*Cell Tracking with the new laprust*, version 0.0.1, source revision
`1dae22e68c4dd1d6c3915a7a416dbfca39284b31`:
https://github.com/raphaelreme/cell_tracking_with_laprust.
The upstream MIT notice is preserved in `licenses/original-benchmark-MIT.txt`.
Raphael Reme is a coauthor of the present paper; this remains separately
credited prior work. Source hashes and reused protocol elements are recorded
in `provenance.json` without private machine paths.

The present driver uses its own November 13 observations. Compared with the
earlier benchmark it adds capped rectangular SciPy, six balanced arm orders,
explicit CPU/thread control, recorded cost/objective checks, stronger track
digests and separate association profiles.

Scientific and data sources, cited in the manuscript:

- Reme et al. (2024), *Particle Tracking in Biological Images with Optical-Flow
  Enhanced Kalman Filtering*, ISBI, DOI: 10.1109/ISBI56570.2024.10635656.
- Hanson et al. (2024), *Automatic Monitoring of Neural Activity with
  Single-Cell Resolution in Behaving Hydra*, Scientific Reports 14, 5083,
  DOI: 10.1038/s41598-024-55608-2. Hydra data:
  https://doi.org/10.5061/dryad.h9w0vt4q3.
- Maska et al. (2023), *The Cell Tracking Challenge: 10 Years of Objective
  Benchmarking*, Nature Methods 20, 1010-1020,
  DOI: 10.1038/s41592-023-01879-y.

The existing Cell Tracking Challenge permission follow-up is unresolved in
the supplied records. Obtain/document permission before public release of the
microscopy-containing preprint or evidence. Software licensing does not grant
permission to redistribute or publicly use the datasets. This directory
contains no images, masks, detections, tracks or cost matrices.
