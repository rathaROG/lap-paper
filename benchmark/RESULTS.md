# Paper results

[`results.zip`](results.zip) contains the final measurements and summaries used
in the paper. New benchmark runs write to `results/full-YYYYMMDD/` and leave this
archive unchanged.

| Archive directory | Contents |
| --- | --- |
| `collections/` | Four Windows/Linux and Python environments: 35,784 timing samples and 1,660 memory observations, with input identities, checks and environment settings |
| `lsape/` | 9,045 timing-round averages over 67 cases and three sessions, paired summaries and verification results |
| `real_world/` | Tracking and replay observations, predictions, metrics, 7,539 captured matrices and paper table summaries |
| `figures.json` | Figure files and their measurement inputs |
| `manifest.json` | SHA-256 and size of every saved file and archive member |

Verify file integrity, sample counts, summary arithmetic and paper table values:

```sh
python benchmark/verify_results.py
```

Verification reads the ZIP through a temporary extraction. It does not run
benchmarks or leave an expanded copy in the repository.

To produce new measurements, follow [the benchmark instructions](README.md).
Data attribution and terms are listed in [the real-world attribution](real_world/ATTRIBUTION.md).
