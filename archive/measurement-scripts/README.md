# Historical measurement source

`run_experiments.py` is the byte-for-byte source used for the retained timing
results, copied from paper commit `dc776900873f86ef98db7f8135eb80bb1335114f` before
correcting the batch-output length check.

Its SHA-256 is
`13d68fc59f2cdd86d67823d3aef2d3e2bdb61fc4e3f1de74beea5a4243e9b37e`, matching
the original experiment environments and the main measurement manifest. Those
records and the timing samples have not been rewritten.

Use the current repository-root script for new experiments. Its
`--validate-batch-only` option checks batch cardinality and objectives without
timing. The retained rechecks are `results/batch_validation.json` for Python
3.14.9 and `archive/python-3.11.16/batch-revalidation/batch_validation.json` for
Python 3.11.16. The artifact
audit verifies the historical source against the recorded measurement hash and
the corrected source against the new validation report. This snapshot is for
provenance, not a standalone reproduction entry point.
