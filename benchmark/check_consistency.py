"""Check that the shared paper-adapter block is identical in every timing suite.

Each suite's run.py carries its own copy of the solver adapters, objectives and
timing loop so that it runs on its own. This check fails if the copies drift.
Standard library only.

    python check_consistency.py
"""
from pathlib import Path
import hashlib
import sys

HERE = Path(__file__).resolve().parent
SUITES = ("base", "formulations", "scaling", "extended")
BEGIN = "# ---- BEGIN SHARED PAPER ADAPTERS"
END = "# ---- END SHARED PAPER ADAPTERS"


def block(path):
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise SystemExit(f"{path}: expected exactly one shared block")
    start = text.index(BEGIN)
    end = text.index("\n", text.index(END))
    return text[start:end]


def main():
    digests = {suite: hashlib.sha256(block(HERE / suite / "run.py").encode()).hexdigest() for suite in SUITES}
    for suite, digest in digests.items():
        print(f"{suite:13s} {digest[:16]}")
    if len(set(digests.values())) != 1:
        print("Shared adapter blocks differ; copy the block from base/run.py into the other suites.")
        return 1
    print("Shared adapter blocks are identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
