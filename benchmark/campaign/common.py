from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers can briefly deny delete sharing. Retry outside measured
    # blocks; a persistent I/O failure remains an infrastructure error.
    for attempt in range(100):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.002)


def append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def input_digest(value):
    """Hash represented bytes, dimensions and dtype, including batch order."""
    import numpy as np
    h = hashlib.sha256()
    values = value if isinstance(value, (list, tuple)) else [value]
    h.update(str(len(values)).encode())
    for array in values:
        a = np.asarray(array)
        h.update(json.dumps([list(a.shape), a.dtype.str]).encode())
        if a.size:
            # Prepared campaign arrays are C-contiguous. Hash their existing
            # buffer directly instead of allocating another potentially-GiB
            # bytes object. A noncontiguous diagnostic input needs one copy.
            h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()

