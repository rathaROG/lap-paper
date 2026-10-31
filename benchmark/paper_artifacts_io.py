"""Portable, deterministic publication exports and saved-input read tracing."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

LARGE_CSV_BYTES = 10_000_000


def utf8_sanity(repo, outputs=False):
    """Reject known UTF-8/Windows-codepage corruption before publication."""
    repo = Path(repo)
    sequences = [(195, 8212), (194, 177), (194, 183), (226, 8364, 8220),
                 (226, 8364, 8221), (226, 8364, 166), (206, 180), (65533,)]
    forbidden = ["".join(map(chr, sequence)) for sequence in sequences]
    paths = list((repo / "benchmark").glob("paper_*.py"))
    paths += list((repo / "benchmark/paper_artifact_panels").glob("*.py"))
    if outputs:
        paths += list((repo / "latex/generated").glob("*.tex"))
        paths += [path for path in (repo / "benchmark/paper-artifacts").glob("*") if path.suffix in (".csv", ".json", ".gz")]
    for path in paths:
        open_source = gzip.open if path.suffix == ".gz" else open
        with open_source(path, mode="rt", encoding="utf-8-sig", newline="") as stream:
            for number, line in enumerate(stream, 1):
                decoded = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), line)
                if any(token in decoded for token in forbidden):
                    raise ValueError(f"UTF-8/codepage corruption in {path.relative_to(repo)}:{number}")
                if outputs and path.parent == repo / 'benchmark/paper-artifacts':
                    absolute = (r'\b[A-Za-z]:[\\/]|(?<![\w/:])/'
                                r'(?:home|usr|opt|tmp|var|mnt|root|Users|workspace|data|build|dev)/')
                    if re.search(absolute, decoded):
                        raise ValueError(f"Absolute host path in export {path.relative_to(repo)}:{number}")


def portable(value, repo):
    """Normalize generated source links without publishing host-local paths.

    Archived inputs are never rewritten. Absolute historical run locations in
    exported copies resolve to the same repository-relative evidence paths.
    External compiler/interpreter locations become explicit descriptors in the
    exported copy; each build's source_file points to the unchanged raw record.
    """
    if isinstance(value, dict):
        return {key: portable(item, repo) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, repo) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            pass
        else:
            return json.dumps(portable(decoded, repo), sort_keys=True, separators=(",", ":"))
    # Apply only to path-shaped values, not TeX commands or prose definitions.
    if "benchmark\\" in value or "benchmark/" in value or re.match(r"^[A-Za-z]:[\\/]", value):
        clean = value.replace("\\", "/")
        while "//" in clean:
            clean = clean.replace("//", "/")
        for marker in ("benchmark/runs/", "benchmark/paper-artifacts/", "latex/generated/", "latex/images/current/"):
            position = clean.find(marker)
            if position >= 0:
                return clean[position:]
        prefix = Path(repo).as_posix().rstrip("/") + "/"
        if clean.lower().startswith(prefix.lower()):
            return clean[len(prefix):]
        if re.match(r"^[A-Za-z]:/", clean) or clean.startswith("/"):
            return "recorded external location; see source_file in archived evidence"
        return clean
    if value.startswith("/") or value.startswith("\\\\"):
        return "recorded external location; see source_file in archived evidence"
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def normalize_exports(repo):
    """Canonicalize generated text only, then gzip CSVs larger than 10 MB.

    Fixed gzip timestamp and empty filename prevent host-specific headers.
    Counterpart deletion is restricted to a just-compressed generated CSV.
    """
    repo = Path(repo).resolve()
    folder = repo / "benchmark/paper-artifacts"
    csv.field_size_limit(64*1024*1024)
    # CSVs must precede JSONs which contain hashes of their backing CSVs.
    csv_paths = sorted(folder.glob("*.csv")) + sorted(path for path in folder.glob("*.csv.gz") if not path.with_suffix("").exists())
    for path in csv_paths:
        was_compressed = path.suffix == ".gz"
        open_source = gzip.open if was_compressed else open
        with open_source(path, mode="rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [portable(row, repo) for row in reader]
        if was_compressed:
            path = path.with_suffix("")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        if was_compressed or path.stat().st_size > LARGE_CSV_BYTES:
            compressed = path.with_suffix(".csv.gz")
            temporary = compressed.with_suffix(".gz.tmp")
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as target:
                    with path.open("rb") as source:
                        while block := source.read(1024*1024): target.write(block)
            temporary.replace(compressed)
            assert path.parent == folder and path.suffix == ".csv"
            path.unlink()
    for path in sorted(folder.glob("*.json")):
        if path.name == "artifact_index.json": continue
        value = portable(json.loads(path.read_text(encoding="utf-8-sig")), repo)
        # F12's backing-file digest must refer to the canonical exported CSV.
        if isinstance(value, dict) and "source_csv" in value and "source_csv_sha256" in value:
            source = repo / value["source_csv"]
            if source.is_file(): value["source_csv_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(path, value)
    tex_paths = list((repo / "latex/generated").glob("*.tex"))
    if (repo / "latex/supplementary_figures.tex").is_file(): tex_paths.append(repo / "latex/supplementary_figures.tex")
    for path in tex_paths:
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(text, encoding="utf-8", newline="\n")


class EvidenceReads:
    """Record files actually opened for reading below benchmark/runs."""
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.root = self.repo / "benchmark/runs"
        self.paths = set()

    def start(self):
        sys.addaudithook(self._audit)
        return self

    def _audit(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)): return
        mode, flags = args[1], args[2]
        if isinstance(mode, str) and any(character in mode for character in "wax+"): return
        if flags is not None and flags & (os.O_WRONLY | os.O_RDWR): return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        try: relative = path.relative_to(self.root)
        except ValueError: return
        self.paths.add("benchmark/runs/" + relative.as_posix())

    def save(self):
        records = []
        for relative in sorted(self.paths):
            path = self.repo / relative
            records.append(dict(path=relative, bytes=path.stat().st_size,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write_json(self.repo / "benchmark/paper-artifacts/generator_input_files.json",
                   {"scope": "Files actually opened by the complete saved-data generator; no benchmark execution.",
                    "files": records})
        return records
