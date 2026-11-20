"""Regressions for exact manuscript excerpts and portable input inventories."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .paper_artifacts_io import normalize_exports, utf8_sanity


def test_normalization_keeps_claim_text_and_normalizes_sources(tmp_path):
    exports = tmp_path / "benchmark/paper-artifacts"
    exports.mkdir(parents=True)
    statements = [
        r"Run \texttt{python} \path{benchmark/paper_evidence.py} using \path{benchmark/runs/}.",
        r"See \path{benchmark/paper-artifacts/}. Results follow in Table~\ref{tab:results}.",
        '{"b": 2, "a": 1}',
    ]
    rows = [{"statement": text, "evidence_keys": "protocol",
             "file": r"D:\recorded\benchmark\runs\trial.json"} for text in statements]
    csv_path = exports / "manuscript_claims.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    json_path = exports / "manuscript_claims.json"
    ledger = {"claims": rows, "evidence": {"compiler": "/usr/bin/compiler"}}
    json_path.write_text(json.dumps(ledger), encoding="utf-8")

    normalize_exports(tmp_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        normalized_rows = list(csv.DictReader(stream))
    normalized_ledger = json.loads(json_path.read_text(encoding="utf-8"))
    assert normalized_ledger["claims"] == normalized_rows
    assert [row["statement"] for row in normalized_rows] == statements
    assert {row["file"] for row in normalized_rows} == {"benchmark/runs/trial.json"}
    assert normalized_ledger["evidence"]["compiler"] == "recorded external location; see source_file in archived evidence"
    utf8_sanity(tmp_path, outputs=True)
    first_pass = (csv_path.read_bytes(), json_path.read_bytes())
    normalize_exports(tmp_path)
    assert (csv_path.read_bytes(), json_path.read_bytes()) == first_pass


def test_microscopy_input_inventory_is_identical_for_cold_warm_and_preloaded_imports(tmp_path):
    source = Path(__file__).resolve().parent
    benchmark = tmp_path / "benchmark"
    microscopy = benchmark / "cell_tracking"
    microscopy.mkdir(parents=True)
    exports = benchmark / "paper-artifacts"
    exports.mkdir()
    for filename in ("paper_artifacts_io.py", "paper_artifacts_microscopy.py"):
        shutil.copyfile(source / filename, benchmark / filename)
    input_names = ("__init__.py", "recalculate.py", "benchmark.py", "manifest.json", "observations.json")
    for filename in input_names:
        shutil.copyfile(source / "cell_tracking" / filename, microscopy / filename)

    script = """\
from pathlib import Path
import sys
sys.path.insert(0, str(Path('benchmark').resolve()))
from paper_artifacts_io import EvidenceReads
if sys.argv[1] == 'preloaded':
    import paper_artifacts_microscopy
reads = EvidenceReads(Path.cwd()).start()
import paper_artifacts_microscopy
paper_artifacts_microscopy.load_saved(Path.cwd())
reads.save()
"""
    inventories = []
    for state in ("cold", "warm", "preloaded"):
        subprocess.run([sys.executable, "-I", "-c", script, state], cwd=tmp_path, check=True)
        inventories.append((exports / "generator_input_files.json").read_bytes())
        assert list((microscopy / "__pycache__").glob("recalculate.*.pyc"))
    assert inventories[0] == inventories[1] == inventories[2]
    files = json.loads(inventories[0])["files"]
    expected = {"benchmark/cell_tracking/" + name for name in input_names}
    assert {record["path"] for record in files} == expected
    for record in files:
        contents = (tmp_path / record["path"]).read_bytes()
        assert record["bytes"] == len(contents)
        assert record["sha256"] == hashlib.sha256(contents).hexdigest()
