"""Generate standalone charts only; never invoke the manuscript builder."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import report_results
import build_pdf
import plot_scaling

ROOT = Path(__file__).resolve().parent


def main():
    assert json.loads((ROOT / "collection_run.json").read_text())["status"] == "completed"
    report_results.main()
    build_pdf.PROJECT = ROOT / "tmp/vector-export"
    build_pdf.figures()
    plot_scaling.PAPER = ROOT / "tmp/scaling-export"
    plot_scaling.generate(ROOT / "results/scaling")
    for source in (build_pdf.PROJECT / "images", plot_scaling.PAPER / "latex/images",
                   plot_scaling.PAPER / "figures"):
        for path in source.iterdir():
            shutil.copy2(path, ROOT / "figures" / path.name)
    subprocess.run([sys.executable, str(ROOT / "plot_extended_results.py"), str(ROOT / "extended")],
                   cwd=ROOT, check=True)
    print("Generated six base and thirteen extended charts in PNG, SVG and PDF.")


if __name__ == "__main__":
    main()
