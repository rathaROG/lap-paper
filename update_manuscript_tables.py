"""Refresh the four numeric result tables while preserving their row labels.

Narrative interpretations must be reviewed separately after a new experiment.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import re

import compare_optional as formulations
import narrative
import run_experiments as base
from report_results import batch_order

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER / "results"


def read_times(name, divisor, results=RESULTS):
    with (results / name).open(newline="") as stream:
        return {(row["case"], row["solver"]): float(row["median_ns"]) / divisor
                for row in csv.DictReader(stream)}


def table_values(results=RESULTS):
    single = read_times("single_summary.csv", 1000, results)
    batch = read_times("batch_summary.csv", 1e6, results)
    optional = read_times("formulations/summary.csv", 1000, results)
    memory = {(row["case"], row["solver"]): row["additional_peak"] / 2**20
              for row in json.loads((results / "memory.json").read_text())}
    datasets = json.loads((results / "formulations" / "datasets.json").read_text())
    batch_cases = [spec["case"] for spec in base.batch_specs()]
    solvers = {solver for case, solver in batch}
    return {
        "6.2": ([[single[case["case"], method] for method in base.SOLVERS]
                 for case in base.case_specs()], 2),
        "6.3": ([[batch[case, method] for case in batch_cases] for method in batch_order(solvers)], 3),
        "6.4": ([[memory[case, method] for case in ("required", "optional")]
                 for method in base.MEMORY_SOLVERS], 3),
        "6.5": ([[min(case["n"], case["m"]) - case["mean_pairs"]] +
                 [optional[case["case"], method] for method in formulations.METHODS]
                 for case in datasets], 2),
    }


def all_table_values():
    """Main tables from results/; Appendix A tables from the archived Python 3.11.16 run."""
    groups = table_values()
    archived = table_values(narrative.ARCHIVE)
    groups.update({"A.1": archived["6.2"], "A.2": archived["6.3"], "A.3": archived["6.5"]})
    old, new = narrative.principal_ratios(narrative.ARCHIVE), narrative.principal_ratios()
    groups["A.4"] = ([[a, b] for a, b in zip(old, new)], 2)
    return groups


def main():
    manuscript = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    for section, (values, places) in all_table_values().items():
        pattern = rf"(### {re.escape(section)} .*?\n)(.*?)(?=\n### |\n## |\Z)"
        match = re.search(pattern, manuscript, re.S)
        assert match, section
        row_index = 0
        lines = []
        for line in match.group(2).splitlines():
            if line.startswith("| "):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                try:
                    [float(cell) for cell in cells[1:]]
                except ValueError:
                    pass
                else:
                    assert len(cells) - 1 == len(values[row_index]), (section, cells[0])
                    line = "| " + cells[0] + " | " + " | ".join(
                        f"{value:.{places}f}" for value in values[row_index]) + " |"
                    row_index += 1
            lines.append(line)
        assert row_index == len(values), (section, row_index, len(values))
        manuscript = manuscript[:match.start(2)] + "\n".join(lines) + "\n" + manuscript[match.end(2):]
    (PAPER / "manuscript.md").write_text(manuscript, encoding="utf-8")
    print("Updated the numeric result tables from archived measurements. Review narrative numbers separately.")


if __name__ == "__main__":
    main()
