"""Render saved real-world benchmark summaries without executing measurements."""

from __future__ import annotations

import argparse
import base64
import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


SOLVER_ORDER = ("lapjv", "lapjvx", "lapjvxa", "lapjvh", "laprust", "lapcpp",
                "scipy", "lapjvs", "lapjvc", "lapjvr")
TRACKING_COLUMNS = (
    "case", "sequence", "source", "tracker", "solver", "precision", "frames", "rounds",
    "median_tracker_ns_per_frame", "q25_tracker_ns_per_frame", "q75_tracker_ns_per_frame",
    "tracker_fps", "scipy_speedup", "idf1", "mota", "num_switches", "num_false_positives",
    "num_misses", "solver_calls", "mean_solver_ns", "solver_fraction_instrumented",
)
REPLAY_COLUMNS = (
    "case", "sequence", "source", "tracker", "precision", "solver", "matrices", "rounds",
    "median_ns_per_assignment", "q25_ns_per_assignment", "q75_ns_per_assignment",
    "scipy_speedup", "objective_max_abs_gap", "objective_mismatch_count",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fmt(value: Any, decimals: int = 2, scale: float = 1) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * scale:,.{decimals}f}"


def _count(value: Any) -> str:
    return _fmt(value, decimals=0)


def _ratio(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.3f}×"


def _interval(row: dict, median_key: str, q25_key: str, q75_key: str) -> str:
    return (f"{_fmt(row.get(median_key), scale=1e-3)} "
            f"[{_fmt(row.get(q25_key), scale=1e-3)}, {_fmt(row.get(q75_key), scale=1e-3)}]")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict], preferred: tuple[str, ...]) -> None:
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _solver_key(name: str) -> tuple[int, str]:
    return (SOLVER_ORDER.index(name) if name in SOLVER_ORDER else len(SOLVER_ORDER), name)


def _case(row: dict) -> str:
    return str(row.get("case") or "/".join(str(row.get(key, "?")) for key in ("sequence", "source", "tracker")))


def _case_label(row: dict, multiline: bool = False) -> str:
    sequence = str(row.get("sequence", "?")).upper().replace("_", "-")
    source = {"gt": "GT boxes", "rfdetr_nano": "RF-DETR Nano"}.get(row.get("source"), str(row.get("source", "?")))
    tracker = str(row.get("tracker", "?")).upper()
    return ("\n" if multiline else " · ").join((sequence, source, tracker))


def _case_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_case(row), []).append(row)
    return [(name, sorted(values, key=lambda r: (str(r.get("precision", "")), _solver_key(str(r.get("solver", ""))))))
            for name, values in sorted(grouped.items())]


def _table(headers: list[str], rows: list[list[str]], first_label: bool = True) -> str:
    head = "".join(f"<th scope='col'>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            tag = "th scope='row'" if first_label and index == 0 else "td"
            close = "th" if tag.startswith("th") else "td"
            cells.append(f"<{tag}>{escape(str(value))}</{close}>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='table-scroll'><table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def _heatmap(output: Path, rows: list[dict], kind: str,
             reference_rows: list[dict] | None = None) -> Path | None:
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    case_references = {name: group for name, group in _case_groups(reference_rows or [])}
    case_references.update(dict(_case_groups(rows)))
    cases = sorted(case_references.items())
    names = [name for name, _ in cases]
    labels = [_case_label(values[0], True) for _, values in cases]
    solvers = sorted({str(row["solver"]) for row in rows}, key=_solver_key)
    precisions = sorted({str(row.get("precision", "unspecified")) for row in rows})
    panels = []
    for precision in precisions:
        matrix = np.full((len(solvers), len(names)), np.nan)
        seen = set()
        for row in rows:
            if str(row.get("precision", "unspecified")) != precision:
                continue
            index = (solvers.index(str(row["solver"])), names.index(_case(row)))
            if index in seen:
                raise ValueError(f"Duplicate {kind} heatmap cell: {precision}/{row['solver']}/{_case(row)}")
            seen.add(index)
            ratio = _number(row.get("scipy_speedup"))
            if ratio is not None and ratio > 0:
                matrix[index] = math.log2(ratio)
        panels.append(matrix)
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in panels])
    # The minimum 0.5x–2x range prevents tiny differences from filling the scale.
    limit = max(1.0, float(math.ceil(float(np.abs(finite_values).max())))) if finite_values.size else 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")
    width = max(8.5, 1.05 * len(names) + 1.5) * len(panels)
    height = max(5.2, 0.49 * len(solvers) + 2.1)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False,
                             layout="constrained", sharey=True)
    for panel_index, (precision, matrix) in enumerate(zip(precisions, panels)):
        ax = axes[0, panel_index]
        plotted = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_yticks(range(len(solvers)), solvers, fontsize=10)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.set_title(precision, fontsize=12, pad=13)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(solvers), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(len(solvers)):
            for j in range(len(names)):
                value = matrix[i, j]
                label = "—" if not np.isfinite(value) else f"{2 ** value:.2f}×"
                color = "white" if np.isfinite(value) and abs(value) > 0.58 * limit else "#172336"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    title = "Tracker update time" if kind == "tracking" else "Captured assignment replay"
    fig.suptitle(title + " · SciPy time / method time", fontsize=15, fontweight="bold")
    ticks = np.arange(-int(limit), int(limit) + 1)
    colorbar = fig.colorbar(plotted, ax=axes.ravel().tolist(), ticks=ticks, shrink=0.78, pad=0.02)
    colorbar.ax.set_yticklabels([f"{2.0 ** tick:g}×" for tick in ticks])
    colorbar.set_label("Ratio above 1 favors method", fontsize=10)
    path = output / f"{kind}_speedup.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def _embedded_figure(path: Path | None, caption: str) -> str:
    if path is None:
        return "<p>No saved observations for this plot.</p>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f"<figure><a href='{escape(path.name, quote=True)}'><img src='data:image/png;base64,{data}' "
            f"alt='{escape(caption, quote=True)}'></a><figcaption>{escape(caption)}</figcaption></figure>")


def _timing_median(metadata: dict, name: str) -> float | None:
    timing = metadata.get("timings_ns", {}).get(name)
    if isinstance(timing, list):
        values = [number for value in timing if (number := _number(value)) is not None]
        return median(values) if values else None
    if isinstance(timing, dict):
        return _number(timing.get("median_ns", timing.get("median")))
    return _number(timing)


def _compact_input(entry: dict) -> dict:
    metadata = entry.get("metadata", entry)
    result = {key: value for key, value in metadata.items() if key not in ("timings_ns", "frames_data")}
    result["timings_ns_summary"] = {}
    for key, value in metadata.get("timings_ns", {}).items():
        if isinstance(value, list) and value and all(_number(item) is not None for item in value):
            result["timings_ns_summary"][key] = {"samples": len(value), "median": median(value), "min": min(value), "max": max(value)}
        else:
            result["timings_ns_summary"][key] = value
    return result


def _input_table(inputs: list[dict]) -> str:
    rows = []
    for entry in inputs:
        metadata = entry.get("metadata", entry)
        video = metadata.get("video", {})
        detector = metadata.get("detector", {})
        rows.append([
            str(metadata.get("sequence", "—")), str(metadata.get("source", "—")),
            _count(metadata.get("frames", video.get("decoded_frames"))),
            "complete" if video.get("fully_decoded") is True else ("partial" if video.get("fully_decoded") is False else "unrecorded"),
            _fmt(_timing_median(metadata, "decode"), 3, 1e-6),
            _fmt(_timing_median(metadata, "detector"), 3, 1e-6),
            _fmt(detector.get("initialization_ns"), 1, 1e-6),
        ])
    return _table(["Sequence", "Input source", "Frames", "Video coverage", "Median decode ms/frame",
                   "Median detector ms/frame", "Model initialization ms"], rows)


def _tracking_tables(rows: list[dict]) -> str:
    if not rows:
        return "<p>No saved tracker measurements.</p>"
    sections = []
    for _, group in _case_groups(rows):
        table_rows = []
        for row in group:
            table_rows.append([
                str(row.get("solver", "—")), str(row.get("precision", "—")),
                _interval(row, "median_tracker_ns_per_frame", "q25_tracker_ns_per_frame", "q75_tracker_ns_per_frame"),
                _fmt(row.get("tracker_fps"), 1), _ratio(row.get("scipy_speedup")),
                _fmt(row.get("idf1"), 2, 100), _fmt(row.get("mota"), 2, 100),
                _count(row.get("num_switches")), _count(row.get("num_false_positives")),
                _count(row.get("num_misses")), _count(row.get("solver_calls")),
                _fmt(row.get("mean_solver_ns"), 2, 1e-3),
                _fmt(row.get("solver_fraction_instrumented"), 2, 100),
            ])
        first = group[0]
        sections.append(f"<h3>{escape(_case_label(first))}</h3><p class='small'>"
                        f"{escape(_count(first.get('frames')))} frames · "
                        f"{escape(_count(first.get('rounds')))} rounds per method. "
                        "Assignment calls and assignment share come from a separate instrumented run.</p>" +
                        _table(["Method", "Precision", "Update µs/frame median [Q1, Q3]", "Update FPS", "SciPy / method",
                                "IDF1 %", "MOTA %", "ID switches", "FP", "FN", "Assignment calls", "Mean solve µs", "Profiled assignment %"], table_rows))
    return "".join(sections)


def _replay_tables(rows: list[dict]) -> str:
    if not rows:
        return "<p>No saved replay measurements.</p>"
    sections = []
    for _, group in _case_groups(rows):
        table_rows = []
        for row in group:
            table_rows.append([
                str(row.get("solver", "—")), str(row.get("precision", "—")),
                _count(row.get("matrices")), _count(row.get("rounds")),
                _interval(row, "median_ns_per_assignment", "q25_ns_per_assignment", "q75_ns_per_assignment"),
                _ratio(row.get("scipy_speedup")), _fmt(row.get("objective_max_abs_gap"), 8),
                _count(row.get("objective_mismatch_count")),
            ])
        sections.append(f"<h3>{escape(_case_label(group[0]))}</h3>" +
                        _table(["Method", "Precision", "Matrices", "Rounds", "Solve µs median [Q1, Q3]", "SciPy / method",
                                "Max absolute objective gap", "Objective mismatches"], table_rows))
    return "".join(sections)


def _raw_links(output: Path) -> str:
    items = []
    for path in sorted(output.iterdir(), key=lambda p: p.name):
        if path.name in ("report.html", "report.txt") or path.name.startswith("."):
            continue
        if path.is_dir() or path.suffix.lower() in (".json", ".csv", ".npz", ".zip", ".txt", ".png"):
            label = path.name + ("/" if path.is_dir() else "")
            items.append(f"<li><a href='{escape(label, quote=True)}'>{escape(label)}</a></li>")
    return "<ul class='raw-links'>" + "".join(items) + "</ul>"


def _text_report(summary: dict) -> str:
    tracking, replay = summary.get("tracking", []), summary.get("replay", [])
    cases = _case_groups(tracking)
    lines = ["Real-world tracking and captured assignment benchmark", "",
             f"Saved scope: {len(cases)} pipelines, {len(tracking)} tracker settings, {len(replay)} replay settings.",
             "GT boxes and cached RF-DETR Nano detections drive actual pyppbox Centroid/SORT tracking.",
             "GT reference identities are hidden from tracker inputs. Evaluation uses supplied boxes at IoU >= 0.5,",
             "without official MOTChallenge ignore/class filtering; these are not leaderboard scores.", "",
             "Primary timings cover tracker.update only: exclude decode, detection and Person construction.",
             "Replay uses common captured matrices. Adapter/conversion/output work is included.",
             "Medians and IQRs describe saved timing rounds; no significance or population claim is made.",
             "SciPy / method > 1 means the method was faster. Near-equality is descriptive.",
             "Stock SORT's unique-feasible association bypass stays enabled; assignment counts are actual calls.",
             "Assignment share is from an instrumented run and is not primary timing.", ""]
    for _, group in cases:
        lines.extend([_case_label(group[0]), "Method | Precision | Update us/frame [Q1,Q3] | FPS | SciPy/method | IDF1 | MOTA | IDSW | FP | FN | Calls"])
        for row in group:
            lines.append(" | ".join((str(row.get("solver")), str(row.get("precision")),
                          _interval(row, "median_tracker_ns_per_frame", "q25_tracker_ns_per_frame", "q75_tracker_ns_per_frame"),
                          _fmt(row.get("tracker_fps"), 1), _ratio(row.get("scipy_speedup")), _fmt(row.get("idf1"), 6),
                          _fmt(row.get("mota"), 6), _count(row.get("num_switches")),
                          _count(row.get("num_false_positives")), _count(row.get("num_misses")), _count(row.get("solver_calls")))))
        lines.append("")
    lines.extend(["Replay results: replay_summary.csv; figures: tracking_speedup.png and replay_speedup.png.",
                  "All CSV metric ratios are fractions. MOTA can be negative; undefined metrics are empty/null.", "",
                  "Validation", _json(summary.get("validation", {})), "", "Protocol", _json(summary.get("protocol", {})),
                  "", "Environment", _json(summary.get("environment", {}))])
    return "\n".join(lines) + "\n"


def generate_report(output: Path) -> dict[str, str]:
    """Build HTML/text/CSV/PNG artifacts from ``output/summary.json`` only."""
    output = Path(output).resolve()
    from realbench_paths import redact
    summary = redact(json.loads((output / "summary.json").read_text(encoding="utf-8")))
    tracking, replay = summary.get("tracking", []), summary.get("replay", [])
    inputs = summary.get("inputs", [])
    _write_csv(output / "tracking_summary.csv", tracking, TRACKING_COLUMNS)
    _write_csv(output / "replay_summary.csv", replay, REPLAY_COLUMNS)
    tracker_figure = _heatmap(output, tracking, "tracking")
    replay_figure = _heatmap(output, replay, "replay", reference_rows=tracking)
    cases = _case_groups(tracking)
    dimensions = [len({row.get(key) for row in tracking}) for key in ("sequence", "source", "tracker")]
    rounds = sorted({row.get("rounds") for row in tracking if row.get("rounds") is not None})
    rounds_text = ", ".join(str(value) for value in rounds) if rounds else "unrecorded"
    pipeline_scope = f"{len(cases)} pipelines: {dimensions[0]} sequences × {dimensions[1]} detection sources × {dimensions[2]} trackers"
    partial_inputs = [entry.get("metadata", entry) for entry in inputs
                      if entry.get("metadata", entry).get("video", {}).get("fully_decoded") is False]
    partial_note = ("<p><strong>Partial-run preview:</strong> at least one input contains only a video prefix. "
                    "These saved results do not represent a complete-clip campaign.</p>" if partial_inputs else "")
    solvers = {row.get("solver") for row in tracking}
    mismatch_values = [_number(row.get("objective_mismatch_count")) for row in replay]
    mismatch_count = sum(value for value in mismatch_values if value is not None) if mismatch_values and all(v is not None for v in mismatch_values) else None
    style = """
    :root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#172336;background:#f4f6f9;line-height:1.5}
    body{margin:0}main{max-width:1500px;margin:auto;padding:32px 36px 64px}h1{font-size:32px;margin:0 0 8px;line-height:1.2}
    h2{font-size:24px;margin:36px 0 12px}h3{font-size:17px;margin:30px 0 8px}p{max-width:1100px;margin:10px 0}
    .lede{font-size:19px;color:#42536b}.scope{background:#fff;border-left:4px solid #245b98;padding:15px 20px;margin:22px 0}
    .facts{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0}.fact{background:#fff;padding:12px 18px;border:1px solid #dce2eb;border-radius:8px;min-width:125px}
    .fact b{font-size:24px;display:block}.fact span,.small,figcaption{color:#526176;font-size:13px}.table-scroll{overflow:auto;background:#fff;border:1px solid #dce2eb;border-radius:6px}
    table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid #e4e8ef;text-align:right}
    th:first-child,td:first-child{text-align:left}thead th{background:#eaf0f7;font-weight:650;position:sticky;top:0}tbody th{font-weight:600}tbody tr:nth-child(even){background:#fafbfd}
    figure{margin:20px 0;background:#fff;padding:16px;border:1px solid #dce2eb;border-radius:8px}figure img{display:block;width:100%;height:auto}figcaption{margin-top:12px}
    details{margin:14px 0;padding:14px 18px;background:#fff;border:1px solid #dce2eb;border-radius:6px}summary{cursor:pointer;font-weight:600}pre{overflow:auto;font-size:12px;max-height:550px}
    a{color:#1d5b99}.raw-links{columns:3;max-width:1100px}.raw-links li{break-inside:avoid;margin:5px 0}code{background:#eaf0f7;padding:2px 5px;border-radius:3px}
    @media(max-width:800px){main{padding:24px 16px}h1{font-size:25px}.raw-links{columns:1}figure{padding:6px}}
    @media print{body{background:white}main{max-width:none;padding:0}.table-scroll{overflow:visible}thead{display:table-header-group}details{break-inside:avoid}figure{break-inside:avoid}}
    """
    content = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
               f"<title>Real-world tracking benchmark</title><style>{style}</style></head><body><main>",
               "<h1>Real-world tracking and assignment benchmark</h1>",
               f"<p class='lede'>{escape(pipeline_scope)}.</p>",
               "<div class='scope'><p>Actual pyppbox Centroid and SORT trackers process GT boxes and cached RF-DETR Nano detections on the supplied lossless MOT16/MOT20 videos. "
               "Ground-truth identities remain hidden from tracker inputs. Each method receives the same detections in a fixed geometric order.</p>"
               "<p><strong>Scope:</strong> accuracy scores evaluate the supplied converted boxes at IoU ≥ 0.5. The original class, visibility and ignore policy is unavailable; "
               "these are supplied-box results, not official MOTChallenge leaderboard scores.</p>" + partial_note + "</div>",
               f"<div class='facts'><div class='fact'><b>{len(cases)}</b><span>pipeline configurations</span></div>"
               f"<div class='fact'><b>{len(solvers)}</b><span>assignment methods</span></div>"
               f"<div class='fact'><b>{escape(rounds_text)}</b><span>timing rounds per method</span></div>"
               f"<div class='fact'><b>{escape(_count(mismatch_count))}</b><span>recorded replay objective mismatches</span></div></div>",
               "<h2>What the measurements include</h2>",
               "<p>The primary measurement is tracker update time. It excludes video decoding, detector inference, construction of pyppbox Person objects, accuracy evaluation and disk writes. "
               "Update FPS is the reciprocal of that measured time; it is not detector-plus-tracker video throughput. "
               "Decoding and detection are performed and recorded separately, then detections are reused across assignment methods.</p>",
               "<p>Tables show the median and interquartile range of saved timing-round averages. The ratio is SciPy time divided by method time within the same pipeline and precision; "
               "values above one favor the method. Near-one differences are descriptive. No statistical significance, application population or universal speedup claim is made.</p>",
               "<p>Stock SORT keeps its unique-feasible association bypass, so not every frame invokes an assignment solver. "
               "Actual assignment call counts and assignment-time shares come from a separate instrumented pass; instrumentation is excluded from the primary timings. "
               "Sequential video frames are not batched into independent assignments.</p>",
               "<h2>Tracker update performance</h2>",
               _embedded_figure(tracker_figure, "Tracker update ratios against SciPy. Colors use a logarithmic ratio scale spanning at least 0.5×–2×, keeping near-equal results visually close to the center."),
               "<h2>Tracking results and accuracy</h2>",
               "<p>IDF1 and MOTA are displayed as percentages here; CSV and JSON retain fractions. ID switches, false positives and misses are counts over the full evaluated sequence. "
               "Predicted identity labels are evaluated by correspondence; they need not equal GT identity numbers. Equal-cost ties can produce different identities and later tracker states. "
               "Primary tracking ratios may therefore include differing trajectories, which the accuracy columns make visible.</p>",
               _tracking_tables(tracking),
               "<h2>Common-matrix assignment replay</h2>",
               "<p>Replay applies each method to the same captured application matrices, separately for float32 and float64. "
               "This isolates assignment-interface runtime from changes to the tracker trajectory. Normal conversion, solving and output adaptation remain included; "
               "matrix loading, capture and objective validation are excluded. Each precision has its own matching SciPy baseline. "
               "Objective agreement does not require identical pairs when optima tie.</p>",
               _embedded_figure(replay_figure, "Common captured-matrix replay. Panels use a shared color scale, with SciPy time divided by method time. Empty cells indicate unavailable observations, including pipelines with no assignment calls."),
               _replay_tables(replay),
               "<h2>Video and detection preparation</h2>",
               "<p>These preparation measurements are separate from tracker-update comparisons. Detector medians exclude initialization and warmup. "
               "RF-DETR initialization is shared across sequence preparation, so repeated entries identify the same initialization rather than additional independent trials. "
               "Decode timings can include cold-start effects and are not an end-to-end pipeline benchmark.</p>",
               _input_table(inputs),
               "<h2>Reproducibility records</h2>",
               f"<details><summary>Protocol</summary><pre>{escape(_json(summary.get('protocol', {})))}</pre></details>",
               f"<details><summary>Environment and installed identities</summary><pre>{escape(_json(summary.get('environment', {})))}</pre></details>",
               f"<details><summary>Validation</summary><pre>{escape(_json(summary.get('validation', {})))}</pre></details>",
               f"<details><summary>Saved source and artifact provenance</summary><pre>{escape(_json(summary.get('reproducibility', {})))}</pre></details>",
               f"<details><summary>Input identities and detector configuration</summary><pre>{escape(_json([_compact_input(entry) for entry in inputs]))}</pre></details>",
               "<h2>Saved data and exports</h2>", _raw_links(output), "</main></body></html>"]
    (output / "report.html").write_text("\n".join(content), encoding="utf-8")
    (output / "report.txt").write_text(_text_report(summary), encoding="utf-8")
    artifacts = {name: str(output / name) for name in ("report.html", "report.txt", "tracking_summary.csv", "replay_summary.csv")}
    if tracker_figure:
        artifacts[tracker_figure.name] = str(tracker_figure)
    if replay_figure:
        artifacts[replay_figure.name] = str(replay_figure)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Saved campaign directory containing summary.json")
    args = parser.parse_args()
    from realbench_paths import redact
    print(json.dumps(redact(generate_report(args.output)), indent=2))


if __name__ == "__main__":
    main()
