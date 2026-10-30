"""Audit archived paper results without running a solver or measurement.

    python benchmark/verify_results.py

The audit checks the four current controlled Windows, controlled Linux,
application and LSAPE bundles, including their manifests, sample coverage,
summary arithmetic and the paper's numeric tables and figure identities.
No solver or measurement program is imported or executed. ZIP contents are
checked and unpacked into temporary directories that are removed afterwards.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import statistics
import sys
from tempfile import TemporaryDirectory
import zipfile

import audit_controlled_update
import audit_lsape_update


HERE = Path(__file__).resolve().parent
UPDATED_SUMMARY_FILES = ("base/single_summary.csv", "base/batch_summary.csv",
                         "formulations/summary.csv", "scaling/summary.csv",
                         "extended/summary.csv")
SINGLE_CASES = ("square128", "square512", "wide64x512", "tall512x64", "near256x320",
                "blocks256x320", "ties128x160", "optional128x160", "optional256x320",
                "tight128x160", "negative128x160", "optional_square128")
SINGLE_METHODS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "jvc", "scipy")
BATCH_CASES = ("batch64x64x80", "batch16x128x160_optional", "batch1024x8x10")
BATCH_METHODS = ("jvrust_loop", "jvrust_batch_t1", "jvrust_batch_t4", "jvhybrid_batch_t1",
                 "jvhybrid_batch_t4", "jvcpp_batch_t1", "jvcpp_batch_t4", "jvx_batch_t1",
                 "jvx_batch_t4", "scipy_loop")
FORMULATION_CASES = ("optional128x160", "optional256x320", "tight128x160", "negative128x160",
                     "optional_square128", "reject128x160_L005", "reject128x160_L001",
                     "reject_tall160x128", "optional_wide64x512", "optional_blocks256x320")
FORMULATION_METHODS = ("jvrust", "jvcpp", "scipy_square", "scipy_dummies", "scipy_capped")
MEMORY_METHODS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "scipy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(root, name):
    name = PurePosixPath(str(name))
    require(not name.is_absolute() and ".." not in name.parts and
            all(":" not in part and "\\" not in part for part in name.parts),
            "Unsafe manifest path")
    result = root.joinpath(*name.parts)
    require(result.resolve().is_relative_to(root.resolve()), "Manifest path escapes its root")
    return result


def default_windows_results():
    archive = HERE / "controlled-windows-20261029.zip"
    return archive if archive.is_file() else None


def default_linux_results():
    archive = HERE / "controlled-linux-20261029.zip"
    return archive if archive.is_file() else None


def default_application_results():
    archive = HERE / "application-windows-20261029.zip"
    return archive if archive.is_file() else None


def default_lsape_results():
    archive = HERE / "lsape-windows-20261029.zip"
    return archive if archive.is_file() else None


def extract_results(archive_path, root, require_manifest=True):
    """Validate every ZIP destination before writing any member to disk."""
    with zipfile.ZipFile(archive_path) as archive:
        planned, seen, files = [], set(), set()
        for info in archive.infolist():
            require(info.orig_filename == info.filename and "\0" not in info.filename,
                    "Unsafe results archive member name")
            relative = PurePosixPath(info.filename)
            require(bool(relative.parts) and relative != PurePosixPath("."),
                    "Unsafe results archive member name")
            target = local(root, info.filename)
            # Windows folds case and strips trailing dots/spaces. Reject these
            # aliases on every platform so verification never drops a member.
            require(all(part == part.rstrip(". ") for part in relative.parts),
                    "Unsafe results archive member name")
            key = tuple(part.casefold() for part in relative.parts)
            require(key not in seen, "Duplicate results archive member")
            seen.add(key)
            kind = stat.S_IFMT(info.external_attr >> 16)
            require(kind in (0, stat.S_IFREG, stat.S_IFDIR),
                    "Results archive links and special files are not supported")
            require(not (kind == stat.S_IFDIR and not info.is_dir()),
                    "Results archive directory has an invalid name")
            if not info.is_dir():
                files.add(key)
            planned.append((info, target, key))
        for _, _, key in planned:
            require(all(key[:length] not in files for length in range(1, len(key))),
                    "Results archive file conflicts with a directory")
        for info, target, _ in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
    if require_manifest:
        require((root / "manifest.json").is_file(),
                "Results archive must contain manifest.json at its root")


@contextmanager
def results_directory(path):
    path = Path(path).resolve()
    if path.is_dir():
        yield path
    else:
        with TemporaryDirectory(prefix="lap-paper-results-") as directory:
            root = Path(directory)
            extract_results(path, root)
            yield root


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=1e-12),
            "Recorded summary arithmetic differs: " + label)


def quartiles(values):
    values = sorted(values)
    require(bool(values), "Empty timing group")
    def percentile(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return statistics.median(values), percentile(.25), percentile(.75)


def verify_manifest(root):
    manifest = read(root / "manifest.json")
    for name, identity in manifest["files"].items():
        current = local(root, name).read_bytes()
        require(len(current) == identity["bytes"] and digest(current) == identity["sha256"],
                "Result file checksum mismatch: " + name)
    members = 0
    for name, records in manifest.get("archives", {}).items():
        with zipfile.ZipFile(local(root, name)) as archive:
            entries = [entry for entry in archive.namelist() if not entry.endswith("/")]
            require(len(entries) == len(set(entries)) and set(entries) == set(records),
                    "Result archive inventory mismatch: " + name)
            for entry, identity in records.items():
                local(root, entry)  # Validate names without extracting any files.
                current = archive.read(entry)
                require(len(current) == identity["bytes"] and digest(current) == identity["sha256"],
                        "Result archive member checksum mismatch: " + name + "/" + entry)
                members += 1
    return {"files": len(manifest["files"]), "archive_members": members}


def verify_lsape(root):
    folder = root / "lsape"
    projection = read(folder / "timing_rounds.json")
    manifest, sessions = projection["manifest"], projection["sessions"]
    require(len(manifest["cases"]) == 67 and len(sessions) == 3, "LSAPE campaign coverage differs")
    methods = {(r["case"], int(r["session_index"]), r["method"]): r for r in rows(folder / "all_case_methods.csv")}
    require(len(methods) == 67 * 3 * 5, "LSAPE method summary coverage differs")
    count = 0
    for session in sessions:
        require(len(session["cases"]) == 67, "LSAPE session case coverage differs")
        for case in session["cases"]:
            for method, values in case["round_average_us"].items():
                require(len(values) == 9 and all(math.isfinite(v) and v > 0 for v in values), "LSAPE timing rounds differ")
                row = methods[case["case"], session["session_index"], method]
                for field, value in zip(("median_us", "q1_us", "q3_us"), quartiles(values)):
                    close(row[field], value, "LSAPE method summary")
                count += len(values)
    for row in rows(folder / "all_cases.csv"):
        points = []
        for session in sessions:
            case = next(case for case in session["cases"] if case["case"] == row["case"])
            values = case["round_average_us"]
            point = math.exp(statistics.mean(math.log(a / b) for a, b in zip(values[row["numerator"]], values[row["denominator"]])))
            close(row[f"s{session['session_index']}_ratio"], point, "LSAPE paired session ratio")
            points.append(point)
        for field, value in (("median_estimate", statistics.median(points)), ("min_session", min(points)), ("max_session", max(points))):
            close(row[field], value, "LSAPE across-session ratio")
    require(count == 9045, "LSAPE round count differs")
    return {"cases": 67, "sessions": 3, "timing_round_averages": count}


def verify_real_world(root):
    archive_path = root / "real_world/observations.zip"
    with zipfile.ZipFile(archive_path) as archive:
        summary = json.loads(archive.read("summary.json"))
        samples = json.loads(archive.read("tracking_samples.json"))
        require(len(summary["tracking"]) == 72 and len(summary["replay"]) == 144 and len(samples) == 648,
                "Real-world tracking/replay coverage differs")
        groups = defaultdict(list)
        for row in samples:
            groups[row["case"], row["solver"]].append(row["nanoseconds"])
        for row in summary["tracking"]:
            values = groups[row["case"], row["solver"]]
            require(len(values) == 9, "Real-world tracking rounds differ")
            for field, value in zip(("median_tracker_ns_per_frame", "q25_tracker_ns_per_frame", "q75_tracker_ns_per_frame"), quartiles(values)):
                close(row[field], value, "Real-world tracker summary")
        replay_groups, replay_count = defaultdict(list), 0
        objective_checks = nonzero_gaps = 0
        maximum_gap = 0
        matrix_count = 0
        for name in archive.namelist():
            if name.startswith("replay/") and name.endswith(".json"):
                data = json.loads(archive.read(name))
                for row in data["samples"]:
                    replay_groups[row["case"], row["solver"], row["precision"]].append(row["nanoseconds"])
                    replay_count += 1
                for validation in data["validation"].values():
                    require(validation["objective_mismatch_count"] == 0, "Recorded real-world objective mismatch")
                    objective_checks += validation["matrices"]
                    nonzero_gaps += validation["objective_nonzero_gap_count"]
                    maximum_gap = max(maximum_gap, validation["objective_max_abs_gap"])
            elif name.startswith("matrices/") and name.endswith(".npz"):
                with zipfile.ZipFile(io.BytesIO(archive.read(name))) as matrices:
                    matrix_count += len(matrices.namelist())
        for row in summary["replay"]:
            values = replay_groups[row["case"], row["solver"], row["precision"]]
            require(len(values) == 9, "Real-world replay rounds differ")
            for field, value in zip(("median_ns_per_assignment", "q25_ns_per_assignment", "q75_ns_per_assignment"), quartiles(values)):
                close(row[field], value, "Real-world replay summary")
        require(replay_count == 1296 and matrix_count == 7539, "Real-world replay sample/matrix count differs")
    return {"tracking_samples": 648, "replay_samples": replay_count, "captured_matrices": matrix_count,
            "tracking_summary_rows": 72, "replay_summary_rows": 144, "recorded_objective_mismatches": 0,
            "objective_checks": objective_checks, "recorded_nonzero_objective_gaps": nonzero_gaps,
            "maximum_absolute_objective_gap": maximum_gap}


def table_data(folder):
    """Read the saved summaries needed for the controlled paper tables."""
    summary_files = UPDATED_SUMMARY_FILES
    prefix = ""
    result = {name: {(row["case"], row["solver"]): float(row["median_ns"]) for row in rows(folder / relative)}
              for name, relative in zip(("single", "batch", "formulations", "scaling", "extended"), summary_files)}
    result.update(
        formulation_specs=read(folder / (prefix + "formulations/datasets.json")),
        scaling_specs=read(folder / (prefix + "scaling/datasets.json")),
        extended_specs=read(folder / "extended/plan.json")["timing"],
        memory={(row["case"], row["solver"]): row["additional_peak"] / 2**20
                for row in read(folder / "base/memory.json")})
    return result


def principal_ratios(data):
    single, batch, form, scale = (data[name] for name in ("single", "batch", "formulations", "scaling"))
    noncongested = [form[case, "scipy_capped"] / form[case, "jvrust"]
                    for case in FORMULATION_CASES if case != "optional_blocks256x320"]
    return [single["blocks256x320", "jvrust"] / single["blocks256x320", "scipy"],
            min(noncongested), max(noncongested),
            form["optional_blocks256x320", "jvrust"] / form["optional_blocks256x320", "scipy_capped"],
            scale["optional_blocks_1024x1280", "jvrust"] / scale["optional_blocks_1024x1280", "scipy_capped"],
            batch["batch1024x8x10", "jvrust_loop"] / batch["batch1024x8x10", "jvrust_batch_t1"],
            batch["batch1024x8x10", "jvrust_loop"] / batch["batch1024x8x10", "jvrust_batch_t4"],
            max(single[case, "jvcpp"] / single[case, "jvrust"] for case in SINGLE_CASES[7:]),
            single["tall512x64", "jvcpp"] / single["tall512x64", "jvrust"],
            form["optional128x160", "jvrust"] / 1000]


def environment_tables(py314, py311):
    def single(data):
        return [[data["single"][case, method] / 1000 for method in SINGLE_METHODS] for case in SINGLE_CASES]

    def batch(data):
        return [[data["batch"][case, method] / 1e6 for case in BATCH_CASES] for method in BATCH_METHODS]

    def formulations(data):
        specs = {row["case"]: row for row in data["formulation_specs"]}
        return [[min(specs[case]["n"], specs[case]["m"]) - specs[case]["mean_pairs"]] +
                [data["formulations"][case, method] / 1000 for method in FORMULATION_METHODS]
                for case in FORMULATION_CASES]

    extended = []
    for spec in py314["extended_specs"]:
        if ((spec["family"] == "large_square" and spec["n"] == 5000) or
            (spec["family"] == "large_rectangle" and (spec["n"], spec["m"]) == (4000, 5000)) or
                spec["family"] == "signed"):
            reference = "scipy" if spec["regime"] == "required" else "scipy_capped"
            extended.append([data["extended"][spec["case"], method] / 1e6
                             for data in (py314, py311) for method in ("jvrust", reference)])
    return {"4": (single(py314), 2), "5": (batch(py314), 3),
            "6": ([[py314["memory"][case, method] for case in ("required", "optional")]
                   for method in MEMORY_METHODS], 3),
            "7": (formulations(py314), 2), "8": (extended, 3), "A1": (single(py311), 2),
            "A2": (batch(py311), 3), "A3": (formulations(py311), 2),
            "A4": ([[left, right] for left, right in zip(principal_ratios(py311), principal_ratios(py314))], 2)}


def native_overview(data):
    comparisons = []
    for suite, specs in (("single", []), ("formulations", data["formulation_specs"]),
                         ("scaling", data["scaling_specs"]), ("extended", data["extended_specs"])):
        table, by_case = data[suite], {row["case"]: row for row in specs}
        for (case, method), value in table.items():
            if method in ("jvrust", "jvcpp", "jvhybrid"):
                reference = "scipy" if suite == "single" or by_case[case].get("regime") == "required" else "scipy_capped"
                comparisons.append((method, value / table[case, reference]))
    require(Counter(method for method, _ in comparisons) == {"jvrust": 201, "jvcpp": 201, "jvhybrid": 76},
            "Native comparison coverage differs")
    return [sum(ratio < 1 for selected, ratio in comparisons if selected == method)
            for method in ("jvrust", "jvcpp", "jvhybrid")] + [max(ratio for _, ratio in comparisons)]


def controlled_paper_tables(windows, linux):
    windows314, windows311 = windows
    linux314, linux311 = linux
    expected = environment_tables(windows314, windows311)
    linux_tables = environment_tables(linux314, linux311)
    expected.update({label: linux_tables[source] for label, source in
                     (("B1", "4"), ("B2", "A1"), ("B3", "5"), ("B4", "A2"), ("B5", "7"), ("B6", "A3"))})
    expected["B7"] = ([native_overview(data) for data in (windows314, windows311, linux314, linux311)], [0, 0, 0, 3])
    return expected


def expected_paper_tables(windows, linux, application, lsape):
    """Project every measured table from the four declared current bundles."""
    controlled = [tuple(table_data(root / name) for name in ("py314-results", "py311-results"))
                  for root in (windows, linux)]
    expected = controlled_paper_tables(*controlled)
    lsape_rows = []
    lsape_root = lsape
    for row in rows(lsape_root / "lsape/paper_table.csv"):
        values = [float(row["O_ms"]), float(row["smaller_side_unmatched_pct"])]
        values.extend(float(row[prefix + "_" + field]) for prefix in ("O_U", "U_P", "O_P", "P_T")
                      for field in ("median_estimate", "min_session", "max_session"))
        lsape_rows.append(values)
    require(len(lsape_rows) == 12, "LSAPE paper row coverage differs")
    expected["10"] = (lsape_rows, [3, 2] + [2] * 12)
    block_rows = [row for row in rows(lsape_root / "lsape/block_counts.csv")
                  if row["comparison"] == "original_over_padded"]
    require([row["block"] for row in block_rows] == list("ABCDEF"), "LSAPE paper block coverage differs")
    counts = [[int(row[field]) for field in ("cases", "denominator_faster", "denominator_slower",
                                             "point_within_practical_band")] for row in block_rows]
    expected["11"] = (counts + [[sum(row[column] for row in counts) for column in range(4)]], 0)
    application_tables = read(application / "real_world/projections.json")["tables"]
    require(set(application_tables) == {"12", "C1", "C2", "C3", "C4"}, "Real-world paper table coverage differs")
    expected.update({label: (table["numeric_values"], table["places"])
                     for label, table in application_tables.items()})
    return expected


def tex_numeric_table(tex, label):
    """Read numeric cells, leaving captions, row labels and other prose free to change."""
    tables = [table for table in re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, re.S)
              if re.search(r"\\label\{tab:" + re.escape(label) + r"\}", table)]
    require(len(tables) == 1, "Missing or duplicated paper table: " + label)
    result = []
    for line in tables[0].splitlines():
        if "&" not in line or not line.rstrip().endswith(r"\\"):
            continue
        cells = line.rstrip()[:-2].split("&")[1:]
        values = []
        number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
        for cell in cells:
            # Numeric cells may contain a range or a median with [Q1, Q3]
            # displayed by shortstack. Remove formatting before reading them.
            plain = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?", "", cell).replace("--", " ")
            tokens = number.findall(plain)
            if not tokens or re.search(r"[A-Za-z]", number.sub("", plain)):
                break
            values.extend(float(token) for token in tokens)
        else:
            require(all(math.isfinite(value) for value in values), "Nonfinite paper table value")
            result.append(values)
    return result


def check_paper_tables(tex, expected):
    for label, (values, places) in expected.items():
        rounded = []
        for row in values:
            precision = [places] * len(row) if isinstance(places, int) else places
            require(len(row) == len(precision), "Paper table precision coverage differs: " + label)
            rounded.append([float(f"{value:.{digits}f}") for value, digits in zip(row, precision)])
        require(tex_numeric_table(tex, label) == rounded,
                "Paper table does not match saved measurements: " + label)
    return len(expected)


def verify_paper_tables(windows, linux, application, lsape):
    paper = HERE.parent / "latex/main.tex"
    require(paper.is_file(), "Authoritative paper source is missing")
    return check_paper_tables(paper.read_text(encoding="utf-8"),
                             expected_paper_tables(windows, linux, application, lsape))


def verify_updated_controlled(root, platform, observations=None):
    """Recompute current observations and check the declared runtime and harness."""
    require(platform in ("Windows", "Linux"), "Unknown controlled platform")
    manifest = verify_manifest(root)
    audited = [audit_controlled_update.audit(root / name) for name in ("py311-results", "py314-results")]
    left, right = audited[0][1], audited[1][1]
    require(left.keys() == right.keys() and len(left) == 3925,
            "Current controlled objective coverage differs")
    require(all(left[key] == right[key] for key in left),
            "Current controlled objectives or recorded cardinalities differ")
    require(audited[0][2] == audited[1][2], "Current controlled schedules differ")
    for name, observed, version, numpy, scipy in zip(
            ("py311-results", "py314-results"), audited, ("3.11.18", "3.14.10"),
            ("2.4.6", "2.5.4"), ("1.17.1", "1.18.1")):
        require(observed[0]["python"].startswith(version + " "),
                "Current controlled interpreter version differs")
        require(observed[0]["numpy"] == numpy and observed[0]["scipy"] == scipy,
                "Current controlled numerical-library versions differ")
        for suite in ("base", "formulations", "scaling", "extended"):
            env = read(root / name / suite / "environment.json")
            require(env["platform"].startswith(platform + "-"),
                    "Current controlled platform differs")
            require(env["script_sha256"] == digest((HERE / suite / "run.py").read_bytes()),
                    "Current controlled harness source differs: " + suite)
    if observations is not None:
        observations.extend(audited)
    return {"manifest": manifest, "environments": 2,
            "timing_samples": sum(item[0]["timing_samples"] for item in audited),
            "memory_observations": sum(item[0]["memory_observations"] for item in audited),
            "exact_objectives_compared": len(left),
            "summary_groups_recomputed": sum(item[0]["settings"] for item in audited),
            "scope": "Current controlled " + platform + " observations; saved-data audit without re-solving."}


def verify_updated_windows(root):
    return verify_updated_controlled(root, "Windows")


def verify_updated_linux(root):
    return verify_updated_controlled(root, "Linux")


def verify_updated_figures(root, platform="windows", other_roots=None):
    """Check declared sources and exports of regenerated publication plots."""
    path = HERE / ("publication-" + platform + "-20261029-figures.json")
    require(path.is_file(), "Publication figure manifest is missing: " + platform)
    manifest = read(path)
    bundles = manifest.get("source_bundles", {platform: manifest.get("source_bundle")})
    for bundle in bundles.values():
        require(bundle is not None and digest(local(HERE, bundle["path"]).read_bytes()) == bundle["sha256"],
                "Publication source bundle checksum differs")
    for name, expected in manifest["scripts"].items():
        require(digest(local(HERE, name).read_bytes()) == expected,
                "Publication generator checksum differs: " + name)
    for name, expected in manifest["inputs"].items():
        source, relative = root, name
        if platform == "linux":
            scope, separator, relative = name.partition("/")
            require(separator and other_roots is not None and scope in ("windows", "linux"),
                    "Publication figure input scope differs: " + name)
            source = other_roots[scope]
        require(digest(local(source, relative).read_bytes()) == expected,
                "Publication figure input checksum differs: " + name)
    outputs = 0
    require(len(manifest["figures"]) == (15 if platform == "windows" else 3),
            "Publication figure coverage differs")
    for figure in manifest["figures"].values():
        for name, expected in figure["outputs"].items():
            require(digest(local(HERE.parent, name).read_bytes()) == expected,
                    "Publication figure output checksum differs: " + name)
            outputs += 1
    require(outputs == (60 if platform == "windows" else 12), "Publication figure export coverage differs")
    return {"figures": len(manifest["figures"]), "output_files": outputs,
            "inputs": len(manifest["inputs"])}


def verify_application_update(root):
    """Audit fresh tracker/replay observations and their publication projection."""
    manifest = verify_manifest(root)
    result = verify_real_world(root)
    with zipfile.ZipFile(root / "real_world/observations.zip") as archive:
        summary = json.loads(archive.read("summary.json"))
    environment = summary["environment"]
    require(environment["python"].startswith("3.11.18 "), "Current application Python version differs")
    require({name: environment["packages"][name] for name in audit_controlled_update.PACKAGES}
            == audit_controlled_update.PACKAGES, "Current application solver versions differ")
    require(summary["reproducibility"]["all_repeated_prediction_hashes_identical"] is True and
            summary["reproducibility"]["all_instrumented_predictions_match"] is True,
            "Application repeated trajectories or profiled outputs differ")
    saved_projection = read(root / "real_world/projections.json")
    tables = saved_projection["tables"]
    require(set(tables) == {"12", "C1", "C2", "C3", "C4"}, "Current application table coverage differs")
    import application_publication
    with TemporaryDirectory(prefix="lap-paper-application-audit-") as directory:
        observations = Path(directory)
        extract_results(root / "real_world/observations.zip", observations, require_manifest=False)
        projected = application_publication.project_campaign(observations)
    for field in ("tables", "numerical_validation", "source_summary_sha256", "input_identities", "capture_identities"):
        require(saved_projection[field] == projected[field],
                "Current application projection differs from saved observations: " + field)
    return {"manifest": manifest, **result, "python": environment["python"],
            "packages": {name: environment["packages"][name] for name in audit_controlled_update.PACKAGES},
            "scope": "Current cached tracking and common-cost replay; detector inference was not rerun."}


def verify_lsape_update(root):
    """Audit fresh LSAPE sessions and the projections used for Tables 10-11."""
    manifest = verify_manifest(root)
    folder = root / "lsape"
    raw = audit_lsape_update.audit(folder)
    for name in ("summary.json", "all_cases.csv", "all_case_methods.csv", "paper_table.csv", "block_counts.csv"):
        require(digest((folder / name).read_bytes()) == digest((folder / "summary" / name).read_bytes()),
                "Current LSAPE publication copy differs from raw summary: " + name)
    timing = read(folder / "timing_rounds.json")
    for index, session in enumerate(timing["sessions"], 1):
        report = read(folder / f"session-{index}/report.json")
        expected = [{"case": case["case"], "round_average_us": {
            method: stats["round_average_us"] for method, stats in case["methods"].items()}}
            for case in report["cases"]]
        require(session["session_index"] == index and session["cases"] == expected and
                timing["manifest"] == report["manifest"], "Current LSAPE round projection differs from raw session")
    summary = verify_lsape(root)
    if (folder / "projections.json").is_file():
        import lsape_publication
        projected = lsape_publication.project_summary(folder / "summary", folder / "verification/report.json")
        require(read(folder / "projections.json")["tables"] == projected["tables"],
                "Current LSAPE paper projection differs from raw summary")
    return {"manifest": manifest, **summary, "independent_raw_audit": raw}


def verify(windows_results=None, linux_results=None, application_results=None, lsape_results=None):
    selected = {
        "windows": default_windows_results() if windows_results is None else windows_results,
        "linux": default_linux_results() if linux_results is None else linux_results,
        "application": default_application_results() if application_results is None else application_results,
        "lsape": default_lsape_results() if lsape_results is None else lsape_results,
    }
    for name, path in selected.items():
        require(path is not None, "Required current results bundle is missing: " + name)
    with ExitStack() as stack:
        roots = {name: stack.enter_context(results_directory(path)) for name, path in selected.items()}
        observations = []
        report = {"status": "passed", "measurements_rerun": False}
        report["updated_windows"] = verify_updated_controlled(roots["windows"], "Windows", observations)
        report["updated_linux"] = verify_updated_controlled(roots["linux"], "Linux", observations)
        require(len(observations) == 4, "Four current controlled environments are required")
        comparisons = 0
        for index, (left_report, left, left_schedule) in enumerate(observations):
            for right_report, right, right_schedule in observations[index + 1:]:
                require(left.keys() == right.keys() and all(left[key] == right[key] for key in left),
                        "Four-environment objectives or recorded cardinalities differ")
                require(left_schedule == right_schedule, "Four-environment schedules differ")
                require(left_report["input_digests"] == right_report["input_digests"],
                        "Four-environment frozen inputs differ")
                comparisons += 1
        report["cross_environment"] = {"environment_pairs": comparisons,
            "exact_objectives_per_pair": 3925, "nonzero_objective_differences": 0,
            "recorded_cardinality_differences": 0, "same_schedules": True, "same_frozen_inputs": True}
        report["updated_application"] = verify_application_update(roots["application"])
        report["updated_lsape"] = verify_lsape_update(roots["lsape"])
        report["paper_tables_recomputed"] = verify_paper_tables(**roots)
        report["updated_figures"] = verify_updated_figures(roots["windows"])
        report["updated_linux_figures"] = verify_updated_figures(roots["linux"], "linux", roots)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows-results", type=Path, default=None,
                        help="Current Windows bundle or directory (default: controlled-windows-20261029.zip)")
    parser.add_argument("--linux-results", type=Path, default=None,
                        help="Current Linux bundle or directory (default: controlled-linux-20261029.zip)")
    parser.add_argument("--application-results", type=Path, default=None,
                        help="Current tracker/replay bundle (default: application-windows-20261029.zip)")
    parser.add_argument("--lsape-results", type=Path, default=None,
                        help="Current LSAPE bundle (default: lsape-windows-20261029.zip)")
    args = parser.parse_args()
    try:
        report = verify(args.windows_results, args.linux_results, args.application_results, args.lsape_results)
    except OSError as error:
        print("Result verification failed:", error.strerror or type(error).__name__, file=sys.stderr)
        return 1
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        print("Result verification failed:", type(error).__name__, str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
