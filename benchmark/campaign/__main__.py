"""Safe entry point: inventory/validation/smoke never start a full session."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .catalog import build_manifest
from .common import read_json, write_json


def validation_report(manifest, build_index=None):
    from .validation import validate_methods, fixtures, batch_fixtures
    index = read_json(build_index) if build_index else None
    methods = sorted({m for c in manifest["cases"] for m in c["methods"]})
    return validate_methods(methods, records=fixtures() + batch_fixtures(), build_index=index)


DEFAULT_NAMES = {"manifest": "campaign-plan.json", "validate": "campaign-validation.json",
                 "diagnose": "campaign-diagnostics", "smoke": "campaign-smoke", "run": "campaign-session"}


def default_output(args):
    """New outputs go in this minute's run folder; derived ones sit beside their input."""
    if args.command == "estimate":
        return args.smoke_root.resolve().parent / "campaign-estimate"
    if args.command == "summarize":
        return args.sessions[0].resolve().parent / "campaign-summary.json"
    from ..paths import default_output as run_output
    return run_output(DEFAULT_NAMES[args.command])


def selected_manifest(args, parser):
    """Derived reports use the recorded protocol, including relocated sessions."""
    if args.command == "summarize":
        manifests = [read_json(path) for session in args.sessions
                     if (path := session / "manifest.json").is_file()]
        profiles = {manifest.get("profile", "full") for manifest in manifests}
        if len(profiles) > 1:
            parser.error("Summarize full and light campaigns separately")
        if manifests:
            if args.profile is not None and args.profile not in profiles:
                parser.error("--profile differs from the recorded session profile")
            return manifests[0]
    elif args.command == "estimate" and args.profile is None:
        path = args.smoke_root / "manifest.json"
        if path.is_file():
            return read_json(path)
    return build_manifest(profile="light") if args.profile == "light" else build_manifest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("manifest", help="Write complete planned coverage without generating inputs")
    inventory.add_argument("--output", type=Path, help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-plan.json")
    commands.add_parser("topology", help="Inspect physical cores and suggest verified placement")
    validate = commands.add_parser("validate", help="Exact tiny oracle gates, including controlled arms")
    validate.add_argument("--build-index", type=Path)
    validate.add_argument("--output", type=Path, help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-validation.json")
    diagnose = commands.add_parser("diagnose", help="Separate bounded input/layout/native-counter diagnostics")
    diagnose.add_argument("--output", type=Path, help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-diagnostics")
    diagnose.add_argument("--build-index", type=Path, help="Separate diagnostic build index; primary artifacts rejected")
    diagnose.add_argument("--full", action="store_true")
    diagnose.add_argument("--allow-full-campaign", action="store_true")
    diagnose.add_argument("--cpus", nargs=4, type=int)
    diagnose.add_argument("--single-cpu", type=int)
    for name in ("smoke", "run"):
        p = commands.add_parser(name, help="Bounded development smoke" if name == "smoke" else "One explicitly enabled environment session")
        p.add_argument("--output", type=Path, help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-" + ("smoke" if name == "smoke" else "session"))
        p.add_argument("--build-index", type=Path, required=True)
        p.add_argument("--cpus", nargs=4, type=int)
        p.add_argument("--single-cpu", type=int)
        if name == "run":
            p.add_argument("--allow-full-campaign", action="store_true",
                           help="Required explicit opt-in; never supplied by smoke")
            p.add_argument("--case-start", type=int, help="First zero-based frozen manifest case index")
            p.add_argument("--case-stop", type=int, help="Exclusive frozen manifest case index")
            p.add_argument("--case-id", action="append", help="Select a case ID; repeat for several cases, in manifest order")
    estimate = commands.add_parser("estimate", help="Estimate runtime from bounded smoke; executes no solver")
    estimate.add_argument("--smoke-root", type=Path, required=True)
    estimate.add_argument("--output", type=Path, help="Default: campaign-estimate beside the smoke root")
    aggregate = commands.add_parser("summarize", help="Strict paired aggregation; incomplete cells remain unavailable")
    aggregate.add_argument("sessions", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, help="Default: campaign-summary.json beside the first session")
    for name in ("manifest", "validate", "diagnose", "estimate", "summarize", "smoke", "run"):
        commands.choices[name].add_argument(
            "--profile", choices=("full", "light"),
            help="Campaign coverage; default full, or the recorded profile for derived reports")
    args = parser.parse_args(argv)
    if args.command != "topology" and args.output is None:
        args.output = default_output(args)
        print(f"Output: {args.output}")
    if args.command == "run" and not args.allow_full_campaign:
        parser.error("Full sessions are disabled by default; --allow-full-campaign is required")
    if args.command == "diagnose" and args.full and not args.allow_full_campaign:
        parser.error("Full diagnostics require --full --allow-full-campaign")
    manifest = selected_manifest(args, parser) if args.command != "topology" else build_manifest()
    selected_cases = None
    if args.command == "run":
        from .runner import select_cases
        try:
            selected_cases = select_cases(manifest, case_start=args.case_start,
                                          case_stop=args.case_stop, case_ids=args.case_id)
        except ValueError as error:
            parser.error(str(error))
    if args.command == "manifest":
        write_json(args.output, manifest)
        print(manifest["counts"])
    elif args.command == "topology":
        from .topology import discover
        import json
        print(json.dumps(discover(), indent=2))
    elif args.command == "validate":
        report = validation_report(manifest, args.build_index)
        write_json(args.output, report)
        print({key: report[key] for key in ("passed", "unsupported", "correctness_failed")})
        return int(bool(report["correctness_failed"]))
    elif args.command == "diagnose":
        from .diagnostics import run_diagnostics
        summary = run_diagnostics(manifest, args.output, args.build_index, full=args.full,
                                  cpus=args.cpus, single_cpu=args.single_cpu)
        return int(any(job["status"] not in ("complete", "resource_excluded", "postponed")
                       for job in summary["jobs"]))
    elif args.command in ("smoke", "run"):
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Choose a fresh output directory; existing evidence is never overwritten")
        # Mandatory correctness gate precedes any measurement. Missing controlled
        # artifacts are fatal here even though the inventory retains unavailable arms.
        index = read_json(args.build_index)
        from .native_build import ALL_ARMS, resolve_build, load_solver
        for language in ("rust", "cpp"):
            for arm in ALL_ARMS:
                load_solver(resolve_build(index, language, arm))
        report = validation_report(manifest, args.build_index)
        if report["correctness_failed"] or report["unsupported"]:
            write_json(args.output / "validation-failed.json", report)
            raise RuntimeError("Tiny validation gate failed or a planned method has no applicable passing fixture")
        from .runner import run_session
        summary = run_session(manifest, args.output, args.build_index, smoke=args.command == "smoke",
                              cpus=args.cpus, single_cpu=args.single_cpu, selected_cases=selected_cases)
        write_json(args.output / "validation.json", report)
        print(f"{summary['scope']}: {summary['wall_seconds']:.1f} seconds; {args.output.resolve()}")
        if summary["status"] == "postponed":
            print(f"Stopped for insufficient host memory; {len(summary['remaining_case_ids'])} selected cases remain. "
                  "Retry the affected case and remaining work in a fresh output directory.")
            return 2
        timing_failed = any(j["status"] != "complete" or
                            any(s["status"] == "correctness_failed" for s in j["statuses"].values())
                            for j in summary["jobs"])
        memory_failed = any(
            item.get("admission", {}).get("methods", {}).get(item.get("method"), {}).get("status") == "postponed"
            or ("watchdog" in item and
                (not item.get("result") or item["result"].get("status") != "complete"
                 or item["watchdog"].get("exit_code") != 0 or item["watchdog"].get("reason")))
            for item in summary["memory"])
        return int(summary["status"] != "finished" or bool(summary.get("remaining_case_ids"))
                   or timing_failed or memory_failed)
    elif args.command == "estimate":
        from .analysis import write_estimate
        write_estimate(manifest, args.smoke_root, args.output)
    elif args.command == "summarize":
        from .analysis import summarize, read_records, summarize_memory
        records = [record for session in args.sessions for record in read_records(session)]
        report = summarize(records, manifest=manifest)
        report["memory"] = summarize_memory(args.sessions)
        write_json(args.output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
