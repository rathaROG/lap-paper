"""Audit saved microscopy evidence, or explicitly launch a new local campaign."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Check saved records; no solver dependencies")
    for option in ("table", "compact-table", "profile-table"):
        audit.add_argument("--" + option, type=Path)
    run = commands.add_parser("run", help="Collect new timings from separately supplied local data")
    run.add_argument("--allow-full-campaign", action="store_true")
    run.add_argument("--ctc", type=Path, required=True)
    run.add_argument("--hydra", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        command = [sys.executable, "-B", str(ROOT / "recalculate.py")]
        for option in ("table", "compact_table", "profile_table"):
            value = getattr(args, option)
            if value is not None:
                command.extend(["--" + option.replace("_", "-"), str(value)])
    else:
        if not args.allow_full_campaign:
            parser.error("New measurements require --allow-full-campaign")
        output = args.output.resolve()
        if output.is_relative_to(ROOT) or output.exists():
            parser.error("Use a new output file outside benchmark/cell_tracking; frozen evidence cannot be replaced")
        if not args.ctc.is_dir() or not args.hydra.is_file():
            parser.error("The CTC directory and Hydra video must already exist locally")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", str(ROOT / "benchmark.py"),
                   "--ctc", str(args.ctc), "--hydra", str(args.hydra), "--output", str(output)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
