"""Build, validate or run the separate five-arm LSAPE experiment."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "smoke", "run", "summarize"))
    args, rest = parser.parse_known_args(argv)
    if args.command == "build":
        from .build_bridge import main as execute
    elif args.command == "verify":
        from .verify_implementation import main as execute
    elif args.command == "summarize":
        from .summarize_rectangular import main as execute
    else:
        from .rectangular_campaign import main as execute
        if args.command == "smoke":
            rest.append("--smoke")
    return execute(rest)


if __name__ == "__main__":
    sys.exit(main())
