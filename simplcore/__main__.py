from __future__ import annotations

import argparse
import json

from . import __version__, report_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m simplcore",
        description="SimplCore project info and environment diagnostics.",
    )
    parser.add_argument("--version", action="store_true",
                        help="print the package version and exit")
    parser.add_argument("--doctor", action="store_true",
                        help="print a compact environment report")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.doctor:
        print(json.dumps(report_environment(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
