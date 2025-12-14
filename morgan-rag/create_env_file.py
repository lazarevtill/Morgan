"""
Utility to generate a working .env from .env.example.

Usage:
  python create_env_file.py               # create .env if missing
  python create_env_file.py --force       # overwrite existing .env
  python create_env_file.py --output foo  # write to a different path
"""

import argparse
from pathlib import Path
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Create .env from .env.example")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(".env"),
        help="Output env file path (default: .env)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        default=Path(".env.example"),
        help="Source env example file (default: .env.example)",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Source file not found: {args.source}", file=sys.stderr)
        return 1

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    try:
        shutil.copyfile(args.source, args.output)
    except Exception as exc:
        print(f"Failed to write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {args.output} from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
