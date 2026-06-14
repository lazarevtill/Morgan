"""Run the text E2E benchmark as a script and emit JSON + markdown reports.

Usage::

    # deterministic (default, zero external services)
    python -m tests.e2e.run_bench

    # live (uses the configured LLM endpoint + qdrant if reachable; skips if not)
    MORGAN_BENCH_LIVE=1 python -m tests.e2e.run_bench

    # custom output dir
    python -m tests.e2e.run_bench --out ./data/bench

Exit code is 0 when every (non-skipped) scenario passes, 1 otherwise. Skipped
scenarios (live mode, services absent) do NOT fail the run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys

from tests.e2e.harness import run_all
from tests.e2e.report import to_json, to_markdown


def _is_live() -> bool:
    return os.environ.get("MORGAN_BENCH_LIVE", "").strip() in ("1", "true", "yes", "on")


async def _amain(out_dir: pathlib.Path, live: bool) -> int:
    report = await run_all(live=live)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "text_e2e_report.json"
    md_path = out_dir / "text_e2e_report.md"
    json_path.write_text(to_json(report), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    # Echo the markdown to stdout for immediate visibility.
    print(to_markdown(report))
    print(f"\nJSON:     {json_path}")
    print(f"Markdown: {md_path}")

    # Skipped scenarios do not fail the run; only real failures do.
    return 1 if report.failed_count > 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Morgan text E2E benchmark")
    parser.add_argument(
        "--out",
        default="./data/bench",
        help="Output directory for JSON + markdown reports (default: ./data/bench)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live mode (equivalent to MORGAN_BENCH_LIVE=1)",
    )
    args = parser.parse_args(argv)
    live = args.live or _is_live()
    return asyncio.run(_amain(pathlib.Path(args.out), live))


if __name__ == "__main__":
    sys.exit(main())
