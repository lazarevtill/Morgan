"""Render a :class:`~tests.e2e.harness.BenchReport` to JSON and markdown."""

from __future__ import annotations

import json

from tests.e2e.harness import BenchReport


def to_dict(report: BenchReport) -> dict[str, object]:
    return {
        "mode": report.mode,
        "generated_at": report.generated_at,
        "summary": {
            "passed": report.passed_count,
            "failed": report.failed_count,
            "skipped": report.skipped_count,
            "recall_accuracy": round(report.recall_accuracy(), 4),
            "latency_p50_ms": round(report.latency_percentile(50), 3),
            "latency_p95_ms": round(report.latency_percentile(95), 3),
            "turns_measured": len(report.all_latencies),
        },
        "scenarios": [
            {
                "name": r.name,
                "category": r.category,
                "status": r.status,
                "detail": r.detail or r.skip_reason,
                "turn_latencies_ms": [round(x, 3) for x in r.turn_latencies_ms],
            }
            for r in report.results
        ],
    }


def to_json(report: BenchReport) -> str:
    return json.dumps(to_dict(report), indent=2)


def to_markdown(report: BenchReport) -> str:
    d = to_dict(report)
    summary = d["summary"]
    assert isinstance(summary, dict)
    lines: list[str] = []
    lines.append("# Morgan text E2E benchmark")
    lines.append("")
    lines.append(f"- **mode:** {report.mode}")
    lines.append(f"- **generated:** {report.generated_at}")
    lines.append(
        f"- **result:** {summary['passed']} passed, "
        f"{summary['failed']} failed, {summary['skipped']} skipped"
    )
    lines.append(f"- **recall accuracy:** {summary['recall_accuracy']}")
    lines.append(
        f"- **latency:** p50={summary['latency_p50_ms']} ms, "
        f"p95={summary['latency_p95_ms']} ms "
        f"({summary['turns_measured']} turns)"
    )
    lines.append("")
    lines.append("| scenario | category | status | detail |")
    lines.append("|----------|----------|--------|--------|")
    for r in report.results:
        detail = (r.detail or r.skip_reason).replace("|", "\\|")
        lines.append(f"| {r.name} | {r.category} | {r.status} | {detail} |")
    lines.append("")
    return "\n".join(lines)
