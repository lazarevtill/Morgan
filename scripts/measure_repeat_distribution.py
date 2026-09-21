"""How much does re-embedding the same text move the vector? -- the number the fingerprint
tolerance (Task 14) is set from.

Morgan's fingerprint check compares a freshly-embedded query against a vector stored earlier.
If the embedding server is not perfectly deterministic -- batching, concurrency, and a model
reload can all perturb floating-point results -- two embeddings of the *same* text are not
bit-identical, and the fingerprint check needs a tolerance wide enough to absorb that noise
without also absorbing a real semantic drift. This script measures the noise.

It takes a fixed sample of stored memories from a snapshot database, re-embeds every one of
them under each of a grid of conditions (batch size x warm-or-cold x client count), and reports
how far each re-embedding drifts (by cosine similarity) from the vector already stored for that
memory. min/p1/median per condition is what ``docs/measurements/2026-09-phase0-baseline.md``
records; the worst condition's p1 sets the tolerance (see the baseline file for the formula).

Reads only -- opens *--db* with a read-only SQLite URI and never runs Morgan's own
``open_db`` (which switches journal mode, a write). Never pass a database another process is
actively writing without taking your own copy first; this script does not need one, since it
performs no write of its own, but a moving target changes the answer between runs.

Resumable: every finished condition is written into *--results-json* immediately, and that
same file is read back on the next invocation to skip whatever is already there -- so an
interrupted run picks back up instead of restarting the whole grid. Nothing about *which* rows
were sampled is stored: the sample is a deterministic function of (db, --rows), so a resumed
run reselects the same rows without ever persisting memory content or ids to disk.

Conditions and their wait cost, once run in full (`--rows 500`, no `--quick`):
    3 batch sizes x 2 (warm/cold) x 2 client counts = 12 conditions x 500 rows = ~6,000 calls.
    Each "cold" condition waits, on its own, for the embedding host to report the model
    unloaded (native Ollama ``/api/ps``) AND for 30 minutes to have passed since this script's
    own last request -- 6 such waits in the full grid. ``--cold-starts N`` adds N more of the
    same wait, each followed by timing a single embed call rather than a whole condition; it
    measures load latency, not drift, and is otherwise independent of the grid above.

``--quick`` shrinks the grid to warm-only, batch sizes {1, 8}, both client counts -- enough to
exercise every code path (sampling, batching, single- and two-client dispatch, progress
persistence) without an idle wait, for a fast smoke test:

    measure_repeat_distribution.py --db SNAPSHOT.db --quick --rows 5

The full run (see docs/measurements/2026-09-phase0-baseline.md for the exact invocation used):

    measure_repeat_distribution.py --db SNAPSHOT.db --rows 500 --cold-starts 3 \\
        --results-json OUT.json --results-md OUT.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import struct
import sys
import time
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import sqlite_vec  # type: ignore[import-untyped]

#: A neutral string for warm-up pings and cold-start timing -- never real memory content.
_PROBE_TEXT = "cold start probe"

FULL_BATCH_SIZES: tuple[int, ...] = (1, 8, 32)
FULL_WARMTH: tuple[str, ...] = ("warm", "cold")
FULL_CONCURRENCY: tuple[int, ...] = (1, 2)

QUICK_BATCH_SIZES: tuple[int, ...] = (1, 8)
QUICK_WARMTH: tuple[str, ...] = ("warm",)
QUICK_CONCURRENCY: tuple[int, ...] = (1, 2)


# --------------------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Condition:
    batch_size: int
    warmth: str  # "warm" | "cold"
    concurrency: int

    @property
    def key(self) -> str:
        return f"batch{self.batch_size}_{self.warmth}_c{self.concurrency}"


def build_plan(*, quick: bool) -> list[Condition]:
    batches = QUICK_BATCH_SIZES if quick else FULL_BATCH_SIZES
    warmths = QUICK_WARMTH if quick else FULL_WARMTH
    concurrencies = QUICK_CONCURRENCY if quick else FULL_CONCURRENCY
    return [
        Condition(batch_size=b, warmth=w, concurrency=c)
        for w in warmths
        for b in batches
        for c in concurrencies
    ]


# ------------------------------------------------------------------------------------ sampling


@dataclass(frozen=True)
class SampleRow:
    id: str
    content: str
    vector: tuple[float, ...]


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """A read-only connection with sqlite-vec loaded -- never ``morgan_brain``'s ``open_db``,
    which switches journal mode (a write) as a side effect of opening."""
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return conn


def sample_rows(conn: sqlite3.Connection, rows: int) -> list[SampleRow]:
    """*rows* memories spread evenly across the table, each with its stored vector.

    Deterministic in (table contents, *rows*): the same call against the same snapshot always
    returns the same sample, so nothing about the sample needs to be persisted for a resume.
    """
    all_rows = conn.execute(
        "SELECT m.id AS id, mm.content AS content, v.embedding AS embedding "
        "FROM vec_items v "
        "JOIN vec_meta m ON m.rowid = v.rowid "
        "JOIN memories mm ON mm.id = m.id "
        "ORDER BY v.rowid"
    ).fetchall()
    total = len(all_rows)
    if total == 0:
        raise SystemExit("no rows found via the vec_items/vec_meta/memories join")

    n = min(rows, total)
    indices: Sequence[int] = range(total) if n == total else [int(i * total / n) for i in range(n)]
    dim = len(all_rows[0]["embedding"]) // 4
    sampled = []
    for i in indices:
        row = all_rows[i]
        blob: bytes = row["embedding"]
        vector = struct.unpack(f"{dim}f", blob)
        sampled.append(SampleRow(id=row["id"], content=row["content"], vector=vector))
    return sampled


# --------------------------------------------------------------------------------------- math


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default method), no numpy required."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


# ------------------------------------------------------------------------------- embedding I/O


@dataclass
class ColdState:
    """When the script last talked to the embedding server -- the clock ``--cold-starts`` and
    every "cold" condition wait against. Persisted across runs so a resume does not reset it."""

    last_request_at: datetime | None = None

    def mark(self) -> None:
        self.last_request_at = datetime.now(UTC)

    def elapsed(self) -> timedelta:
        if self.last_request_at is None:
            return timedelta.max
        return datetime.now(UTC) - self.last_request_at


def native_ps_url(embedding_endpoint: str) -> str:
    """The Ollama-native ``GET .../api/ps`` URL, derived from the OpenAI-compatible endpoint's
    scheme and host:port -- the two commonly differ only in path (``/v1`` vs none)."""
    parts = urlsplit(embedding_endpoint)
    return urlunsplit((parts.scheme, parts.netloc, "/api/ps", "", ""))


async def is_model_unloaded(client: httpx.AsyncClient, ps_url: str, model: str) -> bool:
    try:
        resp = await client.get(ps_url, timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        # The native endpoint is unreachable or erroring -- an unknown state is not a known
        # "unloaded" one, so the wait keeps polling rather than declaring victory.
        return False
    payload = resp.json()
    needle = model.split(":")[0].strip().lower()
    if not needle:
        return True
    for entry in payload.get("models", []):
        name = str(entry.get("model") or entry.get("name") or "").lower()
        if needle in name:
            return False
    return True


async def wait_until_cold(
    client: httpx.AsyncClient,
    ps_url: str,
    model: str,
    state: ColdState,
    *,
    idle: timedelta,
    poll_interval: float,
    log: Any,
) -> None:
    """Block until *model* is unloaded on the host AND *idle* has passed since the script's
    own last request -- both conditions, together, every time this is called."""
    while True:
        elapsed = state.elapsed()
        unloaded = await is_model_unloaded(client, ps_url, model)
        if unloaded and elapsed >= idle:
            return
        remaining = max(0.0, (idle - elapsed).total_seconds())
        log(f"waiting for cold state: unloaded={unloaded} remaining>={remaining:.0f}s")
        await asyncio.sleep(poll_interval)


async def post_embeddings(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    texts: list[str],
    headers: dict[str, str],
    request_timeout: float,
    state: ColdState,
) -> list[list[float]]:
    resp = await client.post(
        url, json={"model": model, "input": texts}, headers=headers, timeout=request_timeout
    )
    state.mark()
    resp.raise_for_status()
    data: list[dict[str, Any]] = resp.json()["data"]
    return [item["embedding"] for item in data]


async def embed_condition(
    *,
    url: str,
    model: str,
    headers: dict[str, str],
    request_timeout: float,
    texts: list[str],
    batch_size: int,
    concurrency: int,
    state: ColdState,
) -> tuple[list[list[float]], float]:
    """Embed *texts* in batches of *batch_size*, spread round-robin over *concurrency* clients
    running concurrently. Returns the vectors in input order, and the wall time taken."""
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results: list[list[list[float]] | None] = [None] * len(batches)

    async def worker(client: httpx.AsyncClient, batch_indices: list[int]) -> None:
        for idx in batch_indices:
            results[idx] = await post_embeddings(
                client, url, model, batches[idx], headers, request_timeout, state
            )

    start = time.monotonic()
    async with AsyncExitStack() as stack:
        clients = [await stack.enter_async_context(httpx.AsyncClient()) for _ in range(concurrency)]
        assignments: list[list[int]] = [[] for _ in range(concurrency)]
        for i in range(len(batches)):
            assignments[i % concurrency].append(i)
        await asyncio.gather(
            *(worker(clients[c], assignments[c]) for c in range(concurrency) if assignments[c])
        )
    elapsed = time.monotonic() - start

    vectors: list[list[float]] = []
    for batch_result in results:
        if batch_result is None:
            raise RuntimeError("a batch was never assigned to a worker")
        vectors.extend(batch_result)
    return vectors, elapsed


# ------------------------------------------------------------------------------------- results


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"meta": {}, "conditions": {}, "cold_starts": {}, "last_request_at": None}
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def save_results(path: Path, data: dict[str, Any]) -> None:
    """Atomic write (temp file + replace) so a crash mid-save never leaves a truncated,
    unparseable progress file behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def render_markdown(data: dict[str, Any]) -> str:
    meta = data["meta"]
    lines = [
        "# Repeat-distribution measurement",
        "",
        f"- db: `{meta.get('db')}`",
        f"- endpoint: `{meta.get('endpoint')}`",
        f"- model: `{meta.get('model')}`",
        f"- rows sampled: {meta.get('rows')}",
        f"- embedding dim: {meta.get('dim')}",
        "",
        "| condition | batch | warmth | clients | n | min cosine | p1 cosine | "
        "median cosine | wall s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(data["conditions"]):
        c = data["conditions"][key]
        lines.append(
            f"| {key} | {c['batch_size']} | {c['warmth']} | {c['concurrency']} | {c['n']} | "
            f"{c['min_cosine']:.5f} | {c['p1_cosine']:.5f} | {c['median_cosine']:.5f} | "
            f"{c['wall_seconds']:.1f} |"
        )
    if data["cold_starts"]:
        lines += ["", "| cold start # | seconds |", "|---:|---:|"]
        for key in sorted(data["cold_starts"]):
            lines.append(f"| {key} | {data['cold_starts'][key]['seconds']:.1f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------- cli


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--db", required=True, type=Path, help="Path to a morgan.db snapshot.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MORGAN_EMBEDDING_ENDPOINT", ""),
        help="OpenAI-compatible embedding endpoint. Defaults to $MORGAN_EMBEDDING_ENDPOINT.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MORGAN_EMBEDDING_MODEL", ""),
        help="Embedding model name. Defaults to $MORGAN_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Outbound bearer token for the embedding endpoint, if it enforces one. Never "
        "the chat/LLM key -- pass explicitly, this script reads no chat setting.",
    )
    parser.add_argument("--rows", type=int, default=500, help="Stored memories to sample.")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between /api/ps polls while waiting for cold.",
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=30.0,
        help="Idle time required before a cold condition or --cold-starts run proceeds.",
    )
    parser.add_argument(
        "--cold-starts",
        type=int,
        default=0,
        help="Extra cold-load timing runs (each waits for idle, then times one embed).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink the grid for a fast smoke test: warm only, batch sizes 1 and 8.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Also the progress file: finished conditions are skipped on a re-run. Defaults "
        "to <db stem>.repeat-distribution.json in the current directory.",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        default=None,
        help="Defaults to <db stem>.repeat-distribution.md in the current directory.",
    )
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _record_last_request(data: dict[str, Any], state: ColdState) -> None:
    at = state.last_request_at
    data["last_request_at"] = at.isoformat() if at else None


async def run_grid(
    args: argparse.Namespace, rows: list[SampleRow], data: dict[str, Any], dim: int
) -> None:
    url = args.endpoint.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    ps_url = native_ps_url(args.endpoint)
    idle = timedelta(minutes=args.idle_minutes)
    state = ColdState(
        last_request_at=(
            datetime.fromisoformat(data["last_request_at"]) if data["last_request_at"] else None
        )
    )
    stored = [r.vector for r in rows]
    texts = [r.content for r in rows]

    async with httpx.AsyncClient() as probe_client:
        for condition in build_plan(quick=args.quick):
            if condition.key in data["conditions"]:
                log(f"skip {condition.key}: already in {args.results_json}")
                continue

            if condition.warmth == "cold":
                log(f"{condition.key}: waiting for a cold model")
                await wait_until_cold(
                    probe_client,
                    ps_url,
                    args.model,
                    state,
                    idle=idle,
                    poll_interval=args.poll_interval,
                    log=log,
                )
            else:
                await post_embeddings(
                    probe_client,
                    url,
                    args.model,
                    [_PROBE_TEXT],
                    headers,
                    args.request_timeout,
                    state,
                )

            log(f"{condition.key}: embedding {len(texts)} rows")
            vectors, elapsed = await embed_condition(
                url=url,
                model=args.model,
                headers=headers,
                request_timeout=args.request_timeout,
                texts=texts,
                batch_size=condition.batch_size,
                concurrency=condition.concurrency,
                state=state,
            )
            scores = [cosine(fresh, ref) for fresh, ref in zip(vectors, stored, strict=True)]
            data["conditions"][condition.key] = {
                "batch_size": condition.batch_size,
                "warmth": condition.warmth,
                "concurrency": condition.concurrency,
                "n": len(scores),
                "min_cosine": min(scores),
                "p1_cosine": percentile(scores, 1.0),
                "median_cosine": median(scores),
                "wall_seconds": elapsed,
            }
            _record_last_request(data, state)
            save_results(args.results_json, data)
            log(
                f"{condition.key}: min={min(scores):.5f} p1={percentile(scores, 1.0):.5f} "
                f"median={median(scores):.5f} ({elapsed:.1f}s)"
            )

        for i in range(1, args.cold_starts + 1):
            key = f"cold_start_{i}"
            if key in data["cold_starts"]:
                log(f"skip {key}: already in {args.results_json}")
                continue
            log(f"{key}: waiting for a cold model")
            await wait_until_cold(
                probe_client,
                ps_url,
                args.model,
                state,
                idle=idle,
                poll_interval=args.poll_interval,
                log=log,
            )
            start = time.monotonic()
            await post_embeddings(
                probe_client, url, args.model, [_PROBE_TEXT], headers, args.request_timeout, state
            )
            seconds = time.monotonic() - start
            data["cold_starts"][key] = {"seconds": seconds}
            _record_last_request(data, state)
            save_results(args.results_json, data)
            log(f"{key}: {seconds:.1f}s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.endpoint:
        log("no --endpoint given and $MORGAN_EMBEDDING_ENDPOINT is unset")
        return 2
    if not args.model:
        log("no --model given and $MORGAN_EMBEDDING_MODEL is unset")
        return 2

    results_json = args.results_json or Path(f"{args.db.stem}.repeat-distribution.json")
    results_md = args.results_md or Path(f"{args.db.stem}.repeat-distribution.md")
    args.results_json, args.results_md = results_json, results_md

    conn = connect_readonly(args.db)
    try:
        rows = sample_rows(conn, args.rows)
    finally:
        conn.close()
    dim = len(rows[0].vector)

    data = load_results(results_json)
    meta = {
        "db": str(args.db),
        "endpoint": args.endpoint,
        "model": args.model,
        "rows": len(rows),
        "dim": dim,
    }
    if data["meta"] and data["meta"] != meta:
        changed = sorted(k for k in meta if data["meta"].get(k) != meta[k])
        log(
            f"{results_json} was written with different parameters ({', '.join(changed)} "
            f"changed); use a different --results-json or matching --db/--endpoint/--model/"
            f"--rows. Never printing the values here -- an endpoint or path is not something "
            f"to echo into a terminal transcript."
        )
        return 2
    data["meta"] = meta
    save_results(results_json, data)

    log(f"sampled {len(rows)} rows (dim={dim}) from {args.db}")
    asyncio.run(run_grid(args, rows, data, dim))

    results_md.write_text(render_markdown(data), encoding="utf-8")
    log(f"wrote {results_json} and {results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
