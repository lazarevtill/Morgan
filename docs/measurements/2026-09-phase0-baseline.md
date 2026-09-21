# Phase 0 baseline -- 2026-09-21

Recorded against commit `24f1566` on `feat/phase0-floor-and-safety`, from the worktree
checkout (`.venv`, Python 3.13.15, win32). Nothing later in phase 0 can be shown to have
moved anything without this file: every number below is what Task 4 onward diffs against.

Endpoints are written as `<chat-endpoint>` / `<embedding-endpoint>` and home paths as `~`
throughout -- see `scripts/measure_repeat_distribution.py`'s own docstring for the redaction
rule this file follows. The unredacted raw outputs referenced below live outside this
repository, under `~/Documents/GitHub/morgan-research-2026-09-19/measurements/`.

## 1. The deterministic numbers

### The four gates

```
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
309 passed, 4 skipped, 1 warning in 132.29s (0:02:12)
```

The one warning is `starlette.testclient`'s `BlockingPortal` deprecation -- pre-existing,
third-party, not from this repository's code.

```
.venv/Scripts/python.exe -m ruff check .
All checks passed!
```

```
.venv/Scripts/python.exe -m ruff format --check .
130 files already formatted
```

```
.venv/Scripts/python.exe -m mypy morgan_brain
Success: no issues found in 48 source files
```

```
.venv/Scripts/python.exe -m bandit -q -c pyproject.toml -r morgan_brain
(no output -- zero findings)
```

### Hash scorecard, with its `RunConfig`

`tests/memory_quality/test_retrieval_quality.py`'s plumbing test
(`test_a_run_completes_and_scores_over_the_labelled_set`) asserts only that a run completes
and scores -- on purpose, per its own docstring: a hash embedder has no semantic similarity,
so any quality number it produced would be an artefact of sha256, and the test prints nothing.
This is the phase-0 exit criterion ("the hash scorecard unchanged to the digit"), so whoever
runs that exit must be able to reproduce this card exactly -- the exact driver, runnable from
the worktree root with the same interpreter as every other command in this file:

```python
import asyncio
import subprocess
from pathlib import Path

from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.eval.retrieval import describe_run, load_probe_set, run_probes, score_run
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db

PROBES = Path("tests/memory_quality/probes.json")
K = 8


def commit() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip() or None


async def main() -> None:
    settings = Settings(embedding_backend="hash", embedding_dim=64)
    conn = open_db(":memory:")
    module = build_memory_module(conn=conn, embedder=FakeEmbedder(dim=64), dim=64)
    probe_set = load_probe_set(PROBES)

    results = await run_probes(probe_set, gate=MemoryGate(module), user_id="owner", k=K)
    card = score_run(results, k=K)
    run = describe_run(
        settings=settings, probe_path=PROBES, probe_set=probe_set, conn=conn,
        k=K, floor_margin=None, commit=commit(),
    )
    print(run.format())
    print(card.format(K))


asyncio.run(main())
```

Save as e.g. `hash_scorecard.py` in the worktree root and run
`.venv/Scripts/python.exe hash_scorecard.py` (not committed -- it duplicates no production
logic, only composes the same public API the live test below already imports). Output:

```
run: embedding=hash stub (dim 64)  k=8  floor=off  probes=probes.json@510b5f20fe94 (86 memories, 60 probes)  db-upgrade=2  commit=24f1566
n=60  recall@8=0.28  mrr=0.09  stale1st=0.60  abstain=0.00
  knowledge_update  n=10   recall@8=0.30  mrr=0.09  stale1st=0.60  abstain=0.00
  multi_hop         n=8    recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
  single_hop        n=20   recall@8=0.30  mrr=0.09  stale1st=0.00  abstain=0.00
  temporal          n=8    recall@8=0.50  mrr=0.21  stale1st=0.00  abstain=0.00
  unanswerable      n=14   recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
```

### `morgan doctor --json` -- the live install of `main`, read-only use

From `~` (home folder):

```json
{
  "database": "~/.local/share/morgan/morgan.db",
  "config_file": "~/.config/morgan/.env",
  "config_file_present": true,
  "project": "default",
  "all_projects": false,
  "embedding_backend": "provider",
  "embedding_dim": 4096,
  "embedding_endpoint": "<embedding-endpoint>",
  "llm_endpoint": "<chat-endpoint>",
  "llm_model": "<chat-model>",
  "sqlite_vec": "v0.1.9",
  "fts5": true,
  "provider": "reachable",
  "embedding_provider": "reachable",
  "vector_rows": 0,
  "memory_rows": 0,
  "fts_rows": 0
}
```

From inside the Morgan checkout (`~/Documents/GitHub/Morgan` -- read-only use, nothing
changed there):

```json
{
  "database": "~/.local/share/morgan/morgan.db",
  "config_file": "~/.config/morgan/.env",
  "config_file_present": true,
  "project": "Morgan",
  "all_projects": false,
  "embedding_backend": "provider",
  "embedding_dim": 4096,
  "embedding_endpoint": "<embedding-endpoint>",
  "llm_endpoint": "<chat-endpoint>",
  "llm_model": "<chat-model>",
  "sqlite_vec": "v0.1.9",
  "fts5": true,
  "provider": "reachable",
  "embedding_provider": "reachable",
  "vector_rows": 0,
  "memory_rows": 0,
  "fts_rows": 0
}
```

`vector_rows`/`memory_rows`/`fts_rows` are scoped to the *current* project (`default` and
`Morgan` respectively, neither of which the archive import used) -- zero there is correct, not
an empty database. See the row counts below for what the database actually holds.

### Live database row counts -- read-only

Database: `~/.local/share/morgan/morgan.db` (the owner's live database -- not the eval
snapshot used elsewhere in this file). Opened with a raw `sqlite3` URI,
`file:~/.local/share/morgan/morgan.db?mode=ro`, `sqlite-vec` loaded directly into that
connection, never through `morgan_brain`'s `open_db` (which switches journal mode -- a write
-- as a side effect of opening). Nothing in this database was written. Raw output (unredacted
path):
`~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-live-db-row-counts.txt`.

Per table:

| table | rows |
|---|---:|
| memories | 3010 |
| vec_meta | 3010 |
| vec_items | 3010 |
| memory_entities | 43936 |
| facts | 0 |
| fts_memories | 3010 |
| session_history | 0 |

Per project (`facts` and `session_history` are empty, so nothing to split):

| table | archive/chatgpt | archive/chatgpt-holdout |
|---|---:|---:|
| memories | 1966 | 1044 |
| vec_meta | 1966 | 1044 |
| vec_items | 1966 | 1044 |
| memory_entities | 32168 | 11768 |

`PRAGMA user_version` = 2 (two migration steps applied).

### `morgan-mcp` stdio: start to `tools/list`, ten runs

Live install on PATH, default (stdio) transport. Each run spawns a fresh `morgan-mcp`
process and times from spawn through `ClientSession.initialize()` + `list_tools()` to the
response, using the `mcp` SDK's own stdio client.

```
run 1: 1.468s   run 2: 1.541s   run 3: 1.405s   run 4: 1.468s   run 5: 1.635s
run 6: 1.548s   run 7: 1.328s   run 8: 1.419s   run 9: 1.499s   run 10: 1.507s

min=1.328s  median=1.483s  max=1.635s
```

## 2. The live numbers

### Live eval scorecard (bundled 60-probe set, `tests/memory_quality/probes.json`)

```
export $(grep -E '^MORGAN_EMBEDDING_(ENDPOINT|MODEL|DIM)=' ~/.config/morgan/.env | xargs)
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider \
  tests/memory_quality/test_retrieval_quality.py -k shares_no_words --live -s -v
```

Only the three embedding settings were exported -- never the chat key.

```
run: embedding=qwen3-embedding:8b (dim 4096)  k=8  floor=off  probes=probes.json@510b5f20fe94 (86 memories, 60 probes)  db-upgrade=2  commit=24f1566
n=60  recall@8=0.89  mrr=0.51  stale1st=0.70  abstain=0.00
  knowledge_update  n=10   recall@8=1.00  mrr=0.50  stale1st=0.70  abstain=0.00
  multi_hop         n=8    recall@8=0.62  mrr=0.16  stale1st=0.00  abstain=0.00
  single_hop        n=20   recall@8=0.90  mrr=0.58  stale1st=0.00  abstain=0.00
  temporal          n=8    recall@8=1.00  mrr=0.66  stale1st=0.00  abstain=0.00
  unanswerable      n=14   recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
1 passed, 2 deselected in 103.35s (0:01:43)
```

### Holdout floor sweep (Task 2), both halves -- from the already-saved run

Not re-run here; recorded in Task 2 and read from
`~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-holdout-floor-sweep.txt`.
Snapshot: qwen3-embedding:8b, 4096 dims, `user_version` 2, 3010 memories; scored with
`all_projects=True` over the 128-label holdout set (fit=90, sealed=38).

```
margin=0.00    fit  n=90   recall@8=0.89  mrr=0.73  abstain=0.00
margin=0.00 sealed  n=38   recall@8=0.88  mrr=0.71  abstain=0.00
margin=0.04    fit  n=90   recall@8=0.89  mrr=0.73  abstain=0.44
margin=0.04 sealed  n=38   recall@8=0.88  mrr=0.71  abstain=0.23
margin=0.08    fit  n=90   recall@8=0.88  mrr=0.72  abstain=0.84
margin=0.08 sealed  n=38   recall@8=0.84  mrr=0.70  abstain=0.85
margin=0.11    fit  n=90   recall@8=0.82  mrr=0.68  abstain=0.96
margin=0.11 sealed  n=38   recall@8=0.80  mrr=0.68  abstain=0.85
margin=0.14    fit  n=90   recall@8=0.80  mrr=0.66  abstain=0.96
margin=0.14 sealed  n=38   recall@8=0.76  mrr=0.67  abstain=0.85
margin=0.18    fit  n=90   recall@8=0.74  mrr=0.62  abstain=0.96
margin=0.18 sealed  n=38   recall@8=0.68  mrr=0.61  abstain=0.85
margin=0.22    fit  n=90   recall@8=0.71  mrr=0.59  abstain=0.96
margin=0.22 sealed  n=38   recall@8=0.64  mrr=0.57  abstain=0.85
```

### Cold-start seconds

From `2026-09-21-repeat-distribution.json`: the three cold starts measured 7.4, 8.0, and 8.4 seconds
respectively. Each was measured with the model unloaded per the native `/api/ps` process list, after
30 idle minutes since the script's own last request. These are substantially faster than the 43 seconds
measured once on 2026-09-19 (a first load from disk); these three reloaded a model file the host still
had cached in filesystem buffers.

## 3. The repeat-distribution script, and the fingerprint tolerance

`scripts/measure_repeat_distribution.py` samples a fixed set of stored memories (spread
evenly across the table) from a snapshot, re-embeds every one of them under a grid of
conditions -- batch size (1/8/32) x warm-or-cold x client count (1/2) -- and reports the
min/p1/median cosine similarity between each re-embedding and the vector already stored for
that memory. `--cold-starts N` separately times N fresh model loads (wait for the native
`/api/ps` to show the model unloaded and 30 idle minutes since the script's own last request,
then time one embed). It reads the snapshot read-only and writes nothing to it.

Resumable: every finished condition is written into `--results-json` immediately, and that
same file (doubling as the progress file) is read back on the next invocation to skip
whatever is already there. Nothing about which rows were sampled is persisted -- the sample
is a deterministic function of `(db, --rows)`, so a resumed run reselects the same rows
without ever writing memory content or ids to disk.

### Smoke check -- `--quick --rows 5` (proves the script runs; not the tolerance verdict)

```
export $(grep -E '^MORGAN_EMBEDDING_(ENDPOINT|MODEL|DIM)=' ~/.config/morgan/.env | xargs)
.venv/Scripts/python.exe scripts/measure_repeat_distribution.py \
  --db ~/Documents/GitHub/morgan-eval-brain-2026-09-19-qwen3-8b/morgan.db \
  --quick --rows 5 \
  --results-json ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-repeat-distribution-quick-smoke.json \
  --results-md ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-repeat-distribution-quick-smoke.md
```

```
sampled 5 rows (dim=4096) from <snapshot>/morgan.db
batch1_warm_c1: min=1.00000 p1=1.00000 median=1.00000 (2.9s)
batch1_warm_c2: min=1.00000 p1=1.00000 median=1.00000 (3.1s)
batch8_warm_c1: min=1.00000 p1=1.00000 median=1.00000 (2.8s)
batch8_warm_c2: min=1.00000 p1=1.00000 median=1.00000 (3.0s)
wrote <results.json> and <results.md>
```

Re-running the identical command against the same `--results-json` skipped all four
conditions (`skip batchN_warm_cC: already in ...`) and exited 0, confirming resumability.
`--quick` also runs both client counts on batch sizes 1 and 8, warm only -- cold conditions
and `--cold-starts` need a genuine 30-minute idle wait and are exercised only by the full run.

### The full run

The script's grid, run with every batch/warmth/concurrency combination and no
`--cold-conditions` restriction, costs up to 9 separate 30-minute idle waits (6 "cold"
conditions, each waiting on its own genuine idle-and-unloaded state, plus 3 for
`--cold-starts 3`) -- more than the brief's "about three 30-minute idle waits" estimate. The
counter-argument: a "cold" condition is 500 rows, i.e. one cold request followed by 499 warm
ones, and the p1 of 500 samples is roughly the 5th-smallest -- one buried cold sample barely
moves the reported p1 away from what a fully-warm condition would show, so six separate
half-hour waits buy very little over one.

**Controller ruling:** the background run uses the full *warm* grid (all 3 batch sizes x
both client counts = 6 warm conditions) plus exactly **one** cold condition (batch 8, one
client) plus `--cold-starts 3` -- about 4 waits total, not 9. The tolerance is set from the
worst condition among *those*, not from the full 12-condition grid. `--cold-conditions "8:1"`
implements this restriction (added to the script for this ruling; the warm half is never
restricted by this option). Exact invocation (env export first, chat key never exported):

```
export $(grep -E '^MORGAN_EMBEDDING_(ENDPOINT|MODEL|DIM)=' ~/.config/morgan/.env | xargs)
.venv/Scripts/python.exe scripts/measure_repeat_distribution.py \
  --db ~/Documents/GitHub/morgan-eval-brain-2026-09-19-qwen3-8b/morgan.db \
  --rows 500 --cold-conditions "8:1" --cold-starts 3 \
  --results-json ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-repeat-distribution.json \
  --results-md ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-repeat-distribution.md
```

This produces 6 warm conditions + 1 cold condition x 500 rows = 3,500 embeddings, plus 3
single cold-start timings -- read the numbers when they land as measuring 7 conditions, not
12; batch sizes 1 and 32 have no cold measurement in this run.

Results from `2026-09-21-repeat-distribution.md`:

| condition | batch | warmth | clients | n | min | p1 | median | wall seconds |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| batch1_warm_c1 | 1 | warm | 1 | 500 | 0.99837 | 0.99866 | 1.00000 | 273.3 |
| batch1_warm_c2 | 1 | warm | 2 | 500 | 0.99820 | 0.99878 | 1.00000 | 252.4 |
| batch32_warm_c1 | 32 | warm | 1 | 500 | 0.99820 | 0.99866 | 1.00000 | 261.2 |
| batch32_warm_c2 | 32 | warm | 2 | 500 | 0.99837 | 0.99883 | 1.00000 | 253.6 |
| batch8_cold_c1 | 8 | cold | 1 | 500 | 0.99837 | 0.99860 | 1.00000 | 271.0 |
| batch8_warm_c1 | 8 | warm | 1 | 500 | 0.99837 | 0.99871 | 1.00000 | 261.7 |
| batch8_warm_c2 | 8 | warm | 2 | 500 | 0.99837 | 0.99861 | 1.00000 | 253.0 |

**tolerance: 0.995** (worst p1 0.99860, batch8_cold_c1, above 0.998)

## 4. The `PARTITION KEY` measurement

`scripts/measure_partition_key.py`: copies `--db` first (never touches the source -- opened
read-only, copied via `shutil.copy2`, the copy deleted afterward unless `--keep-copy`), builds
a second `vec_items_pk` table on the copy with `project TEXT PARTITION KEY` (sqlite-vec
0.1.9) alongside the existing metadata-column `vec_items`, populated from the same rows. Runs
`--queries` KNN queries, evenly spread across the table, against both tables -- each query
uses a sampled row's own stored vector (no embedding call needed) and the same
`user_id`/`project` filter both tables carry -- and reports wall time and returned-id
equality per table, `--runs` times for stability. No host, model, or path is hardcoded; every
value above comes from the command below.

```bash
.venv/Scripts/python.exe scripts/measure_partition_key.py \
  --db ~/Documents/GitHub/morgan-eval-brain-2026-09-19-qwen3-8b/morgan.db \
  --queries 200 --k 8 --runs 2 \
  --results-json ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-partition-key.json
```

Raw output (unredacted path):
`~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-21-partition-key.json`.

```
run 1: metadata median=58.63ms mean=58.88ms (min=47.23ms max=73.09ms) | partition median=41.94ms mean=42.13ms (min=32.45ms max=53.97ms) | mismatches=0/200
run 2: metadata median=58.71ms mean=59.15ms (min=47.24ms max=72.04ms) | partition median=41.96ms mean=42.12ms (min=32.93ms max=52.55ms) | mismatches=0/200
```

**Verdict: recall time drops clearly (~28% faster on both median and mean, in both runs)
with identical ids returned (0/200 mismatches, both runs) -- take the `PARTITION KEY`.**

## 5. Commit

This file, `scripts/measure_repeat_distribution.py`, and `scripts/measure_partition_key.py`
are committed together. No file under `morgan_brain/` changed in this task.
