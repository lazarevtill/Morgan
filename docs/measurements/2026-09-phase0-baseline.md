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
This card was produced with the same module's public API
(`load_probe_set`/`run_probes`/`score_run`/`describe_run`, `Settings(embedding_backend="hash",
embedding_dim=64)`, `FakeEmbedder(dim=64)`) run ad hoc for this record; the live scorecard
below is the one the checked-in live test actually prints with `-s`.

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

Opened with a raw `sqlite3` URI (`file:...?mode=ro`), `sqlite-vec` loaded directly into that
connection, never through `morgan_brain`'s `open_db` (which switches journal mode -- a write
-- as a side effect of opening). Nothing in this database was written.

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

Pending -- background run. Belongs to `scripts/measure_repeat_distribution.py
--cold-starts 3` (Section 3).

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

### The full run -- not run by this task; the controller runs it in the background

~6,000 embeddings (3 batch sizes x 2 warmth x 2 client counts x 500 rows) plus
`--cold-starts 3`. Exact invocation (env export first, chat key never exported):

```
export $(grep -E '^MORGAN_EMBEDDING_(ENDPOINT|MODEL|DIM)=' ~/.config/morgan/.env | xargs)
.venv/Scripts/python.exe scripts/measure_repeat_distribution.py \
  --db ~/Documents/GitHub/morgan-eval-brain-2026-09-19-qwen3-8b/morgan.db \
  --rows 500 --cold-starts 3 \
  --results-json ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-XX-repeat-distribution.json \
  --results-md ~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-XX-repeat-distribution.md
```

Each of the grid's 6 "cold" conditions waits for its own genuine idle-and-unloaded state
before running (per-condition, not shared across the cold half of the grid), which costs up
to 6 waits there plus 3 more for `--cold-starts 3` -- more than the brief's "about three
30-minute idle waits" estimate. Before spending that time, weigh it against what it buys: a
"cold" condition is one cold request followed by 499 warm ones, and the p1 of 500 samples is
roughly the 5th-smallest -- one buried cold sample is statistically invisible next to its
warm twin's distribution. Six separate half-hour waits may not be worth more than one shared
wait covering the whole cold half of the grid (only the first sub-condition run then
genuinely cold, the rest warm-but-labelled-cold); this script does not make that call for
the controller, it just implements the literal per-condition reading.

**tolerance: pending (default 0.995)** -- Task 3b writes the full run's numbers and the
tolerance verdict into this section, by this rule: **tolerance = 0.995, unless the worst
condition's p1 is below 0.998, in which case tolerance = that p1 rounded down to three
digits minus 0.003 -- naming the condition that set it.**

## 4. The `PARTITION KEY` measurement

On a scratch copy of the eval snapshot (never the original -- verified byte-for-byte
identical by checksum immediately after copying, before anything touched the copy), a second
`vec_items` table was built with `project TEXT PARTITION KEY` (sqlite-vec 0.1.9) alongside the
existing metadata-column table, populated from the same rows. 200 queries, evenly spread
across the table, used each row's own stored vector as the query vector (no embedding call
needed) with `k=8`, filtered by the same `user_id`/`project` metadata both tables carry. Run
twice, independently, for stability:

```
run 1  metadata-column vec_items   : min=47.85ms  median=59.99ms  mean=60.57ms  max=79.62ms
       PARTITION KEY   vec_items_pk: min=32.73ms  median=41.18ms  mean=41.48ms  max=51.63ms
       id mismatches: 0 / 200

run 2  metadata-column vec_items   : min=50.32ms  median=62.36ms  mean=62.22ms  max=86.50ms
       PARTITION KEY   vec_items_pk: min=32.38ms  median=42.74ms  mean=42.85ms  max=56.82ms
       id mismatches: 0 / 200
```

**Verdict: recall time drops clearly (~30% faster on both median and mean, in both runs)
with identical ids returned (0/200 mismatches, both runs) -- take the `PARTITION KEY`.**

## 5. Commit

This file and `scripts/measure_repeat_distribution.py` are committed together. No file under
`morgan_brain/` changed in this task.
