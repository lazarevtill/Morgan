# Phase 0 closing note -- 2026-09-23

The acceptance run of `SPEC-phase0` section 6, item by item, every number beside the value
`2026-09-phase0-baseline.md` recorded for it, on `feat/phase0-floor-and-safety` from the
worktree checkout (`.venv`, Python 3.13.15, win32), at three commits:

- **`2f213ff`** -- items 1 to 10 and the live metrics.
- **`7b79da9`** -- items 3 and 6, and the hash scorecard, measured again. It is `2f213ff`
  plus six commits, which between them: make the memory commands' connection wait
  `MORGAN_DB_BUSY_TIMEOUT_MS`; add the model-change remedy to the mismatch messages; add
  `doctor`'s width check; read `consolidate`'s per-project switch through the gate; give three
  functions that other modules read public names; keep the suite from reading the shell's
  `MORGAN_*` variables; and correct comments, docstrings and docs.
- **`4af8672`** -- the final head, one docstring after `7b79da9`, where the four gates ran.

Each section names the commit it was measured at.

Endpoints are written as `<chat-endpoint>` / `<embedding-endpoint>` and home paths as `~`,
the rule the baseline follows. A message quoted in a text block below is one line as printed,
wrapped here at 100 columns. The unredacted outputs -- every raw log, and the harness that
produced them -- live outside this repository, under
`~/Documents/GitHub/morgan-research-2026-09-19/measurements/2026-09-23-acceptance/`.

## What this ran against

Not the live database. Under the task's pre-flight ruling, every step ran against a fresh
`VACUUM INTO` copy of `~/.local/share/morgan/morgan.db`, taken through a plain `sqlite3`
connection opened read-only (`file:...?mode=ro`) into a scratch `MORGAN_DATA_DIR`. The live
install still runs the pre-phase-0 code; a live file migrated to `user_version` 7 underneath it
would be written by code that does not know the new columns. Migrating the live file is the
owner's, after the merge -- see "What is still owed" below.

The live file was 96,149,504 bytes at `user_version` 2 before the copy, and unchanged in size
and modification time after it. The copy holds:

| table | rows | the spec's frozen input |
| --- | ---: | ---: |
| memories | 3011 | 3010 |
| vec_meta | 3011 | 3010 |
| vec_items | 3011 | 3010 |
| memory_entities | 43938 | 43936 |
| fts_memories | 3011 | 3010 |
| facts | 0 | 0 |
| session_history | 0 | 0 |

Per project: `archive/chatgpt` 1,966, `archive/chatgpt-holdout` 1,044, and **`Morgan` 1** -- one
memory the live install wrote after the baseline was taken. Every count below is against 3,011.

## 1. The baseline file and the labels -- pass, at `2f213ff`

`docs/measurements/2026-09-phase0-baseline.md` holds the four gates, the hash scorecard with its
`RunConfig` and its driver, `doctor --json` from two working directories, the live row counts,
the `morgan-mcp` start-to-`tools/list` distribution, the live scorecard, the holdout floor
sweep, the cold-start seconds, the repeat-distribution grid with its tolerance verdict (0.995)
and the `PARTITION KEY` measurement with its verdict.

The 128 labels sit beside the snapshot. Their recorded sha256 matches the file:
`9175ae87a048fa0d3cd71973c45e710d9306e5bc9a7b616ddfc77d8475c1e686`.

## 2. `migrate`, `restore`, diff, `migrate` again -- pass, at `2f213ff`

`migrate --dry-run` lists exactly steps 3 to 7, `user_version` 2 against `code_version` 7:

| step | name | kind |
| ---: | --- | --- |
| 3 | create `embedding_spaces` and `projects` | light |
| 4 | provenance columns | heavy |
| 5 | rename `default` to `personal` | heavy |
| 6 | rebuild vec0 and FTS5 | heavy |
| 7 | seed `projects` | light |

**Wave 1**, 62.2 s wall, snapshot taken first, `from_version` 2 to `user_version` 7,
`quick_check: ok`:

| step | counts | what SPEC section 4 predicted |
| ---: | --- | --- |
| 4 | memories 3011, facts 0 | "an `UPDATE` over 3,010 rows, about a second" |
| 5 | every table 0 | "0 rows today" |
| 6 | vec_items 3011, fts_memories 3011 | "seconds to low tens of seconds" |
| 7 | projects 3 | "3 rows" -- see "Where this differs from the spec" |

`before` and `after` are equal: memories 3011, facts 0, session_history 0.

**Restore.** The preview (no `--yes`) changes nothing. With `--yes`, a safety snapshot is taken
first, then the file replaced. The restored file opens read-only at `PRAGMA user_version` **2**,
and a row-hash diff against the snapshot -- sha256 over every row of every table in column
order, the vec0 shadow tables included -- is **empty over all 21 tables**. `sqldiff` is not
installed on this host; SPEC section 6 item 2 allows a row-hash script, and one was used.

**Wave 2**, 56 s wall: `from_version` 2 to 7 again, `quick_check: ok`, identical step counts.
Against wave 1's result, only `embedding_spaces` and `projects` differ, and only in their
volatile columns (`created_at`, `fingerprint_recorded_at`, and the 81,920-byte fingerprint blob
of five freshly embedded vectors). `memories`, `vec_meta`, `vec_items`, `fts_memories`,
`memory_entities`, `facts` and `session_history` hash identically across both waves.

The space is registered and fingerprinted **after** the wave commits: `qwen3-embedding:8b`,
4,096 dims, `min_cosine` 0.99905 on wave 1 and 0.99999999999999978 on wave 2 (warm), both above
the recorded tolerance of 0.995.

A third wave, run on a copy of the eval snapshot with the embedding endpoint on a closed port,
committed with `quick_check: ok` and reported the space as `fingerprint: unverified` carrying
the reason. The migration stands whatever the embedding server does.

## 3. The embedder stopped, the host cold, and a model that is not the one -- pass

**Stopped**, at `2f213ff` (a counting proxy for the tool list, a closed loopback port for the
rest):

| what | observed | the baseline |
| --- | --- | --- |
| `morgan-mcp` `initialize` + `tools/list`: requests reaching the embedder | **0** | -- |
| the same, seconds, min / median / max | 0.710 / 0.727 / 0.737, over 3 fresh processes | 1.328 / 1.483 / 1.635, over 10, the live `uv` install |
| `morgan facts`, embedder stopped | answers, exit 0, 0.449 s | -- |
| `morgan recall`, embedder stopped | `unreachable`, 5.317 s wall; the retry loop used 4.6 s of `MORGAN_EMBEDDING_UNREACHABLE_BUDGET_SECONDS=5.0` | -- |

The seconds are recorded, not judged: the 0.5 s target on the tool list was struck on
2026-09-21 because it measured Python start-up. What is asserted is the zero. These numbers are
not comparable with the baseline's -- they invoke the module on this worktree's interpreter
rather than the console script the baseline timed.

The error names the endpoint, the setting and the class:

```text
model endpoint http://127.0.0.1:<closed>/v1/embeddings is unreachable: ConnectError after
2 attempts over 4.6 s (All connection attempts failed); check MORGAN_EMBEDDING_ENDPOINT and
run `morgan doctor`
```

(The endpoint here is a closed loopback port standing in for the stopped host, so the owner's
own endpoint is never involved.)

**Cold**, at `2f213ff`. **409.9 minutes** had passed since this run's last embedding request,
and `/api/ps` reported no model loaded when the measurement fired -- far past the 30 minutes the
spec asks for. (That is what was observed: the host is the owner's and serves their live
install too, so "no model loaded at fire time" is the stronger of the two statements.) The
first `recall` ran beside a concurrent `facts`, both released from one barrier:

| command | wall | outcome |
| --- | ---: | --- |
| `recall --all-projects` | **14.023 s** | exit 0, **8 results**, `abstained: false`, `reason: no_floor` |
| `facts --all-projects` | **1.104 s** | exit 0, answered while the model was still loading |

```text
embed.done   attempts=1 inputs=6 latency_ms=12218.4 outcome=ok
recall.done  degraded=None embed_latency_ms=12223.8 embed_outcome=ok query_language=en reason=no_floor
```

The first recall after idle **answers**; its `recall.done` line carries the latency
(12,223.8 ms) and the query language (`en`); it took **one attempt**, no retry; and the command
that does not embed finished in 1.1 s rather than waiting out the load. That is the whole of
item 3's cold condition.

The 12.2 s load is longer than the baseline's three cold starts (7.4 / 8.0 / 8.4 s) and far
shorter than the 43 s it records for a first load from disk. A second cold load, measured later
under the baseline's own condition -- 31.4 idle minutes with the machine awake -- came back at
**7.8 s**, inside that interval; see "Cold starts" under the live metrics. The 12.2 s was
measured under a different condition and its cause was not isolated. Either way it took **one
attempt**, inside `MORGAN_EMBEDDING_TIMEOUT_SECONDS=50` -- sized for a 43 s first load -- and
inside the 60 s retry budget; no test and no setting depends on the value.

**A model that is not the one**, at `7b79da9`. A `tests/fakes.py` model server answering at
the same width, 4,096, was put behind `MORGAN_EMBEDDING_ENDPOINT` with `MORGAN_EMBEDDING_MODEL`
left as the name the space recorded -- so only the fingerprint can tell the two apart, which is
the swap the invariant is about: a different model behind the same name. `recall` is refused
before it searches anything, exit 1:

```text
embedding space 1 (qwen3-embedding:8b, 4096 dims) does not match the model at
MORGAN_EMBEDDING_ENDPOINT: fingerprint cosine -0.0232 < 0.995 on 5 of 5 strings; stored vectors
would be searched with the wrong model; check MORGAN_EMBEDDING_MODEL or run
`morgan doctor --vectors`, or point MORGAN_DATA_DIR at a new database if the model was changed
on purpose: a database keeps the model its vectors were written with
```

It names both sides -- the space by id, model and width, the model answering by the setting that
addresses it -- then what went wrong, what it means, what to check, and what to do when the
change was meant. `doctor` against the same server:

```text
embedding_space: 1 (qwen3-embedding:8b, 4096 dims): fingerprint MISMATCH (min cosine -0.0232)
```

And `doctor` against the real embedding host, one run, its probe the only embedding request:

```text
embedding_dim: 4096
provider: reachable (0.3 s)
embedding_provider: slow (6.2 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)
embedding_space: 1 (qwen3-embedding:8b, 4096 dims): fingerprint matches (min cosine 1.0000)
```

The space **matches**. The probe took 6.2 s: the previous embedding request -- item 6's cold
load, below -- was twelve minutes earlier, past the host's five-minute keep-alive. `/api/ps` was
not polled for this run, so the 6.2 s is not counted as a cold start.

## 4. A dropped connection, a scripted 502, and the per-call log -- pass, at `2f213ff`

**A dropped connection mid-import.** A 24-conversation export imported through a proxy that
closes the 4th embedding request's connection without a word: **exit 0 in 16.45 s**, 16
conversations, 8 held out, 48 memories, 0 skipped turns. The drop cost one retry
(`embed.done attempts=2 ... outcome=ok`) and nothing else.

**A scripted 502** names all four things:

```text
model endpoint http://127.0.0.1:<fake>/v1/embeddings answered too slowly or dropped: HTTP 502
after 5 attempts over 5.5 s; a cold host loads the model in seconds, 43 s on a first load from
disk; check MORGAN_EMBEDDING_ENDPOINT and run `morgan doctor`
```

Endpoint, setting (`MORGAN_EMBEDDING_ENDPOINT`), attempts (5) and class (`HTTP 502`, filed as
retryable rather than `unreachable`).

**Every recall in this note's logs carries `embed_latency_ms` and `query_language`:**

```text
recall.done  degraded=None embed_latency_ms=316.8 embed_outcome=ok query_language=en reason=no_floor
recall.done  degraded=None embed_latency_ms=233.7 embed_outcome=ok query_language=ru reason=no_floor
```

The second is a Cyrillic query; the language is decided by script, with no model call.

## 5. The env files, the `./.env` trap, and the chat key -- pass, at `2f213ff`

`doctor` lists each env file with its presence:

```text
env_file: ~/.config/morgan/.env (present)
env_file: ~/.env (absent)
```

With a `./.env` in the working directory setting `MORGAN_DATA_DIR` elsewhere,
`settings_for("cli")` resolves to that folder's database -- by design, the CLI's working
directory is the owner's -- and `settings_for("mcp")` resolves to `~/.local/share/morgan/
morgan.db` regardless. The `./.env` does not move `morgan-mcp`'s database. (Measured by
resolving both surfaces' settings from that directory, not by starting a server: a
`morgan-mcp` started for the test would open the owner's live database, which this run must
not do.)

An embedding request sent to a separate `MORGAN_EMBEDDING_ENDPOINT` while `MORGAN_LLM_API_KEY`
is set and `MORGAN_EMBEDDING_API_KEY` is not carries **no `authorization` header at all**, and
the chat key appears in no header: only `accept`, `accept-encoding`, `connection`,
`content-length`, `content-type`, `host`, `user-agent`.

## 6. What `doctor` counted and what it probed -- pass

**Unchanged at `7b79da9`, and one new line.** `doctor`'s text was captured
at `2f213ff` from the home folder, on the migrated copy, with `MORGAN_EMBEDDING_DIM=4096` -- the
space's width -- and both servers on closed loopback ports, so no line depends on a host. The
same command at `7b79da9`, the same ports reused, is **byte-identical**: 2,232 bytes and 29
lines both times, no line differs. The counts, `rows_missing_provenance`, `migration` and
`projects` lines quoted below are in that capture, so they hold at both commits; the `slow` and
`data_flow` lines come from other runs, named where they appear.

`--json` has a new key, `embedding_dim_error`, which is `null` when the width agrees:

```text
"embedding_dim": 4096, "embedding_dim_error": null
```

With `MORGAN_EMBEDDING_DIM=1024` against the same 4,096-wide space, **exactly one line** of the
text changes, line 7, and `--json`'s `embedding_dim_error` carries the same message:

```text
embedding_dim: 1024 (embedding space 1 (qwen3-embedding:8b) holds 4096-dimensional vectors but
MORGAN_EMBEDDING_DIM is 1024; the two must agree (set MORGAN_EMBEDDING_DIM=4096, the width this
database was written at, or point MORGAN_DATA_DIR at a new database if the model was changed on
purpose))
```

`doctor` still exits 0 and reads only: it reports the disagreement rather than failing on it,
and the migrated copy row-hashes unchanged after every `doctor` and `recall` run at `7b79da9`.

**The counts**, at `2f213ff` (and in the byte-identical capture at `7b79da9`), from the home
folder, on the migrated copy:

```text
memories: 0 in project 'personal' (3011 across all projects)
fts: 0 in project 'personal' (3011 across all projects)
vectors: 0 in project 'personal' (3011 across all projects)
```

A host answering after 4 s, at `2f213ff`, with `MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0`:

```text
provider: slow (4.4 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)
embedding_provider: slow (4.2 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)
```

**Two data-flow lines.** The owner's own configuration addresses chat and embeddings at
different hosts, so `doctor` gives two, one per host, each naming the setting that addresses it
and what each command sends there. The wording below is as printed in the loopback capture
above -- at `2f213ff`, and byte-identically at `7b79da9` -- with `localhost` and `127.0.0.1`
for hosts. Against the owner's configuration, item 3's real-host `doctor` at `7b79da9` printed
the same two lines, and item 9's `doctor --json` at `2f213ff` carries the same two entries: the
same words each time, the hosts aside, which are written here as placeholders:

```text
data_flow: <chat-endpoint> (MORGAN_LLM_ENDPOINT) receives ask: the question, the memories
  recalled for it and the recent history; consolidate: up to 50 memories per project, with the
  project's current facts
data_flow: <embedding-endpoint> (MORGAN_EMBEDDING_ENDPOINT) receives remember: the memory's
  text; recall: the query; import: every imported message; doctor --vectors: a sample of stored
  memories; a process's first embedding call: up to 5 stored memories, while the embedding
  space's fingerprint is unrecorded
```

The entries are grouped by host, so an install that serves both from one host gets a single
line naming both settings -- also correct, and checked.

**And after a cold load, `slow` -- never `unreachable`**, at `2f213ff`. Measured on the real
embedding host 31.4 minutes after the previous embedding request, by the client's clock: that
request ended at 07:34:33, and this run started at 08:05:56. `/api/ps`, sampled every two
minutes, showed that request's model still loaded at 07:36 and 07:38, due to unload at 07:39:32
by the host's own stamp, and no model loaded at all 13 samples from 07:40:04 to 08:04:04:

```text
provider: reachable (0.3 s)
embedding_provider: slow (7.8 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)
embedding_space: 1 (qwen3-embedding:8b, 4096 dims): fingerprint matches (min cosine 1.0000)
```

`doctor`'s embedding probe *is* the process's first embedding request -- it sends the five
fingerprint strings in one call, one attempt, no retry -- so it pays the cold load itself, and
the 7.8 s it reports is that load. A host that answers is never called `unreachable`: the probe
record carries `timeout_seconds: 60.0` and `slow_after_seconds: 2.0`, so a load of this length
lands in `slow` with room to spare. The same call also confirms the space `matches`, so one
probe answers both "is it served" and "is it the model that wrote the stored vectors".

## 7. Counts, provenance and the `projects` rows -- pass, at `2f213ff`

Counts equal across steps 4 and 5 in both waves, and the restored file diffs empty against the
snapshot. Row counts for every table the wave touched -- the vec0 and FTS5 indexes' own shadow
tables aside -- taken from the row-hash of the copy before the first wave and of the migrated
file after the second:

| table | before | after |
| --- | ---: | ---: |
| memories | 3011 | 3011 |
| vec_meta | 3011 | 3011 |
| vec_items | 3011 | 3011 |
| memory_entities | 43938 | 43938 |
| fts_memories | 3011 | 3011 |
| facts | 0 | 0 |
| session_history | 0 | 0 |
| embedding_spaces | no table | 1 |
| projects | no table | 3 |

After `migrate`, as `doctor` printed them -- `logs/doctor-two-flows.txt` in the measurements
folder named at the top, and byte-identically at `7b79da9`:

```text
rows_missing_provenance: 0
rows_missing_provenance_reason: None
migration: user_version 7 of 7, nothing pending
```

`projects` holds three rows, each with its classification, capture and consolidation state:

```text
project 'Morgan': unclassified, capture on, consolidate on
project 'archive/chatgpt': unclassified, capture on, consolidate on
project 'archive/chatgpt-holdout': unclassified, capture on, consolidate on
```

All three are `unclassified`: step 7 seeds names only. A classification, a remote and a root are
recorded by a CLI write from inside the repository, which this run deliberately never made
against the copy.

## 8. `abstained` in its shapes -- pass, at `2f213ff`

| shape | observed |
| --- | --- |
| `empty` | a project with nothing in it: `abstained: true`, `reason: empty`, no results |
| `too_few_to_judge` | three memories stored: all three returned, `abstained: false`, `reason: too_few_to_judge` |
| `no_floor` | `MORGAN_RECALL_FLOOR_MARGIN` unset: `recall.done ... reason=no_floor` |
| `declined` | with `MORGAN_RECALL_FLOOR_MARGIN=0.08`, the baseline's accepted value: **5 of 8** unanswerable holdout probes returned `abstained: true`, `reason: declined`, no results |

Eight probes are too few to read as a rate; the sealed 128-label sweep re-run below still
abstains on **0.85** of its unanswerable half at that margin, exactly the baseline's figure.
Every one of the eight `recall.done` lines carried `embed_latency_ms` and `query_language`, and
the logged reason agreed with the payload's on all eight.

## 9. `doctor --vectors --clients 2` -- pass, at `2f213ff`

Re-embedding a sample of the stored vectors and comparing each answer against what is stored,
through two concurrent clients, `--all-projects` over the 3,011-memory copy:

| | this run | on record |
| --- | ---: | ---: |
| rows sampled | 180 | 180 |
| min cosine | 0.99842 | 0.99879; 0.99820, the worst min over the repeat grid's seven conditions |
| median cosine | 0.99973 | 1.00000 |
| rows below the 0.995 tolerance | **0** | none below 0.99 |
| rows the two clients disagreed on | **0** | -- |

The 180 rows, 0.99879, 1.00000 and "none below 0.99" are the evaluation snapshot's own audit
record, in the `README.txt` beside
`~/Documents/GitHub/morgan-eval-brain-2026-09-19-qwen3-8b/morgan.db`; 0.99820 is the
repeat-distribution grid in `2026-09-phase0-baseline.md`.

Per client: client-1 min 0.99842, median 0.99973, 263.4 s; client-2 min 0.99843,
median 0.99975, 262.8 s. Every sampled row is within tolerance for both, and the minimum sits
inside the interval the repeat-distribution grid measured.

Run without `--all-projects`, the audit samples the current project, which in this worktree is
`Morgan` -- one memory, one sampled row. That is the same project scoping the baseline's
`doctor --json` records; `--all-projects` is what audits the archive.

## 10. The ADR and the table registry -- pass, at `2f213ff`

`docs/decisions/0001-fact-key-and-forget-reach.md` is in place.
`tests/unit/memory/test_forget_reaches_every_project_keyed_table.py`: 12 passed in 7.13 s.

## The exit

### The four gates -- at `4af8672`

`4af8672` is the last code commit; it differs from `7b79da9` by one docstring. On Windows,
Python 3.13, from the worktree's own environment: `pytest` 690 passed, 4 skipped, 1 warning (a
third-party `DeprecationWarning` from starlette) in 272.11 s; `ruff check` all checks passed;
`ruff format --check` 180 files already formatted; `mypy` (strict) no issues in 56 source
files; `bandit` no issues identified. On Linux, Python 3.12, the repository's CI at `7b79da9`:
tests, ruff and mypy, bandit, CodeQL and secret scanning all pass.

The baseline's, for comparison: `pytest` 309 passed, 4 skipped in 132.29 s; `ruff check` all
checks passed; `ruff format --check` 130 files already formatted; `mypy` no issues in 48 source
files; `bandit` no findings.

**The suite reads none of the shell's `MORGAN_*` variables** (`fee1579`). `tests/conftest.py`
removes every one of them from each test's environment, matched without regard to case, except
in tests marked `live`, which read their endpoint and width from them. The fix's author ran the
whole suite with `fee1579` applied and `MORGAN_DATA_DIR`, `MORGAN_TEMPORAL_DB_URL` and
`MORGAN_SNAPSHOT_DIR` exported at a throwaway folder: 689 passed, 4 skipped, 1 failed, and
nothing was written to the exported paths. The one failure, `test_wheel_install`, failed the
same way with nothing exported, because that environment had no `pip`; the run at `4af8672`
above, which has `pip`, passes it. This acceptance run re-checked two files only, at
`7b79da9`: the two that fail with those three variables exported against `2f213ff` -- 14
passed, and nothing written.

### The hash scorecard -- unchanged to the digit, at `7b79da9`

The baseline's driver, verbatim, fed to the worktree's interpreter on stdin from the worktree
root, so it imports exactly as it would saved there:

```text
run: embedding=hash stub (dim 64)  k=8  floor=off  probes=probes.json@510b5f20fe94 (86 memories, 60 probes)  db-upgrade=7  commit=7b79da9
n=60  recall@8=0.28  mrr=0.09  stale1st=0.60  abstain=0.00
  knowledge_update  n=10   recall@8=0.30  mrr=0.09  stale1st=0.60  abstain=0.00
  multi_hop         n=8    recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
  single_hop        n=20   recall@8=0.30  mrr=0.09  stale1st=0.00  abstain=0.00
  temporal          n=8    recall@8=0.50  mrr=0.21  stale1st=0.00  abstain=0.00
  unanswerable      n=14   recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
```

All six score lines diff empty against the baseline's. Only the header moved, in the two fields
that must: `db-upgrade` 2 to 7 (the five new steps) and the commit. The probe set hashes the
same. The card was identical at `2f213ff` as well.

### The live metrics -- at `2f213ff`

Not measured again after `2f213ff`. What they rely on is that no later change touches
embedding, ranking or scoring arithmetic. The code changes on the path a recall takes are of
two kinds: `composition.py` now opens its connection with `MORGAN_DB_BUSY_TIMEOUT_MS` as the
busy timeout, which matters only while another process holds the lock; and `wire.py`'s
mismatch message and `composition.py`'s width refusal each gained a remedy clause, which runs
only when a model or a width is refused. The direct evidence is the hash scorecard above,
which runs the same ranking and scoring code: digit-identical at `7b79da9`. `4af8672` adds
one docstring.

No live metric moved outside the baseline's interval; the two that are directly comparable came
back identical to the digit.

**The 60-probe live scorecard** (`tests/memory_quality/test_retrieval_quality.py -k
shares_no_words --live`; only the three embedding settings exported, never the chat key):

```text
run: embedding=qwen3-embedding:8b (dim 4096)  k=8  floor=off  probes=probes.json@510b5f20fe94 (86 memories, 60 probes)  db-upgrade=7  commit=2f213ff
n=60  recall@8=0.89  mrr=0.51  stale1st=0.70  abstain=0.00
  knowledge_update  n=10   recall@8=1.00  mrr=0.50  stale1st=0.70  abstain=0.00
  multi_hop         n=8    recall@8=0.62  mrr=0.16  stale1st=0.00  abstain=0.00
  single_hop        n=20   recall@8=0.90  mrr=0.58  stale1st=0.00  abstain=0.00
  temporal          n=8    recall@8=1.00  mrr=0.66  stale1st=0.00  abstain=0.00
  unanswerable      n=14   recall@8=0.00  mrr=0.00  stale1st=0.00  abstain=0.00
```

Every line matches the baseline's, per kind as well as overall. Only `db-upgrade` (2 to 7) and
the commit moved.

**The holdout floor sweep at the accepted margin**, re-run over the sealed 128-label set:

| margin 0.08 | n | recall@8 | mrr | abstain | the baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| fit | 90 | 0.88 | 0.72 | 0.84 | 0.88 / 0.72 / 0.84 |
| sealed | 38 | 0.84 | 0.70 | 0.85 | 0.84 / 0.70 / 0.85 |

Identical. The floor value the branch documents for `qwen3-embedding:8b` -- 0.08 -- still keeps
what it kept and silences what it silenced.

**The vector audit** is item 9 above: min 0.99842 against a worst-condition 0.99820 and a
tolerance of 0.995.

**Cold starts.** Two cold loads were measured, under two different conditions:

| condition | cold load | the baseline |
| --- | ---: | --- |
| 31.4 idle minutes, machine awake -- **the baseline's own condition** | **7.8 s** | 7.4 / 8.0 / 8.4 s |
| 409.9 minutes and a machine sleep in between | 12.2 s | (the baseline records 43 s for a first load from disk) |

The first is **inside the baseline's interval**. The second is outside it, and its cause was
not isolated: the latency is timed on the client, and the client had just woken from sleep, so
re-established connections account for the gap as readily as anything on the host does. What
can be said is that a cold load measured under the baseline's own condition is inside the
baseline's range; the 7.8 s is the comparable number, and treating it as such is a judgement,
stated here rather than buried.

**So no live metric is outside the baseline's interval:** the 60-probe scorecard and the floor
sweep are equal to it digit for digit, the audit minimum is inside it, and the cold start is
inside it when measured the way the baseline measured it.

## Where this differs from the spec

1. **The database moved.** SPEC section 1 freezes 3,010 memories and 43,936 entity rows; the
   copy taken for this run holds 3,011 and 43,938, in three projects rather than two. The live
   install wrote one memory in project `Morgan` after the baseline. Nothing is wrong; the
   numbers in this note are the live ones.
2. **Step 7 seeds one row per project that exists, which is two on the spec's own input.**
   SPEC section 4's table predicts "3 rows" for step 7 and section 6 item 7 asks for three;
   section 3.7 defines the step as one row per distinct project in `memories`, `facts` and
   `session_history`, which is what the code does. Migrating a copy of the eval snapshot -- the
   spec's frozen input exactly, two projects -- seeds **2** rows. Item 7 reads three today only
   because of the third project above. The code matches section 3.7; section 4's cell is an
   arithmetic slip against its own input.
3. **The counts line carries no thousands separator.** SPEC sections 3.6 and 6 quote
   `(3,010 across all projects)`; the renderer emits a bare integer and
   `tests/unit/surfaces/test_doctor_counts.py` pins that. The spec writes every number in prose
   with a separator, so this reads as prose style rather than a required format. Left as it is.

## What is still owed, and only the owner can do it

1. **Rotate the chat model server's API key.** It was printed in the open on 2026-09-19 and
   must be treated as compromised. Set the new value in `~/.config/morgan/.env`
   (`MORGAN_LLM_API_KEY`) and on the server.
2. **Close every running `morgan-mcp`**, every session that uses one -- not only a pre-phase-0
   one (`docs/WIRING.md`) -- and keep them closed until the migration is done. A tool call
   made while the migration holds the lock waits up to `MORGAN_DB_BUSY_TIMEOUT_MS` (an older
   `morgan-mcp` waits 5 seconds) and then fails with "database is locked"; a pre-phase-0 server
   left running writes rows in the old shape after the migration, which `doctor`'s
   `rows_missing_provenance` counts.
3. **Then, with this branch merged and the new `morgan` installed, run `morgan migrate` first,
   before any other command opens the file.** It takes a `VACUUM INTO` snapshot before it runs
   a step, into `MORGAN_SNAPSHOT_DIR`, for `morgan restore`; run first, that snapshot is the
   untouched version-2 file. `morgan doctor`, `morgan migrate --dry-run` and `morgan snapshot`
   open the file without changing it. `remember`, `recall`, `facts`, `forget`, `ask`,
   `consolidate` and `import`, and every `morgan-mcp` tool call, apply the light step 3 on
   their first open: it creates `embedding_spaces` and `projects` and moves `user_version`
   from 2 to 3. After that, every write is refused by name until `morgan migrate` runs. Restart
   the sessions once it has. The wave measured here took about a minute on a copy of the file,
   reached `user_version` 7 with `quick_check: ok` and left `rows_missing_provenance: 0`;
   restoring its snapshot put the file back at `user_version` 2, diffing empty.
