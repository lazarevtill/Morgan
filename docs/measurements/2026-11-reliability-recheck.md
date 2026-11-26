# Reliability recheck — 2026-11-26

This is a verification record for existing Morgan behavior before AOA integration.
Synthetic fixtures are not evidence of semantic quality, delivery to a person, or actual
consumer acceptance. No private memory, external recipient, deployment or persistent
service is used. Historical measurements elsewhere retain their original dates.

## Starting point and scope

- Clean `main` at `3eb3ec4253214e930c2edc09046add60a7d99f6b`.
- `origin`: `https://github.com/lazarevtill/Morgan.git`; `git fetch origin --prune`
  confirmed `HEAD`, `origin/main` and `origin/HEAD` name that same commit.
- Working branch: `fix/reliability-recheck-2026-11`.
- Baseline tests use an untouched `git archive HEAD` under `/tmp/morgan-recheck-baseline`.
  Changes and regression tests are evaluated separately in the checkout.
- Local context read: aoasis research, Tessi's product/consumer boundaries, Heimdall's
  deterministic authority and planned memory boundaries, and ml-knowledge-base's
  agent-workflow evidence guidance. These are context, not integrations tested here.
  All source changes remain in Morgan.

## Behavior checklist

| Supported behavior | Implementation | Existing verification | Boundary still requiring separate evidence |
| --- | --- | --- | --- |
| Scoped capture, immutable replay and revisions | `memory/gate.py`, `module.py`, `store/episodic.py` | `test_project_scoping`, `test_immutable_events`, `test_event_revisions`, frozen revision contract | Attribution is reported provenance, not authenticated identity |
| Ranked recall, facts and exact evidence | `memory/recall/`, `evidence.py`, `knowledge/` | `test_each_signal_is_project_scoped`, `test_fact_budget`, `test_durable_evidence`, retrieval probes | Semantic quality needs a real embedding server; multi-hop remains unsupported |
| Model consolidation with source validation | `memory/knowledge/consolidation.py`, `basis.py` | `test_consolidation_basis`, `test_consolidation_concurrency` | Synthetic operations do not prove model extraction/entailment quality |
| Atomic indexes, concurrent writes and erasure | `memory/store/`, `module.py` | `test_transaction_recovery`, `test_forget_atomicity`, process/concurrency tests and table registry check | Backups deliberately retain forgotten content; blocked WAL checkpoint is reported |
| Upgrade, snapshot and restore | `memory/migrations.py`, `snapshot.py` | `test_migrate_cli`, `test_upgrade_concurrency`, `test_restore_cli`, future-schema refusal | Restore requires every database user to stop; no online restore guarantee |
| Embedding width/model identity | `composition.py`, `checked_embedder.py` | width and fingerprint tests, import canary tests | Process cache sequencing needs cross-context coverage; no automatic re-embedding |
| CLI, JSON and seven MCP tools | `surfaces/cli/`, `mcp_server.py` | CLI subprocess tests, SDK sessions, raw stdio JSON-RPC tests | Real consumer interoperability is a separate gate |
| Settings, provider failure, bind/auth controls | `config.py`, `providers/`, `surfaces/network.py` | settings precedence, retry budgets, bind/listener and key-routing tests | Real model endpoint unavailable initially; HTTP SDK defaults need transport checks |
| Credential admission | `memory/credentials.py`, capture/consolidation/import paths | `test_credentials` | Narrow explicit shapes only; metadata and separate journals are outside admission |
| Explicit tasks and execution receipts | `tasks.py` | task/batch/receipt units and process crash recovery | Registered trusted handlers only; synthetic external effects are not delivery evidence |
| Finite approved notes and supervision | `local_notes.py`, intake/recurrence/worker/supervisor modules | note process recovery, scheduled/recurring/supervisor integration tests | No installed scheduler or service; approval does not renew itself |
| Current plans and reviewed owner input | `daily_brief.py`, `guided_plan.py`, `owner_input.py` | daily brief, completion/cancellation/reopen/renewal/status process tests | Model draft quality and physical GUI interaction need independent trials |
| Optional local model/UI/push/runtime adapters | `model_cycle.py`, `owner_model.py`, `owner_ui.py`, `owner_push_cli.py`, runtime tooling | guardian/STOP, scripted dialog, loopback push, wheel/pinned-runtime tests | No real push recipient, native human interaction or runtime activation tested |

## Reproduced defects and changes

1. **Embedding proof crossed database boundaries.** The process cache keyed an embedding
   space by endpoint/model/width/integer row ID. Independent databases commonly both had
   ID 1, so an incompatible or unrecorded fingerprint could reuse another database's proof;
   a stricter tolerance could reuse a loose check too. Cache keys now include the persisted
   fingerprint and tolerance; unrecorded spaces still validate their own stored sample.
2. **Failed canaries retained prior trust.** An explicit mismatch left the successful cache
   entry behind. Checks now revoke it before the request, so subsequent calls recheck until
   recovery. Four cache/canary regressions failed before these fixes.
3. **Concurrent fingerprint initialization overwrote the winner.** Two first requests could
   both observe NULL before model I/O. A short immediate transaction now rereads and compares
   the winner instead of overwriting it. Separate-connection tests cover matching and
   mismatching winners; no lock spans the model await. The mismatching case failed before
   the fix. All three changes are in `memory/checked_embedder.py`.
4. **Chat HTTP errors escaped the surface contract.** The adapter caught connection errors
   but not SDK HTTP status errors. `providers/openai_compat.py` now maps refusal and transient
   failures to the same named domain errors used by embeddings. Timeouts report slow rather
   than unreachable. `providers/redaction.py` shares the existing credential-redaction
   policy with both adapters; suppressed raw exception chains cannot echo the response key.
   Eighteen real-socket status regressions failed before the fix. Independent review then
   reproduced 307/308 redirects forwarding prompts to a different origin: the SDK followed
   `Location` by default. Its owned public HTTP client now disables redirects, with four
   streaming/nonstreaming cases asserting zero destination requests. The final independent
   advisor review found a second gap after successful stream headers: SSE error events and
   read failures escaped the boundary. Stream creation, iteration and cleanup now share it.
   Six real HTTP regressions first failed, then passed: credential-reflecting SSE errors,
   read timeouts and truncated streams, each before and after partial text. Delivered text
   is not replayed; chat still makes one attempt. The broader provider checks passed 89 cases
   on both OpenAI 3.60.0 and the declared minimum 3.59.0 after this correction.
5. **Authenticated remote MCP returned 421.** FastMCP was constructed with its default
   loopback Host policy; changing the bind later retained that policy. `run_http_async`
   now sets the policy for the actual bind after the API-key guard. Two full-ASGI remote
   initialization cases failed before the fix; loopback Host protection remains tested.
6. **Configuration errors broke CLI JSON.** Settings were constructed outside the output
   error boundary. Invalid values and undecodable CLI dotenv files produced tracebacks and
   empty stdout. Four regressions failed before the fix. `surfaces/cli/__main__.py` now
   preserves exit 1 and the existing JSON/text error envelope, omits validation input values,
   and never creates a database for these failures. Independent review also reproduced
   unknown `~user` expansion escaping as `RuntimeError`; configuration path expansion now
   normalizes that into a named validation error. Six JSON/text cases cover data, snapshot
   and code-root paths. Settings validation also hides input values in rendered exceptions.
7. **Credential refusal stranded reviewed workflows.** `owner_input` and `guided_plan`
   saved attempt/pending markers before memory admission. A forbidden source could strand
   the review; a forbidden derived title could first leave an orphan source or partially
   capture a selection. New reviews and first confirmations now preflight all new payloads
   before journalling attempts. Existing started attempts retain their fence and historical
   replay semantics. Existing journals already stranded by older code are not reset or
   automatically repaired. Seven public CLI cases reproduced the defect before the fixes; further
   regressions cover corrected reviews, renewals, completion and historical replay.
8. **Fractional/bool run limits violated finite-count contracts.** `RunLimits(max_actions=1.5)`
   allowed two effects and wrote metadata diagnostics refused to read. Counts now require
   integers and durations require nonboolean numbers before existing range/finite checks.
   Eight negative cases cover this public SDK boundary; CLI integer arguments are unchanged.

The source changes introduce no schema migration, new tool, execution authority or automatic
retry. Existing accepted evidence and execution receipts are not rewritten. New integration
coverage connects CLI subprocesses to the actual MCP stdio SDK, scoped evidence, restart,
forget and snapshot recovery. Documentation now names seven MCP tools and four read-only
tools, describes the implemented bounded workflows, distinguishes development from pinned
deployment, and corrects immutable import replay and consolidation attribution guidance.

## Environment and commands

macOS arm64, CPython 3.12.13. A new disposable development environment resolved
pytest 9.2.0, pytest-asyncio 1.5.0, Ruff 0.16.10, mypy 1.20.2, Bandit 1.9.4,
mcp 1.30.0, openai 3.60.0, httpx 0.29.0, sqlite-vec 0.1.9 and Pydantic 2.14.4.
The macOS whole-suite check used pytest-xdist 3.8.0 with four isolated workers to reduce
wall time; the unmodified baseline ran serially. This verifies those resolved versions, not
every version permitted by the dependency ranges.

Setup (no global install or persistent service):

```sh
UV_CACHE_DIR=/tmp/morgan-recheck-uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=/tmp/morgan-recheck-uv-cache uv pip install --python .venv/bin/python -e '.[dev]' bandit build
UV_CACHE_DIR=/tmp/morgan-recheck-uv-cache uv pip install --python .venv/bin/python pip
UV_CACHE_DIR=/tmp/morgan-recheck-uv-cache uv pip install --python .venv/bin/python pytest-xdist
mkdir -p /tmp/morgan-recheck-baseline
git archive HEAD | tar -x -C /tmp/morgan-recheck-baseline
```

The archive command ran before changes at the starting commit. Baseline `pytest` ran in
that archive with `PYTHONPATH=/tmp/morgan-recheck-baseline` and the checkout's absolute
`.venv/bin/python -m pytest -q --tb=short`. That path is inherited by subprocess tests.

| Check | Observed result |
| --- | --- |
| Unmodified archive `python -m pytest -q --tb=short` | 2112 passed, 4 skipped in 1001.26s |
| macOS checkout `.venv/bin/python -m pytest -q -n 4 --dist worksteal --tb=short` | 2181 passed, 4 skipped in 721.20s, before final config/dependency/runtime follow-ups |
| `.venv/bin/ruff check .` | Passed, baseline and final |
| `.venv/bin/ruff format --check .` | Passed: 328 baseline / 334 final files |
| `.venv/bin/mypy morgan_brain` | Passed: 104 baseline / 105 final source files |
| `.venv/bin/bandit -c pyproject.toml -r morgan_brain` | Zero findings, baseline and final; existing nosec-comment warnings remain |
| `git diff --check` | Passed |
| `.venv/bin/python -m tests.revision_contract.run_actual --output <new-file.json>` followed by `.venv/bin/python -m tests.revision_contract.score_gold --actual <new-file.json>` | 68/68 frozen authored cases passed; fixture unchanged |
| `.venv/bin/python -m pytest -q tests/integration/test_cli_mcp_recovery.py` | 1 passed; actual CLI/stdio subprocesses, hash backend |

Focused independently owned checks also passed: 648 memory/persistence cases, 539
provider/surface cases, 265 task/owner recovery cases, 58 owner/limits cases and 53 guided
review cases. These overlap the full suite and must not be added to its total.
Pytest emits the existing unspecified async fixture-loop-scope deprecation warning.
The later [dependency audit](2026-11-dependency-audit.md) records the final runtime/tool
versions, minimum-version checks, image scans and Linux test result. Subsequent configuration
and provider follow-ups passed 55 focused tests on macOS; these overlap the whole suite.

## End-to-end evidence

**Synthetic integration.** `test_cli_mcp_recovery.py` runs actual subprocess CLI commands
and an MCP SDK stdio session over a disposable database. It verifies the seven tool schemas
and four read-only annotations, CLI-to-MCP and MCP-to-CLI persistence, exact Unicode text,
project and owner isolation, explicit all-project recall, conflicting immutable replay,
source-only context with no authority, forget's snapshot, preview-only restore, and recovery.
The server is closed before restore. Hash vectors make no semantic-quality claim.

The full package was also built with `python -m pip wheel --no-deps --wheel-dir
/tmp/morgan-recheck-wheel .`, installed using `uv pip install --python
/tmp/morgan-recheck-wheel/runtime/bin/python <wheel>` into a new scratch venv, and imported
with `python -I` from outside the checkout. Installed `morgan remember`/`recall` persisted
the same synthetic ID and `morgan-mcp --help` exposed its transport option. This is a real
wheel-install smoke check; no runtime was activated or deployed.

A separate disposable HTTP subprocess bound to an ephemeral port with a synthetic token.
An unauthenticated request returned 401; an authenticated request with a remote Host name
initialized through the real streamable-HTTP SDK, listed seven tools, stored a fixture and
read its evidence. It shut down after the check. This supplements the full-ASGI regressions.

Separate actual `owner_input` invocations exercised init → submit → review → refused wrong
digest → confirm → duplicate confirm → status → STOP → refused submission → resume. Exactly
one task existed, and neither submit nor review authorized it. Actual `local_notes_cli`
approve/run/run delivered one byte-exact Markdown file, with zero effects on repeat; STOP
blocked work, resume retained deduplication, and diagnose read the separate task journal.
Crash, expiry and uncertain-outcome recovery are covered by the existing process suites;
no external recipient was used.

**Real provider integration.** The owner's supplied LAN Ollama endpoint was reachable
(Ollama 0.40.18). Only synthetic text was sent. The private endpoint was supplied through
environment variables, never committed to configuration. With `OLLAMA_BASE` set to that
base URL, the existing measurement ran as:

```sh
MORGAN_LLM_ENDPOINT="$OLLAMA_BASE/v1" \
MORGAN_EMBEDDING_ENDPOINT="$OLLAMA_BASE/v1" \
MORGAN_EMBEDDING_MODEL=qwen3-embedding:8b MORGAN_EMBEDDING_DIM=4096 \
.venv/bin/python -m pytest -q -s --live \
  tests/memory_quality/test_retrieval_quality.py::test_recall_finds_the_right_memory_when_the_query_shares_no_words_with_it
```

Result: **1 passed in 106.16s**, 86 synthetic memories, 60 probes, k=8, floor off,
fixture digest prefix `510b5f20fe94`, schema 11. The run reports starting Git HEAD `3eb3ec4`;
it executed the changed working tree, not an immutable build of that starting commit.

| Probe kind | n | recall@8 | MRR | Stale first | Abstain |
| --- | --- | --- | --- | --- | --- |
| Overall | 60 | 0.89 | 0.51 | 0.70 | 0.00 |
| Single hop | 20 | 0.95 | 0.59 | 0.00 | 0.00 |
| Temporal | 8 | 1.00 | 0.66 | 0.00 | 0.00 |
| Knowledge update | 10 | 0.90 | 0.51 | 0.70 | 0.00 |
| Multi-hop | 8 | 0.62 | 0.16 | 0.00 | 0.00 |
| Unanswerable | 14 | 0.00 | 0.00 | 0.00 | 0.00 |

A separate temporary database used `gemma4:latest` for chat and
`qwen3-embedding:8b` (4096) for embeddings. Actual CLI `remember --source user_stated`,
`recall`, `ask`, `consolidate`, `facts`, and MCP stdio `recall`/`evidence` succeeded.
Gemma answered `PostgreSQL`; consolidation persisted one inferred PostgreSQL fact with
`support_event_ids=["database-choice"]`. Reopened recall returned both the fact and source;
another project could not fetch that source. A deliberate 1024-width configuration failed
with exit 1 and the named mismatch; restoring 4096 recovered without rewriting the database.
This is one observed consolidation example, not a model-extraction accuracy evaluation.
A subsequent `doctor --project harbor --vectors --clients 2 --json` on this same scratch
database re-embedded all three stored episodics: both clients matched at approximately
1.0000 minimum cosine, with zero below-tolerance rows and zero disagreements. The five-string
space fingerprint also matched. This sample does not certify every model or production row.

**Actual consumer interoperability.** MCP SDK stdio and HTTP transports were exercised.
Tessi, Heimdall, Claude Desktop/Code and another independent consumer were not connected.
No AOA integration was built, and no such interoperability acceptance is claimed.

## Test quality and remaining limits

- Existing SQLite suites test rollback, persisted rows, competing processes and crashes;
  they are stronger than return-code-only checks. Frozen revision acceptance uses production
  state and leaves its declared gold unchanged. Deterministic embeddings test wiring only.
- The old per-test cache reset hid sequences across databases/policies in one process;
  new regressions cover those sequences. Direct MCP dispatch and isolated auth tests missed
  FastMCP's full Host middleware; new full-ASGI and socket checks close that gap. Chat mocks
  missed real SDK status behavior; new raw HTTP servers exercise it.
- One known advertised startup limitation remains: FastMCP reads an unrelated cwd `.env`
  and an invalid UTF-8 file still causes startup failure. Reproduced with disposable data.
  Morgan's own CLI config error is now structured, but the SDK offers no clean public
  constructor opt-out for its dotenv read. No global monkeypatch was introduced.
- Model results still show stale-first ranking, weak multi-hop and no abstention with
  floor disabled. Consolidation still selects through an empty-query recall and requires
  eligible attributed evidence. No broad correctness claim follows from one successful fact.
- Physical macOS GUI interaction, real push delivery/read receipts, installed guardian or
  login-agent operation, production backup restoration, Windows execution and the
  entire dependency-version range remain unverified here. No service or deployed pilot was modified.

The later Linux 3.12.16 full run had 2147 passes, 40 deadline/contention failures and four
skips under concurrent host load and a bind-mounted source tree. Native serial follow-up
cleared the transient cases and reproduced one race-test budget mismatch on unchanged
baseline code. The race tests now use their existing overall contention budget, while a
new unit preserves the production default and tests bounded refusal/recovery. Final native
affected checks: 16 passed. The [dependency audit](2026-11-dependency-audit.md) preserves
the failed runs, exact follow-ups and remaining platform/security limits.
