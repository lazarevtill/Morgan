# Morgan reshape — local-first, durable, reusable across projects

**Status:** approved design, not started
**Date:** 2026-08-02
**Supersedes:** the Personal Agent OS horizon plan
([vision](2026-06-09-personal-agent-os-vision.md), [horizons](2026-06-09-horizons-roadmap.md),
[ports](2026-06-09-ports-design.md),
[profiles & sync](2026-06-09-deployment-profiles-and-sync-design.md)) — see §10.

## 1. Diagnosis

Morgan was built as an architecture and never as a tool. The domain semantics are sound; the
wiring, the defaults, and the persistence are not. Nothing in the repo has ever run against real
data on the owner's machine.

Evidence, verified against the working tree on 2026-08-02:

| Claim | Evidence |
|---|---|
| Two of three retrieval signals die on restart | `modules/memory/store.py:40-42,65-67` — the BM25 index, `_by_id`, and the entity map are process-local dicts, written only in `store()`, never rehydrated. Vector payloads carry enough to reconstruct a `Memory` (`store.py:50-63`), so under Qdrant the vector signal survives and the other two do not. |
| The keyword index cannot read Russian | `modules/memory/retrieval/bm25.py:9` tokenises on `[a-z0-9]+`. Cyrillic is dropped entirely, so keyword recall returns nothing for Russian text — a large fraction of the intended corpus. |
| The default vector backend is ephemeral | `config.py:42` defaults `vector_backend="memory"`. Combined with the above, a fresh process recalls nothing at all. |
| Qdrant was never initialised | `ensure_collection()` is defined at `modules/memory/stores/vector.py:97`; `composition.py:81-86` constructs `QdrantVectorIndex` and never calls it. The only caller is one unit test. |
| The optimize/eval path cannot execute | `composition.py:440` asks the router for role `judge`; `config.py:114-118` creates only `strong` and `fast`; `RoleRouter.chat_for` raises `LookupError` (`providers/router.py:75`). The job body runs under `asyncio.ensure_future` (`scheduling/learning_jobs.py:217`), so the failure surfaces only as an unretrieved-task warning. Consolidation is unaffected — it uses role `strong` (`learning/consolidation.py:97`). |
| The promotion gate is unsound | `learning/champion_trainer.py:120-127` scores the first candidate and then **ignores the score**, promoting unconditionally when no champion exists; thereafter a bare `>` on a single run over a 12-item golden set (one item = 8.3 points), scheduled hourly by the worker. Currently unreachable in production only because the scorer raises `LookupError` first. |
| No deletion exists | Zero `delete`/`forget`/`purge`/`erase` functions across `morgan_brain/`, against a stated "forget me" premise. |
| "Cold path never blocks" is false | `bus/inproc.py:19-21` awaits every subscriber inline; `composition.py:218-219` registers the consolidation subscriber on that bus; the orchestrator publishes `RESPONSE_GENERATED` inline. Consolidation therefore runs inside the request. |
| Six documented settings are dead | `redact_egress`, `encryption`, `mlflow_tracking_uri`, `enable_channels`, `enable_mcp`, `telegram_token` — zero readers outside `config.py`. |
| Provider-agnosticism breaks at the composition root | `config.py:106-117` hardcodes the `"ollama"` provider key; `providers/factory.py:29-30,83-85` resolves it; `composition.py:341` constructs `OllamaEmbedder` directly. (The adapter seam itself is clean, and `OllamaEmbedder` in fact speaks OpenAI-compatible `/embeddings`, so llama-server works once the name and wiring are fixed.) |
| Defaults are internally inconsistent | `config.py:27` selects an embedding model whose own comment says dimension 2560; `config.py:45` defaults `embedding_dim` to 1024. |
| A non-editable install cannot start | The built wheel contains neither `morgan_brain/eval/data/` nor `morgan_brain/providers/data/`, so `CapabilityRegistry.from_packaged()` (`composition.py:189`) fails outright. Cause: the root `.gitignore:37` `data/` pattern, whose negations do not apply when hatchling builds from `morgan-brain/` (there is no `morgan-brain/.gitignore`). |
| The docs overstate the code | CLAUDE.md claims "820 tests, mypy-strict clean". Actual: 883 collected — 871 passed, 11 skipped, 1 xfailed in ~13 s — and `mypy morgan_brain` reports 1 error (`channels/telegram.py:57`). The two "latent bugs" it lists as H1 blockers are already fixed (`orchestrator.py:236`, `composition.py:334-339`). |
| The suite proves little | It runs in ~13 seconds because nothing crosses a process boundary, touches disk, or contacts a real model. |

What is genuinely worth keeping, and is rare in the 2026 landscape: the single `MemoryGate`
chokepoint (`security/memory_gate.py`, 39 lines), `MemorySource` actor attribution, temporal facts
that supersede rather than overwrite, the per-item scratch gate (`eval/runner.py:60-72`), the
provider role router, and surprise-gated consolidation.

**The reshape is therefore not a rewrite.** It makes one claim true end to end — durable,
project-scoped memory reachable from anywhere — and re-arms the rest on real data.

## 2. Constraints (owner decisions)

1. **One brain, homelab-hosted.** A single always-on deployment; the three laptops and all AI tools
   connect to it. There is one store and one truth, so device sync, the read-only replica, and the
   phone client are out of scope permanently.
2. **Two first-class consumers.** An MCP server so every AI tool shares the brain, and an importable
   Python library so the owner's other projects use it directly. The chat assistant is demoted to
   one app among several, kept working, with the door open to becoming universal later.
3. **llama.cpp, not Ollama.** `llama-server` serves chat, `/v1/embeddings`, `/v1/rerank`, and GBNF
   grammars from one binary on a 24 GB+ VRAM GPU. Local judge and reflection models are viable.
4. **One store, project-tagged.** Every memory carries a project; recall defaults to the current
   project and crossing projects is explicit.
5. **Self-learning is the point, so it gets fixed rather than parked.**
6. **The homelab counts as remote.** Transport authentication and at-rest encryption matter. No
   third-party model endpoint is in scope, so egress PII redaction is not.
7. **Cut hard.** Subsystems without production call sites are deleted, not kept "for later".

## 3. Target architecture

One deployment, one data directory, four surfaces over one core.

```
                 ┌───────── surfaces ─────────┐
   CLI  ·  MCP server  ·  Python library  ·  HTTP /api
                 └─────────────┬──────────────┘
                          MemoryGate            ← the only way in, hot AND cold
                               │
              ┌────────────────┴────────────────┐
              │   ONE SQLite database           │
              │   episodics · facts · entities  │
              │   history · signals             │
              │   FTS5 (keyword) · sqlite-vec   │
              └────────────────┬────────────────┘
                               │
              providers → llama-server (OpenAI-compatible)
```

| Surface | Consumer | State |
|---|---|---|
| CLI `morgan …` with `--json` | the owner's shell, scripts, any repo | new, thin |
| MCP server | Claude Code, Cursor, any MCP client | new, 5 tools |
| Python library `Brain` | the owner's other projects | new, thin facade |
| HTTP `/api` | the assistant, remote clients | exists; auth hardening |

The library facade is the implementation the other three consume, so behaviour cannot diverge
between them. **`MemoryGate` is the only way to reach a store — on the cold path as well as the
hot path** (§4.3 makes this true; today it is not).

### 3.1 Cold-path execution model

The service is one process. Consolidation and signal mining must not run inside the request, which
they do today. `RESPONSE_GENERATED` handlers are dispatched to a **bounded background task queue**
drained by a worker task in the same process; `publish()` returns as soon as the item is enqueued.
Queued work is derived from durable signal rows, so a crash loses scheduling, not data — the drain
resumes from the rows on restart. If the queue is full, the item is dropped with a logged counter
rather than blocking the response.

The separate `learning-worker` process remains supported for the homelab, as an opt-in that
requires the Redis bus extra. It is no longer required for the loop to close.

Because more than one process may open the database, SQLite runs in **WAL mode with a busy
timeout**, set at connection open.

## 4. The five load-bearing fixes

### 4.1 Durability — one database

One SQLite database is the source of truth for episodics, facts, entities, session history, and
signals. **FTS5 replaces the in-process BM25 index** and **`sqlite-vec` holds the vectors**, both
inside that same database, behind the existing `VectorIndex` Protocol.

Rationale for one engine rather than two:

- **`forget()` becomes a single transaction** across vectors, FTS rows, facts, and signals. A
  two-engine design (LanceDB or Qdrant beside SQLite) has no cross-engine transaction, so a crash
  mid-delete leaves orphaned vectors and erasure has to be re-run idempotently.
- **At-rest encryption covers everything at once** (§6), including vectors — which a separate
  vector engine would leak in plaintext.
- It makes constraint #4, "one store", literally true.

At ~323 conversations for one user, `sqlite-vec`'s brute-force search is comfortably adequate; ANN
is not needed. The `VectorIndex` Protocol keeps the choice contained, and **LanceDB is the named
escape hatch** if corpus growth ever outpaces brute force — a decision to be triggered by measured
latency, not anticipation.

Contract changes required (contract-first, per repo rule):
- `VectorIndex` gains `delete(ids)`.
- Real erasure requires explicit `VACUUM` after bulk deletes; `sqlite-vec` row deletes are not
  space-reclaiming on their own.

`ensure_collection()` and the Qdrant backend are retained as the opt-in scale path, and finally get
a production call site. A startup assertion fails loudly when the provider's reported embedding
dimension disagrees with the store's.

### 4.2 llama.cpp as the default provider — remote first, local as fallback

**The baseline topology is remote.** `llama-server` runs on the homelab GPU box; the three
laptops reach it over the NetBird overlay. A laptop running its own `llama-server` is the
fallback, for offline work or development. Both are the same OpenAI-compatible protocol, so the
difference is only the endpoint URL plus what a network hop demands:

- **Outbound auth.** `llama-server --api-key` is supported, so Morgan needs a setting for the key
  it *presents to the model server*. This is distinct from `MORGAN_API_KEY`, which is the key
  *clients present to Morgan* — the two point in opposite directions and must not be conflated.
- **Timeouts assume a network.** A 24 GB GPU generating a long completion across NetBird has a
  different latency profile than a loopback socket. The request timeout is configurable, with a
  default sane for a remote box under load.
- **Unreachable is normal, not fatal.** A laptop off the network or a rebooting homelab must not
  crash-loop the service. The startup probe raises on a *dimension mismatch* but only warns on an
  *unreachable host*, naming the endpoint.
- **No hardcoded hostname.** The endpoint comes from env, per the project's config rule. A
  localhost default exists so a fresh clone runs with zero configuration; it is a development
  convenience, not the expected production value.
- **`morgan doctor` reports the endpoint and whether it answers** — the first question anyone asks
  when this breaks is which server they are actually talking to.


Remove the hardcoded `"ollama"` from `config.py:106-117` and `providers/factory.py:29-30,83-85`, and
replace the direct `OllamaEmbedder` construction at `composition.py:341` with a routed embedder.
Default the provider to `llama-server`'s OpenAI-compatible endpoint, and **bind the `judge` and
`reflection` roles**, whose absence is the sole reason the optimize path has never run. GBNF
grammar-constrained decoding becomes the top rung of the structured-output ladder, replacing
prompted-JSON as the practical default.

### 4.3 Project scoping

Add `project` to `Memory` and `TemporalFact`.

**The gate contract is extended, not just filtered.** Today `composition.py:190-196` hands the raw
`SqliteTemporalStore` to `MemoryConsolidator`, which uses it directly for `current_facts`
(`consolidation.py:185,262,314`), `close_fact` (`:241`), and `set_confidence` (`:341`) — operations
the gate does not expose. A project filter applied only at `MemoryGate` would therefore **not** bind
nightly consolidation, which could merge facts across projects and silently defeat the guarantee.
So `MemoryGate` gains `current_facts`, `close_fact`, and `set_confidence`, all project-scoped, and
consolidation is handed the gate rather than the raw store. `SessionHistoryStore` and `SignalStore`
move behind the gate for the same reason.

**How "current project" is determined**, per surface:
- **CLI** — the git repository root of the working directory, by name; `--project` overrides,
  `--all-projects` crosses.
- **MCP server** — the client's declared workspace root, falling back to the server's configured
  default; a `project` argument overrides per call.
- **Library** — required explicitly at `Brain(project=…)`. No implicit default.
- **HTTP** — a required request field.

Recall defaults to the current project; crossing is always explicit. This is the precondition for
pointing Morgan at company repositories at all.

### 4.4 Deletion

`forget()` performs cascading erasure in one transaction across: episodics, vectors, FTS rows,
entities, fact intervals, session history, **and `signals.db` — which stores the full `query`,
`original_reply`, and `user_edit` text of every turn** (`learning/signals.py:41-43,69-71`), exactly
the data the premise covers. Followed by `VACUUM`.

Two consequences are named rather than ignored:
- A **promoted champion preprompt may embed text mined from now-forgotten conversations.** `forget()`
  flags affected champion versions for review; it cannot un-learn them. The registry keeps versions,
  so rollback is available.
- `workspace_path` contents are **out of scope** for `forget()` and documented as such.

### 4.5 Seed from real data

Import the owner's ChatGPT export (`conversations.json`, 323 conversations) through consolidation,
so the brain is useful on first run rather than an empty box. The import lands in a dedicated
`archive/chatgpt` project, not in any working project.

**A conversation-level holdout is selected at import time** and recorded, before any consolidation
runs — see §5.

## 5. Self-learning, made sound

The mechanism stays; the gate is rebuilt. **Auto-promotion is disarmed by an explicit
`MORGAN_ENABLE_CHAMPION_PROMOTION` flag, defaulting to false, checked where the trainer registers
its job** — not by convention. Milestone 1 binds the `judge` role, which makes the unsound gate
*reachable* for the first time, so the flag must land in the same milestone as the binding.

1. **Measure the noise floor.** Score one fixed candidate ten times against the current judge and
   report the spread. Until that number exists, no gate can be calibrated. Roughly an afternoon.
2. **Grow the golden set** from 12 items to 100+, drawn **only from the held-out conversations
   reserved at import (§4.5)**. This is the correction to a false assumption: the scratch gate at
   `eval/runner.py:60-72` isolates eval *writes during eval runs*; it does nothing to stop the
   reflection optimizer from mining the same conversations that golden items came from. Without an
   import-time holdout, the gate measures memorisation. Labeling 100+ items is days of work, not a
   side effect of import.
3. **Replace the gate.** Delete the unconditional first-candidate promotion at
   `champion_trainer.py:123-127`. Require a paired-bootstrap win over multiple runs with an effect
   exceeding the measured noise floor. Only then set the promotion flag true.

Judge and reflection models run locally on the 24 GB GPU. The versioned registry and alias rollback
are kept unchanged — they are sound.

## 6. What is deleted

Verified to have no production importers: `channels/` · `voice/` + `interfaces/voice.py` +
`apps/perception_gpu/` · `modules/mcp/` (the MCP **host** stub, whose `connect()` body is a
placeholder at `client.py:142-144` — unrelated to the MCP **server** built in §3) ·
`providers/resilience.py` · `interfaces/rerank.py` (no implementations) · the duplicate `Embedder`
in `interfaces/embedding.py` · the `NotImplementedError` stubs in `learning_lifecycle/` · the
top-level `clients/` package, which squats a generic name in `site-packages` ·
`docs/ARCHITECTURE_V2.md`, which specifies ten microservices contradicting the shipped design.

Three of these are **not** clean removals and require edits to kept code:
- **`proactivity/` is flag-gated production wiring, not dead code.** `apps/learning_worker/__main__.py:47,156-173,274-277`
  builds a `ProactivityEngine` and subscribes to `HEARTBEAT` behind `enable_proactivity`. That
  wiring must be excised with it.
- **Cutting `scheduling/heartbeat.py` breaks the kept `scheduling` package** — `scheduling/__init__.py:14,19`
  imports and re-exports `HeartbeatManager`. That import and `__all__` entry must be edited.
- **Cutting `interfaces/embedding.py`** also requires removing `providers/factory.py:14,77-88`
  (`build_embedder`, itself uncalled).

**Privacy is deleted, and replaced by an operational control.** Both `privacy/egress.py` (there is
no third party to redact for) and `privacy/crypto.py` are removed. Field-level encryption is
incompatible with the FTS5 keyword index of §4.1 — you cannot full-text index ciphertext — and
would not have covered vectors anyway. At-rest protection is therefore **volume-level encryption on
the homelab host**, which covers the entire database including vectors and signals, costs no code,
and is documented in the operations guide. Transport protection is a bearer token over the
**NetBird** overlay network (self-hostable WireGuard, already in use across the owner's machines)
or TLS at a reverse proxy. Shipping an advertised but unreachable security control is worse than shipping none.

A docs truth pass lands in the same commits: the six dead settings, the false test and mypy counts,
the two already-fixed "latent bugs", and the word "bi-temporal" — the schema carries valid time only
(`valid_from`/`valid_to`/`superseded_by`/`last_confirmed`), with no ingestion-time column.

## 7. Sequence

| # | Deliverable | Acceptance | Size |
|---|---|---|---|
| 0 | `.gitattributes` to end the CRLF churn; tag `legacy-v0.0.4-full`; delete §6 including its tests and the three dependent edits; docs truth pass; fix the wheel data-file exclusion | Wheel installs into a clean venv and starts; suite green; `mypy` reports 0 errors | ~1 day |
| 1 | One-database store: SQLite + FTS5 + `sqlite-vec` + WAL; project scoping through an extended `MemoryGate`; `forget()`; cold-path task queue; llama.cpp defaults + `judge`/`reflection` roles + the promotion flag defaulting off; `morgan remember\|recall\|facts\|forget\|ask\|doctor` | Store a fact in one repo, reboot, recall it from a different repo with `--all-projects`, with a real embedder, on disk — all three retrieval signals returning, including for Russian text | 4–5 days |
| 2 | ChatGPT import with import-time holdout; MCP server (stdio local, streamable-HTTP + bearer token to the homelab) + `SKILL.md` | An external MCP client writes an episodic that survives consolidation into a fact | 3 days |
| 3 | Noise floor → 100+ labeled golden items from the holdout → paired-bootstrap gate → set the promotion flag | A promotion requires a bootstrap win exceeding the measured noise floor | ~1.5 weeks |

Milestones 0 and 1 are planned in detail now. Milestones 2 and 3 are re-planned after 1 lands,
because what 1 teaches about retrieval quality changes both.

## 7.1 Concept annotation and query expansion — milestone 2, decided 2026-08-03

Today `Entity` is `{name, type}` with no relations, and `EntityIndex.search` matches lowercased
names exactly. A memory tagged `horse` is reachable by the literal token "horse" and nothing else.
The owner wants recall to follow meaning: asking about a pet, an animal, or a sport should reach
the horse memory, and the tag structure should reorganise itself as relationships change.

**Decided source: an LLM proposes, a curated file overrides.** The consolidation worker (cold
path, `strong` role, local GPU) extracts concepts per memory; a hand-editable concepts file pins
the relationships the owner cares about — `Harbor → registry → infrastructure` — and wins over the
model's guess. This was chosen over a lexical resource, which is English-only and would fail on a
corpus that is substantially Russian, and over embeddings-alone, which offers no inspectable tags
and nothing to correct when a match is wrong.

Shape, consistent with the existing invariants:

- **Cold path writes.** Concepts and edges are produced during consolidation, never in a request.
  Store `concepts`, `memory_concepts(weight, source=extracted|curated)`, and
  `concept_edges(relation=broader|narrower|related, weight)`.
- **Hot path reads.** Query terms resolve to concepts and expand by SQL lookup — no model call —
  producing a **fourth ranking** fed into the existing reciprocal-rank fusion beside vector, FTS
  and entity.
- **Restructuring is deferred work.** When edges change, mark affected memories dirty; the next
  cold-path pass re-annotates. The graph is never rewritten mid-request.

**The constraint that governs the design: bounded expansion.** Unbounded traversal turns
`horse → animal → pet → dog` into a horse query returning the dog. Expansion is one hop, weighted
strictly below exact matches, and never able to outrank them.

**Sequencing note.** This lands alongside the retrieval-quality measurement work, deliberately.
`tests/memory_quality/` is currently 3 tests over a hash embedder — there is no benchmark that
could tell whether expansion improves recall or merely adds noise. Building a precision-sensitive
feature without that measurement would be guessing. It also interacts with the open relevance-floor
question (§11): expansion makes "recall always returns something" materially worse, so the two
should be decided together.

## 8. Testing

The current suite is a refactoring net, not evidence. Milestone 1 adds integration tests that
**reopen the store in a second process** against `tmp_path`, exercising all three retrieval signals
after restart — the failure mode that makes today's system useless — plus a Cyrillic recall case,
since the current tokeniser silently drops it. A wheel-install test asserts that packaged data
files resolve in a non-editable install.

Tests belonging to deleted modules are removed in milestone 0, with those modules. Tests for
surviving behaviour are only touched after the new integration tests are green.

Two implementation traps to cover: raw user text is not a valid FTS5 `MATCH` expression — tokens
must be quoted or syntax errors surface as recall failures — and the entity signal currently feeds
RRF an unordered dict-iteration list (`store.py:84-88`), so its ordering must be defined when it
moves to SQL. RRF itself is rank-only (`retrieval/fusion.py:7-12`), so the fusion mechanism survives
the swap unchanged.

## 9. Open decisions

None blocking. The `sqlite-vec` choice is the one to revisit first if measured recall latency
degrades; LanceDB behind the same Protocol is the escape hatch.

## 10. What this supersedes, and non-goals

Superseded: device sync, the read-only memory replica, the reference phone client, deployment
profiles, and the `/v1` OpenAI-compatible facade — all consequences of a multi-device assumption
ruled out in favour of one homelab host.

Non-goals, unchanged: voice GPU serving; LoRA fine-tuning (still gated behind the four-condition
escalation test); multi-tenancy beyond keeping everything `user_id`-keyed.

## 11. Known risks

- **Retrieval quality is unmeasured.** `tests/memory_quality/` is 3 tests over a SHA-256 hash
  embedder. Milestone 1 makes retrieval durable but does not prove it is good; measuring it needs
  the seeded corpus from §4.5.
- **`sqlite-vec` is pre-1.0.** Mitigated by the Protocol boundary and by a corpus size where brute
  force suffices, but it is a young dependency in the critical path.
- **The cheapest competing hypothesis is untested.** Git-backed markdown with provenance may serve
  cross-project recall better than a vector store. If, after milestone 1, `morgan recall` is used
  mainly for exact-phrase lookups, the embedder should come out.
- **The learning loop may not clear its own bar.** A single user producing a few thousand turns may
  never generate an effect exceeding the judge's noise floor. Step 5.1 exists to find that out in an
  afternoon, before the labeling work in 5.2.
