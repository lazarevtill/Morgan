# Morgan Local-First Reshape — Milestones 0 & 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Morgan from an unused architecture into a durable, project-scoped memory brain that survives a restart and is reachable from any repository via a CLI.

**Architecture:** Collapse all persistence into **one SQLite database** (episodics, facts, entities, history, signals, FTS5 keyword index, `sqlite-vec` vectors) behind the existing `MemoryGate`, which is extended to cover the cold path so project scoping cannot be bypassed. Delete every subsystem with no production call site. Default the provider stack to `llama-server`.

**Tech Stack:** Python 3.12, SQLite (FTS5 + `sqlite-vec`), pydantic v2, FastAPI, `openai` SDK against llama-server, pytest.

## Global Constraints

- Python `>=3.12`; line-length 100.
- **Lint gates, stated honestly.** `ruff check .` does **not** currently pass: there are **341
  pre-existing errors** (137 inside `morgan_brain/`) as of commit `1ad5642`. Cause:
  `pyproject.toml:28` pins `ruff>=0.7`, an unpinned floor, while the installed ruff is 0.16.1
  with a much larger default rule set — which also means the CI lint job is failing on `main`.
  Fixing that baseline is **out of scope for this plan** (it would bury every task's diff).
  The gates for each task are therefore:
  - `python -m ruff check . 2>&1 | tail -1` must report a count **≤ the count at the task's base
    commit**. Never introduce a new error; deleting files may legitimately lower it.
  - `ruff format --check .` must pass — it currently does (265 files clean). This is a hard gate.
  - `mypy morgan_brain` must report **0 errors** from Task 3 onward. It reports exactly 1 today
    (`channels/telegram.py:57`), in a file Task 3 deletes. Hard gate.
  - `pytest -q` must be green, with any change in counts explained. Baseline at `1ad5642`:
    **872 passed, 11 skipped, 1 xfailed**.
- Pinning ruff and clearing the 341-error baseline is real work, but it belongs in its own
  change, not smuggled into a reshape task. Recorded as a follow-up.
- All settings are `MORGAN_`-prefixed and read only via `get_settings()`. Never re-read env directly.
- Everything that persists is `user_id`-keyed (`UserScoped`) **and, after Task 12, `project`-keyed**.
- All memory access goes through `MemoryGate` — after Task 13 this includes consolidation, history, and signals.
- Facts evolve, never overwrite: update closes the old interval and opens a new one.
- Contract-first: change the `Protocol` in `morgan_brain/interfaces/` (or `stores/vector.py` for `VectorIndex`) **before** changing an implementation.
- No provider SDK is imported above `providers/adapters/`.
- Commit messages are plain descriptions. No co-author trailers, no generated-by footers.
- Work happens on branch `reshape/local-first-foundation`. Never commit to `main`.

---

# Phase 0 — Make the repository honest and installable

## Task 1: End the CRLF churn and tag the pre-cut state

**Files:**
- Create: `.gitattributes`
- Modify: every tracked text file (normalization only)

**Interfaces:**
- Consumes: nothing
- Produces: a clean `git status` on checkout, so later diffs are readable

- [ ] **Step 1: Tag the current state before anything is deleted**

```bash
git tag -a legacy-v0.0.4-full -m "Full platform before the local-first reshape cut"
git rev-parse legacy-v0.0.4-full
```

- [ ] **Step 2: Create `.gitattributes`**

```gitattributes
* text=auto eol=lf
*.png binary
*.jpg binary
*.ico binary
*.db binary
```

- [ ] **Step 3: Renormalize the working tree**

```bash
git add --renormalize .
git status --porcelain | wc -l
```
Expected: a large number of staged modifications, all line-ending only.

- [ ] **Step 4: Verify nothing but line endings changed**

```bash
git diff --cached --stat | tail -1
git diff --cached --ignore-all-space --numstat | awk '$1!=0 || $2!=0' | head
```
Expected: the second command prints **nothing** — every change is whitespace.

- [ ] **Step 5: Commit**

```bash
git add .gitattributes
git commit -m "chore: normalize line endings to LF and pin them via .gitattributes"
```

---

## Task 2: Make a non-editable install actually start

The built wheel contains neither `morgan_brain/eval/data/` nor `morgan_brain/providers/data/`, so `CapabilityRegistry.from_packaged()` (`composition.py:189`) fails on any real install. Cause: the root `.gitignore` `data/` pattern; hatchling builds from `morgan-brain/`, where the negations do not apply.

**Files:**
- Create: `morgan-brain/.gitignore`
- Modify: `morgan-brain/pyproject.toml`
- Test: `morgan-brain/tests/integration/test_wheel_install.py`

**Interfaces:**
- Consumes: nothing
- Produces: a wheel whose packaged JSON data resolves at runtime

- [ ] **Step 1: Write the failing test**

```python
"""A built wheel must carry its packaged data files."""
import subprocess
import sys
import zipfile
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]


def test_wheel_contains_packaged_data(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp_path), str(PKG_ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = next(tmp_path.glob("morgan_brain-*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())

    assert "morgan_brain/eval/data/golden_set.json" in names
    assert any(n.startswith("morgan_brain/providers/data/") for n in names), sorted(
        n for n in names if "providers" in n
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd morgan-brain && pytest tests/integration/test_wheel_install.py -v`
Expected: FAIL — `assert 'morgan_brain/eval/data/golden_set.json' in names`

- [ ] **Step 3: Add a package-local `.gitignore` that does not exclude packaged data**

Create `morgan-brain/.gitignore`:

```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
build/
dist/
# Runtime data only — packaged data under morgan_brain/**/data/ must ship.
/data/
```

- [ ] **Step 4: Force-include packaged data in the wheel**

In `morgan-brain/pyproject.toml`, under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["morgan_brain"]
artifacts = [
    "morgan_brain/eval/data/*.json",
    "morgan_brain/providers/data/*.json",
]
```

Note: `clients` is intentionally dropped from `packages` here — Task 3 deletes it.

- [ ] **Step 5: Run the test again**

Run: `cd morgan-brain && pytest tests/integration/test_wheel_install.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add morgan-brain/.gitignore morgan-brain/pyproject.toml morgan-brain/tests/integration/test_wheel_install.py
git commit -m "build: ship packaged eval and provider data in the wheel"
```

---

## Task 3: Delete the subsystems with no production importers

**Files:**
- Delete: `morgan_brain/channels/`, `morgan_brain/voice/`, `morgan_brain/interfaces/voice.py`, `morgan_brain/apps/perception_gpu/`, `morgan_brain/modules/mcp/`, `morgan_brain/providers/resilience.py`, `morgan_brain/interfaces/rerank.py`, `morgan_brain/interfaces/embedding.py`, `clients/`
- Delete: the matching test directories/files
- Modify: `morgan_brain/providers/factory.py` (remove `build_embedder`), `morgan-brain/pyproject.toml` (drop `channels`, `perception`, `voice`, `mcp` extras and the `morgan` script entry)

**Interfaces:**
- Consumes: nothing
- Produces: a smaller package that still imports and passes its remaining tests

- [ ] **Step 1: Prove each target has no production importer**

Count importers that live **outside every module being deleted in this task**. A naive
per-module grep produces false positives, because the deleted modules legitimately import each
other (`voice/persona_bridge.py` imports `interfaces/voice.py`, and both are going).

```bash
cd morgan-brain
DOOMED='^morgan_brain/(channels|voice|apps/perception_gpu|modules/mcp)/|^morgan_brain/interfaces/(voice|rerank|embedding)\.py|^morgan_brain/providers/resilience\.py'
for m in channels voice interfaces.voice modules.mcp providers.resilience interfaces.rerank interfaces.embedding; do
  printf '%-28s ' "$m"
  grep -rn "morgan_brain\.${m}" morgan_brain --include='*.py' | grep -Ev "$DOOMED" | wc -l
done
```

Expected: `0` for every module except `interfaces.embedding`, which must report exactly `1` —
`providers/factory.py:14`, removed in Step 3 of this task.

**If any other line is non-zero, stop and report it** — the plan's premise is wrong for that
module. (Verified by the controller on 2026-08-02: this is the expected output.)

- [ ] **Step 2: Delete the modules and their tests**

```bash
cd morgan-brain
git rm -r --quiet morgan_brain/channels morgan_brain/voice morgan_brain/apps/perception_gpu \
  morgan_brain/modules/mcp morgan_brain/interfaces/voice.py morgan_brain/interfaces/rerank.py \
  morgan_brain/interfaces/embedding.py morgan_brain/providers/resilience.py
git rm -r --quiet tests/unit/voice tests/unit/channels 2>/dev/null || true
cd .. && git rm -r --quiet clients
```

Then remove the remaining test files for deleted modules. **Use this explicit list — do NOT use
a grep-and-`xargs git rm`.** A dry run of the obvious grep also matches
`tests/unit/providers/test_factory.py`, `test_router.py`, `test_fake_adapter.py`,
`test_openai_compat.py`, `test_interfaces.py`, and `tests/unit/test_learning_worker.py` — all of
which test **kept** code. Deleting them would leave the suite green because the tests were gone.

```bash
cd morgan-brain
git rm -r --quiet tests/unit/modules/mcp
git rm --quiet tests/unit/voice/test_fake.py
```

(`tests/unit/voice/test_interfaces.py`, `tests/unit/voice/test_persona_bridge.py`, and
`tests/unit/channels/` are already removed by Step 2's directory deletions.)

Then **edit, do not delete**, the files that merely reference a deleted module:

```bash
cd morgan-brain
grep -rln 'resilience\|interfaces\.rerank\|interfaces\.embedding' tests
```

For each hit, remove only the import and the specific test functions that exercise the deleted
module; leave every test of kept behaviour intact. `tests/unit/providers/test_factory.py` is the
expected hit — it tests `build_embedder`, which Step 3 removes.

- [ ] **Step 3: Remove `build_embedder` from the factory**

In `morgan_brain/providers/factory.py`, delete the `build_embedder` function (lines ~77-88) and the now-unused `Embedder` import (line ~14). Verify it had no callers:

```bash
grep -rn 'build_embedder' morgan_brain tests
```
Expected: no output after the edit.

- [ ] **Step 4: Drop the dead extras and the CLI entry point**

In `morgan-brain/pyproject.toml`, delete the `channels`, `perception`, `voice`, and `mcp` entries from `[project.optional-dependencies]`, and delete the `[project.scripts]` block entirely — Task 18 re-adds it pointing at the new CLI.

- [ ] **Step 5: Verify the package still imports and the suite is green**

```bash
cd morgan-brain
python -c "import morgan_brain, morgan_brain.composition; print('ok')"
pytest -q
ruff check . && mypy morgan_brain
```
Expected: import ok; suite green; `mypy` now reports **0 errors** (the only prior error was `channels/telegram.py:57`, now deleted).

- [ ] **Step 6: Commit**

```bash
git commit -am "refactor: delete channels, voice, perception-gpu, mcp host, resilience and dead interfaces"
```

---

## Task 4: Excise proactivity and the heartbeat, including their live wiring

Unlike Task 3's targets, proactivity **is** wired in production behind a flag (`apps/learning_worker/__main__.py:47,156-173,274-277`), and `scheduling/__init__.py:14,19` imports `HeartbeatManager` — so deleting `heartbeat.py` breaks the **kept** `scheduling` package.

**Files:**
- Delete: `morgan_brain/proactivity/`, `morgan_brain/scheduling/heartbeat.py`, `morgan_brain/modules/proactivity/`, their tests
- Modify: `morgan_brain/scheduling/__init__.py`, `morgan_brain/apps/learning_worker/__main__.py`, `morgan_brain/config.py`

**Interfaces:**
- Consumes: nothing
- Produces: a `scheduling` package that still exports `CronService` and `InProcessScheduler`

- [ ] **Step 1: Delete the modules and their tests**

**Use this explicit list — do NOT use a grep-and-`xargs git rm`.** A dry run of the obvious grep
also matches `tests/unit/scheduling/test_cron.py` (tests the **kept** `CronService`),
`tests/live/test_redis_bus_live.py` (kept bus), `tests/unit/test_learning_worker.py`, and
`tests/unit/test_learning_worker_smoke.py`. Deleting those would hide real regressions.

```bash
cd morgan-brain
git rm -r --quiet morgan_brain/proactivity morgan_brain/modules/proactivity morgan_brain/scheduling/heartbeat.py
git rm -r --quiet tests/unit/proactivity
git rm --quiet tests/unit/scheduling/test_heartbeat.py
```

Then **edit, do not delete**, the four survivors that reference proactivity or the heartbeat:
`tests/unit/test_learning_worker.py`, `tests/unit/test_learning_worker_smoke.py`,
`tests/unit/scheduling/test_cron.py`, `tests/live/test_redis_bus_live.py`. Remove only the
proactivity/heartbeat imports and the test functions that exercise them.

- [ ] **Step 2: Repair `scheduling/__init__.py`**

Remove the `from morgan_brain.scheduling.heartbeat import HeartbeatManager` import (line 14) and the `"HeartbeatManager"` entry from `__all__` (line 19). Update the module docstring's first line to name only what remains:

```python
"""Scheduling package — CronService and InProcessScheduler."""
```

- [ ] **Step 3: Excise the worker wiring**

In `morgan_brain/apps/learning_worker/__main__.py`, delete: the `ProactivityEngine` import (line ~47), the `_build_proactivity_engine` function and its helpers (~152-208), the `HEARTBEAT` subscription block (~274-277), and the proactivity paragraph in the module docstring (~14-16).

- [ ] **Step 4: Remove the dead flag**

In `morgan_brain/config.py`, delete `enable_proactivity` (line 48). Then confirm nothing reads it:

```bash
grep -rn 'enable_proactivity' morgan_brain tests ../docs ../README.md ../CLAUDE.md
```
Expected: only docs hits, which Task 6 fixes.

- [ ] **Step 5: Verify**

```bash
cd morgan-brain
python -c "from morgan_brain.scheduling import CronService, InProcessScheduler; print('ok')"
python -m morgan_brain.apps.learning_worker --help >/dev/null 2>&1 || python -c "import morgan_brain.apps.learning_worker.__main__; print('worker imports')"
pytest -q && ruff check . && mypy morgan_brain
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git commit -am "refactor: remove proactivity engine, heartbeat manager and their worker wiring"
```

---

## Task 5: Replace the privacy layer with an operational control

Field-level encryption is incompatible with the FTS5 keyword index built in Task 9 — you cannot full-text index ciphertext — and would not have covered vectors. At-rest protection moves to volume-level encryption on the homelab host.

**Files:**
- Delete: `morgan_brain/privacy/`, its tests
- Modify: `morgan_brain/config.py`, `morgan_brain/composition.py`, `morgan-brain/pyproject.toml`
- Create: `docs/OPERATIONS.md`

**Interfaces:**
- Consumes: nothing
- Produces: no privacy imports anywhere; an operations doc stating the real control

- [ ] **Step 1: Confirm the privacy layer has no live call sites**

```bash
cd morgan-brain
grep -rn 'morgan_brain\.privacy' morgan_brain --include='*.py' | grep -v '^morgan_brain/privacy/'
```
Expected: no output. If there are hits, list them and stop.

- [ ] **Step 2: Delete it**

```bash
cd morgan-brain
git rm -r --quiet morgan_brain/privacy
grep -rln 'privacy\|redaction\|Presidio\|presidio' tests | xargs -r git rm --quiet
```

- [ ] **Step 3: Remove the dead settings and the extra**

In `morgan_brain/config.py`, delete `redact_egress`, `encryption`, and `passphrase` (lines ~89-100). In `pyproject.toml`, delete the `privacy` optional-dependency block.

- [ ] **Step 4: Write the operations doc**

Create `docs/OPERATIONS.md`:

```markdown
# Operations

## At-rest protection

Morgan stores everything — episodics, facts, session history, training signals, and vectors —
in one SQLite database under `MORGAN_DATA_DIR`. There is no field-level encryption: it cannot
coexist with the FTS5 keyword index, and it would not cover vectors.

At-rest protection is therefore a property of the host. The homelab volume backing
`MORGAN_DATA_DIR` must be encrypted (LUKS or the equivalent for your storage layer). This
covers the entire database, including vectors and signal text.

## Transport protection

The homelab instance is reachable from three laptops. All `/api/*` routes require
`Authorization: Bearer $MORGAN_API_KEY`. Reach the service over the NetBird overlay network, or
terminate TLS at a reverse proxy. Never expose it on a public interface with the default key.

## Backups

Back up the single database file with `sqlite3 morgan.db ".backup 'morgan-backup.db'"` while the
service runs — a filesystem copy of a WAL-mode database mid-write is not consistent.
```

- [ ] **Step 5: Verify**

```bash
cd morgan-brain && pytest -q && ruff check . && mypy morgan_brain
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add docs/OPERATIONS.md
git commit -am "refactor: replace the privacy layer with host-level at-rest encryption"
```

---

## Task 6: Docs truth pass

**Files:**
- Modify: `CLAUDE.md`, `README.md`, **`morgan-brain/README.md`**, `docs/ROADMAP.md`, `docs/WIRING.md`, `morgan-brain/.env.example`, `docs/OPERATIONS.md`
- Delete: `docs/ARCHITECTURE_V2.md`

**Carried from the Task 5 review:** `morgan-brain/README.md` still documents the deleted privacy
layer in four places — line 3 ("privacy-first"), line 38 (a `privacy/` package-map row), line 97
(`pip install -e ".[privacy]"`, an extra that no longer exists), and line 121 (a kernel-layer
list including `privacy/`). Line 97 is the worst of them: it tells a reader to install an extra
that was removed, which fails outright.

**Interfaces:**
- Consumes: the deletions from Tasks 3-5
- Produces: documentation that matches the code

- [ ] **Step 1: Delete the contradictory architecture doc**

```bash
git rm --quiet docs/ARCHITECTURE_V2.md
grep -rn 'ARCHITECTURE_V2' docs README.md CLAUDE.md
```
Fix every link the grep finds to point at the reshape design spec instead.

- [ ] **Step 2: Correct the false claims in `CLAUDE.md`**

Apply each of these:
- Replace "820 tests, mypy-strict clean" with the number printed by `pytest -q` after Task 5, and state `mypy` is clean only once it is.
- Delete the two "latent bugs" bullets — both are fixed (`orchestrator.py:236`, `composition.py:334-339`).
- Delete every reference to channels, voice, perception-gpu, proactivity, MCP host, and the privacy layer from the package map, the service table, the extras list, and the invariants.
- Remove the "provider SDKs isolated" claim's implication that composition is provider-neutral until Task 16 makes it true.
- Replace the "Current direction (H1)" section with a pointer to `docs/superpowers/specs/2026-08-02-morgan-reshape-local-first-design.md`.
- Fix the archive reference: the tag is `legacy-v0.0.3-monolith`; the branch is `origin/legacy/v0.0.3-monolith`.

- [ ] **Step 0: Append the stack rationale to `docs/OPERATIONS.md`**

Morgan runs entirely on open-source infrastructure, and the reasoning behind each choice is
useful to anyone deploying it — including the owner six months from now. Append this section
verbatim to the end of `docs/OPERATIONS.md`. Keep it factual; it is a rationale, not an
advertisement. Licenses below were verified on 2026-08-02.

```markdown
## The stack, and why

Morgan depends on four pieces of infrastructure. All are open source and self-hostable, which
is the point: a personal brain that only works while someone else's service is up is not a
personal brain. If you are choosing your own stack, these are worth knowing about.

### NetBird — the network

[netbird.io](https://netbird.io) · [github.com/netbirdio/netbird](https://github.com/netbirdio/netbird)
· BSD-3-Clause (the `management/`, `signal/` and `relay/` components are AGPLv3)

A WireGuard-based overlay network that connects the laptops to the homelab without opening a
single public port. Devices find each other peer-to-peer through a signal server, with a
TURN relay only as a fallback when NAT traversal fails.

**Why this and not a hosted mesh VPN:** the coordination server is software you can run
yourself. With most alternatives the client is open source but the control plane is not, so
the thing that decides which of your devices may reach your brain is somebody else's service.
Morgan's whole premise is that the owner holds their own data and policy, and a proprietary
control plane in the middle of that is a contradiction. NetBird also carries SSO, MFA and
granular ACLs, so "which device may reach the brain" stays an explicit, auditable decision.

### llama.cpp — the inference

[github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) · MIT

One binary, `llama-server`, serves chat completions, `/v1/embeddings`, and `/v1/rerank` over an
OpenAI-compatible API, with GBNF grammars for genuinely constrained structured output.

**Why this and not an inference wrapper:** Morgan needs four model roles (fast, strong, judge,
reflection) plus embeddings and reranking. llama-server covers all of it from one process on
one GPU, and GBNF means structured output is enforced by the decoder rather than requested in
a prompt and hoped for. It is also the layer everything else in the local ecosystem is built
on, so depending on it directly removes a translation layer rather than adding one.

### SQLite + FTS5 — the store

[sqlite.org](https://sqlite.org) · public domain

Every durable thing Morgan owns — episodics, facts, entities, session history, training
signals, and the keyword index — lives in one SQLite file.

**Why one file:** erasure. `forget()` has to remove a memory from six places at once, and a
single database makes that one transaction instead of a distributed delete that can half-fail.
It also means at-rest encryption is one encrypted volume rather than a per-store problem, and
backup is one `.backup` command. FTS5's `unicode61` tokenizer additionally indexes non-Latin
scripts correctly, which the hand-rolled BM25 index it replaced did not.

### sqlite-vec — the vectors

[github.com/asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) ·
[alexgarcia.xyz/sqlite-vec](https://alexgarcia.xyz/sqlite-vec/) · MIT or Apache-2.0

Vector search as a SQLite extension, written in pure C with no dependencies, by Alex Garcia.
A Mozilla Builders project.

**Why this and not a vector database:** it puts the vectors in the same file as everything
else, which is what makes the erasure and backup arguments above hold for vectors too. A
separate vector service would reintroduce exactly the cross-engine transaction problem that
choosing one store was meant to avoid. At single-user corpus sizes its brute-force search is
comfortably fast, and its `vec0` metadata columns filter *inside* the nearest-neighbour search
— which matters, because filtering after the fact silently drops results that should have
been returned.
```

- [ ] **Step 2a: Fix package docstrings that still promise deleted services**

Carried from the Task 3 review. Two docstrings still name a service whose directory is gone:
- `morgan_brain/apps/__init__.py` — lists "perception_gpu — voice/vision (deferred; interface
  only)" as one of three entrypoints. There are now two.
- `morgan_brain/__init__.py:12` — "apps — brain_api, learning_worker, perception_gpu entrypoints."

Then sweep for any others:

```bash
cd morgan-brain
grep -rn 'perception_gpu\|perception-gpu\|PersonaPlex\|Telegram\|Discord\|MCP host' morgan_brain --include='*.py'
```
Expected after the edits: no output.

- [ ] **Step 3a: Delete the dead config fields themselves**

Tasks 3-5 removed the code that read these, but the fields survive in `config.py` and Step 3's
verification loop below only scans `.env.example` — so it is blind to them. Delete from
`morgan_brain/config.py`: `enable_channels`, `enable_mcp`, `telegram_token`, `discord_token`,
and `mcp_servers`. Confirm each has no reader first:

```bash
cd morgan-brain
for v in enable_channels enable_mcp telegram_token discord_token mcp_servers; do
  printf '%-18s %s\n' "$v" "$(grep -rn "settings\.$v\b" morgan_brain --include='*.py' | wc -l)"
done
```
Expected: `0` for every one.

- [ ] **Step 3: Strip dead settings from `.env.example` and `docs/WIRING.md`**

Remove `MORGAN_REDACT_EGRESS`, `MORGAN_ENCRYPTION`, `MORGAN_PASSPHRASE`, `MORGAN_ENABLE_CHANNELS`, `MORGAN_ENABLE_MCP`, `MORGAN_ENABLE_PROACTIVITY`, `MORGAN_TELEGRAM_TOKEN`, `MORGAN_DISCORD_TOKEN`, and `MORGAN_MLFLOW_TRACKING_URI` if `learning_backend` stays `local`.

Verify none of the remaining documented variables is dead:

```bash
cd morgan-brain
for v in $(grep -o '^MORGAN_[A-Z_]*' .env.example | sed 's/MORGAN_//' | tr 'A-Z' 'a-z'); do
  n=$(grep -rn "settings\.$v\b" morgan_brain --include='*.py' | wc -l)
  [ "$n" -eq 0 ] && echo "DEAD: $v"
done
```
Expected: no `DEAD:` lines except settings introduced by later tasks.

- [ ] **Step 4: Correct the "bi-temporal" claim**

In `docs/ROADMAP.md`, `CLAUDE.md`, and `modules/memory/stores/temporal.py`'s docstring, replace "bi-temporal" with "valid-time" — the schema has `valid_from`/`valid_to`/`superseded_by`/`last_confirmed` and no ingestion-time column.

- [ ] **Step 5: Mark the superseded specs**

Add to the top of each of `2026-06-09-personal-agent-os-vision.md`, `-horizons-roadmap.md`, `-ports-design.md`, and `-deployment-profiles-and-sync-design.md`:

```markdown
> **Superseded** by [the local-first reshape](2026-08-02-morgan-reshape-local-first-design.md)
> (2026-08-02). Device sync, the memory replica, the phone client, deployment profiles, and the
> `/v1` facade are out of scope.
```

- [ ] **Step 6: Commit**

```bash
git commit -am "docs: correct claims to match the code after the cut"
```

---

# Phase 1 — One durable, project-scoped store

## Task 7: One SQLite connection, WAL mode, extension loading

**Files:**
- Create: `morgan_brain/modules/memory/stores/db.py`
- Test: `morgan-brain/tests/unit/memory/test_db.py`

**Interfaces:**
- Consumes: nothing
- Produces: `open_db(path: str) -> sqlite3.Connection` — WAL-enabled, busy-timeout set, `sqlite-vec` loaded. Every store in Phase 1 takes this connection.

- [ ] **Step 1: Add the dependency**

In `morgan-brain/pyproject.toml`, add `"sqlite-vec>=0.1.9"` to `[project].dependencies`, and move `redis` and `qdrant-client` out of `dependencies` into a new extra.

The `>=0.1.9` floor is deliberate: Task 8 depends on vec0 **metadata columns** and
`distance_metric=cosine`, both verified working on 0.1.9. Earlier releases were not verified to
support them, and silently losing metadata filtering would reintroduce the crowd-out bug Task 8
exists to prevent.

```toml
scale = ["redis>=5.2", "qdrant-client>=1.12"]
```

- [ ] **Step 2: Write the failing test**

```python
import sqlite3
from morgan_brain.modules.memory.stores.db import open_db


def test_open_db_enables_wal_and_vec(tmp_path):
    conn = open_db(str(tmp_path / "m.db"))
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    # sqlite-vec is loaded when vec_version() resolves
    assert conn.execute("SELECT vec_version()").fetchone()[0]


def test_open_db_is_reopenable(tmp_path):
    path = str(tmp_path / "m.db")
    open_db(path).execute("CREATE TABLE t (a TEXT)")
    conn2 = open_db(path)
    assert conn2.execute("SELECT count(*) FROM t").fetchone()[0] == 0


def test_memory_path_is_supported_for_tests():
    conn = open_db(":memory:")
    assert conn.execute("SELECT vec_version()").fetchone()[0]
```

- [ ] **Step 3: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: morgan_brain.modules.memory.stores.db`

- [ ] **Step 4: Implement**

```python
"""The one SQLite connection factory.

Every store in the memory subsystem shares a single database file so that erasure is one
transaction and at-rest encryption is one volume. WAL mode lets the API process and an
optional worker process read concurrently; the busy timeout absorbs writer contention.
"""

from __future__ import annotations

import sqlite3

import sqlite_vec

_BUSY_TIMEOUT_MS = 5000


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the Morgan database with WAL, a busy timeout, and sqlite-vec loaded."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    # ":memory:" has no journal to switch; WAL is meaningless and PRAGMA returns "memory".
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    return conn
```

Note the `:memory:` branch: the first test asserts WAL only for file paths. Adjust
`test_open_db_enables_wal_and_vec` to use a file path (it already does) and leave
`test_memory_path_is_supported_for_tests` asserting only that `vec_version()` resolves.

- [ ] **Step 5: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/test_db.py -v`
Expected: PASS (3 tests)

If `sqlite_vec.load` raises `AttributeError: enable_load_extension`, the interpreter was built without extension support — stop and report it; that is an environment blocker, not a code bug.

- [ ] **Step 6: Commit**

```bash
git add morgan-brain/pyproject.toml morgan_brain/modules/memory/stores/db.py tests/unit/memory/test_db.py
git commit -m "feat(memory): single SQLite connection factory with WAL and sqlite-vec"
```

---

## Task 8: Persistent vector index on sqlite-vec

**Files:**
- Create: `morgan_brain/modules/memory/stores/sqlite_vector.py`
- Modify: `morgan_brain/modules/memory/stores/vector.py:31-34` (extend the `VectorIndex` Protocol)
- Test: `morgan-brain/tests/unit/memory/test_sqlite_vector.py`

**Interfaces:**
- Consumes: `open_db` from Task 7
- Produces: `SqliteVectorIndex(conn, dim)` satisfying `VectorIndex`, plus `VectorIndex.delete(ids: list[str]) -> None` added to the Protocol. Existing `InMemoryVectorIndex` and `QdrantVectorIndex` must gain `delete` too.

- [ ] **Step 1: Extend the Protocol first (contract-first)**

In `morgan_brain/modules/memory/stores/vector.py`:

```python
@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(self, record: VectorRecord) -> None: ...
    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]: ...
    async def delete(self, ids: list[str]) -> None: ...
```

- [ ] **Step 2: Write the failing test**

```python
import pytest
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.vector import VectorRecord


def _idx(tmp_path, dim=4):
    return SqliteVectorIndex(open_db(str(tmp_path / "m.db")), dim=dim)


async def test_upsert_then_search_returns_the_record(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0], payload={"content": "x"}))
    hits = await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]
    assert hits[0].payload["content"] == "x"


async def test_search_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="b", user_id="u2", vector=[1, 0, 0, 0]))
    hits = await idx.search(user_id="u1", vector=[1, 0, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    idx = SqliteVectorIndex(open_db(path), dim=4)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[0, 1, 0, 0]))
    reopened = SqliteVectorIndex(open_db(path), dim=4)
    hits = await reopened.search(user_id="u", vector=[0, 1, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_delete_removes_the_vector(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0]))
    await idx.delete(["a"])
    assert await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=5) == []


async def test_upsert_replaces_rather_than_duplicates(tmp_path):
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="a", user_id="u", vector=[0, 1, 0, 0]))
    hits = await idx.search(user_id="u", vector=[0, 1, 0, 0], top_k=5)
    assert [h.id for h in hits] == ["a"]


async def test_scoping_happens_inside_the_knn_not_after(tmp_path):
    """Regression: post-filtering a global KNN silently drops the caller's own neighbours.

    Two users share the store. u1 owns the exact match AND the second-nearest vector, but
    u2's identical vector would crowd the top-k of an unscoped query. With top_k=2, a
    correct implementation returns BOTH of u1's vectors.
    """
    idx = _idx(tmp_path)
    await idx.upsert(VectorRecord(id="u1-exact", user_id="u1", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="u2-exact", user_id="u2", vector=[1, 0, 0, 0]))
    await idx.upsert(VectorRecord(id="u1-near", user_id="u1", vector=[0.9, 0.1, 0, 0]))
    await idx.upsert(VectorRecord(id="u2-near", user_id="u2", vector=[0.9, 0.1, 0, 0]))

    hits = await idx.search(user_id="u1", vector=[1, 0, 0, 0], top_k=2)
    assert [h.id for h in hits] == ["u1-exact", "u1-near"]
```

- [ ] **Step 3: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_sqlite_vector.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 4: Implement**

```python
"""Persistent vector index backed by sqlite-vec, inside the one Morgan database.

Scoping happens **inside** the KNN via vec0 metadata columns, not by over-fetching and
filtering afterwards. This is not a style choice — post-filtering is incorrect. vec0 selects
its ``k`` nearest neighbours globally, before any join or WHERE on a joined table, so with
several users or projects in one store the caller's own near neighbours can be crowded out
and never returned at all. Verified on sqlite-vec 0.1.9: with two users interleaved and
``k=2``, an unscoped query returned only the *other* user's rows, while the metadata-scoped
query returned the correct two.

vec0 also defaults to L2. The index this replaces ranked by cosine (``_cosine`` in vector.py,
and Qdrant's ``Distance.COSINE``), so ``distance_metric=cosine`` is set explicitly — otherwise
ranking silently changes for unnormalised llama-server embeddings.
"""

from __future__ import annotations

import json
import sqlite3
import struct

from morgan_brain.modules.memory.stores.vector import VectorHit, VectorRecord


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class SqliteVectorIndex:
    def __init__(self, conn: sqlite3.Connection, *, dim: int) -> None:
        self._conn = conn
        self._dim = dim
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS vec_meta (
                rowid   INTEGER PRIMARY KEY,
                id      TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vec_meta_id ON vec_meta (id);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                embedding float[{dim}] distance_metric=cosine,
                user_id TEXT
            );
            """
        )
        conn.commit()

    async def upsert(self, record: VectorRecord) -> None:
        if len(record.vector) != self._dim:
            raise ValueError(
                f"embedding dimension {len(record.vector)} does not match store dimension "
                f"{self._dim}"
            )
        cur = self._conn.execute("SELECT rowid FROM vec_meta WHERE id = ?", (record.id,))
        row = cur.fetchone()
        if row is not None:
            rowid = row["rowid"]
            self._conn.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "UPDATE vec_meta SET user_id = ?, payload = ? WHERE rowid = ?",
                (record.user_id, json.dumps(record.payload), rowid),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO vec_meta (id, user_id, payload) VALUES (?, ?, ?)",
                (record.id, record.user_id, json.dumps(record.payload)),
            )
            rowid = int(cur.lastrowid or 0)
        self._conn.execute(
            "INSERT INTO vec_items (rowid, embedding, user_id) VALUES (?, ?, ?)",
            (rowid, _pack(record.vector), record.user_id),
        )
        self._conn.commit()

    async def search(
        self, *, user_id: str, vector: list[float], top_k: int
    ) -> list[VectorHit]:
        # user_id is a vec0 metadata column, so the filter applies INSIDE the KNN.
        rows = self._conn.execute(
            """
            SELECT m.id AS id, m.payload AS payload, v.distance AS distance
            FROM vec_items v
            JOIN vec_meta m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ? AND v.user_id = ?
            ORDER BY v.distance
            """,
            (_pack(vector), top_k, user_id),
        ).fetchall()
        return [
            VectorHit(id=r["id"], score=-float(r["distance"]), payload=json.loads(r["payload"]))
            for r in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        """Protocol-level delete, for callers outside the single-database path.

        ``forget()`` deletes these rows with plain SQL inside its own transaction instead —
        see Task 16 — because committing here would break its atomicity.
        """
        for mid in ids:
            row = self._conn.execute("SELECT rowid FROM vec_meta WHERE id = ?", (mid,)).fetchone()
            if row is None:
                continue
            self._conn.execute("DELETE FROM vec_items WHERE rowid = ?", (row["rowid"],))
            self._conn.execute("DELETE FROM vec_meta WHERE rowid = ?", (row["rowid"],))
        self._conn.commit()
```

- [ ] **Step 5: Add `delete` to the other two implementations**

In `morgan_brain/modules/memory/stores/vector.py`, `InMemoryVectorIndex`:

```python
    async def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._records.pop(mid, None)
```

In `QdrantVectorIndex` (same file), delete by the same UUID5-derived point ids that `upsert` uses.

- [ ] **Step 6: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/ -v && mypy morgan_brain`
Expected: PASS, mypy clean.

- [ ] **Step 7: Commit**

```bash
git add -A morgan_brain/modules/memory/stores tests/unit/memory
git commit -m "feat(memory): persistent sqlite-vec vector index with delete support"
```

---

## Task 9: FTS5 keyword index that can read Russian

`retrieval/bm25.py:9` tokenises on `[a-z0-9]+`, so Cyrillic is dropped entirely and keyword recall returns nothing for Russian text. FTS5 with `unicode61` fixes that.

**Files:**
- Create: `morgan_brain/modules/memory/retrieval/fts.py`
- Test: `morgan-brain/tests/unit/memory/test_fts.py`

**Interfaces:**
- Consumes: `open_db` from Task 7
- Produces: `FtsIndex(conn)` with `add(memory_id, content, user_id)`, `search(text, *, user_id, top_k) -> list[str]` (ranked ids), `delete(ids)`

- [ ] **Step 1: Write the failing test**

```python
from morgan_brain.modules.memory.retrieval.fts import FtsIndex, to_match_query
from morgan_brain.modules.memory.stores.db import open_db


def _idx(tmp_path):
    return FtsIndex(open_db(str(tmp_path / "m.db")))


def test_finds_english_term(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "the harbor mirror was misconfigured", user_id="u")
    assert idx.search("harbor", user_id="u", top_k=5) == ["a"]


def test_finds_cyrillic_term(tmp_path):
    """The old [a-z0-9]+ tokenizer dropped Cyrillic entirely."""
    idx = _idx(tmp_path)
    idx.add("a", "реестр Harbor был настроен неверно", user_id="u")
    assert idx.search("реестр", user_id="u", top_k=5) == ["a"]


def test_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u1")
    idx.add("b", "harbor", user_id="u2")
    assert idx.search("harbor", user_id="u1", top_k=5) == ["a"]


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    FtsIndex(open_db(path)).add("a", "harbor mirror", user_id="u")
    assert FtsIndex(open_db(path)).search("harbor", user_id="u", top_k=5) == ["a"]


def test_delete_removes_the_row(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u")
    idx.delete(["a"])
    assert idx.search("harbor", user_id="u", top_k=5) == []


def test_raw_punctuation_does_not_raise():
    """Raw user text is not a valid MATCH expression; it must be tokenised and quoted."""
    assert to_match_query('what about ACME-14802 "quoted" AND?') != ""


def test_query_with_no_indexable_tokens_is_empty(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", "harbor", user_id="u")
    assert idx.search("!!! ???", user_id="u", top_k=5) == []
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_fts.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement**

```python
"""FTS5 keyword index — the persistent replacement for the in-process BM25 index.

Two traps this module exists to handle:

* Raw user text is **not** a valid FTS5 ``MATCH`` expression. Hyphens, quotes and bare
  ``AND``/``OR`` produce syntax errors that surface as silent recall failures, so every
  token is extracted and quoted.
* The previous tokenizer was ``[a-z0-9]+``, which dropped Cyrillic entirely. ``unicode61``
  indexes it, so keyword recall works for the intended corpus.
"""

from __future__ import annotations

import re
import sqlite3

_TOKEN = re.compile(r"\w+", re.UNICODE)


def to_match_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression (OR over quoted tokens)."""
    tokens = _TOKEN.findall(text)
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


class FtsIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                memory_id UNINDEXED,
                user_id   UNINDEXED,
                content,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        conn.commit()

    def add(self, memory_id: str, content: str, *, user_id: str) -> None:
        self._conn.execute(
            "DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.execute(
            "INSERT INTO fts_memories (memory_id, user_id, content) VALUES (?, ?, ?)",
            (memory_id, user_id, content),
        )
        self._conn.commit()

    def search(self, text: str, *, user_id: str, top_k: int) -> list[str]:
        match = to_match_query(text)
        if not match:
            return []
        rows = self._conn.execute(
            """
            SELECT memory_id FROM fts_memories
            WHERE fts_memories MATCH ? AND user_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, user_id, top_k),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (mid,))
        self._conn.commit()
```

- [ ] **Step 4: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/test_fts.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Do NOT delete bm25.py yet**

`modules/memory/store.py:21` still imports `Bm25Index`. Deleting the module here would break
the package import and leave the Tasks 9 and 10 commits red, violating the global constraint
that every commit is green. The deletion happens in Task 11, in the same commit that removes
the import.

Verify the new index coexists with the old one:

```bash
cd morgan-brain && python -c "import morgan_brain.composition; print('ok')" && pytest -q
```
Expected: import ok, suite green.

- [ ] **Step 6: Commit**

```bash
git add morgan_brain/modules/memory/retrieval/fts.py tests/unit/memory/test_fts.py
git commit -m "feat(memory): FTS5 keyword index with unicode tokenization"
```

---

## Task 10: Persistent entity index

**Files:**
- Create: `morgan_brain/modules/memory/retrieval/entities.py`
- Test: `morgan-brain/tests/unit/memory/test_entities.py`

**Interfaces:**
- Consumes: `open_db` from Task 7
- Produces: `EntityIndex(conn)` with `add(memory_id, names, *, user_id)`, `search(terms, *, user_id, top_k) -> list[str]`, `delete(ids)`. Ordering is **deterministic**: by number of matched entities descending, then `memory_id` ascending — the old dict-iteration ordering was undefined.

- [ ] **Step 1: Write the failing test**

```python
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.stores.db import open_db


def _idx(tmp_path):
    return EntityIndex(open_db(str(tmp_path / "m.db")))


def test_matches_on_entity_name(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor"], user_id="u")
    assert idx.search({"harbor"}, user_id="u", top_k=5) == ["a"]


def test_is_user_scoped(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor"], user_id="u1")
    idx.add("b", ["Harbor"], user_id="u2")
    assert idx.search({"harbor"}, user_id="u1", top_k=5) == ["a"]


def test_ordering_is_deterministic_by_match_count(tmp_path):
    idx = _idx(tmp_path)
    idx.add("b", ["Harbor"], user_id="u")
    idx.add("a", ["Harbor", "Qdrant"], user_id="u")
    assert idx.search({"harbor", "qdrant"}, user_id="u", top_k=5) == ["a", "b"]


def test_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    EntityIndex(open_db(path)).add("a", ["Harbor"], user_id="u")
    assert EntityIndex(open_db(path)).search({"harbor"}, user_id="u", top_k=5) == ["a"]


def test_delete_removes_all_rows_for_the_memory(tmp_path):
    idx = _idx(tmp_path)
    idx.add("a", ["Harbor", "Qdrant"], user_id="u")
    idx.delete(["a"])
    assert idx.search({"harbor"}, user_id="u", top_k=5) == []
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_entities.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement**

```python
"""Persistent entity-overlap index — the third recall signal.

Ordering is defined here rather than left to dict iteration: most matched entities first,
then memory id, so fusion input is stable across processes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable


class EntityIndex:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                user_id   TEXT NOT NULL,
                name      TEXT NOT NULL,
                PRIMARY KEY (memory_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_lookup
                ON memory_entities (user_id, name);
            """
        )
        conn.commit()

    def add(self, memory_id: str, names: Iterable[str], *, user_id: str) -> None:
        self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO memory_entities (memory_id, user_id, name) VALUES (?, ?, ?)",
            [(memory_id, user_id, n.lower()) for n in names],
        )
        self._conn.commit()

    def search(self, terms: Iterable[str], *, user_id: str, top_k: int) -> list[str]:
        wanted = [t.lower() for t in terms]
        if not wanted:
            return []
        placeholders = ",".join("?" * len(wanted))
        rows = self._conn.execute(
            f"""
            SELECT memory_id, COUNT(*) AS hits
            FROM memory_entities
            WHERE user_id = ? AND name IN ({placeholders})
            GROUP BY memory_id
            ORDER BY hits DESC, memory_id ASC
            LIMIT ?
            """,
            (user_id, *wanted, top_k),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (mid,))
        self._conn.commit()
```

- [ ] **Step 4: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/test_entities.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/retrieval/entities.py tests/unit/memory/test_entities.py
git commit -m "feat(memory): persistent entity index with deterministic ordering"
```

---

## Task 11: Rewire MemoryModule onto the durable indexes

This is the task that fixes the headline defect: recall currently loses two of three signals on restart.

**Files:**
- Create: `morgan_brain/modules/memory/stores/episodic.py`
- Modify: `morgan_brain/modules/memory/store.py` (whole file)
- Test: `morgan-brain/tests/unit/memory/test_store_durability.py`

**Interfaces:**
- Consumes: `open_db` (Task 7), `SqliteVectorIndex` (8), `FtsIndex` (9), `EntityIndex` (10)
- Produces: `EpisodicStore(conn)` with `put(memory)`, `get(memory_id) -> Memory | None`, `delete(ids)`; `MemoryModule(embedder=…, vectors=…, temporal=…, clock=…, fts=…, entities=…, episodics=…)`

- [ ] **Step 1: Write the failing test**

```python
"""Recall must survive a restart on all three signals."""
import hashlib

import pytest
from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryQuery
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from datetime import datetime, timezone


class HashEmbedder:
    """Deterministic 4-d embedder; identical text embeds identically across processes.

    Uses sha256, NOT the builtin ``hash()`` — PYTHONHASHSEED randomises ``hash()`` per
    process, so a builtin-hash embedder would produce different vectors in the CLI
    subprocess than in the store, silently breaking cross-process vector recall while the
    FTS signal masked the failure.
    """

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().strip().encode("utf-8")).digest()
        return [float(digest[i]) for i in range(4)]


def _module(path: str) -> MemoryModule:
    conn = open_db(path)
    return MemoryModule(
        embedder=HashEmbedder(),
        vectors=SqliteVectorIndex(conn, dim=4),
        temporal=SqliteTemporalStore(path),
        clock=lambda: datetime.now(timezone.utc),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )


async def test_keyword_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(
        Memory(user_id="u", content="the Harbor mirror blocked the deploy")
    )
    got = await _module(path).recall(MemoryQuery(user_id="u", text="Harbor"))
    assert any("Harbor" in m.content for m in got)


async def test_cyrillic_keyword_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(Memory(user_id="u", content="реестр Harbor заблокировал деплой"))
    got = await _module(path).recall(MemoryQuery(user_id="u", text="реестр"))
    assert any("реестр" in m.content for m in got)


async def test_entity_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(
        Memory(user_id="u", content="a note", entities=[Entity(name="Harbor", type="org")])
    )
    got = await _module(path).recall(MemoryQuery(user_id="u", text="harbor"))
    assert any(m.content == "a note" for m in got)


async def test_recall_is_user_scoped_after_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(Memory(user_id="u1", content="secret harbor note"))
    got = await _module(path).recall(MemoryQuery(user_id="u2", text="harbor"))
    assert got == []
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_store_durability.py -v`
Expected: FAIL — `EpisodicStore` does not exist, and `MemoryModule` takes no `fts`/`entities`/`episodics`

- [ ] **Step 3: Implement `EpisodicStore`**

```python
"""Durable episodic records — the rehydration source that in-process dicts used to be."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource


class EpisodicStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                source     TEXT NOT NULL,
                content    TEXT NOT NULL,
                importance REAL NOT NULL,
                entities   TEXT NOT NULL,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
            """
        )
        conn.commit()

    def put(self, memory: Memory) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, user_id, kind, source, content, importance, entities, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.user_id,
                memory.kind.value,
                memory.source.value,
                memory.content,
                memory.importance,
                json.dumps([{"name": e.name, "type": e.type} for e in memory.entities]),
                memory.created_at.isoformat() if memory.created_at else None,
            ),
        )
        self._conn.commit()

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            kind=MemoryKind(row["kind"]),
            source=MemorySource(row["source"]),
            content=row["content"],
            importance=row["importance"],
            entities=[Entity(**e) for e in json.loads(row["entities"])],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    def delete(self, ids: list[str]) -> None:
        for mid in ids:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        self._conn.commit()
```

- [ ] **Step 4: Rewrite `MemoryModule`**

Replace `__init__`, `store`, `recall`, and `_owned` in `morgan_brain/modules/memory/store.py`. Delete the `Bm25Index` import and the `_memory_from_payload` helper — rehydration now comes from `EpisodicStore`, not from vector payloads.

```python
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: VectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
        fts: FtsIndex,
        entities: EntityIndex,
        episodics: EpisodicStore,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._fts = fts
        self._entities = entities
        self._episodics = episodics

    async def store(self, memory: Memory) -> str:
        if memory.created_at is None:
            memory.created_at = self._clock()
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        self._episodics.put(memory)
        await self._vectors.upsert(
            VectorRecord(
                id=memory.id,
                user_id=memory.user_id,
                vector=vector,
                payload={"content": memory.content, "user_id": memory.user_id},
            )
        )
        self._fts.add(memory.id, memory.content, user_id=memory.user_id)
        self._entities.add(
            memory.id, [e.name for e in memory.entities], user_id=memory.user_id
        )
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        q_vector = await self._embedder.embed(query.text)
        vec_hits = await self._vectors.search(
            user_id=query.user_id, vector=q_vector, top_k=query.top_k * 2
        )
        vector_ranking = [h.id for h in vec_hits]
        fts_ranking = self._fts.search(
            query.text, user_id=query.user_id, top_k=query.top_k * 2
        )
        entity_ranking = self._entities.search(
            {t for t in query.text.split()}, user_id=query.user_id, top_k=query.top_k * 2
        )

        fused_ids = reciprocal_rank_fusion([vector_ranking, fts_ranking, entity_ranking])
        episodic = [m for m in (self._episodics.get(mid) for mid in fused_ids) if m is not None]

        facts = await self._temporal.current_facts(user_id=query.user_id)
        fact_memories = [
            Memory(
                user_id=query.user_id,
                kind=MemoryKind.SEMANTIC,
                content=f"{f.subject} {f.predicate} {f.object}".replace("_", " "),
                source=f.source,
            )
            for f in facts
        ]
        return (fact_memories + episodic)[: query.top_k]
```

- [ ] **Step 5: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/ -v`
Expected: PASS (4 new tests plus the earlier ones)

- [ ] **Step 6: Delete BM25 and update every construction site**

The constructor signature changed, so every caller breaks. This is the complete list — it is
**nine files, one of which is production code that appears in no other task**:

Production:
- `morgan_brain/composition.py:182` — the main assembly
- `morgan_brain/eval/runner.py:66` — the eval scratch gate. Build its three indexes over
  `open_db(":memory:")` so eval writes stay isolated.

Tests:
- `tests/memory_quality/conftest.py:18`
- `tests/integration/test_cross_process_recall.py:25`
- `tests/unit/test_memory_module.py:12`
- `tests/unit/learning/test_anti_amnesia.py:39`
- `tests/unit/learning/test_profile.py:40`
- `tests/unit/learning/test_learner.py:41`
- `tests/unit/learning/test_consolidation.py:53`

Extract a shared `_module(path)` helper into `tests/unit/memory/conftest.py` and reuse it —
Task 12 references the same helper, and duplicating it nine times guarantees drift.

Now delete the old index in the same commit that removes its import:

```bash
cd morgan-brain
git rm --quiet morgan_brain/modules/memory/retrieval/bm25.py tests/unit/test_bm25.py
grep -rn 'Bm25Index\|retrieval.bm25' morgan_brain tests
pytest -q && ruff check . && mypy morgan_brain
```
Expected: no grep output; suite green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(memory): durable multi-signal recall that survives a restart"
```

---

## Task 12: Add `project` to the domain model and the stores

**Files:**
- Modify: `morgan_brain/models/memory.py`, `morgan_brain/modules/memory/stores/episodic.py`, `.../stores/temporal.py`, `.../stores/sqlite_vector.py`, `.../retrieval/fts.py`, `.../retrieval/entities.py`, `.../store.py`
- Test: `morgan-brain/tests/unit/memory/test_project_scoping.py`, `morgan-brain/tests/unit/memory/conftest.py`

**Note on `sqlite_vector.py`:** it is in this list deliberately. `project` must become a
**second vec0 metadata column** alongside `user_id`, and `VectorIndex.search` must take a
`project: str | None` argument, for the same reason Task 8 scopes on `user_id` inside the KNN —
filtering afterwards lets another project's vectors crowd out the caller's own. Leaving vectors
unscoped would also make `test_recall_defaults_to_the_query_project` below fail: the
cross-project id comes back in the vector ranking and `episodics.get(mid)` resurrects it.
`MemoryModule.recall` must additionally drop any fused id whose stored project does not match.

**Interfaces:**
- Consumes: Tasks 8-11
- Produces: `Memory.project: str`, `TemporalFact.project: str`, `MemoryQuery.project: str | None`, `MemoryQuery.all_projects: bool`. Default project constant `DEFAULT_PROJECT = "default"`.

- [ ] **Step 1: Write the failing test**

```python
from morgan_brain.models.memory import Memory, MemoryQuery
# ... same _module helper as Task 11, imported from a shared conftest ...


async def test_recall_defaults_to_the_query_project(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor mirror note"))
    await m.store(Memory(user_id="u", project="personal", content="harbor sailing note"))
    got = await m.recall(MemoryQuery(user_id="u", project="acme", text="harbor"))
    assert [x.content for x in got] == ["harbor mirror note"]


async def test_all_projects_crosses_the_boundary(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor mirror note"))
    await m.store(Memory(user_id="u", project="personal", content="harbor sailing note"))
    got = await m.recall(
        MemoryQuery(user_id="u", text="harbor", all_projects=True, top_k=10)
    )
    assert len(got) == 2


async def test_project_is_required_to_be_non_empty():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Memory(user_id="u", project="", content="x")
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_project_scoping.py -v`
Expected: FAIL — `Memory` has no `project` field

- [ ] **Step 3: Add the field to the models**

In `morgan_brain/models/memory.py`:

```python
DEFAULT_PROJECT = "default"


class Memory(UserScoped):
    project: str = Field(default=DEFAULT_PROJECT, min_length=1)
    kind: MemoryKind = MemoryKind.EPISODIC
    # ... unchanged ...


class TemporalFact(UserScoped):
    project: str = Field(default=DEFAULT_PROJECT, min_length=1)
    subject: str
    # ... unchanged ...


class MemoryQuery(BaseModel):
    user_id: str
    project: str = DEFAULT_PROJECT
    all_projects: bool = False
    text: str
    top_k: int = 8
    kinds: list[MemoryKind] | None = None
    include_superseded: bool = False
```

- [ ] **Step 4: Add the column to every store**

Add `project TEXT NOT NULL DEFAULT 'default'` to the `memories`, `facts`, and `memory_entities` schemas and to `fts_memories` as an `UNINDEXED` column; thread it through every `INSERT` and add it to each `WHERE` clause, skipped when `all_projects` is true. Because each schema uses `CREATE TABLE IF NOT EXISTS`, add an idempotent migration in each store's `__init__`:

```python
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
        if "project" not in cols:
            conn.execute(
                "ALTER TABLE memories ADD COLUMN project TEXT NOT NULL DEFAULT 'default'"
            )
            conn.commit()
```

FTS5 virtual tables cannot be `ALTER`ed — if `project` is missing from `fts_memories`, drop and rebuild it from `memories` in the same transaction.

- [ ] **Step 5: Thread project through `MemoryModule`**

`store()` passes `memory.project` to each index; `recall()` passes `None` when `query.all_projects` is true, else `query.project`. `current_facts` gains the same parameter.

- [ ] **Step 6: Run the tests**

Run: `cd morgan-brain && pytest tests/unit/memory/ -q && mypy morgan_brain`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(memory): project-scoped memories, facts and retrieval"
```

---

## Task 13: Extend MemoryGate to cover the cold path

`composition.py:190-196` hands the raw `SqliteTemporalStore` to `MemoryConsolidator`, which uses it for `current_facts`, `close_fact`, and `set_confidence` — operations the gate does not expose. A project filter only at the gate would leave nightly consolidation free to merge facts across projects.

**Files:**
- Modify: `morgan_brain/security/memory_gate.py`, `morgan_brain/interfaces/memory.py`, `morgan_brain/learning/consolidation.py`, `morgan_brain/composition.py:190-196`
- Test: `morgan-brain/tests/unit/security/test_gate_cold_path.py`

**Interfaces:**
- Consumes: Task 12
- Produces: `MemoryGate.close_fact(fact_id, *, user_id, project, now)`, `MemoryGate.set_confidence(fact_id, *, user_id, project, value)`, and `current_facts(..., project, all_projects)`. `MemoryConsolidator` takes `gate: MemoryGate` instead of `temporal: SqliteTemporalStore`.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from morgan_brain.models.memory import TemporalFact
from morgan_brain.security.memory_gate import MemoryGate


async def test_close_fact_is_exposed_on_the_gate(gate: MemoryGate):
    fid = await gate.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="s", predicate="p", object="o")
    )
    await gate.close_fact(fid, user_id="u", project="p")
    assert await gate.current_facts(user_id="u", project="p") == []


async def test_current_facts_is_project_scoped(gate: MemoryGate):
    await gate.upsert_fact(
        TemporalFact(user_id="u", project="acme", subject="s", predicate="p", object="o")
    )
    assert await gate.current_facts(user_id="u", project="personal") == []


async def test_gate_rejects_empty_project(gate: MemoryGate):
    with pytest.raises(PermissionError):
        await gate.current_facts(user_id="u", project="")


def test_consolidator_does_not_hold_a_raw_store():
    """Regression: consolidation must go through the gate, not around it."""
    import inspect
    from morgan_brain.learning.consolidation import MemoryConsolidator

    params = inspect.signature(MemoryConsolidator.__init__).parameters
    assert "temporal" not in params
    assert "gate" in params
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/security/test_gate_cold_path.py -v`
Expected: FAIL — `MemoryGate` has no `close_fact`

- [ ] **Step 3: Extend the `MemoryStore` Protocol first**

In `morgan_brain/interfaces/memory.py`, add `close_fact` and `set_confidence` with project parameters, and add `project`/`all_projects` to `current_facts`.

- [ ] **Step 4: Extend the gate**

```python
    async def close_fact(
        self, fact_id: str, *, user_id: str, project: str, now: datetime | None = None
    ) -> None:
        self._require_scope(user_id, project)
        await self._store.close_fact(fact_id, user_id=user_id, project=project, now=now)

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        self._require_scope(user_id, project)
        await self._store.set_confidence(
            fact_id, user_id=user_id, project=project, value=value
        )

    @staticmethod
    def _require_scope(user_id: str, project: str | None = None) -> None:
        if not user_id:
            raise PermissionError("memory access requires a user_id")
        if project is not None and not project:
            raise PermissionError("memory access requires a project")
```

- [ ] **Step 5: Switch the consolidator to the gate**

In `morgan_brain/learning/consolidation.py`, replace the `temporal` constructor parameter with `gate: MemoryGate` and route lines 185, 241, 262, 314, and 341 through it. Update `composition.py:190-196` to pass the gate. Delete the comment at `consolidation.py:76` claiming these ops are "not exposed on the gate" — they now are.

- [ ] **Step 6: Run the tests**

Run: `cd morgan-brain && pytest -q && mypy morgan_brain`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(security): route consolidation through MemoryGate with project scoping"
```

---

## Task 13A: Wire the durable stack into production composition

**This is the load-bearing task of the whole milestone.** Tasks 7-13 build durable stores that
nothing constructs. Without this task, `composition.py:81` keeps building
`InMemoryVectorIndex`, `bus.start()` is never called so consolidation never runs, signals and
history stay in separate database files (making Task 16's `forget()` impossible), and the
milestone acceptance test passes on FTS alone while production memory stays exactly as
ephemeral as the spec's §1 diagnosis says it is.

**Files:**
- Modify: `morgan_brain/config.py`, `morgan_brain/composition.py` (`_build_vector_index` ~81, `_assemble`, `build_app_context`, `build_worker_context`, the `signals.db`/`history.db` derivation at 338-339 and 415), `morgan_brain/apps/brain_api/app.py`, `morgan_brain/apps/learning_worker/__main__.py`, `morgan_brain/eval/runner.py:66`
- Test: `morgan-brain/tests/integration/test_composition_durable.py`

**Interfaces:**
- Consumes: Tasks 7-13
- Produces: `Settings.data_dir` (`MORGAN_DATA_DIR`, default `./data`); `vector_backend` gains `"sqlite"` **as the default**; one `sqlite3.Connection` per process shared by every store; `bus.start()`/`stop()` bound to the brain-api lifespan

- [ ] **Step 1: Write the failing test**

```python
"""The assembled app must use durable stores, not in-memory ones."""
import sqlite3
from morgan_brain.composition import build_app_context
from morgan_brain.config import Settings
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex


def test_app_context_uses_the_sqlite_vector_index(tmp_path, monkeypatch):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    ctx = build_app_context(Settings())
    assert isinstance(ctx.vectors, SqliteVectorIndex), type(ctx.vectors)


def test_every_store_shares_one_database_file(tmp_path, monkeypatch):
    """signals and history must be reachable from the memory connection, or forget() cannot work."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    build_app_context(Settings())
    dbs = sorted(p.name for p in tmp_path.glob("*.db"))
    assert dbs == ["morgan.db"], f"expected one database, found {dbs}"

    conn = sqlite3.connect(tmp_path / "morgan.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memories", "facts", "interaction_signals", "session_history"} <= tables, tables


def test_brain_api_starts_and_stops_the_bus(tmp_path, monkeypatch):
    """Nothing called bus.start() before this task, so queued cold-path work never ran."""
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    from morgan_brain.apps.brain_api.app import create_app

    app = create_app()
    with TestClient(app) as client:          # __enter__ runs the lifespan
        assert client.get("/health").status_code == 200
        assert app.state.ctx.bus.is_running is True
    assert app.state.ctx.bus.is_running is False
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/integration/test_composition_durable.py -v`
Expected: FAIL — `InMemoryVectorIndex` is built, three `.db` files appear, no lifespan exists.

- [ ] **Step 3: Add `data_dir` and the sqlite backend to Settings**

```python
    data_dir: str = "./data"
    vector_backend: Literal["sqlite", "memory", "qdrant"] = "sqlite"
```

Derive every store path from `data_dir`. Replace the `signals.db` / `history.db` siblings at
`composition.py:338-339` and `:415` with the single `data_dir/morgan.db`, and default
`temporal_db_url` to that same file.

- [ ] **Step 4: Build one connection and share it**

In `_assemble` (and the worker's equivalent), call `open_db` once and pass that connection to
`SqliteTemporalStore`, `SqliteVectorIndex`, `FtsIndex`, `EntityIndex`, `EpisodicStore`,
`SignalStore`, and `SessionHistoryStore`. `SignalStore` and `SessionHistoryStore` currently open
their own files — change their constructors to accept a `sqlite3.Connection` like the others.
Add `"sqlite"` to `_build_vector_index`.

- [ ] **Step 5: Fix the eval scratch gate**

`eval/runner.py:66` constructs a `MemoryModule` and is production code in no other task's file
list. Build its three new indexes over `open_db(":memory:")` so the scratch gate still isolates
eval writes.

- [ ] **Step 6: Wire the bus lifecycle**

Add an `is_running` property to `InProcessBus`. In `apps/brain_api/app.py`, add a FastAPI
`lifespan` that calls `await ctx.bus.start()` on entry and `await ctx.bus.stop()` on exit, and
store the context on `app.state.ctx`. Do the same in the worker's `main()`.

- [ ] **Step 7: Run the tests**

```bash
cd morgan-brain && pytest tests/integration/test_composition_durable.py -v && pytest -q && mypy morgan_brain
```
Expected: the three new tests pass; suite green.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(composition): assemble the durable single-database stack and run the bus"
```

---

## Task 13B: Thread `project` through every caller

After Task 12 the model carries `project`, but five production sites still build `MemoryQuery`
without one, so they silently pin to `"default"`. The worst consequence is not the API: nightly
**consolidation** would only ever consolidate the `default` project, permanently excluding
everything the CLI writes under a real project name from fact extraction.

**Files:**
- Modify: `morgan_brain/core/orchestrator.py:128,191,264`, `morgan_brain/learning/consolidation.py:258`, `morgan_brain/modules/tools/builtin/memory_search.py:55`, `morgan_brain/composition.py:472`, `morgan_brain/apps/brain_api/app.py`
- Test: `morgan-brain/tests/unit/test_project_threading.py`

**Interfaces:**
- Consumes: Tasks 12, 13, 13A
- Produces: `project` as a required parameter on `Orchestrator.handle_turn`, `handle_turn_with_id`, and `stream_turn`; a required `project` field on the chat request models

- [ ] **Step 1: Write the failing test**

```python
import inspect
from morgan_brain.core.orchestrator import Orchestrator


def test_orchestrator_turn_methods_require_a_project():
    for name in ("handle_turn", "handle_turn_with_id", "stream_turn"):
        params = inspect.signature(getattr(Orchestrator, name)).parameters
        assert "project" in params, f"{name} does not accept a project"


async def test_consolidation_does_not_hardcode_the_default_project(consolidator, gate):
    """Regression: consolidating only 'default' would exclude everything the CLI stores."""
    await gate.store(Memory(user_id="u", project="acme", content="harbor mirror note"))
    await consolidator.run(user_id="u", project="acme")
    assert await gate.current_facts(user_id="u", project="acme") != []


def test_no_production_site_builds_an_unscoped_MemoryQuery():
    """Guard against reintroducing the bug."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[2] / "morgan_brain"
    offenders = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if "MemoryQuery(" in line and "project" not in line and "class " not in line:
                offenders.append(f"{py.relative_to(root)}:{i}")
    assert offenders == [], offenders
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/test_project_threading.py -v`
Expected: FAIL — `handle_turn` has no `project`; the scan lists five offenders.

- [ ] **Step 3: Thread it through**

Add `project: str` to the three orchestrator turn methods and pass it into every `MemoryQuery`.
Add a required `project` field to the chat/stream request models in `apps/brain_api/app.py` and
pass it down.

**Consolidation — the exact sites.** There is no `run` method; do not look for one. Task 13
routed the consolidator through the gate but hardcoded `DEFAULT_PROJECT` at every call, because
threading it belongs here. As of commit `6db1318` the hardcoded sites are:

- `learning/consolidation.py:187` — `current_facts` inside `apply`
- `learning/consolidation.py:244` — `close_fact` inside `apply`
- `learning/consolidation.py:266` — `current_facts` inside `consolidate`
- `learning/consolidation.py:318` — `current_facts` inside `decay_confidence`
- `learning/consolidation.py:346` — `set_confidence` inside `decay_confidence`

Add a `project: str` parameter to the public methods that need it — `apply(user_id, batch)`,
`consolidate(user_id)`, and `decay_confidence(...)` — and thread it to those five calls. Then, at
the scheduling layer that invokes consolidation, **iterate over the distinct projects present for
the user** rather than consolidating one. Verify the count afterwards:

```bash
cd morgan-brain && grep -c 'DEFAULT_PROJECT' morgan_brain/learning/consolidation.py
```
Expected: 0 remaining hardcoded uses (the import may also go).

Until this lands, nightly consolidation only ever consolidates the `default` project, so anything
the CLI writes under a real project name is excluded from fact extraction entirely.

- [ ] **Step 4: Run the tests**

Run: `cd morgan-brain && pytest -q && mypy morgan_brain`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(memory): thread project through the orchestrator, API and consolidation"
```

---

## Task 14: `forget()` — cascading erasure in one transaction

**Files:**
- Modify: `morgan_brain/modules/memory/store.py`, `morgan_brain/security/memory_gate.py`, `morgan_brain/interfaces/memory.py`, `morgan_brain/learning/signals.py`, `morgan_brain/learning/history.py`
- Test: `morgan-brain/tests/unit/memory/test_forget.py`

**Interfaces:**
- Consumes: Tasks 11-13, and **Task 13A** (which puts signals and history in the same database — without it this task cannot work)
- Produces: `MemoryGate.forget(*, user_id: str, project: str) -> ForgetReport`, and `ForgetReport` as a dataclass with `memories: int`, `facts: int`, `signals: int`, `history: int`, `champions_flagged: list[str]`

- [ ] **Step 1: Write the failing test**

```python
async def test_forget_removes_from_every_index(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor mirror secret"))
    report = await m.forget(user_id="u", project="p")
    assert report.memories == 1
    reopened = _module(path)
    assert await reopened.recall(MemoryQuery(user_id="u", project="p", text="harbor")) == []


async def test_forget_is_project_scoped(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor"))
    await m.store(Memory(user_id="u", project="personal", content="harbor"))
    await m.forget(user_id="u", project="acme")
    left = await m.recall(MemoryQuery(user_id="u", text="harbor", all_projects=True))
    assert len(left) == 1


async def test_forget_erases_signal_text(tmp_path):
    """signals.db holds query/original_reply/user_edit — the premise covers it."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    # store a signal row for (u, p), then:
    report = await m.forget(user_id="u", project="p")
    assert report.signals >= 0  # exact count asserted with a seeded signal row


async def test_forget_is_idempotent(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    await m.forget(user_id="u", project="p")
    second = await m.forget(user_id="u", project="p")
    assert second.memories == 0
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/memory/test_forget.py -v`
Expected: FAIL — `MemoryModule` has no `forget`

- [ ] **Step 3: Implement**

Collect the affected memory ids first, then delete from `memories`, `vec_items`/`vec_meta`, `fts_memories`, `memory_entities`, `facts`, `signals`, and `history` inside a single `BEGIN IMMEDIATE` transaction, commit, then `VACUUM` (which cannot run inside a transaction). Return the counts.

Four things in this implementation are easy to get wrong, and all four were wrong in the first
draft of this plan:

1. **The table names are `interaction_signals` and `session_history`**, not `signals`/`history`
   (`learning/signals.py:63`, `learning/history.py:37`). The wrong names fail at first run.
2. **Those tables must already live in the same database** — Task 13A moves them. Without that
   prerequisite no single-connection DELETE can reach them.
3. **Vector rows are deleted with plain SQL inside the transaction.** Calling
   `self._vectors.delete()` would commit separately and break atomicity — the point of choosing
   one engine.
4. `EpisodicStore`'s attribute is `_conn`, not `conn`.

First define the report type in the same module:

```python
from dataclasses import dataclass, field


@dataclass
class ForgetReport:
    memories: int = 0
    facts: int = 0
    signals: int = 0
    history: int = 0
    champions_flagged: list[str] = field(default_factory=list)
```

```python
    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        """Erase everything this user stored under this project, in one transaction.

        Champion preprompts are NOT erased: a promoted champion may embed text mined from a
        forgotten conversation and cannot be un-learned, only rolled back. Affected versions
        are returned in ``champions_flagged`` for the owner to review.
        """
        conn = self._episodics._conn
        ids = [
            str(r["id"])
            for r in conn.execute(
                "SELECT id FROM memories WHERE user_id = ? AND project = ?", (user_id, project)
            )
        ]
        report = ForgetReport(memories=len(ids))
        placeholders = ",".join("?" * len(ids))

        conn.execute("BEGIN IMMEDIATE")
        try:
            if ids:
                conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM fts_memories WHERE memory_id IN ({placeholders})", ids)
                conn.execute(
                    f"DELETE FROM memory_entities WHERE memory_id IN ({placeholders})", ids
                )
                # Vectors live in this same database, so they go inside the transaction.
                conn.execute(
                    f"DELETE FROM vec_items WHERE rowid IN "
                    f"(SELECT rowid FROM vec_meta WHERE id IN ({placeholders}))",
                    ids,
                )
                conn.execute(f"DELETE FROM vec_meta WHERE id IN ({placeholders})", ids)
            report.facts = conn.execute(
                "DELETE FROM facts WHERE user_id = ? AND project = ?", (user_id, project)
            ).rowcount
            report.signals = conn.execute(
                "DELETE FROM interaction_signals WHERE user_id = ? AND project = ?",
                (user_id, project),
            ).rowcount
            report.history = conn.execute(
                "DELETE FROM session_history WHERE user_id = ? AND project = ?",
                (user_id, project),
            ).rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        conn.execute("VACUUM")  # cannot run inside a transaction
        return report
```

`champions_flagged` stays empty in this task. Flagging requires the `PromptRegistry`, which
`MemoryModule` does not hold and no task wires in; promising it here and returning nothing would
be worse than scoping it out. Record it in the ledger as deferred, and say so in the docstring.

The gate method takes no `query` parameter — erasure is per (user, project). Update the
Interfaces block above accordingly.

- [ ] **Step 4: Add `project` to the signals and history schemas**

`learning/signals.py:63-71` and `learning/history.py` need a `project` column with the same idempotent `ALTER TABLE` migration used in Task 12, and both move behind the gate.

- [ ] **Step 5: Run the tests**

Run: `cd morgan-brain && pytest -q && mypy morgan_brain`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(memory): cascading forget across every store in one transaction"
```

---

## Task 15: Take the cold path off the request

`bus/inproc.py:19-21` awaits every subscriber inline, so consolidation runs inside the response.

**Files:**
- Modify: `morgan_brain/bus/inproc.py`
- Test: `morgan-brain/tests/unit/test_inproc_bus_async.py`

**Interfaces:**
- Consumes: nothing
- Produces: `InProcessBus(queue_size=…)` with `publish()` returning after enqueue, plus `await bus.drain()` for deterministic tests

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.interfaces.events import Event, EventType


async def test_publish_returns_while_the_handler_is_still_running():
    """Against the old inline bus, `publish` would not return until `slow` finished."""
    bus = InProcessBus()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_event):
        started.set()
        await release.wait()

    bus.subscribe(EventType.RESPONSE_GENERATED, slow)
    await bus.start()

    # A timeout, not a bare assert: the old bus HANGS here rather than failing cleanly.
    await asyncio.wait_for(
        bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={})),
        timeout=1.0,
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert not release.is_set()          # handler is genuinely still blocked
    release.set()
    await bus.drain()
    await bus.stop()


async def test_drain_runs_every_queued_handler():
    bus = InProcessBus()
    seen: list[Event] = []

    async def collect(event):
        seen.append(event)

    bus.subscribe(EventType.RESPONSE_GENERATED, collect)
    await bus.start()
    for _ in range(3):
        await bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={}))
    await bus.drain()
    assert len(seen) == 3
    await bus.stop()


async def test_full_queue_drops_rather_than_blocks():
    """Back-pressure must never block the request path, even when the drain stalls."""
    bus = InProcessBus(queue_size=1)
    gate = asyncio.Event()

    async def stalled(_event):
        await gate.wait()

    bus.subscribe(EventType.RESPONSE_GENERATED, stalled)
    await bus.start()

    for _ in range(5):
        await asyncio.wait_for(
            bus.publish(Event(type=EventType.RESPONSE_GENERATED, user_id="u", payload={})),
            timeout=1.0,
        )
    assert bus.dropped > 0, "a full queue must drop and count, not block"

    gate.set()
    await bus.drain()
    await bus.stop()
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/test_inproc_bus_async.py -v`
Expected: FAIL — `publish` blocks; `drain` does not exist

- [ ] **Step 3: Implement the bounded queue**

`publish()` puts onto an `asyncio.Queue(maxsize=queue_size)` with `put_nowait`, catching `asyncio.QueueFull` and logging a dropped-event counter. `start()` spawns the drain task; `stop()` cancels it after draining. `drain()` awaits `queue.join()`.

Recovery is documented, not coded: queued work derives from durable signal rows written synchronously by `Orchestrator._persist_turn`, so a crash loses scheduling, not data.

- [ ] **Step 4: Run the tests**

Run: `cd morgan-brain && pytest -q`
Expected: green. Tests that relied on synchronous delivery must now `await bus.drain()` — update them.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(bus): dispatch in-process subscribers off the request path"
```

---

## Task 16: llama.cpp defaults, the missing roles, and the promotion flag

**Files:**
- Modify: `morgan_brain/config.py`, `morgan_brain/providers/factory.py:29-30,83-85`, `morgan_brain/composition.py:341`, `morgan_brain/learning/champion_trainer.py`, `morgan_brain/apps/learning_worker/__main__.py`
- Test: `morgan-brain/tests/unit/test_provider_defaults.py`

**Interfaces:**
- Consumes: nothing
- Produces: settings `llm_endpoint` defaulting to llama-server, `role_bindings` covering `strong`/`fast`/`judge`/`reflection`, and `enable_champion_promotion: bool = False`

- [ ] **Step 1: Write the failing test**

```python
from morgan_brain.config import Settings


def test_default_provider_is_not_ollama():
    s = Settings()
    assert "ollama" not in s.providers
    assert all("ollama:" not in b for bs in s.role_bindings.values() for b in bs)


def test_judge_and_reflection_roles_are_bound():
    s = Settings()
    assert set(s.role_bindings) >= {"strong", "fast", "judge", "reflection"}


def test_promotion_is_disarmed_by_default():
    assert Settings().enable_champion_promotion is False


def test_hash_backend_is_stable_across_processes():
    """PYTHONHASHSEED randomises builtin hash(); the stub must not use it."""
    import subprocess, sys, json
    code = (
        "import asyncio,json;"
        "from morgan_brain.providers.factory import build_hash_embedder;"
        "print(json.dumps(asyncio.run(build_hash_embedder().embed('harbor'))))"
    )
    runs = [
        json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                  text=True, check=True).stdout)
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
```

Note what is deliberately **not** here: an assertion that `Settings().embedding_dim == 1024`.
That compares a constant to itself and proves nothing about the model. The real check is the
startup probe in Step 5, which asks the live provider for a vector and measures it.

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/unit/test_provider_defaults.py -v`
Expected: FAIL on all four

- [ ] **Step 3: Rewrite the provider defaults**

In `config.py`: rename the default provider key to `llamacpp`, default `llm_endpoint` to `http://localhost:8081/v1`, pick a default `embedding_model` whose dimension is genuinely 1024 (or set `embedding_dim` to the chosen model's real dimension — the two must agree), add `enable_champion_promotion: bool = False`, add `embedding_backend: Literal["provider", "hash"] = "provider"` (the `hash` value selects a deterministic stub embedder, which Tasks 17 and 18 require to run the CLI without a live model), and extend `_fill_provider_defaults` to bind all four roles:

```python
        if not self.role_bindings:
            self.role_bindings = {
                "strong": [f"llamacpp:{self.llm_model}"],
                "fast": [f"llamacpp:{self.llm_fast_model}"],
                "judge": [f"llamacpp:{self.llm_model}"],
                "reflection": [f"llamacpp:{self.llm_model}"],
            }
```

In `providers/factory.py`, replace the `provider == "ollama"` branch with `"llamacpp"` mapped to the OpenAI-compatible adapter, keeping `"ollama"` as a still-supported non-default key. In `composition.py:341`, build the embedder through the factory instead of constructing `OllamaEmbedder` directly.

- [ ] **Step 4: Gate promotion behind the flag**

In `champion_trainer.py`, delete the unconditional first-candidate branch at lines 123-127 so a missing champion no longer auto-promotes, and in `apps/learning_worker/__main__.py` register the optimize job only when `settings.enable_champion_promotion` is true. Add a startup log line stating the flag's state.

- [ ] **Step 5: Add the dimension probe**

The embedder exposes no dimension property, so "reported dimension" needs a mechanism. In
`composition.py`, after the embedder is built, embed the single token `"probe"` once and compare
`len(vector)` against `settings.embedding_dim`; raise `RuntimeError` naming both values on
mismatch. Skip the probe when `embedding_backend == "hash"` (no provider to ask) and when the
provider endpoint is unreachable — a startup probe must not turn a temporarily-down model server
into a crash loop; log a warning instead.

This is what actually catches the shipped default being wrong: `embedding_model` currently
defaults to a model emitting 2560 while `embedding_dim` says 1024.

- [ ] **Step 6: Run the tests**

Run: `cd morgan-brain && pytest -q && mypy morgan_brain`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(providers): default to llama-server, bind judge and reflection, disarm promotion"
```

---

## Task 17: The `morgan` CLI

**Files:**
- Create: `morgan_brain/cli/__init__.py`, `morgan_brain/cli/__main__.py`, `morgan_brain/cli/project.py`
- Modify: `morgan-brain/pyproject.toml` (`[project.scripts]`)
- Test: `morgan-brain/tests/integration/test_cli.py`

**Interfaces:**
- Consumes: everything above
- Produces: `morgan remember|recall|facts|forget|ask|doctor`, each accepting `--project`, `--all-projects`, and `--json`

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys


def _run(args, env, cwd):
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", *args],
        capture_output=True, text=True, env=env, cwd=cwd, check=False,
    )


def test_remember_then_recall_across_processes(tmp_path, monkeypatch):
    env = {**os.environ, "MORGAN_DATA_DIR": str(tmp_path), "MORGAN_EMBEDDING_BACKEND": "hash"}
    assert _run(["remember", "the Harbor mirror blocked the deploy"], env, tmp_path).returncode == 0
    out = _run(["recall", "harbor", "--json"], env, tmp_path)
    assert out.returncode == 0
    assert "Harbor" in json.loads(out.stdout)["results"][0]["content"]


def test_doctor_reports_actionable_status(tmp_path):
    out = _run(["doctor", "--json"], {**os.environ, "MORGAN_DATA_DIR": str(tmp_path)}, tmp_path)
    report = json.loads(out.stdout)
    assert set(report) >= {"database", "sqlite_vec", "fts5", "provider", "embedding_dim"}


def test_project_defaults_to_the_git_repo_name(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    from morgan_brain.cli.project import detect_project
    assert detect_project(tmp_path) == tmp_path.name
```

- [ ] **Step 2: Run and watch it fail**

Run: `cd morgan-brain && pytest tests/integration/test_cli.py -v`
Expected: FAIL — no `morgan_brain.cli`

- [ ] **Step 3: Implement project detection**

```python
"""Which project a command belongs to, per the reshape spec §4.3."""

from __future__ import annotations

import subprocess
from pathlib import Path

from morgan_brain.models.memory import DEFAULT_PROJECT


def detect_project(cwd: Path | None = None) -> str:
    """Return the git repository's directory name, or DEFAULT_PROJECT outside a repo."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return DEFAULT_PROJECT
    return Path(root).name or DEFAULT_PROJECT
```

- [ ] **Step 4: Implement the commands**

`argparse` with one subparser per command. Every command resolves `--project` (default
`detect_project()`), builds the gate from `composition`, and prints either human text or `--json`.

`doctor --json` must emit at least these keys, because Task 18 asserts on them:

```json
{
  "database": "/abs/path/to/morgan.db",
  "sqlite_vec": "v0.1.9",
  "fts5": true,
  "provider": "reachable" ,
  "embedding_dim": 1024,
  "vector_rows": 12,
  "memory_rows": 12,
  "fts_rows": 12
}
```

`vector_rows` is `SELECT count(*) FROM vec_meta`. It is the check that catches an unwired vector
store — the failure mode where recall still works via FTS and nobody notices the vectors were
never persisted.

- [ ] **Step 5: Register the entry point**

```toml
[project.scripts]
morgan = "morgan_brain.cli.__main__:main"
```

- [ ] **Step 6: Run the tests**

Run: `cd morgan-brain && pytest tests/integration/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cli): morgan remember, recall, facts, forget, ask and doctor"
```

---

## Task 18: The milestone acceptance test

**Files:**
- Create: `morgan-brain/tests/integration/test_cross_repo_recall.py`

**Interfaces:**
- Consumes: every task above
- Produces: the executable form of the spec's milestone-1 acceptance criterion

- [ ] **Step 1: Write the test**

```python
"""Spec §7 milestone 1: store in one repo, restart, recall from another."""
import json
import os
import subprocess
import sys


def _morgan(args, *, cwd, data_dir):
    env = {**os.environ, "MORGAN_DATA_DIR": str(data_dir), "MORGAN_EMBEDDING_BACKEND": "hash"}
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", *args],
        capture_output=True, text=True, env=env, cwd=cwd, check=False,
    )


def test_cross_repo_recall_after_restart(tmp_path):
    data = tmp_path / "brain"
    repo_a, repo_b = tmp_path / "acme", tmp_path / "personal"
    for r in (repo_a, repo_b):
        r.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=r, check=True)

    stored = _morgan(
        ["remember", "ACME-14802 blocked on the Harbor mirror, not the chart"],
        cwd=repo_a, data_dir=data,
    )
    assert stored.returncode == 0, stored.stderr

    # A separate process — nothing is shared but the database on disk.
    out = _morgan(["recall", "harbor", "--all-projects", "--json"], cwd=repo_b, data_dir=data)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert any("Harbor mirror" in r["content"] for r in results)
    assert results[0]["project"] == "acme"


def test_vectors_are_actually_persisted_not_just_fts(tmp_path):
    """Without this, the whole suite passes on the FTS signal alone.

    Under the hash embedder there is no semantic similarity, so a recall assertion cannot
    distinguish "vectors work" from "vectors are unwired and FTS carried it" — which is exactly
    how a milestone declares itself done while production memory stays ephemeral. Assert the
    rows exist on disk instead.
    """
    data = tmp_path / "brain"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _morgan(["remember", "the Harbor mirror blocked the deploy"], cwd=repo, data_dir=data)

    doctor = json.loads(_morgan(["doctor", "--json"], cwd=repo, data_dir=data).stdout)
    assert doctor["vector_rows"] > 0, doctor
    assert doctor["database"].endswith("morgan.db"), doctor


@pytest.mark.live
def test_real_embedder_round_trip(tmp_path):
    """The spec's acceptance says 'with a real embedder'. The hash stub does not satisfy that.

    Skipped by default; run with a live llama-server on MORGAN_LLM_ENDPOINT.
    """
    data = tmp_path / "brain"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    env_real = {**os.environ, "MORGAN_DATA_DIR": str(data)}  # no hash backend
    subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", "remember", "the deploy was blocked by the registry mirror"],
        cwd=repo, env=env_real, check=True, capture_output=True,
    )
    out = subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", "recall", "what stopped the release?", "--json"],
        cwd=repo, env=env_real, capture_output=True, text=True, check=True,
    )
    # A semantic match with no shared keywords — only a real embedder can do this.
    assert "registry mirror" in out.stdout


def test_cyrillic_survives_the_same_round_trip(tmp_path):
    data = tmp_path / "brain"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _morgan(["remember", "реестр Harbor заблокировал деплой"], cwd=repo, data_dir=data)
    out = _morgan(["recall", "реестр", "--json"], cwd=repo, data_dir=data)
    assert "реестр" in json.loads(out.stdout)["results"][0]["content"]
```

- [ ] **Step 2: Run it**

Run: `cd morgan-brain && pytest tests/integration/test_cross_repo_recall.py -v`
Expected: PASS. If it fails, milestone 1 is not done — fix the cause, not the test.

- [ ] **Step 3: Full verification**

```bash
cd morgan-brain
pytest -q
ruff check . && ruff format --check .
mypy morgan_brain
```
Expected: all green, `mypy` 0 errors.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(integration): cross-repo recall after restart"
```

---

## Task 19: Update the docs to the delivered state

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/WIRING.md`, `docs/ROADMAP.md`, `morgan-brain/.env.example`

- [ ] **Step 1: Rewrite the package map and invariants in `CLAUDE.md`**

Reflect: one SQLite database, the four surfaces, `MemoryGate` covering the cold path, project scoping, `forget()`, llama-server defaults, and the promotion flag. Replace the old build/run commands with the CLI.

- [ ] **Step 2: Rewrite `docs/WIRING.md` for llama-server**

Document starting `llama-server` with chat, embedding, and rerank models, the `MORGAN_ROLE_BINDINGS` shape for all four roles, and `morgan doctor` as the verification step.

- [ ] **Step 3: Record the real numbers**

Run `pytest -q` and put the actual counts in `CLAUDE.md`. Never write a number you have not just observed.

- [ ] **Step 4: Commit**

```bash
git commit -am "docs: describe the delivered local-first brain"
```

---

## Self-review notes

**Spec coverage.** §4.1 → Tasks 7-11 + 13A; §4.2 → Task 16; §4.3 → Tasks 12, 13, 13B; §4.4 →
Task 14; §3.1 → Tasks 15 + 13A (the queue is useless until something starts the bus); §6 →
Tasks 3-6; §7 M0 → Tasks 1-6; §7 M1 → Tasks 7-19; §8 → Tasks 2, 11, 18. §4.5 (ChatGPT import)
and §5 (learning gate) are milestone 2/3 and deliberately out of this plan.

**Execution order.** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → **13A** → **13B**
→ 14 → 15 → 16 → 17 → 18 → 19. Task 13A must precede Task 14: `forget()` cannot delete from
`interaction_signals` and `session_history` until 13A moves them into the same database.

**Corrections applied after adversarial review (2026-08-02).** The first revision of this plan
would have gone green without delivering the milestone. Recorded here so the same holes are not
reintroduced:

- No task wired the durable stores into `composition.py`, so production would have kept using
  `InMemoryVectorIndex` while Task 18 passed → **Task 13A**.
- `forget()` targeted tables named `signals` and `history`. The real names are
  `interaction_signals` and `session_history`, and they lived in **separate database files**.
- Nothing called `bus.start()`; `apps/brain_api/app.py` has no lifespan hook, so every queued
  cold-path event would have sat unprocessed forever.
- Vector search was scoped by over-fetching and post-filtering. Proven wrong on sqlite-vec
  0.1.9: with two users interleaved and `k=2`, the unscoped query returns the *other* user's
  rows and the caller's own second-nearest never appears. vec0 metadata columns fix it.
- Two `grep | xargs git rm` steps would have deleted tests for **kept** code — the entire
  provider suite, `test_cron.py`, `test_redis_bus_live.py`, both learning-worker tests — leaving
  the suite green because the tests were gone. Both replaced with explicit reviewed lists.
- `project` never reached the orchestrator, the API, `memory_search`, or consolidation, so
  nightly consolidation would only ever have consolidated the `default` project → **Task 13B**.

**Known follow-ups, not gaps.**
- `ForgetReport.champions_flagged` always returns empty in Task 14. Flagging needs the
  `PromptRegistry`, which `MemoryModule` does not hold. Scoped out deliberately rather than
  promised and stubbed.
- `QdrantVectorIndex.delete` (Task 8, Step 5) is exercised only by the existing skipped live test.
- The real-embedder half of the spec's acceptance criterion is covered by one
  `@pytest.mark.live` test that needs a running llama-server; it does not run in CI.
