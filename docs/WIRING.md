# Wiring and running

## 1. Install

Python 3.12 or later. From the repository root:

```bash
uv tool install --editable .   # morgan and morgan-mcp, on your PATH in every directory
pip install -e ".[dev]"        # in a virtualenv, to work on Morgan: plus pytest, ruff, mypy
```

A tool install gets its own environment and puts both commands on your user `PATH`, which is
where MCP clients, agents' shells and OpenResearch sessions look for them. A virtualenv's
commands are on `PATH` only while it is active. The install is editable, so a `git pull` in
this checkout upgrades it.

## 2. The model server

Any OpenAI-compatible endpoint. The default and documented one is
[`llama-server`](https://github.com/ggml-org/llama.cpp), which serves one model per process,
so run a chat model and an embedding model:

```bash
llama-server -m qwen2.5-7b-instruct.gguf --port 8081                 # chat
llama-server -m mxbai-embed-large.gguf --embedding --port 8082       # embeddings
```

Run them on the machine with the GPU and reach them over your overlay network, or on
`localhost` for offline work; only the endpoint URL differs. Ollama's `/v1` and vLLM speak the
same protocol.

## 3. Configure

Copy `.env.example` to **`~/.config/morgan/.env`** (`$XDG_CONFIG_HOME/morgan/.env`). That
file is read from every working directory, which is what the CLI needs: it is meant to run
from inside whichever repository you are working in. The CLI reads a `./.env` in the current
directory after it, which overrides it; real environment variables override both.
`morgan-mcp` reads the user file and the environment only: a client starts it in whatever
folder it has open, and a `./.env` there is that project's, not Morgan's.

```bash
MORGAN_LLM_ENDPOINT=http://localhost:8081/v1   # the chat model
MORGAN_LLM_MODEL=qwen2.5-7b-instruct
MORGAN_EMBEDDING_MODEL=mxbai-embed-large      # the embedding model
MORGAN_EMBEDDING_DIM=1024                     # must match the embedding model
# MORGAN_EMBEDDING_ENDPOINT=http://localhost:8082/v1   # only if it is a separate server
# MORGAN_LLM_API_KEY=                         # only if the chat server enforces --api-key
# MORGAN_EMBEDDING_API_KEY=                   # only if a separate embedding server enforces one
# MORGAN_LLM_JSON_MODE=json_schema            # how consolidation asks for JSON
# MORGAN_DATA_DIR=~/.local/share/morgan       # the one database
# MORGAN_API_KEY=                             # required before morgan-mcp binds beyond loopback
```

Three keys, and the direction and host each goes to matter: `MORGAN_LLM_API_KEY` is what Morgan
presents *to* the chat server; `MORGAN_EMBEDDING_API_KEY` is what it presents *to* the embedding
server, only when `MORGAN_EMBEDDING_ENDPOINT` addresses a separate one -- without one, embeddings
go to the chat host and carry `MORGAN_LLM_API_KEY`, since there is no second host to give a key
to. Neither ever reaches the other server: the chat credential does not appear in the embedding
host's logs, and vice versa. `MORGAN_API_KEY` is unrelated to both -- what MCP clients present
*to* Morgan over HTTP.

When one server answers both, leave `MORGAN_EMBEDDING_ENDPOINT` empty and everything goes to
`MORGAN_LLM_ENDPOINT`. Set it when they are separate, which is the common case:
`llama-server` loads one model per process, and a chat server started without `--embedding`
answers `/embeddings` with a 501.

## 4. Check: `morgan doctor`

```
database: ~/.local/share/morgan/morgan.db
env_file: ~/.config/morgan/.env (present)
env_file: ~/code/my-repo/.env (absent)
project: my-repo
all_projects: False
embedding_backend: provider
embedding_dim: 1024
embedding_endpoint: http://gpu-box:8082/v1
llm_endpoint: http://gpu-box:8081/v1
llm_model: qwen2.5-7b-instruct
data_flow: gpu-box (MORGAN_LLM_ENDPOINT) receives ask: the question, the memories recalled for it and the recent history; consolidate: up to 50 memories per project, with the project's current facts
data_flow: gpu-box (MORGAN_EMBEDDING_ENDPOINT) receives remember: the memory's text; recall: the query; import: every imported message; doctor --vectors: a sample of stored memories; a process's first embedding call: up to 5 stored memories, while the embedding space's fingerprint is unrecorded
sqlite_vec: v0.1.9
fts5: True
provider: reachable (0.3 s)
embedding_provider: slow (41.8 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)
embedding_space: 1 (mxbai-embed-large, 1024 dims): fingerprint matches (min cosine 0.9993); strings sha256 82b4fe781cb3d981d562c08cb1de3ce802abc22ea5d2e956e11672f66b340f6f
migration: user_version 7 of 7, nothing pending
snapshots: 1 in ~/.local/share/morgan/snapshots, newest morgan-20260921T101500Z-migrate.db, 1843200 bytes in all
memories: 12 in project 'my-repo' (128 across all projects)
fts: 12 in project 'my-repo' (128 across all projects)
vectors: 12 in project 'my-repo' (128 across all projects)
rows_by_project: {'my-repo': 12, 'personal': 116}
rows_missing_provenance: 0
rows_missing_provenance_reason: None
project 'my-repo': unclassified, capture on, consolidate on
project 'personal': unclassified, capture on, consolidate on
code_root: ~/code
code_root: ~/work (not a directory)
vector_audit: none (not requested; pass --vectors)
```

Paths print in full; `~` stands for your home folder here.

`doctor` only reads. It opens the database read-only and changes nothing in it -- no table
is created, no migration step runs, the journal mode is left as it is -- so it is safe to run
on a database another install still writes to, on one waiting for `morgan migrate`, or on one
`morgan restore` just put back. On a fresh install the first line reads `(no database yet)`
and the counts are `None`: the first command that stores a memory creates the file. A file
that cannot be read that way is named on the same line, with the reason.

Every probe is independent, so one failure does not hide the rest. The first lines answer
"why is my brain empty?": a database somewhere other than where you expect, or a `.env` file
read, or missing, where you did not expect it. Each `env_file` line is a file the CLI read, in
order, and whether it was there.

The `data_flow` lines say which host receives which of your text, one line per endpoint, by
host name alone; when one server answers both, it is one line.

The two model servers are probed separately, each with one request. `provider` is the chat
endpoint, which only `ask` and `consolidate` need. `embedding_provider` is the endpoint every
`remember` and `recall` embeds with, probed by embedding the five fixed strings the embedding
space is fingerprinted with, so a chat server that serves no embeddings shows as refused here;
under the hash backend it reads `not used`. Each reads one of four:

- `reachable`: it answered, within `MORGAN_DOCTOR_SLOW_AFTER_SECONDS` (2 s).
- `slow`: it answered after that, or answered a 429 or a 5xx other than 501, which a retry may
  mend. An embedding host that unloads its model when idle takes tens of seconds over its
  first answer: that is `slow`, and it works.
- `refused`: it answered, and refused the request. A 401 or 403 names the key setting the
  request carried (`MORGAN_EMBEDDING_API_KEY` when embeddings have their own endpoint, else
  `MORGAN_LLM_API_KEY`); any other 4xx, a redirect, a 501, or a 200 that is not an
  embeddings response (a proxy's web page) names the endpoint setting.
- `unreachable`: no answer came within `MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS` (60 s), which
  names that setting, or no connection was made, which names the endpoint setting. A host that
  answered is never unreachable. When this machine cannot build an HTTP client at all, the
  line says so and names neither.

Each line says why: how long the answer took, the timeout it missed, the HTTP status, the
setting to check. `--json` gives each probe's `seconds`, `timeout_seconds`,
`slow_after_seconds` and `error` under `provider_probe` and `embedding_probe`.

`embedding_space` compares the five strings' fresh vectors with the fingerprint the database
recorded for its active space: `matches`, `MISMATCH` (a different model is answering: stored
vectors would be searched with the wrong one), or `unrecorded` (a process's first embedding
call records it, once a sample of stored memories shows the model answering wrote them).
`doctor` never records one. `migration` lists the steps `morgan migrate`
would run, heavy or light; `doctor` runs none. `snapshots` counts the files in
`MORGAN_SNAPSHOT_DIR` that `snapshot`, `migrate`, `forget` and `restore` left there, which
Morgan never deletes.

The `memories`/`fts`/`vectors` lines are scoped to `project`, with the total across every
project beside them, so a zero from the wrong directory does not read as an empty database.
`rows_by_project` gives every project's own count, and `rows_missing_provenance` counts rows
that an older Morgan, still running somewhere, wrote after `morgan migrate` (an empty
`author_id`, or a NULL `vec_items` status); it reads `None` with a reason on a database that
has not been through `morgan migrate` yet, where the columns it counts do not exist. Each
`project` line is a row of the `projects` table: its classification and whether capture and
consolidation are on. Migration step 7 seeds one row, `unclassified`, for each project a
database already held; nothing adds a row for a project written to later, and a project with
no row is consolidated. A `MORGAN_CODE_ROOTS` entry that is not a directory is marked so;
nothing walks the roots yet.

The plain embedding probe above already sends the five fingerprint strings to the embedding
host on every `doctor` run, to say whether it answers at all. `morgan doctor --vectors` sends
more: a sample of up to `MORGAN_VECTOR_AUDIT_SAMPLE_ROWS` (180) stored vectors in `project`
(every project with `--all-projects`), spread evenly across the active embedding space's
table, re-embedded and compared against what is stored, at
`MORGAN_EMBEDDING_FINGERPRINT_TOLERANCE`. A line to stderr says so first -- the host itself is
in `data_flow`, above -- because stdout still carries `--json`. `--clients N` runs the same
sample through N independent, concurrent embedders instead of one, and reports each client's
own numbers beside any id whose fresh vectors disagreed between clients -- catching a server
that only sometimes answers a request wrong, which one client alone cannot see. The audit is
skipped, with a reason naming the probe's own verdict, when that plain probe already found the
embedding host unreachable or refused -- otherwise every client would wait out the full import
retry budget (`MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS`, 600 s by default) only to report
what the probe already knows:

```
vector_audit: 180 sampled, min 0.9982, median 0.9998, 0 below tolerance
  client-1: min 0.9985, median 1.0000, 0 below tolerance, 173.786 s
  client-2: min 0.9982, median 0.9996, 0 below tolerance, 173.537 s
  wall_seconds includes this process's own per-request client setup, which grows with
  --clients on its own; Ollama specifically may also serialise concurrent clients rather than
  answer them in parallel. A slower wall time with --clients than without it can be either, or
  both -- not evidence on its own that the host serialised anything.
```

`--json` gives the same numbers under `vector_audit`: `sampled`, `min`, `median`,
`below_tolerance` (the ids, pooled across every client), `per_client` (each client's own `min`,
`median`, `below_tolerance` and `wall_seconds`, or an `error` if that client could not be
reached at all) and `disagreements` (the ids clients answered differently for). `vector_audit`
and `vector_audit_reason` are always both present, like `embedding_space` and its own reason,
so a `--json` consumer reading one never gets a `KeyError` depending on whether `--vectors` was
passed: without it, or on the hash backend, or before an embedding space is registered, or when
the embedding host is unreachable or refused, `vector_audit` reads `None` and
`vector_audit_reason` says why -- nothing runs a model unasked.

## 5. The CLI

Every command takes `--json`. `remember`, `recall`, `facts`, `forget`, `ask`, `consolidate`
and `doctor` also take `--project` (default: the current git repository's name; a linked
worktree counts as the repository it came from; outside a repository, `personal`) and, where
it makes sense, `--all-projects`.

```bash
morgan remember "the Harbor mirror blocked the deploy"   # embedding model only
morgan recall "what blocked the deploy"                   # vector + FTS5, fused
morgan facts                                              # currently-valid facts
morgan ask "what do you know about the deploy"            # chat model: recall, answer, remember
morgan consolidate                                        # chat model: memories → facts
morgan forget                                             # everything under this project
morgan import ~/Downloads/conversations.json              # seed from a ChatGPT export
morgan doctor                                             # see section 4
morgan snapshot                                           # a verified copy of the database
morgan restore <snapshot> --yes                           # put a snapshot back
morgan migrate                                            # upgrade the database, behind a snapshot
morgan install-skill                                      # see section 7
```

`remember` with no `--project` outside a repository stores in `personal`, and its `--json`
result says so with `project_defaulted: true`. `recall`'s `--json` result carries `abstained`
and `reason`: an empty result is `empty` (nothing stored in scope) or `declined` (with
`MORGAN_RECALL_FLOOR_MARGIN` set, nothing stood out above the background); results carry
`too_few_to_judge` or `no_floor` when the floor did not judge them, and `null` when it did.
`forget` takes a snapshot first and prints its path: `morgan restore` with that path is the
undo.

### Snapshots, restore and migrate

`snapshot`, `restore` and `migrate` act on the whole database file and take no `--project`.

`morgan snapshot [--reason <word>]` writes a `VACUUM INTO` copy of the database into
`MORGAN_SNAPSHOT_DIR` (default `snapshots/` beside the database), named by UTC time and
reason, and checks it with `PRAGMA quick_check` before reporting it; `--list` lists them. The
copy is consistent while other processes hold the file open. Morgan never deletes a snapshot
that passed its check.

`morgan restore <file>` alone only prints what it would replace, and exits 2. With `--yes` it
checks the snapshot, refuses one written by a newer Morgan, takes a `before-restore` snapshot
of the database as it is, and swaps the file in. On Windows, close every running `morgan-mcp`
first: a file another process holds open cannot be replaced.

After an upgrade of Morgan, `morgan doctor`'s `migration` line says whether the database waits
for `morgan migrate`. Light steps run by themselves when the database is opened. A heavy step
(one that rewrites, moves or deletes rows) waits for `morgan migrate`, and until it runs every
write -- `remember`, `ask`, `consolidate`, `forget`, `import`, and the MCP tools that write --
fails with an error containing "writes are blocked until `morgan migrate` runs" and naming the
pending steps; `recall` and `facts` still answer. `morgan migrate --dry-run` lists the pending
steps. `morgan migrate` takes a `migrate` snapshot, runs every pending step in one
transaction, checks the result, prints the row counts before and after, and then checks the
embedding space against the embedding server. A step that fails rolls the whole run back, and
the error names the snapshot. Close sessions still running an older `morgan-mcp` before you
migrate: an older process writing afterwards is what `rows_missing_provenance` counts.

`import` seeds memory from a ChatGPT export so a fresh brain is not an empty box. It writes
to `archive/chatgpt`, not to your working project, and takes no `--project`: a fifth of the
conversations go to `archive/chatgpt-holdout` instead, reserved for evaluating the memory
against conversations nothing has learned from. Expect it to take a while, since every turn
costs an embedding call; progress goes to stderr. Re-running updates in place.

An import runs a canary: every `MORGAN_IMPORT_CANARY_EVERY` (50) memories it actually stores
-- never one skipped because it was already there unchanged -- it re-sends the five
fingerprint strings alone and compares them against the embedding space's recorded
fingerprint, and once more at the end for whatever was stored since the last good check. This
bounds how many memories a model that is *still* answering wrong when a check runs can reach
before it is named; it does not catch a vector that was wrong only in between two checks and
has since recovered -- `morgan doctor --vectors`'s full re-embed sample is the check for that.
A mismatch raises `ImportStopped`, printed on stderr (and under `suspect_ids` in the `--json`
error object), naming the suspect range, the suspect memory ids and the setting that
addresses the model, and pointing at `morgan doctor --vectors`. **The suspects are not
repaired.** They are already stored, in every index, with whatever vectors they were given --
the canary runs after the stretch is stored, not before -- and re-running the import skips
every id already there unchanged, suspects included, rather than re-embedding them. Nothing
re-embeds named memory ids, and nothing holds a stretch back until its own canary passes.

`MORGAN_EMBEDDING_BACKEND=hash` replaces the embedding call with a deterministic stub, so the
memory commands run with no model server at all (keyword and entity search still work; vector
similarity does not mean anything).

When a model server is down, too slow or refuses the request, every command that needs it
says which endpoint and which setting to check, and exits 1; under `--json` the error is the
whole of stdout. An embedding call is retried first, within
`MORGAN_EMBEDDING_RETRY_BUDGET_SECONDS` (60 s; `MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS`,
600 s, for an import) when the host answered slowly, with a 5xx other than 501 or a 429, or
dropped the connection -- a cold host loading its model looks like that -- and within
`MORGAN_EMBEDDING_UNREACHABLE_BUDGET_SECONDS` (5 s) when no connection could be made. A
refusal (a 4xx other than 429, a 501, a redirect) is not retried.

## 6. The MCP server

The same five operations for any MCP client, through the same gate, returning the same
results the CLI's `--json` prints. `project` is a tool argument (the server is a daemon; its
own working directory means nothing to a client). A call that names no project works in
`personal`: `remember` says so with `project_defaulted: true`, and `forget` without a
`project` erases `personal`. Every tool declares MCP's hints: `recall` and `facts` are
read-only, `remember` and `ask_morgan` write (a turn stores the exchange), `forget` is
destructive. Errors come back as the tool's error result, carrying the message the CLI prints: an
unreachable, slow or refusing model server, an `EmbeddingSpaceMismatch`, or a database waiting
for `morgan migrate`, which refuses the three tools that write while `recall` and `facts`
still answer.

```bash
claude mcp add -s user morgan -- morgan-mcp --transport stdio   # Claude Code, every project
morgan-mcp --transport http                                 # loopback, MORGAN_MCP_HOST/PORT
MORGAN_API_KEY=… morgan-mcp --transport http --host <overlay-address>   # other machines
```

`-s user` matters: Claude Code's default scope registers a server only for the project the
command ran in, and a memory meant for every repository would be missing from all the others.
`claude mcp get morgan` starts the server and reports whether it connected.

The HTTP transport enforces `MORGAN_API_KEY` as a bearer token. With no key set it serves
loopback only, and refuses to start on any other host: these tools include `forget`. See
[`OPERATIONS.md`](OPERATIONS.md) for client configuration.

## 7. Teach your agents: `morgan install-skill`

The tools say what Morgan can do; the skill says when. It tells an agent to recall before a
task or a design decision, to remember decisions, corrections, conventions and measured
conclusions with their evidence, never to store secrets, and which project to name.

```bash
morgan install-skill                         # lists every path, then asks
morgan install-skill --yes --json            # for scripts
morgan install-skill --mcp-server brain      # if morgan-mcp is registered under another name
```

It writes `morgan/SKILL.md` for each agent installed here: Claude Code (`~/.claude/skills`,
or `CLAUDE_CONFIG_DIR`), Codex (`~/.agents/skills`), OpenCode
(`$XDG_CONFIG_HOME/opencode/skills`), Cursor (`~/.cursor/skills`). In Claude Code's
`settings.json` it allows `mcp__morgan__recall` and `mcp__morgan__facts`, the tools that
declare themselves read-only, so recalling does not stop for a prompt. The tools that write
stay behind the prompt, and under `"defaultMode": "dontAsk"`, which refuses every call no
rule allows, agents cannot `remember` until you allow `mcp__morgan__remember` yourself.
A `morgan` skill it did not write is left alone, and so is a settings file it cannot parse; both
are checked again as each file is written. Run it again after upgrading Morgan.

**OpenResearch.** Its agent sessions run with their own configuration and load neither the
skills above nor your MCP servers. The installer puts the skill in OpenResearch's upload
store (`user-skills/global/morgan/SKILL.md` under `ORX_DATA_DIR`, or
`~/.local/share/openresearch`), which every session receives, and the skill falls back to the
`morgan` command when the tools are absent -- so `morgan` must be on the session's `PATH`. A
session runs in its own git worktree; the command still files memories under the
repository's project, so a conclusion recorded in one experiment session is recalled in the
next, and `--all-projects` finds it from another research project. If you moved
OpenResearch's data folder in its settings, set `ORX_DATA_DIR` before installing.

## 8. Docker

`docker compose up -d` builds the image and runs `morgan-mcp --transport http` on port 8090
with `./data` mounted as the database directory. Put `MORGAN_LLM_ENDPOINT` and a real
`MORGAN_API_KEY` in `.env` next to the compose file; the published port is not loopback, so
the server refuses to start without the key.

After an upgrade that leaves a heavy migration step pending, the server refuses every write
until `morgan migrate` has run against `/app/data`. Run it in a one-off container of the same
service, which gets the same `.env` and the same `./data` mount, with the old server stopped:

```bash
docker compose build
docker compose stop morgan-mcp
docker compose run --rm morgan-mcp morgan migrate
docker compose up -d
```

The snapshot `migrate` takes lands in `./data/snapshots`.

## 9. Consolidation on a schedule

Nothing in Morgan runs a model unasked. If you want nightly consolidation, that is one cron
line on the machine that holds the database:

```
0 3 * * * morgan consolidate --all-projects --json >> ~/.local/share/morgan/consolidate.log
```
