# Wiring and running

This guide covers the memory CLI and MCP server. Optional task and owner workflows have
separate contracts in the [decision index](decisions/README.md).

## 1. Install

For development, follow [the virtual-environment setup](../CONTRIBUTING.md). To make the
commands available on your user PATH, an alternative is:

```bash
uv tool install --editable .
```

That installation uses its own environment. It does not install dependencies into your
system Python, and a checkout change changes what it runs. Use a
[pinned, verified runtime](decisions/pinned-runtime.md) for deployment.

## 2. The model server

Morgan uses OpenAI-compatible chat and embedding endpoints. Compatibility depends on the
server's supported request modes, model IDs and embedding dimensions; test them with
`morgan doctor` before capture. Strict native answers additionally need the
[counted llama.cpp backend](decisions/strict-native-context.md).

A typical llama.cpp setup uses separate processes for chat and embeddings:

```bash
llama-server -m /path/to/chat-model.gguf --port 8081
llama-server -m /path/to/embedding-model.gguf --embedding --port 8082
```

Use the model IDs advertised by those servers. Ollama exposes compatible endpoints under
`/v1`; a server that serves both models can use one endpoint. A remote server receives the
text sent for embedding or generation. Choose endpoints you trust.

## 3. Configure

The user file is `${XDG_CONFIG_HOME:-~/.config}/morgan/.env`.
The CLI reads that file, then `./.env`, then environment variables. Later values win.
Morgan's MCP settings read the user file and environment only.

Create the user file without overwriting an existing configuration:

```bash
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/morgan"
mkdir -p "$config_dir"
test -e "$config_dir/.env" || cp .env.example "$config_dir/.env"
```

Edit it to match your servers. This example assumes separate endpoints and 1,024-dimensional
embeddings; replace the model IDs and dimension with the actual served values:

```dotenv
MORGAN_OWNER_USER_ID=owner
MORGAN_LLM_ENDPOINT=http://localhost:8081/v1
MORGAN_LLM_MODEL=your-chat-model
MORGAN_EMBEDDING_ENDPOINT=http://localhost:8082/v1
MORGAN_EMBEDDING_MODEL=your-embedding-model
MORGAN_EMBEDDING_DIM=1024
```

| Setting | Purpose |
| --- | --- |
| `MORGAN_LLM_API_KEY` | Outbound authentication to the chat endpoint. |
| `MORGAN_EMBEDDING_API_KEY` | Outbound authentication when a separate embedding endpoint is set. |
| `MORGAN_API_KEY` | Inbound bearer token for Morgan's HTTP MCP listener. |
| `MORGAN_LLM_JSON_MODE` | Structured output mode: `json_schema`, `json_object` or `prompted`. |
| `MORGAN_DATA_DIR` | Memory directory; defaults to `~/.local/share/morgan`. |
| `MORGAN_TEMPORAL_DB_URL` | Explicit SQLite URL; otherwise derived from the data directory. |
| `MORGAN_SNAPSHOT_DIR` | Snapshot directory; otherwise `snapshots/` under the data directory. |
| `MORGAN_RECALL_FLOOR_MARGIN` | Optional model-specific relevance threshold. |
| `MORGAN_EMBEDDING_BACKEND` | `provider` for semantic embeddings; `hash` for offline plumbing checks. |

Without a separate embedding endpoint, embeddings go to the chat endpoint and use its key.
With a separate endpoint, its key is used instead. The inbound MCP key is independent.

See [.env.example](../.env.example) for retry, timeout, fingerprint and project-classification
settings. Do not change the embedding model or dimension for an existing database: its stored
vectors belong to one fingerprinted space. Use a new data directory for a different model,
and also change any explicit database URL. A snapshot retains the original model's vectors.

FastMCP itself still attempts to decode a `./.env` in the client's startup directory.
An invalid UTF-8 file there can stop MCP startup even though Morgan's settings ignore it.
This known SDK limitation is distinct from CLI configuration errors.

## 4. Check: `morgan doctor`

```bash
morgan doctor --project demo --json
```

Doctor reads database state without migrations or writes and explicitly probes the model
endpoints. It reports configuration sources, provider status, embedding identity, pending
migrations and scoped counts. See [diagnostics](DIAGNOSTICS.md) for interpretation and
`doctor --vectors` sampling.

## 5. The CLI

Use `morgan --help` and `morgan COMMAND --help` for current arguments. The examples below use
an explicit project so their scope does not depend on the working directory.

```bash
morgan remember "The deployment uses PostgreSQL" --project demo --source user_stated
morgan recall "Which database does deployment use?" --project demo --json
morgan facts --project demo --json
morgan evidence SOURCE_ID --project demo --json
morgan context inspect SOURCE_ID --project demo --json
morgan ask "Which database does deployment use?" --project demo
morgan consolidate --project demo
```

Replace `SOURCE_ID` with an actual stored ID. `evidence` accepts up to 32 IDs;
`context inspect` accepts up to 16. Both read an existing database without embedding or
chat calls and refuse cross-project lookup. See the [revision](REVISION_CLIENT.md) and
[inspection](CONTINUATION_CONTEXT.md) contracts for effective time and exact selections.

Capture actual user statements as `user_stated`, tool observations as `tool_observed`,
and agent conclusions as `agent_inferred`. Omitted source is `unknown`. Optional
`--author-id` reports attribution, not authentication. Consolidation uses eligible active
user/tool evidence; unknown captures are not promoted. Replies to `ask` are agent inferences.

Without `--project`, the CLI uses the enclosing repository name, including linked worktrees,
or `personal` outside one. Same-named repositories share a project. Repository-derived writes
also record project classification; explicit project names do not imply a filesystem location.

### Snapshots, restore and migrate

These commands act on the whole database:

```bash
morgan snapshot
morgan snapshot --list
morgan migrate --dry-run
```

A heavy migration requires `morgan migrate` and takes a snapshot first. Restore replaces the
database and requires all users of that file to stop. Follow the
[upgrade and recovery procedure](OPERATIONS.md#upgrade-and-recover) before either operation.
`morgan forget --project NAME` erases that scope and keeps an undo snapshot; retained snapshots
still contain forgotten data.

### Import

```bash
morgan import /path/to/conversations.json
```

Import accepts an explicitly selected ChatGPT export. It writes to `archive/chatgpt` and
reserves one fifth of conversations in `archive/chatgpt-holdout`; it takes no `--project`.
Import resumes per event. Exact replay skips stored events; changed content under an existing
identity is refused. See [conversation export](decisions/explicit-conversation-export.md).

The import canary checks embedding identity periodically and at completion. A mismatch stops
the run and reports suspect IDs already stored. It does not repair their vectors; replay skips
them. A transient wrong vector between successful checks can escape detection.
`doctor --vectors` samples stored vectors; it is not a complete repair operation.

### Errors and retries

Under `--json`, failures use the JSON error envelope on stdout; diagnostics go to stderr.
Invalid configuration exits 1 without creating a database. Provider failures name the endpoint
and relevant setting, with bounded, credential-redacted excerpts.

Embedding calls retry transient failures within configured budgets: 60 seconds interactively,
600 seconds for import, and five seconds when no connection can be made. Refusals such as
401, 403, redirects and 501 are not retried. Chat calls are single-attempt; partial streaming
output is not replayed after failure.

## 6. The MCP server

The seven tools are `remember`, `recall`, `facts`, `forget`, `ask_morgan`, `evidence`
and `inspect_context`. Pass `project` explicitly. Omitting it uses `personal`, including
for `forget`. The configured owner applies to all tool calls.

| Tool behavior | Tools |
| --- | --- |
| Read-only | `recall`, `facts`, `evidence`, `inspect_context` |
| Writes | `remember`, `ask_morgan` |
| Destructive, with snapshot | `forget` |

```bash
morgan-mcp --transport stdio
morgan-mcp --transport http
```

HTTP defaults to loopback on port 8090. A non-loopback bind requires a real `MORGAN_API_KEY`.
For client configuration, transport protection and explicit remote binding, see
[operations](OPERATIONS.md#mcp-clients). No service is installed by these commands.

## 7. Teach your agents: `morgan install-skill`

```bash
morgan install-skill
morgan install-skill --mcp-server brain
```

The installer previews its changes and asks before writing. `--yes --json` supports explicit
scripted installation. It writes the packaged usage skill for detected Claude Code, Codex,
OpenCode and Cursor installations, and supports OpenResearch's upload store.

Claude Code settings permit the four read-only tools. Write tools retain their existing
permission boundary. An unrecognized skill or unparseable settings file is left alone.
Run the installer again after an upgrade. For OpenResearch with a custom data directory,
set `ORX_DATA_DIR`; its sessions need `morgan` on PATH for the CLI fallback.

## 8. Docker

The Compose service publishes port 8090 and mounts `./data` at `/app/data`. Create a local
`.env` beside the Compose file with reachable model endpoints, matching model IDs/dimension
and a real `MORGAN_API_KEY`. Container localhost is not the host's model server.

```bash
docker compose build
docker compose up -d
```

The image binds to `0.0.0.0` and refuses startup without the key. Protect the published port
with your network policy. To upgrade with a migration:

```bash
docker compose build
docker compose stop morgan-mcp
docker compose run --rm morgan-mcp morgan migrate
docker compose up -d
```

Stop other processes using the same database as well. Snapshots remain in `./data/snapshots`.
The [dependency audit](measurements/2026-11-dependency-audit.md) records the image's known
unresolved findings; a pinned base image is not a claim of zero vulnerabilities.

## 9. Consolidation on a schedule

Morgan does not schedule consolidation. An owner-managed scheduler can invoke
`morgan consolidate --all-projects --json` with an absolute executable path and explicit
configuration. That command sends eligible memory text to the configured model endpoint.
Account for logs, failures and the per-project consolidation switch in the surrounding job.
