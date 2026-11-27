# Morgan

Project-scoped memory for AI tools, with a CLI and an MCP server.

Morgan stores source records, retrieves them by meaning and keyword, and uses an explicitly
invoked model to derive dated facts. The memory core keeps its data in one SQLite database.
Optional planning and task workflows use separate journals and require explicit review or approval.

## Capabilities

| Operation | Behavior |
| --- | --- |
| Capture | Store attributed source text with stable IDs and immutable corrections. |
| Retrieve | Combine sqlite-vec and FTS5 rankings within an owner and project. |
| Inspect | Read exact source IDs, provenance and revision state without a model call. |
| Answer | Recall context, call the configured chat model and store the exchange. |
| Consolidate | Derive facts from eligible evidence; retain prior validity intervals. |
| Recover | Snapshot, migrate and restore the database; erase a project with a snapshot for undo. |

The CLI derives the project from the enclosing Git repository, or uses `personal` outside one.
MCP clients must pass the project explicitly; the server cannot infer the client's repository.
`--all-projects` enables cross-project reads where supported. Two repositories with the same
folder name share a project unless the caller chooses distinct names.

## Local development quick start

Requires Python 3.12+ and a compatible chat/embedding endpoint. CI and the Docker image use
Python 3.12.16. Run these commands from the checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/morgan"
mkdir -p "$config_dir"
test -e "$config_dir/.env" || cp .env.example "$config_dir/.env"
```

Edit that file before running Morgan. Set the endpoint URLs, served model IDs and embedding
dimension to match your server. Separate chat and embedding servers require
`MORGAN_EMBEDDING_ENDPOINT`. See [configuration and model setup](docs/WIRING.md).

```bash
morgan doctor
morgan remember "The demo uses PostgreSQL" --project demo --source user_stated
morgan recall "Which database does the demo use?" --project demo
morgan evidence SOURCE_ID --project demo --json
```

Replace `SOURCE_ID` with the ID returned by `remember`. Capture and recall call the embedding
endpoint. `evidence` reads an existing database without model calls. The deterministic
`MORGAN_EMBEDDING_BACKEND=hash` backend supports offline checks in a separate scratch database;
its vectors do not provide semantic similarity.

The default database is `~/.local/share/morgan/morgan.db`. An editable installation follows
checkout changes. For a deployed pilot, build a [pinned runtime](docs/decisions/pinned-runtime.md)
from an exact trusted commit and verify it before use.

## Connect an MCP client

Morgan exposes seven tools over stdio or streamable HTTP: `remember`, `recall`, `facts`,
`forget`, `ask_morgan`, `evidence` and `inspect_context`. Four are read-only: `recall`, `facts`,
`evidence` and `inspect_context`. `ask_morgan` stores the exchange.

Configure your client to launch the installed executable. With the virtual environment above:

```json
{
  "mcpServers": {
    "morgan": {
      "command": "/absolute/path/to/Morgan/.venv/bin/morgan-mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

Replace the executable path with your installation. Pass the intended project on each call.
For HTTP authentication and remote access, see [operations](docs/OPERATIONS.md).
`morgan install-skill` previews and installs usage guidance for supported local coding agents.

## Explicit workflows

| Workflow | Guide |
| --- | --- |
| Trusted task handlers, finite batches and recovery receipts | [Task execution](docs/decisions/explicit-task-execution.md) |
| Approved local notes, delivery windows and bounded supervision | [Local notes](docs/decisions/local-approved-notes.md) |
| Reviewed commitments, current plans and completion | [Personal planning](docs/decisions/current-personal-plan-pilot.md) |
| Local owner messages and macOS review | [Owner input](docs/decisions/local-owner-input.md) |
| Finite local model drafting | [Model lifecycle](docs/decisions/bounded-model-cycle.md) |
| Explicit local alerts and reviewed push messages | [Owner channels](docs/decisions/owner-channel-hosting.md) |

These workflows do not install services or renew their authority. Memory content does not
authorize execution. A push server accepting a message does not prove that a person received it.

## Verification and limits

The [November reliability audit](docs/measurements/2026-11-reliability-recheck.md) records
2,223 passing tests and four skips on its final PR revision, plus live Ollama checks.
The [dependency audit](docs/measurements/2026-11-dependency-audit.md) lists verified versions,
security fixes and unresolved upstream findings. These are dated results, not a certification
of every deployment or dependency version.

Retrieval still has measured limits in stale-answer ranking and multi-hop questions. Explicit
revision links filter superseded sources, but Morgan does not infer those links from prose.
Native model answers remain unverified; valid citations do not establish truth or permission.
See [current limitations](docs/ROADMAP.md) and the [evidence index](docs/measurements/README.md).

## Documentation

- [Documentation index](docs/README.md): installation, operation, integration and design records.
- [Architecture](docs/ARCHITECTURE.md): storage, retrieval and write boundaries.
- [Development](CONTRIBUTING.md): setup, checks and documentation conventions.
- [Agent guidance](AGENTS.md): repository invariants and implementation constraints.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
