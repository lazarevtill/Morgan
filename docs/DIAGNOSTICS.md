# Diagnostics

Use `morgan doctor` to check configuration, database state and model connectivity. It opens
the database read-only, creates no tables, runs no migration and records no fingerprint.
It does make explicit provider probes. Run it with the same environment as the failing client.

```bash
morgan doctor --project demo
morgan doctor --project demo --json
```

## Read the result

| Field | Interpretation |
| --- | --- |
| `database`, `env_file` | Confirm the database path and configuration files actually selected. |
| `project`, `all_projects` | Check the requested scope before treating zero results as missing data. |
| `data_flow` | Names the endpoint hosts and the text each operation sends. |
| `provider`, `embedding_provider` | Independent chat and embedding probes; one failure does not hide the other. |
| `embedding_space` | Compares five fixed strings with the stored fingerprint; never writes it. |
| `migration` | Stored versus supported schema and pending steps. Current code supports schema 11. |
| `snapshots` | Counts and sizes retained snapshots; Morgan does not prune them. |
| `memories`, `fts`, `vectors` | Scoped counts with all-project totals. |
| `rows_missing_provenance` | Legacy or incomplete metadata; unavailable schemas include a reason. |
| `project` rows | Classification and capture/consolidation switches, not repository credentials. |
| `code_root` | Configured paths and whether they exist. No repository discovery walk runs. |

A fresh installation may have no database yet. Doctor reports that without creating one.
The first ordinary memory context can initialize it. Exact evidence reads require an existing
database and do not initialize one.

## Provider status

| Status | Meaning | Next check |
| --- | --- | --- |
| `reachable` | Probe completed within the slow threshold. | Continue with the intended operation. |
| `slow` | Late answer, throttling or a transient server failure. | Inspect server load/model loading and configured budgets. |
| `refused` | Authentication, unsupported endpoint, redirect or invalid response. | Check the named key or endpoint setting. |
| `unreachable` | Connection failed or the probe deadline expired. | Check address, network route and server availability. |

Defaults are two seconds for the slow threshold and 60 seconds for each probe timeout.
JSON exposes `seconds`, `timeout_seconds`, `slow_after_seconds` and `error` under
`provider_probe` and `embedding_probe`. Hash embeddings report the embedding provider as
unused. Chat status does not establish that the same server supports embeddings.

An `embedding_dim_error` reports configured and stored widths. A fingerprint mismatch can
also detect a model change at the same width. Restore the database's original model/configuration,
or use a new database for the new model. Restoring a snapshot does not convert its vectors.

## Audit stored vectors

```bash
morgan doctor --project demo --vectors --clients 2 --json
```

This re-embeds a sample of up to `MORGAN_VECTOR_AUDIT_SAMPLE_ROWS` stored memories
(default 180). `--all-projects` broadens the sample. The selected text is sent to the
embedding endpoint; a notice goes to stderr.

Each client independently embeds the same sample. JSON `vector_audit` reports `sampled`,
`min`, `median`, `below_tolerance`, `per_client` and `disagreements`. Client timings include
local setup and server work, so slower concurrent results alone do not prove server serialization.
The audit is skipped when the initial embedding probe is refused or unreachable.

`vector_audit_reason` explains an absent audit. A passing sample does not certify every row.
No vectors are repaired or replaced. Import canary failures require investigation of the
reported suspect IDs; simply replaying an import skips unchanged stored events.

## Project classification

A repository-derived CLI write records its remote/root and classifies it using
`MORGAN_WORK_REMOTE_GLOBS`. A read records nothing. MCP-only projects remain unclassified
until such a CLI write occurs. Remote/root values are not printed in doctor.

Git URL rewriting and included Git config files are not followed. Unreadable configuration
leaves existing classification unchanged. Same-named checkouts share the project row.
Configured code roots are listed but not scanned.

For a pending migration, restore, blocked WAL checkpoint or backups, use
[operations](OPERATIONS.md). For known SDK and retrieval limits, use [the roadmap](ROADMAP.md).
