# Operations

This runbook covers the memory database and MCP server. Optional task and owner workflows
keep separate journals; backing up or forgetting memory does not manage those journals.
Use each workflow's [decision record](decisions/README.md) for STOP and receipt recovery.

## Protect stored data

The SQLite database contains source text, facts, vectors, search indexes and session history.
Snapshots are complete copies. Morgan provides no field-level encryption; protect the volumes
holding the database and snapshots with host encryption and account access controls.

New memory admission rejects a narrow set of explicit credential shapes. It does not scan
all metadata, historical data, backups or separate task journals. See
[credential admission](decisions/credential-admission.md).

## MCP clients

### Local stdio

Configure the client with the absolute path to the installed `morgan-mcp` executable and
arguments `["--transport", "stdio"]`. This avoids depending on a desktop application's PATH.
The [README](../README.md#connect-an-mcp-client) has a configuration example.

With Claude Code and an executable already on PATH:

```bash
claude mcp add -s user morgan -- morgan-mcp --transport stdio
```

Registration scope controls where Claude Code sees the server. It does not select Morgan's
memory project. Pass the intended project on each tool call; omitting it selects `personal`,
including for `forget`, which erases that scope after taking its undo snapshot.

### Remote HTTP

Set a real `MORGAN_API_KEY` in the server environment, then bind explicitly:

```bash
morgan-mcp --transport http --host OVERLAY_ADDRESS
```

Replace `OVERLAY_ADDRESS` with the server's interface address. Port defaults to 8090.
Clients use `/mcp` and `Authorization: Bearer TOKEN`, with the same token. Keep credentials
out of checked-in client configuration. Follow the client's supported secret-storage mechanism;
a JSON string containing a shell variable is not automatically expanded by every client.

For Claude Code behind a TLS proxy on port 443, with the token supplied in your shell environment:

```bash
claude mcp add -s user --transport http morgan https://YOUR_HOST/mcp \
  --header "Authorization: Bearer ${MORGAN_API_KEY:?Set MORGAN_API_KEY first}"
```

The shell expands this variable before registration; treat the resulting client configuration
as credential-bearing. Replace `YOUR_HOST` with the TLS proxy hostname. For a direct encrypted
overlay connection to the listener above, use `http://OVERLAY_ADDRESS:8090/mcp` instead;
Morgan itself serves plain HTTP, so that alternative depends on overlay encryption.

A key is required beyond loopback. With a configured key, HTTP requests require it on loopback
too. Without a key, loopback access relies on the local account/host boundary.
Morgan does not terminate TLS: use an encrypted overlay or a TLS reverse proxy and restrict
network access. A bearer token authenticates access to the configured owner; a supplied
`author_id` is only provenance.

## Backups

```bash
morgan snapshot
morgan snapshot --list
```

Snapshot uses `VACUUM INTO` and verifies the copy with `PRAGMA quick_check`. It produces a
consistent backup while other processes use the database. Copying only the live database file
while WAL writes are active is not an equivalent backup.

Default snapshot directory: `MORGAN_DATA_DIR/snapshots`, overridden by
`MORGAN_SNAPSHOT_DIR`. An explicit `MORGAN_TEMPORAL_DB_URL` does not relocate snapshots.
Keep an independently stored copy if you need recovery from host loss.
Morgan never prunes verified snapshots. Set your own retention policy: snapshots can retain
content later removed with `forget`.

## Upgrade and recover

1. Record the active runtime revision, configuration and database location. Build and verify
   the new [pinned runtime](decisions/pinned-runtime.md) without activating it.
2. Stop every process using the database, including MCP clients that may restart a stdio server.
   Do not mix older writers with a newer schema or workflow journal.
3. With the active runtime, run `morgan snapshot` and retain that pre-upgrade copy before any
   ordinary database open by the new runtime. Keep the old runtime and its configuration for rollback.
4. With the intended new runtime, run `morgan doctor` and `morgan migrate --dry-run`.
   Review pending steps. Doctor is read-only. The migration dry-run applies no schema or data
   migrations, but its writable SQLite open can switch a restored file to WAL journal mode.
5. Run `morgan migrate` when required. It takes its own snapshot, applies pending steps in one
   transaction and checks the result. A failed migration step rolls back the wave. After commit,
   when `MORGAN_EMBEDDING_BACKEND=provider`, migrate checks identity at the embedding endpoint.
   A failed post-commit identity check does not undo the migration. Resolve it before resuming use.
6. Check doctor and the relevant scoped operations before restarting clients. Verify journal
   reader compatibility separately when upgrading owner/task workflows.

Light migrations can run on ordinary writable open. Heavy steps block writes until explicit
migration. Read-only evidence and doctor do not migrate. Newer-than-supported schemas are refused.

To restore, keep all database users stopped:

```bash
morgan restore /absolute/path/to/snapshot.db
```

The preview exits 2. Review the named database and snapshot before confirming:

```bash
morgan restore /absolute/path/to/snapshot.db --yes
```

Confirmation verifies the snapshot, creates a before-restore snapshot and replaces the database
and WAL/SHM state. Concurrent writers can lose writes against the replaced file; online restore is
unsupported.
Restore with a compatible runtime and the original embedding configuration.

Memory restore does not roll back separate task receipts or external effects. Follow the
workflow's recovery contract before resuming any execution.

## Erasure and retained copies

`morgan forget --project NAME` snapshots first, then erases registered data for that owner/project.
After commit it vacuums the database and requests WAL truncation. Another reader can block the
checkpoint; Morgan reports the warning and a later checkpoint is needed. This is functional
database erasure, not certified removal from storage media, exports or backups.

## Security and verification records

The [dependency audit](measurements/2026-11-dependency-audit.md) lists unresolved container,
upstream tooling and host findings. The
[reliability report](measurements/2026-11-reliability-recheck.md)
states which transports, recovery paths and model calls were tested. Neither report means an
installed pilot has been upgraded or a physical device has received a notification.
