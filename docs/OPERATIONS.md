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
`Authorization: Bearer $MORGAN_API_KEY`. Expose the service over Tailscale, or terminate TLS at
a reverse proxy. Never expose it on a public interface with the default key.

## Backups

Back up the single database file with `sqlite3 morgan.db ".backup 'morgan-backup.db'"` while the
service runs — a filesystem copy of a WAL-mode database mid-write is not consistent.
