# Operations

## At-rest protection

Everything is in one SQLite database under `MORGAN_DATA_DIR` (default
`~/.local/share/morgan/`): memories, facts, vectors, the keyword and entity indexes, the
semantic index, session history. There is no field-level encryption: it cannot coexist with
the FTS5 index and would not cover vectors. At-rest protection is a property of the host:
encrypt the volume (LUKS or the equivalent).

## Transport protection

`morgan-mcp --transport http` requires `Authorization: Bearer $MORGAN_API_KEY`. Reach it over
your overlay network or terminate TLS at a reverse proxy. It will not start on a non-loopback
address without a real key.

## Backups

```bash
sqlite3 ~/.local/share/morgan/morgan.db ".backup 'morgan-backup.db'"
```

Safe while the server runs; a filesystem copy of a WAL-mode database mid-write is not
consistent. One file is the whole backup, and the whole exposure.

## MCP clients

### On the same machine — stdio

```bash
claude mcp add morgan -- morgan-mcp --transport stdio
```

```json
{ "mcpServers": { "morgan": { "command": "morgan-mcp", "args": ["--transport", "stdio"] } } }
```

### On another machine — streamable-HTTP

Run `morgan-mcp --transport http --host <overlay address>` (or set `MORGAN_MCP_HOST`; the port
is `MORGAN_MCP_PORT`, default 8090) with `MORGAN_API_KEY` set.

```bash
claude mcp add --transport http morgan http://<host>:8090/mcp \
  --header "Authorization: Bearer $MORGAN_API_KEY"
```

```json
{
  "mcpServers": {
    "morgan": {
      "url": "http://<host>:8090/mcp",
      "headers": { "Authorization": "Bearer $MORGAN_API_KEY" }
    }
  }
}
```

## The stack

All open source and self-hostable, which is the point: a memory that only works while someone
else's service is up is not yours.

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT). `llama-server` serves chat,
  `/v1/embeddings` and native JSON-schema constrained output from one binary on one GPU.
- **[SQLite](https://sqlite.org) + FTS5** (public domain). One file holds everything, which is
  what makes `forget()` one transaction, backup one command, and encryption one volume. FTS5's
  `unicode61` tokenizer indexes non-Latin scripts correctly.
- **[sqlite-vec](https://github.com/asg017/sqlite-vec)** (MIT / Apache-2.0). Vectors in the same
  file, filtered *inside* the nearest-neighbour search via vec0 metadata columns; filtering
  afterwards silently drops results that should have been returned.
- **[NetBird](https://netbird.io)** (BSD-3-Clause), or any overlay network you run yourself, to
  reach the machine that holds the database without opening a public port.
