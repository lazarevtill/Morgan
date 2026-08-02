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
