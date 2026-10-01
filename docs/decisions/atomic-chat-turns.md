# Atomic native chat turns

A native `ask` previously committed both history rows before embedding either source
event. A failed assistant embedding could leave a transcript and only the input memory.
A forget during recall or generation was also followed by new store calls that captured
the newer erasure generation and recreated the old in-flight turn.

Chat now captures the global erasure generation before its first await. Both events are
prepared outside SQLite's write transaction, using the same preparation/persistence helpers
as a single-event store. A bounded turn operation writes exactly two new events, all their
indexes, and two matching history rows under one existing transaction. It checks the captured
generation and immutable source boundaries again under the lock before persistence.

An embedding, identity, index or history failure commits no part of the turn. A committed
forget cancels the in-flight turn with the existing explicit retry error. A fresh call after
forget remains possible. The model may already have generated an answer, but `ask` does not
return a successful durable turn until its persistence transaction commits.

Turn history must share the memory connection, owner, context and session. Preparation
refuses an existing caller-owned transaction because embeddings must not await under its
lock. This operation rejects previously stored event IDs; it does not treat generic event
replay as permission to append duplicate history. Native chat retry IDs are not introduced.
Single-event immutable replay remains supported.
Both preparation paths refuse an existing transaction before embedding. Input event time
is captured before recall/generation; reply event time is captured after generation.
Event recorded time remains server-owned at insertion. Session history keeps its existing
injected-clock recording timestamp and insertion ordering; no historical rows are rewritten.

Reported input source defaults to UNKNOWN; assistant output remains AGENT_INFERRED with
`model:<configured model>` authorship. Reported metadata does not authenticate a caller.
No database migration, token counter, evidence packing, prompt-role change or citation
validation is part of this change. Those require separate measured acceptance slices.

Native chat reads history with the explicit owner predicate before its limit. Existing
colon-separated session keys may collide for different owner/session pairs; retaining
those keys requires owner filtering rather than trusting the encoded key as authorization.
No stored keys or historical rows are rewritten. Low-level callers may keep their existing
unfiltered ``recent`` behavior by omitting its optional owner argument.
