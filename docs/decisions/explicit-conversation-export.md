# Explicit conversation export memory

The existing ChatGPT JSON importer accepts only an explicitly selected file. It does not
look for sessions, inspect agent directories, install hooks, or capture live conversations.
Use a new isolated data directory for a first trial. No new database layer or service is
introduced: all writes and search use MemoryGate and the existing SQLite indexes.

## Accepted evidence

Each conversation must supply a stable `conversation_id` (or `id`) and a `mapping` object.
The entire list is checked for these fields before writing. Each accepted message must
supply a stable `id`, a user/assistant author, and nonempty string parts of text content.
Missing message IDs are skipped; identities are never generated from list positions.

Assistant answers require an explicit `final` channel (message or metadata), or both
`end_turn: true` and `status: finished_successfully`. Any non-final channel, tool recipient,
tool/function call, named internal author, hidden message, or unfinished answer is excluded.
Unknown finality is skipped even if this loses older answers. This is a structural filter
for the supplied format, not a detector of reasoning disguised as ordinary final text.
Non-text objects are not stored. Unknown content formats must get a reviewed adapter.

New memories record `session_id = chatgpt:<conversation-id>`, source event time when supplied,
import origin, and actor attribution. A `user` role does not identify a human in a shared account. New user-role text has
`source=unknown` and empty `author_id`; `user_id` controls access, not authorship.
`unresolved_user_turns` reports accepted user-role turns, including unchanged replays,
without counting chunks separately. Text, names and conversation titles never establish identity.
Assistant answers are
`agent_inferred`, authored by `chatgpt:assistant`, never owner commitments. Existing memory
IDs remain SHA256(`chatgpt:<message-id>#<part>`) truncated to 32 hex characters, allowing a
source export to locate each split part. The export itself is not retained automatically.
All supplied branches are independent evidence; sorting is presentation, not inferred ancestry.

## Replay and search

Unchanged reimports add no memories and do not re-embed. Reusing a message identity with
changed content, actor, or known conversation provenance fails closed. Legacy records with
empty session IDs are not retroactively attributed. Previously owner-attributed user records
now raise an identity conflict on replay; they are not silently accepted or rewritten.
Existing assistant compatibility is unchanged. Old records
that a previous importer wrongly accepted are not silently deleted or reclassified.
An interrupted import may have committed earlier messages; replay safely resumes them.

The existing archive/holdout project split remains: roughly 20% of conversations are in
`archive/chatgpt-holdout`, separate from `archive/chatgpt`. Search an explicit scope, or use
existing all-project retrieval deliberately. This change does not wire the entire archive
into the pilot's bounded inbox search or make every recalled statement current truth.

Import uses the configured embedding backend. For an offline plumbing trial explicitly set
`MORGAN_EMBEDDING_BACKEND=hash` and use a fresh data directory; hash embeddings do not measure
semantic quality. With the provider backend, embedding text goes to that configured provider;
review configuration before importing personal data. No chat-model generation is needed.

## Reproducible measurement, 2026-11-06

A synthetic eight-message export had a user note, a final answer, analysis, commentary, a
tool call, a tool result, an answer of unknown finality, and a hidden user message. Before:
seven stored, including five excluded markers, no conversation provenance. After: two stored,
six skipped, zero excluded markers, both with `chatgpt:c2` provenance. Reimport: zero new
memories and zero embedding calls. Russian keyword `билет` retrieves both correctly attributed
quotes through existing recall. This measures filtering, replay and search plumbing only.
No real export, private session, model service, or active pilot data was used or changed.


## Shared-account validation

Two synthetic user-role messages naming different speakers remain searchable with unknown
source and empty author IDs. Both share verified export conversation provenance and remain
isolated to the importing account. Replay reports two unresolved turns and no new memories.
Even syntactically valid commitment JSON from that export is rejected by the existing daily
planner's user-stated evidence requirement. No author-mapping or bulk promotion option is
provided; verified human authorship needs separate explicit evidence, not an account-wide guess.

Existing wrongly attributed rows are not repaired by this release. A replay conflict preserves
the database and requires review. Use a separate fresh database for an unverified shared export;
do not treat the unchanged old archive as corrected. No real export was supplied for this test.

A same-input before/after run confirms the boundary: the previous importer labelled a
synthetic user-role commitment `user_stated` / importing owner and the planner accepted it.
The corrected importer stores `unknown` / empty author; search still retrieves it, while the
planner rejects it. No model or live-pilot data was used.
