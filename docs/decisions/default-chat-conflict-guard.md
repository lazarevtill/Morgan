# Default chat stops at unresolved revision forks

Default Chat formerly rendered only recalled content into the model prompt. When two active
corrections revised the same event, both could be recalled while their unresolved-fork metadata
was omitted. A model could then choose whichever assertion appeared later. The original 12-case
default acceptance pilot remains frozen and NO_GO; this change does not replace or regrade it.

Default `Chat.ask(strict_context=False)` now checks the existing recalled `revision_state`. Any
`conflicted` record produces a bounded constant clarification before chat generation. The notice
includes no source values, durable IDs or copied instructions. It names the existing
recall/evidence/remember route. Russian queries use the existing script-based language helper;
other queries receive the English notice. No extra retrieval, counter, schema, service or
relevance heuristic is introduced.

The clarification goes through the normal atomic turn write, erasure-generation check and commit-time
revision-basis validation. Every recalled default record is revalidated inside the write
transaction: a concurrent fork or resolution refuses the stale turn without partial persistence. Its
assistant memory is `agent_inferred` with reported author `morgan:conflict-guard`, because no
model generated it. Normal generated answers retain `model:<configured model>` attribution.
`Chat.ask` retains its string return. The per-call immutable `ask_with_provenance` result
lets CLI/MCP report the program author and `model_used: null` for notices without shared state.
Default nonconflict prompts/options and the strict answer contract retain their existing
behavior.

This is a conservative whole-turn guard: even an unrelated fork among recalled records blocks
the answer. Morgan does not have reliable claim-to-question relevance validation in the default
path. This small change prevents arbitrary branch selection; selective conflict answering
remains a separate evaluated design. A fork absent from recall is outside this guard's coverage.
No broader answer-quality gain is claimed.

To resolve a fork in a named owner/project scope, obtain the original question's recall records,
inspect their revision metadata and exact sources, and ask the user/source author for a
supported correction. For example:

```sh
morgan recall "Which glaze did I choose?" --project personal --json
morgan evidence amber cobalt --project personal --json
morgan remember "I chose jade glaze" --project personal --event-id glaze-join \
  --source user_stated --author-id person --effective-at 2026-10-01T00:00:00Z \
  --revises-event-id amber --revises-event-id cobalt --json
```

Here `amber` and `cobalt` stand for the actual current conflicting leaf IDs returned by
recall/evidence. The correction must preserve the actual source/author lineage; `user_stated`
requires an actual user statement. MCP clients use the existing `recall`, `evidence`, and
`remember` tools with the same explicit project and `revises_event_ids` list. Check every
eligible leaf using `eligible_leaf_count` and `revision_truncated` before resolving. Each
remember correction accepts at most **8 parent IDs**. For a larger family, make a supported
join of up to 8 verified leaves, then join that new leaf with up to 7 remaining leaves; repeat
until all branches are covered. Re-read current metadata after each step. A truncated inventory
is not a complete family: retrieve the missing exact records before claiming resolution.
Never silently omit a branch or fabricate a statement to bypass the parent limit. Once a
supported join resolves the fork, the next default ask resumes ordinary generation. Reported
owner/source/author labels do not establish
authenticated per-agent isolation.

Offline integration tests cover real Gate fork metadata, RU/EN clarification, unrelated-fork
refusal, user/agent provenance, normal generation after an explicit join, unchanged nonconflict
wire, embedding failure and concurrent erasure. Existing strict and atomic-chat tests remain
required. No endpoint or live memory is used by these tests.


Guard turns use the additive `origin_kind: ask_conflict_guard` on both input and notice.
Repeated procedural questions and notices are excluded from competitive vector/keyword recall
**before candidate limits** so they cannot displace the unresolved sources that prompted them.
The input's source, author, text and history remain durable; exact get/evidence, export and replay
retain both records. Consolidation recalls through the same eligibility filter and does not turn
these procedural turns into semantic facts. Ordinary ask/remember origins retain their behavior.

This deliberately excludes even a substantive user assertion inside a guarded turn from ranked
recall. Use explicit `remember` to record such an assertion as competitive evidence, preserving
its true source and actor. No input is erased or relabeled as agent-authored. This origin is an
additive contract value, requiring current Morgan to import/replay; older clients that exhaustively
validate origin enum values may refuse it and should update rather than silently rewrite it.
Unrelated stored evidence can still crowd a fork outside recall; this change prevents the guard's
own persisted turns from doing so and does not claim globally exhaustive conflict discovery.
