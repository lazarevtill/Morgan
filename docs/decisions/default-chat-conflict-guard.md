# Default chat stops at unresolved revision forks

Default Chat formerly rendered only recalled content into the model prompt. When two active corrections revised the same event, both could be recalled while their unresolved-fork metadata was omitted. A model could then choose whichever assertion appeared later. The original 12-case default acceptance pilot remains frozen and NO_GO; this change does not replace or regrade it.

Default `Chat.ask(strict_context=False)` now checks the existing recalled `revision_state`. Any `conflicted` record produces a bounded constant clarification before chat generation. The notice includes no source values, durable IDs or copied instructions. It names the existing recall/evidence/remember route. Russian queries use the existing script-based language helper; other queries receive the English notice. No extra retrieval, counter, schema, service or relevance heuristic is introduced.

The clarification goes through the normal atomic turn write and erasure-generation check. Its assistant memory is `agent_inferred` with reported author `morgan:conflict-guard`, because no model generated it. Normal generated answers retain `model:<configured model>` attribution. Default nonconflict prompts/options and the strict answer contract retain their existing behavior.

This is a conservative whole-turn guard: even an unrelated fork among recalled records blocks the answer. Morgan does not have reliable claim-to-question relevance validation in the default path. This small change prevents arbitrary branch selection; selective conflict answering remains a separate evaluated design. A fork absent from recall is outside this guard's coverage. No broader answer-quality gain is claimed.

To resolve a fork in a named owner/project scope, obtain the original question's recall records, inspect their revision metadata and exact sources, and ask the user/source author for a supported correction. For example:

```sh
morgan recall "Which glaze did I choose?" --project personal --json
morgan evidence amber cobalt --project personal --json
morgan remember "I chose jade glaze" --project personal --event-id glaze-join \
  --source user_stated --author-id person --effective-at 2026-10-01T00:00:00Z \
  --revises-event-id amber --revises-event-id cobalt --json
```

Here `amber` and `cobalt` stand for the actual current conflicting leaf IDs returned by recall/evidence. The correction must preserve the actual source/author lineage; `user_stated` requires an actual user statement. MCP clients use the existing `recall`, `evidence`, and `remember` tools with the same explicit project and `revises_event_ids` list. Check every eligible leaf and truncation diagnostic before resolving; respect the existing bounded revision API rather than silently omit parents. Once a supported join resolves the fork, the next default ask resumes ordinary generation. Reported owner/source/author labels do not establish authenticated per-agent isolation.

Offline integration tests cover real Gate fork metadata, RU/EN clarification, unrelated-fork refusal, user/agent provenance, normal generation after an explicit join, unchanged nonconflict wire, embedding failure and concurrent erasure. Existing strict and atomic-chat tests remain required. No endpoint or live memory is used by these tests.
