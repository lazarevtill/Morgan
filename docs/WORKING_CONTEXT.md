# Continuing personal work

Working contexts are optional, named organizers for work that crosses sessions or
agents. They select exact source quotations for decisions and their reported reasons,
open questions, intentions, and reported progress. They do not replace durable events
or turn model classifications into facts or action permissions.

These commands default to `personal`, including inside a repository. Pass the same
explicit `--project` to every command when organizing project work.

```console
morgan recall "our gift zine" --project personal --json
morgan evidence SOURCE_EVENT_ID --project personal --json
morgan context propose gift-zine --event-id SOURCE_EVENT_ID --json > proposal.json
morgan context apply proposal.json
morgan context list
morgan context show gift-zine
morgan context resume gift-zine "Continue what we discussed; draft the next page." --session-id fresh-agent-session
```

Review and edit the proposal before applying it. Quotations must remain unique exact
spans of their original events; offsets and evidence snapshots are checked again at
apply time. A changed head, corrected source, or forget operation rejects an obsolete
proposal. Proposed reason links and category choices remain unverified.

Continuation requires an unused session ID; `default` is reserved. An occupied key is
rejected again atomically at commit, including occupation during model or embedding
work. Continue subsequent work with another fresh session and the same context ID.
Continuation produces a draft and persists the user/agent turn atomically. It has no
action tools. Its source-basis IDs identify the organizer's inputs, not independently
validated citations for every sentence in the generated draft. Reported USER, AGENT,
and TOOL authorship is preserved; caller labels are not authentication.

A correction can invalidate a view. Listing keeps its name visible while showing
`needs_rebuild`; reading hides stale quotations. Rebuild explicitly from current
source IDs, review, and apply:

```console
morgan context propose gift-zine --rebuild --event-id CURRENT_SOURCE_ID --json > proposal.json
morgan context apply proposal.json
```

The MCP equivalents are `working_context_read`, `working_context_propose`,
`working_context_apply`, and `working_context_resume`. Propose is read-only but uses
the configured model and is excluded from installer auto-approval; apply and resume
write. Resume requires an explicit fresh session ID.
Personal and project scopes are independent of the reported client or author.

Versioned contracts use `morgan.working_context.v1`,
`morgan.working_context.preview.v1`, `morgan.working_context.result.v1`,
`morgan.working_context.list.v1`, and `morgan.continuation.v1`. Existing recall and
consolidation exclude organizer JSON. No database migration or new service is needed;
the organizer uses existing durable fact, evidence, revision, and erasure contracts.

Bounds: 20 new source IDs per proposal, 16 selected source IDs, four items per
category, 240 characters per quote, 16 KiB serialized state, and 49 KiB model-input
bytes. Listing returns at most 32 names with a truncation flag. These byte limits are
not a tokenizer budget: real token cost and usefulness require measured evaluation.

## Experimental evaluation status

This opt-in prototype has no demonstrated continuation-quality advantage and is not
approved for deployment or merge on that basis. A frozen synthetic comparison used
four closed continuations per arm: ordinary retrieval was useful in 1/4, rolling
summary in 1/4, and working context in 0/4. All failures stayed in the denominator;
no failed slot was retried and no holdout prompts were tuned. Independent grading
was not fully blind because context classifications leaked during packet extraction.

Three candidate continuations were unavailable after rejected maintenance proposals
(schema shape, exact quote, or malformed JSON). Its one persisted continuation omitted
required older layout and measurement grounds. Exact-source validation establishes
attribution of selected quotes, not completeness of the selected working context.
The fixed all-four-useful acceptance gate failed. Missing usage from failed requests
also prevents claiming an efficiency advantage.

Correctness fixes for consolidation's structural-fact inventory and reported backend
attribution are separate from product usefulness. The next design question is how a
bounded update retains or recovers essential earlier source grounds after maintenance
failure, rather than accepting invalid proposals or adding unbounded benchmark runs.
