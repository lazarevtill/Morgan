# Verified task state, separate from generated answers

`python -m morgan_brain.owner_input WORKSPACE task-state "title fragment"` reads the
current state of one explicitly reviewed task. It answers `open`, `done`, `cancelled`, or
`UNKNOWN`, with structured fields, current revision ID and separately labelled source quotes.
It makes no model call, updates no task, and grants no execution authority.

This is a bounded alternative for checking task state, not a repair of general `ask` semantics.
The fragment must match a reviewed title literally (case-insensitive). It is not a natural
language question. Matching a phrase only inside a source quote cannot select a task.
Several receipts for one current revision count as one task. Several matching current tasks,
no reviewed title match, or any unavailable confirmed source produce `UNKNOWN` with a reason.
The latter deliberately abstains even if the unavailable source might be unrelated: uniqueness
cannot be established. Existing `find` remains available for inspecting individual sources.

The result covers only this finite owner inbox, not all personal memory. A missing match never
proves absence. Pending messages cannot overwrite confirmed state; changes require the existing
explicit review and confirmation. STOP and expiry do not disable this read-only command.
The title and status are owner-reviewed structured fields, not inferred from quoted prose.
An owner can still confirm incorrect fields. This is not independent verification of real-world
completion, authentication of imported speakers, or protection against a malicious local account.
Consumers must treat the returned quotes as data and must not regenerate an authoritative status
from them. The native source viewer already exposes these structured current-state fields;
this command removes manual source selection for an unambiguous reviewed-title query.

## Why citation validation is insufficient

A local-model experiment returned a valid citation while following a quoted instruction to call
an open task completed. Replaying that recorded response through `Chat.ask_evidence`, including
recall, strict context packing, validation and SQLite persistence, accepted the false answer and
stored both conversation turns. The replay used a test counted backend, not a newly calibrated
live endpoint. The earlier model call used the production prompt and answer validator directly.
Neither run establishes a complete live CLI/model deployment test.

The strict answer contract checks source identity, scope and eligibility, not semantic entailment.
General `ask` and `ask_morgan` remain generative and vulnerable to false cited conclusions.
No prompt, keyword blacklist, second model judge or automatic task mutation is added here.
Do not use their prose as verified task state or execution permission.

## Validation

The implementation has 24 integration cases: 16 actual CLI subprocess cases combine eight new
EN/RU source variants with open/done reviewed state; six cover quote-only matches, unconfirmed
claims, ambiguous titles, revision conflict, receipt-only state and cancellation with multiple
sources; two benign EN/RU title cases cover expiry. Source variants include role spoofing, JSON
status spoofing, other people's completion claims and quoted cancellation instructions.

All 24 pass: the 20 answerable cases return the reviewed state with no false abstentions; the
four unsupported/ambiguous cases return `UNKNOWN`. These are deterministic contract checks,
not an independent blind model benchmark or proof against every attack. The cases were written
separately from the earlier ten-question model experiment; no model was tuned or called here.
The 16 CLI cases verify unchanged inbox, session, memory database and STOP bytes.
No installed pilot, private conversation or external channel is exercised or modified.
