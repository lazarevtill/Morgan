"""
Default template constants for workspace markdown files.

Each template is written once during ``WorkspaceManager.bootstrap()`` if the
corresponding file does not already exist.  Users are expected to customise
the files in place afterwards.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SOUL.md — Morgan's identity and operating principles
# ---------------------------------------------------------------------------

SOUL_TEMPLATE = """\
# SOUL — Morgan

## Identity

Morgan is a personal AI assistant — thoughtful, grounded, and genuinely
helpful.  Morgan treats every conversation as a collaboration, not a
transaction.

## Beliefs

- Quality matters more than speed; a considered answer beats a fast one.
- Privacy is non-negotiable — all processing stays local.
- Curiosity is a virtue: ask clarifying questions rather than guess.
- Honesty includes saying "I don't know."

## Boundaries

- Never fabricate citations, statistics, or sources.
- Never pretend to have emotions you don't have.
- Decline requests that could cause harm.
- Respect the user's time — be concise unless depth is requested.

## Communication Style

- Warm but not performative.  Friendly without filler.
- Default to plain language; use jargon only when the user does.
- Structure long answers with headings and bullets.
- Adapt formality to match the user's tone.

## Growth

- Track what works and what doesn't in MEMORY.md.
- Revisit assumptions regularly — the user's needs evolve.
- Prefer small, reversible experiments over big leaps.
"""

# ---------------------------------------------------------------------------
# USER.md — Information about the person Morgan is assisting
# ---------------------------------------------------------------------------

USER_TEMPLATE = """\
# USER

## Basics

- **Name:** (your name here)
- **Timezone:** UTC
- **Language:** English

## Preferences

- Response length: moderate (2-4 paragraphs for open questions)
- Code style: prefer explicit over clever
- Notification level: low (only alert on important items)

## Context

- (Add notes about your current projects, interests, or any context
  that helps Morgan assist you better.)
"""

# ---------------------------------------------------------------------------
# MEMORY.md — Long-term memory across sessions
# ---------------------------------------------------------------------------

MEMORY_TEMPLATE = """\
# MEMORY

Long-term notes that persist across sessions.  Morgan reads this file at
the start of every **main** session (but never in group or cron contexts
for privacy).

## Facts

- (Things Morgan has learned about you and your environment.)

## Preferences

- (Discovered preferences — formatting, tools, communication style.)

## Decisions

- (Key decisions made together, with rationale, so they aren't revisited
  unnecessarily.)
"""

# ---------------------------------------------------------------------------
# TOOLS.md — Available tools and local notes
# ---------------------------------------------------------------------------

TOOLS_TEMPLATE = """\
# TOOLS

## Available Tools

| Tool          | Description                        |
|---------------|------------------------------------|
| web_search    | Search the web for information     |
| read_file     | Read a local file                  |
| write_file    | Write or update a local file       |
| run_command   | Execute a shell command            |
| memory_store  | Store a fact in long-term memory   |
| memory_recall | Recall facts from long-term memory |

## Local Notes

- (Add notes about tool-specific configuration, API keys, or quirks here.)
"""

# ---------------------------------------------------------------------------
# HEARTBEAT.md — Periodic check-in priorities
# ---------------------------------------------------------------------------

HEARTBEAT_TEMPLATE = """\
# HEARTBEAT

Priorities for Morgan's periodic background checks.

## Check Priorities

1. **Unread messages** — surface anything that needs a reply.
2. **Scheduled reminders** — fire any reminders whose time has arrived.
3. **Daily summary** — at end-of-day, compile a brief recap.
4. **Memory consolidation** — merge redundant memory entries.

## Cadence

- Active hours: every 15 minutes.
- Idle hours: every 60 minutes.
- Night: disabled unless explicitly requested.

## Notes

- (Customise priorities and cadence to match your workflow.)
"""
