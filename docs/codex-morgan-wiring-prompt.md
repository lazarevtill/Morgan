# Codex Agent Prompt: Map & Wire All Parts in Morgan

> **Purpose:** Hand this prompt to an OpenAI Codex agent (or any agentic coding LLM) so it can autonomously audit, map, and fully wire every module in the Morgan project — drawing patterns from OpenClaw (Telegram/channels/routing/cron) and Claude Code (tools/memory/agents/compaction/skills).

---

## Context

You are working on **Morgan**, a self-hosted personal AI assistant at `Morgan/` inside a monorepo. Morgan is a Python 3.11+ project with three packages:

- `morgan-rag/` — Core intelligence library (`morgan/` package)
- `morgan-server/` — FastAPI server (`morgan_server/` package)
- `morgan-cli/` — Click-based CLI (`morgan_cli/` package)

Morgan was recently extended with **12 new modules** ported from two reference codebases that live in the same monorepo:

1. **Claude Code** (`src/`) — TypeScript CLI for Claude. Contributed patterns for: Tool system (`src/Tool.ts`, `src/tools/`), Agent/subagent spawning (`src/tools/AgentTool/`), Skills (`src/skills/`), Context compaction (`src/services/compact/`), Task management, Hook system, App state store, Memory system.

2. **OpenClaw** (`OpenClaw/`) — TypeScript multi-channel AI gateway. Contributed patterns for: Telegram adapter (`src/telegram/`), multi-channel gateway + routing (`src/channels/`, `src/routing/`), SOUL.md workspace pattern (`AGENTS.md`), heartbeat/cron scheduling, daily memory consolidation, channel allowlists + pairing, plugin SDK.

## Your Mission

**Audit every module, find all broken wiring, missing imports, dead code paths, unconnected interfaces, and stub implementations — then fix them so Morgan runs end-to-end.**

The 12 modules exist on disk but many are partially wired or have placeholder implementations. Your job is to make them all actually work together as a connected system.

---

## Phase 1: Map — Full Module Audit

Read every file in every module below. For each one, determine:
- Does it import correctly? Are all internal cross-references valid?
- Does it have a working `__init__.py` that exports the right symbols?
- Is it actually called from anywhere (server lifespan, assistant, API routes, CLI)?
- Are there stub/placeholder methods that do nothing?
- Are there TODO/FIXME/NotImplementedError markers?

### Modules to audit (all under `morgan-rag/morgan/`):

| # | Module | Key files | Wired into |
|---|--------|-----------|------------|
| 1 | `tools/` | `base.py`, `executor.py`, `permissions.py`, `builtin/*.py` | `core/assistant.py` — tool executor in `ask()` loop |
| 2 | `workspace/` | `manager.py`, `templates.py`, `paths.py` | `morgan-server/app.py` — lifespan bootstrap |
| 3 | `compaction/` | `auto_compact.py`, `compactor.py`, `token_counter.py` | `core/assistant.py` — after each response, check threshold |
| 4 | `memory_consolidation/` | `daily_log.py`, `consolidator.py`, `hybrid_search.py` | `core/assistant.py` — append daily log after response |
| 5 | `channels/` | `gateway.py`, `base.py`, `routing.py`, `telegram_channel.py`, `synology_channel.py`, `discord_channel.py` | `morgan-server/app.py` — lifespan init + agent handler |
| 6 | `agents/` | `base.py`, `spawner.py`, `loader.py`, `builtin/*.py` | Tool executor (AgentTool) or direct spawn from assistant |
| 7 | `skills/` | `loader.py`, `executor.py`, `registry.py`, `bundled/*.md` | API route + CLI command |
| 8 | `scheduling/` | `cron_service.py`, `heartbeat.py`, `jobs.py` | `morgan-server/app.py` — lifespan init |
| 9 | `task_manager/` | `manager.py`, `types.py`, `progress.py` | `morgan-server/app.py` — lifespan init |
| 10 | `hook_system/` | `manager.py`, `types.py` | `morgan-server/app.py` — lifespan init |
| 11 | `app_state/` | `store.py` | `morgan-server/app.py` — lifespan init |
| 12 | `security/` | `permission_modes.py`, `memory_gating.py`, `allowlist.py` | `channels/gateway.py`, `workspace/manager.py` |

### Also audit these integration points:

- `morgan-server/morgan_server/app.py` — lifespan function: does it init ALL 12 modules? Does shutdown clean up all of them?
- `morgan-server/morgan_server/api/routes/tools_api.py` — are tool/skill/task/channel/workspace API endpoints implemented and registered?
- `morgan-server/morgan_server/api/routes/__init__.py` — does it export `tools_router`?
- `morgan-cli/morgan_cli/cli.py` — are CLI commands for tools/skills/tasks/workspace/channels/schedule implemented?
- `morgan-rag/morgan/core/assistant.py` — does `ask()` or `chat()` actually call the ToolExecutor? Does it run auto-compact? Does it append to daily log?
- `morgan-rag/morgan/config/settings.py` — are all feature flags defined (`morgan_enable_tools`, `morgan_enable_channels`, `morgan_enable_scheduling`, `morgan_enable_agents`, `morgan_enable_workspace`, `morgan_enable_compaction`)?
- `morgan-rag/pyproject.toml` — are optional dependency groups defined (`[channels]`, `[scheduling]`, `[tokens]`)?

---

## Phase 2: Wire — Fix Everything

After mapping, fix every issue you found. Follow these rules:

### Rule 1: Tool Execution Loop in Assistant

The `MorganAssistant.ask()` (or `chat()`) method must support a **tool-use loop** modeled on Claude Code's `QueryEngine.ts`:

```
User message → LLM call → if tool_use in response → execute tool → feed result back → LLM call again → repeat until text response
```

This means:
1. `MorganAssistant.__init__` must create a `ToolExecutor` and register all built-in tools
2. The LLM prompt must include tool schemas (JSON format matching Anthropic tool_use spec)
3. The response parser must detect tool_use blocks in LLM output
4. Tool results feed back as tool_result messages
5. Loop continues until LLM returns a text response (no more tool calls)

Reference: Claude Code `src/QueryEngine.ts` lines handling `tool_use` content blocks.

### Rule 2: Telegram Channel Must Work End-to-End

The Telegram adapter (`channels/telegram_channel.py`) must:
1. Receive messages via python-telegram-bot polling
2. Route through `ChannelGateway` → `RouteResolver` → agent handler
3. Agent handler calls `MorganAssistant.chat()` (with tool loop!)
4. Response flows back through gateway → Telegram `send_message`

Reference: OpenClaw's Telegram channel at `OpenClaw/src/telegram/` for patterns like:
- Allowlist filtering
- Group chat @mention detection
- Typing indicator while processing
- Message chunking for long responses
- Rate limiting with retry
- `/start`, `/help`, `/clear`, `/status` command handling

### Rule 3: Memory System Must Be Connected

Three memory layers must all work:
1. **Conversation memory** — `morgan/memory/memory_processor.py` (existing) stores per-conversation context in Qdrant
2. **Daily logs** — `morgan/memory_consolidation/daily_log.py` appends a summary after each conversation turn to `~/.morgan/memory/YYYY-MM-DD.md`
3. **MEMORY.md consolidation** — `morgan/memory_consolidation/consolidator.py` periodically (via cron or manual) reads recent daily logs and updates `~/.morgan/MEMORY.md`
4. **Hybrid search** — `morgan/memory_consolidation/hybrid_search.py` combines BM25 keyword search over daily logs with vector search over Qdrant embeddings

Wire daily log append into the assistant's response flow. Wire consolidation as a cron job. Wire hybrid search as the `memory_search` built-in tool.

Reference: Claude Code memory system at `~/.claude/projects/*/memory/MEMORY.md` pattern.

### Rule 4: Context Compaction Must Trigger Automatically

After each assistant response:
1. `AutoCompactTracker` counts tokens in conversation
2. When tokens exceed threshold (default: 80% of 200K context window), trigger compaction
3. `Compactor.compact()` summarizes old messages via LLM, replaces them with summary
4. Circuit breaker trips after 3 consecutive compaction failures

Reference: Claude Code `src/services/compact/autoCompact.ts`.

### Rule 5: Workspace Loads into System Prompt

On server startup:
1. `WorkspaceManager.bootstrap()` creates `~/.morgan/SOUL.md`, `USER.md`, `MEMORY.md` if missing
2. `load_session_context(session_type)` returns concatenated workspace content
3. This content becomes part of the LLM system prompt in every `ask()` call
4. Security gate: MEMORY.md only loaded for main/DM sessions, NOT group chats

Reference: OpenClaw `AGENTS.md` workspace pattern.

### Rule 6: Hooks Fire at the Right Times

Wire `HookManager.trigger()` calls at these points:
- `MESSAGE_INBOUND` — when assistant receives a message (before processing)
- `MESSAGE_REPLY` — when assistant sends a response (after generation)
- `PRE_TOOL_USE` / `POST_TOOL_USE` — around each tool execution in the tool loop
- `SESSION_START` / `SESSION_END` — in server lifespan and session manager
- `PRE_COMPACT` / `POST_COMPACT` — around compaction in the compactor
- `CONFIG_CHANGE` — when settings are updated via API

Reference: Claude Code hook system in `settings.json`.

### Rule 7: Skills Execute Through Agent Spawner

When a skill is invoked (via API or CLI):
1. `SkillRegistry.get(name)` returns the `Skill` definition
2. `SkillExecutor.execute(skill, args)` creates an `AgentDefinition` from the skill
3. `AgentSpawner.spawn(definition, prompt)` runs the agent with the skill's allowed tools and system prompt
4. Result returns to caller

Reference: Claude Code `src/skills/` loader pattern with YAML frontmatter in `.md` files.

### Rule 8: All API Routes Must Exist and Work

Verify `morgan-server/morgan_server/api/routes/tools_api.py` has working endpoints:
- `GET /api/tools` — calls `ToolExecutor.list_tools()`, returns JSON schemas
- `POST /api/tools/{name}` — calls `ToolExecutor.execute()`, returns `ToolResult`
- `GET /api/skills` — calls `SkillRegistry.list_all()`
- `POST /api/skills/{name}` — calls `SkillExecutor.execute()`
- `GET /api/tasks` — calls `TaskManager.list_tasks()`
- `GET /api/tasks/{id}` — calls `TaskManager.get_task()`
- `GET /api/channels` — returns registered channels from gateway
- `GET /api/workspace` — returns workspace status
- `POST /api/workspace/consolidate` — triggers memory consolidation

All routes must be imported in `routes/__init__.py` and included in the FastAPI app.

### Rule 9: CLI Commands Must Work

Verify `morgan-cli/morgan_cli/cli.py` has Click commands that call the server API:
- `morgan tools` / `morgan tools run <name> <input>`
- `morgan skills` / `morgan skills run <name>`
- `morgan tasks`
- `morgan workspace` / `morgan workspace consolidate`
- `morgan channels`
- `morgan schedule list` / `morgan schedule add`

### Rule 10: Config and Dependencies

Verify `morgan-rag/morgan/config/settings.py` has all feature flags as Pydantic fields with env var bindings:
```python
morgan_enable_tools: bool = True
morgan_enable_workspace: bool = True
morgan_enable_compaction: bool = True
morgan_enable_channels: bool = False
morgan_enable_scheduling: bool = False
morgan_enable_agents: bool = True
morgan_workspace_path: Optional[str] = None
morgan_telegram_token: Optional[str] = None
morgan_discord_token: Optional[str] = None
morgan_synology_token: Optional[str] = None
morgan_synology_incoming_url: Optional[str] = None
morgan_synology_webhook_path: str = "/synology-webhook"
morgan_synology_webhook_port: int = 8765
morgan_synology_bot_name: str = "Morgan"
morgan_synology_rate_limit: int = 30
```

Verify `morgan-rag/pyproject.toml` has optional dependency groups:
```toml
[project.optional-dependencies]
channels = ["python-telegram-bot>=20.0", "discord.py>=2.0"]
scheduling = ["apscheduler>=3.10"]
tokens = ["tiktoken>=0.5"]
```

---

## Phase 3: Verify

After all wiring is complete, run:

```bash
# Unit tests
cd morgan-rag && python -m pytest tests/ -v --tb=short

# Import smoke test — every module must import cleanly
python -c "
from morgan.tools import ToolExecutor
from morgan.tools.base import BaseTool, ToolResult, ToolContext
from morgan.tools.permissions import PermissionContext, PermissionMode
from morgan.workspace import WorkspaceManager
from morgan.compaction import AutoCompactTracker, Compactor
from morgan.memory_consolidation import DailyLogManager, MemoryConsolidator, HybridMemorySearch
from morgan.channels import ChannelGateway
from morgan.channels.base import BaseChannel, InboundMessage
from morgan.channels.routing import SessionKey, RouteResolver
from morgan.agents import AgentSpawner, AgentDefinition
from morgan.skills import SkillRegistry, SkillExecutor
from morgan.scheduling import CronService, HeartbeatManager
from morgan.task_manager import TaskManager
from morgan.hook_system import HookManager, HookType
from morgan.app_state import AppStateStore
from morgan.security import MemoryGate, ChannelAllowlist
print('All modules import successfully')
"

# Server startup test
cd morgan-server && python -c "
from morgan_server.app import create_app
app = create_app()
print('Server app created successfully')
"
```

---

## Reference Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    morgan-server (FastAPI)                   │
│                                                             │
│  lifespan: init workspace, state, hooks, tasks, cron,       │
│            channels, gateway                                │
│                                                             │
│  routes: /chat, /tools, /skills, /tasks, /channels,         │
│          /workspace, /memory, /knowledge, /health           │
└──────────┬──────────────────────────────────┬───────────────┘
           │ HTTP/WS                          │ polling
           ▼                                  ▼
┌──────────────────┐              ┌───────────────────────┐
│  morgan-cli      │              │  ChannelGateway       │
│  (Click CLI)     │              │  ├─ TelegramChannel   │
│                  │              │  ├─ SynologyChannel    │
│  tools, skills,  │              │  ├─ DiscordChannel     │
│  tasks, schedule │              │  └─ RouteResolver      │
└──────────────────┘              └───────────┬───────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MorganAssistant                           │
│                                                             │
│  ask(message) loop:                                         │
│    1. Load workspace context (SOUL.md + MEMORY.md)          │
│    2. Hook: MESSAGE_INBOUND                                 │
│    3. Build messages with conversation history               │
│    4. LLM call with tool schemas                            │
│    5. If tool_use → ToolExecutor.execute() → loop to 4      │
│    6. Auto-compact check (AutoCompactTracker)               │
│    7. Append to daily log                                   │
│    8. Hook: MESSAGE_REPLY                                   │
│    9. Return response                                       │
│                                                             │
│  Components:                                                │
│    ToolExecutor ← tools/builtin/* (bash, file, web, calc,   │
│                   memory_search)                            │
│    AgentSpawner ← agents/builtin/* (researcher, coder,      │
│                   planner)                                  │
│    SkillExecutor ← skills/bundled/*.md                      │
│    Compactor + AutoCompactTracker                           │
│    DailyLogManager + MemoryConsolidator                     │
│    HookManager                                              │
│    WorkspaceManager                                         │
│    SecurityGate (MemoryGate, ChannelAllowlist)              │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌────────────┐    ┌──────────────┐    ┌──────────────────┐
    │ LLMService │    │ Qdrant       │    │ ~/.morgan/       │
    │ (Ollama)   │    │ (vectors)    │    │  SOUL.md         │
    └────────────┘    └──────────────┘    │  USER.md         │
                                          │  MEMORY.md       │
                                          │  memory/         │
                                          │   2026-04-01.md  │
                                          └──────────────────┘
```

---

## What NOT To Do

- Do not rewrite modules from scratch. Fix wiring, not architecture.
- Do not add new modules or features beyond what's listed.
- Do not change the Telegram channel from polling to webhook unless it's broken.
- Do not remove existing functionality (emotional intelligence, companion system, RAG pipeline).
- Do not mock or stub things that should be real implementations.
- Do not add type: ignore or noqa comments to hide import errors — fix the imports.
- Do not modify OpenClaw or Claude Code source — they are read-only references.

## Output Format

For each fix, state:
1. **File**: full path
2. **Issue**: what's broken/missing
3. **Fix**: the exact code change (diff or full replacement)
4. **Verify**: how to confirm it works

After all fixes, run the Phase 3 verification commands and report results.
