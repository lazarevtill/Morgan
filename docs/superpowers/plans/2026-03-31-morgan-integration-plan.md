# Morgan Integration Plan: Claude Code + OpenClaw Patterns

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate 12 production-grade features from Claude Code and OpenClaw into Morgan, transforming it from a chat-only assistant into a fully agentic, multi-channel, extensible personal AI platform.

**Architecture:** Each feature is a self-contained module added to `morgan-rag/morgan/` with API routes in `morgan-server/` and CLI commands in `morgan-cli/`. Features build on each other in dependency order: Tool System (foundation) → Workspace/Memory → Compaction → Channels → Agents → Skills → Scheduling → Tasks → Hooks → State → Security.

**Tech Stack:** Python 3.11+, Pydantic v2 (schemas/validation), asyncio (concurrency), APScheduler (cron), tiktoken (token counting), Click (CLI), FastAPI (API), SQLite + Qdrant (hybrid memory search)

**Sources:**
- Claude Code (`/home/lazarev/ClaudeCode/src/`) — Tool system, agents, skills, compaction, tasks, MCP, permissions
- OpenClaw (`/home/lazarev/ClaudeCode/OpenClaw/`) — SOUL.md workspace, multi-channel gateway, daily memory, heartbeat/cron, plugin SDK
- Morgan current (`/home/lazarev/ClaudeCode/Morgan/`) — Emotional intelligence, RAG services, companion system

---

## File Structure Overview

All new modules go under `morgan-rag/morgan/`. Server routes go in `morgan-server/morgan_server/api/routes/`. CLI commands extend `morgan-cli/morgan_cli/cli.py`.

```
morgan-rag/morgan/
├── tools/                          # NEW - Tool system (Task 1)
│   ├── __init__.py                 # Tool registry & get_tools()
│   ├── base.py                     # BaseTool ABC, ToolResult, ToolContext
│   ├── permissions.py              # Permission checking (allow/deny/ask)
│   ├── builtin/                    # Built-in tool implementations
│   │   ├── web_search.py
│   │   ├── file_read.py
│   │   ├── file_write.py
│   │   ├── bash_tool.py
│   │   ├── calculator.py
│   │   └── memory_search.py
│   └── executor.py                 # Tool execution loop
│
├── workspace/                      # NEW - SOUL.md workspace (Task 2)
│   ├── __init__.py
│   ├── manager.py                  # WorkspaceManager - loads/saves workspace files
│   ├── templates.py                # Default SOUL.md, USER.md, TOOLS.md templates
│   └── paths.py                    # Workspace path resolution
│
├── compaction/                     # NEW - Context compaction (Task 3)
│   ├── __init__.py
│   ├── auto_compact.py             # Token tracking, threshold detection
│   ├── compactor.py                # Summary generation, message replacement
│   └── token_counter.py            # Token estimation (tiktoken)
│
├── memory_consolidation/           # NEW - Daily logs + consolidation (Task 4)
│   ├── __init__.py
│   ├── daily_log.py                # memory/YYYY-MM-DD.md management
│   ├── consolidator.py             # Daily -> MEMORY.md consolidation
│   └── hybrid_search.py            # Vector + BM25 hybrid search
│
├── channels/                       # NEW - Multi-channel gateway (Task 5)
│   ├── __init__.py
│   ├── gateway.py                  # ChannelGateway - routes messages
│   ├── base.py                     # BaseChannel ABC
│   ├── routing.py                  # Session key resolution, route matching
│   ├── telegram_channel.py         # Telegram adapter
│   ├── discord_channel.py          # Discord adapter
│   └── whatsapp_channel.py         # WhatsApp adapter (placeholder)
│
├── agents/                         # NEW - Agent/subagent system (Task 6)
│   ├── __init__.py
│   ├── base.py                     # AgentDefinition, AgentResult
│   ├── spawner.py                  # Agent spawning (async subprocess)
│   ├── builtin/                    # Built-in agent definitions
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   └── planner.py
│   └── loader.py                   # Load agent definitions from .md files
│
├── skills/                         # NEW - Skill/plugin system (Task 7)
│   ├── __init__.py
│   ├── loader.py                   # Skill discovery & loading
│   ├── executor.py                 # Skill execution
│   ├── registry.py                 # Skill registry
│   └── bundled/                    # Bundled skill .md files
│       ├── web_research.md
│       └── code_review.md
│
├── scheduling/                     # NEW - Heartbeat + Cron (Task 8)
│   ├── __init__.py
│   ├── cron_service.py             # APScheduler-based cron
│   ├── heartbeat.py                # Periodic conversational checks
│   └── jobs.py                     # Job definitions & persistence
│
├── task_manager/                   # NEW - Task management (Task 9)
│   ├── __init__.py
│   ├── manager.py                  # TaskManager - register, track, update
│   ├── types.py                    # TaskState, TaskProgress, TaskType
│   └── progress.py                 # Progress tracking
│
├── hook_system/                    # NEW - Hook system (Task 10)
│   ├── __init__.py
│   ├── manager.py                  # HookManager - register, trigger
│   └── types.py                    # Hook types, HookResult
│
├── app_state/                      # NEW - Centralized state (Task 11)
│   ├── __init__.py
│   └── store.py                    # AppStateStore with observer pattern
│
└── security/                       # NEW - Security & permissions (Task 12)
    ├── __init__.py
    ├── permission_modes.py         # PermissionMode enum, rules
    ├── memory_gating.py            # Context isolation for shared sessions
    └── allowlist.py                # DM pairing / channel allowlists
```

---

## Task 1: Tool System

**Source:** Claude Code `src/Tool.ts`, `src/tools/BashTool/`, `src/tools/FileReadTool/`

**Files:**
- Create: `morgan-rag/morgan/tools/__init__.py`
- Create: `morgan-rag/morgan/tools/base.py`
- Create: `morgan-rag/morgan/tools/permissions.py`
- Create: `morgan-rag/morgan/tools/executor.py`
- Create: `morgan-rag/morgan/tools/builtin/web_search.py`
- Create: `morgan-rag/morgan/tools/builtin/file_read.py`
- Create: `morgan-rag/morgan/tools/builtin/bash_tool.py`
- Create: `morgan-rag/morgan/tools/builtin/calculator.py`
- Create: `morgan-rag/morgan/tools/builtin/memory_search.py`
- Create: `morgan-rag/morgan/tools/builtin/__init__.py`
- Modify: `morgan-rag/morgan/core/assistant.py` — wire tool executor into `ask()`
- Modify: `morgan-server/morgan_server/api/routes/chat.py` — pass tool context to assistant
- Test: `morgan-rag/tests/test_tools.py`
- Test: `morgan-rag/tests/test_tool_executor.py`

- [ ] **Step 1: Write failing test for BaseTool**

```python
# morgan-rag/tests/test_tools.py
import pytest
from morgan.tools.base import BaseTool, ToolResult, ToolContext, ToolInputSchema


def test_base_tool_is_abstract():
    """BaseTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTool()


def test_tool_result_success():
    result = ToolResult(output="hello", is_error=False)
    assert result.output == "hello"
    assert result.is_error is False


def test_tool_result_error():
    result = ToolResult(output="fail", is_error=True, error_code="PERMISSION_DENIED")
    assert result.is_error is True
    assert result.error_code == "PERMISSION_DENIED"


def test_tool_input_schema():
    schema = ToolInputSchema(
        type="object",
        properties={"query": {"type": "string", "description": "Search query"}},
        required=["query"],
    )
    assert schema.type == "object"
    assert "query" in schema.properties
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lazarev/ClaudeCode/Morgan/morgan-rag && python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'morgan.tools'`

- [ ] **Step 3: Implement BaseTool, ToolResult, ToolContext**

Create `morgan-rag/morgan/tools/__init__.py`, `morgan-rag/morgan/tools/base.py`:

```python
# morgan-rag/morgan/tools/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ToolInputSchema:
    type: str = "object"
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "properties": self.properties, "required": self.required}


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    working_directory: str = "."
    session_type: str = "main"
    environment: Dict[str, str] = field(default_factory=dict)
    abort_signal: Optional[Callable[[], bool]] = None


class BaseTool(ABC):
    """Abstract base class for all tools. Ported from Claude Code src/Tool.ts."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def aliases(self) -> List[str]:
        return []

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> ToolInputSchema: ...

    @abstractmethod
    async def call(self, input_data: Dict[str, Any], context: ToolContext) -> ToolResult: ...

    def is_read_only(self, input_data: Dict[str, Any]) -> bool:
        return False

    def is_destructive(self, input_data: Dict[str, Any]) -> bool:
        return False

    def user_facing_name(self, input_data: Dict[str, Any]) -> str:
        return self.name

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        for req in self.input_schema.required:
            if req not in input_data:
                return f"Missing required field: {req}"
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lazarev/ClaudeCode/Morgan/morgan-rag && python -m pytest tests/test_tools.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Implement permissions module** (`morgan-rag/morgan/tools/permissions.py`)

PermissionMode enum (DEFAULT, PLAN, BYPASS, AUTO), PermissionRule (tool_pattern, decision, source), PermissionContext, check_permission() with deny-first resolution using fnmatch.

- [ ] **Step 6: Write test for ToolExecutor** (`morgan-rag/tests/test_tool_executor.py`)

Test: executor runs tool, unknown tool returns error, missing input returns validation error.

- [ ] **Step 7: Implement ToolExecutor** (`morgan-rag/morgan/tools/executor.py`)

Manages tool registry by name+aliases, validates input, checks permissions, executes with try/except, logs results.

- [ ] **Step 8: Run all tool tests**

Run: `cd /home/lazarev/ClaudeCode/Morgan/morgan-rag && python -m pytest tests/test_tools.py tests/test_tool_executor.py -v`
Expected: PASS (7 tests)

- [ ] **Step 9: Implement 5 built-in tools**

- `calculator.py` — Safe math via ast.parse (no exec/eval of arbitrary code), supports +,-,*,/,**,%
- `file_read.py` — Path validation, line numbering, offset/limit params
- `bash_tool.py` — asyncio.create_subprocess_shell, timeout, blocked pattern list (rm -rf /, mkfs, etc.)
- `web_search.py` — Delegates to morgan.services.external_knowledge.web_search.WebSearchService
- `memory_search.py` — Delegates to morgan.memory.memory_processor.get_memory_processor()

- [ ] **Step 10: Commit Task 1**

```bash
git add morgan-rag/morgan/tools/ morgan-rag/tests/test_tools.py morgan-rag/tests/test_tool_executor.py
git commit -m "feat: add tool system with BaseTool, permissions, executor, 5 built-in tools"
```

---

## Task 2: SOUL.md + Workspace Pattern

**Source:** OpenClaw `AGENTS.md`, workspace structure

**Files:**
- Create: `morgan-rag/morgan/workspace/__init__.py`
- Create: `morgan-rag/morgan/workspace/manager.py`
- Create: `morgan-rag/morgan/workspace/templates.py`
- Create: `morgan-rag/morgan/workspace/paths.py`
- Test: `morgan-rag/tests/test_workspace.py`

- [ ] **Step 1: Write failing test for WorkspaceManager**

Tests: bootstrap creates SOUL.md/USER.md/MEMORY.md/memory/, load_soul returns content, update_soul persists, load_session_context for main includes memory, group session does NOT include MEMORY.md (security gate).

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement workspace module**

- `paths.py` — get_morgan_home() (~/.morgan), get_workspace_path() with MORGAN_WORKSPACE_PATH override
- `templates.py` — SOUL_TEMPLATE, USER_TEMPLATE, MEMORY_TEMPLATE, TOOLS_TEMPLATE, HEARTBEAT_TEMPLATE
- `manager.py` — WorkspaceManager: bootstrap(), load_soul/user/memory/tools/heartbeat(), update_soul/user/memory(), load_daily_log(date), append_daily_log(entry), load_session_context(session_type) with security gate (MEMORY.md only for main/dm sessions, truncated to 200 lines / 25KB)

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit Task 2**

```bash
git commit -m "feat: add SOUL.md workspace pattern with session-gated memory"
```

---

## Task 3: Context Compaction

**Source:** Claude Code `src/services/compact/autoCompact.ts`, `compact.ts`

**Files:**
- Create: `morgan-rag/morgan/compaction/__init__.py`
- Create: `morgan-rag/morgan/compaction/token_counter.py`
- Create: `morgan-rag/morgan/compaction/auto_compact.py`
- Create: `morgan-rag/morgan/compaction/compactor.py`
- Test: `morgan-rag/tests/test_compaction.py`

- [ ] **Step 1: Write failing test**

Tests: estimate_tokens, tracker initial state, record_success resets failures, circuit breaker after 3 failures, warning state calculation, compactor summarizes messages.

- [ ] **Step 2: Implement token_counter.py**

Uses tiktoken if available (encoding_for_model gpt-4), falls back to len(text)//4.

- [ ] **Step 3: Implement auto_compact.py**

Constants: MAX_OUTPUT_TOKENS_FOR_SUMMARY=20000, AUTOCOMPACT_BUFFER_TOKENS=13000, MAX_CONSECUTIVE_FAILURES=3, DEFAULT_CONTEXT_WINDOW=200000. AutoCompactTracker with circuit_breaker_tripped property. calculate_token_warning_state returns percent_left, threshold booleans.

- [ ] **Step 4: Implement compactor.py**

Compactor.compact(messages, keep_recent=4): splits old/recent, generates LLM summary of old via morgan.services.get_llm_service(), builds compacted list with summary system message. Falls back to truncated text on LLM failure.

- [ ] **Step 5: Run tests and commit**

```bash
git commit -m "feat: add context compaction with auto-compact and circuit breaker"
```

---

## Task 4: Memory Consolidation (Daily Logs + Hybrid Search)

**Source:** OpenClaw `extensions/memory-core/`, Claude Code `src/memdir/`

**Files:**
- Create: `morgan-rag/morgan/memory_consolidation/__init__.py`
- Create: `morgan-rag/morgan/memory_consolidation/daily_log.py`
- Create: `morgan-rag/morgan/memory_consolidation/consolidator.py`
- Create: `morgan-rag/morgan/memory_consolidation/hybrid_search.py`
- Test: `morgan-rag/tests/test_memory_consolidation.py`

- [ ] **Step 1: Write failing test**

Tests: daily log write/read, creates dated file, read empty returns None, list_recent, keyword search finds matching content.

- [ ] **Step 2: Implement daily_log.py**

DailyLogManager(memory_dir): append(entry) with timestamp, read_today(), read_date(date), list_recent(days=7).

- [ ] **Step 3: Implement hybrid_search.py**

HybridMemorySearch: keyword_search with BM25 scoring (k1=1.5, b=0.75), hybrid_search combining keyword + vector cosine similarity with configurable weights (0.6 vector, 0.4 keyword).

- [ ] **Step 4: Implement consolidator.py**

MemoryConsolidator: consolidate(days_to_review=7) reads recent daily logs + current MEMORY.md, generates updated MEMORY.md via LLM prompt.

- [ ] **Step 5: Run tests and commit**

```bash
git commit -m "feat: add memory consolidation with daily logs and hybrid search"
```

---

## Task 5: Multi-Channel Gateway

**Source:** OpenClaw `src/routing/`, `src/channels/`, `src/gateway/`

**Files:**
- Create: `morgan-rag/morgan/channels/__init__.py`
- Create: `morgan-rag/morgan/channels/base.py`
- Create: `morgan-rag/morgan/channels/gateway.py`
- Create: `morgan-rag/morgan/channels/routing.py`
- Create: `morgan-rag/morgan/channels/telegram_channel.py`
- Create: `morgan-rag/morgan/channels/discord_channel.py`
- Test: `morgan-rag/tests/test_channels.py`

- [ ] **Step 1: Write failing test**

Tests: SessionKey.for_dm/for_group, RouteResolver default, RouteResolver with binding, gateway registers channel.

- [ ] **Step 2: Implement base.py**

InboundMessage, OutboundMessage dataclasses. BaseChannel ABC with name, start(), stop(), send(), set_message_handler(), on_message().

- [ ] **Step 3: Implement routing.py**

SessionKey (frozen dataclass with for_dm/for_group/for_main static methods). ResolvedRoute. RouteResolver with binding resolution: peer -> group -> channel -> default.

- [ ] **Step 4: Implement gateway.py**

ChannelGateway: register_channel, set_agent_handler, start/stop all channels, _handle_message routes via resolver and dispatches to agent handler.

- [ ] **Step 5: Implement telegram_channel.py and discord_channel.py**

Telegram: python-telegram-bot, polling mode, allowlist filtering. Discord: discord.py, intents, guild filtering.

- [ ] **Step 6: Run tests and commit**

```bash
git commit -m "feat: add multi-channel gateway with routing and Telegram/Discord adapters"
```

---

## Task 6: Agent/Subagent System

**Source:** Claude Code `src/tools/AgentTool/`

**Files:**
- Create: `morgan-rag/morgan/agents/__init__.py`
- Create: `morgan-rag/morgan/agents/base.py`
- Create: `morgan-rag/morgan/agents/spawner.py`
- Create: `morgan-rag/morgan/agents/loader.py`
- Create: `morgan-rag/morgan/agents/builtin/researcher.py`
- Create: `morgan-rag/morgan/agents/builtin/coder.py`
- Create: `morgan-rag/morgan/agents/builtin/planner.py`
- Create: `morgan-rag/morgan/agents/builtin/__init__.py`
- Test: `morgan-rag/tests/test_agents.py`

- [ ] **Step 1: Write failing test**

Tests: AgentDefinition fields, builtin agents exist (researcher/coder/planner), AgentResult, spawner runs agent with mock run_fn.

- [ ] **Step 2: Implement agent base and spawner**

AgentDefinition: agent_type, when_to_use, get_system_prompt (callable), tools, model, effort, max_turns, source. AgentSpawner: spawn(definition, prompt) calls run_fn or default (get_llm_service). Loader: parse_frontmatter from .md, load_agents_from_dir.

- [ ] **Step 3: Implement 3 built-in agents**

researcher (web_search/memory_search/file_read, thorough), coder (bash/file_read/file_write/calculator, thorough), planner (file_read/web_search/memory_search, balanced).

- [ ] **Step 4: Run tests and commit**

```bash
git commit -m "feat: add agent/subagent system with spawner and built-in agents"
```

---

## Task 7: Skill/Plugin System

**Source:** Claude Code `src/skills/`, OpenClaw plugin SDK

**Files:**
- Create: `morgan-rag/morgan/skills/__init__.py`
- Create: `morgan-rag/morgan/skills/loader.py`
- Create: `morgan-rag/morgan/skills/executor.py`
- Create: `morgan-rag/morgan/skills/registry.py`
- Create: `morgan-rag/morgan/skills/bundled/web_research.md`
- Create: `morgan-rag/morgan/skills/bundled/code_review.md`
- Test: `morgan-rag/tests/test_skills.py`

- [ ] **Step 1: Write failing test**

Tests: load_skill_from_file parses frontmatter, prompt substitution replaces ${var}, load_skills_from_dir, registry register/get/list.

- [ ] **Step 2: Implement skill loader**

Skill dataclass: name, description, when_to_use, allowed_tools, argument_names, model, effort, agent, source, _content. get_prompt(args) does ${var} substitution. load_skill_from_file parses YAML frontmatter, extracts ${var} patterns. load_skills_from_dir scans *.md.

- [ ] **Step 3: Implement registry and executor**

SkillRegistry: register, get, list_all, list_user_invocable. SkillExecutor: creates AgentDefinition from skill, runs through AgentSpawner.

- [ ] **Step 4: Create bundled skills** (web_research.md, code_review.md)

- [ ] **Step 5: Run tests and commit**

```bash
git commit -m "feat: add skill/plugin system with .md frontmatter loading"
```

---

## Task 8: Heartbeat + Cron Scheduling

**Source:** OpenClaw `src/cron/`, heartbeat pattern

**Files:**
- Create: `morgan-rag/morgan/scheduling/__init__.py`
- Create: `morgan-rag/morgan/scheduling/cron_service.py`
- Create: `morgan-rag/morgan/scheduling/heartbeat.py`
- Create: `morgan-rag/morgan/scheduling/jobs.py`
- Test: `morgan-rag/tests/test_scheduling.py`

- [ ] **Step 1: Write failing test**

Tests: CronJob definition, HeartbeatCheck, add/remove job, heartbeat register_check.

- [ ] **Step 2: Implement scheduling**

CronJob: job_id, schedule (cron expression), prompt, channel, model, isolated. HeartbeatCheck: name, fn, priority, last_run. CronService: add/remove/list jobs, disk persistence (JSON), APScheduler integration. HeartbeatManager: interval with jitter (0.8-1.2x), batch 2-3 checks per beat sorted by priority+last_run.

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add heartbeat + cron scheduling system"
```

---

## Task 9: Task Management

**Source:** Claude Code `src/tasks/types.ts`

**Files:**
- Create: `morgan-rag/morgan/task_manager/__init__.py`
- Create: `morgan-rag/morgan/task_manager/types.py`
- Create: `morgan-rag/morgan/task_manager/manager.py`
- Create: `morgan-rag/morgan/task_manager/progress.py`
- Test: `morgan-rag/tests/test_task_manager.py`

- [ ] **Step 1: Write failing test**

Tests: task creation, status transitions (pending->in_progress->completed), list tasks, progress tracker tool use counting.

- [ ] **Step 2: Implement task management**

TaskType enum (AGENT, SHELL, CRON, DREAM). TaskStatus enum. TaskState dataclass. TaskManager: create_task, get_task, update_status, list_tasks, delete_task. ProgressTracker: record_tool_use, record_tokens, recent_activities (capped at 20).

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add task management system with progress tracking"
```

---

## Task 10: Hook System

**Source:** Claude Code hooks, OpenClaw plugin hooks

**Files:**
- Create: `morgan-rag/morgan/hook_system/__init__.py`
- Create: `morgan-rag/morgan/hook_system/manager.py`
- Create: `morgan-rag/morgan/hook_system/types.py`
- Test: `morgan-rag/tests/test_hooks.py`

- [ ] **Step 1: Write failing test**

Tests: register and trigger, multiple hooks fire in order, short-circuit on {abort: True}.

- [ ] **Step 2: Implement hook system**

HookType enum: MESSAGE_INBOUND, MESSAGE_REPLY, PRE_TOOL_USE, POST_TOOL_USE, SESSION_START, SESSION_END, PRE_COMPACT, POST_COMPACT, CONFIG_CHANGE. HookManager: register, unregister, trigger (supports async+sync handlers, short-circuits on abort).

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add hook system for event-driven extensibility"
```

---

## Task 11: App State Store

**Source:** Claude Code `src/state/AppStateStore.ts`

**Files:**
- Create: `morgan-rag/morgan/app_state/__init__.py`
- Create: `morgan-rag/morgan/app_state/store.py`
- Test: `morgan-rag/tests/test_app_state.py`

- [ ] **Step 1: Write failing test**

Tests: initial state, set_state with updater, subscribe/unsubscribe.

- [ ] **Step 2: Implement AppStateStore**

AppState dataclass: verbose, main_model, permission_mode, tasks, channels, plugins, skills, status_text. AppStateStore: get_state, set_state(updater), subscribe returns unsubscribe callable. Thread-safe with RLock.

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add centralized AppStateStore with observer pattern"
```

---

## Task 12: Security and Permissions

**Source:** Claude Code permissions, OpenClaw memory gating + allowlists

**Files:**
- Create: `morgan-rag/morgan/security/__init__.py`
- Create: `morgan-rag/morgan/security/permission_modes.py`
- Create: `morgan-rag/morgan/security/memory_gating.py`
- Create: `morgan-rag/morgan/security/allowlist.py`
- Test: `morgan-rag/tests/test_security.py`

- [ ] **Step 1: Write failing test**

Tests: memory gate allows main, blocks group/cron. Allowlist open policy, restricted policy, pairing flow (request code -> approve -> allowed).

- [ ] **Step 2: Implement security module**

MemoryGate: should_load_memory(session_type) returns True only for main/dm. ChannelAllowlist: policy (open/allowlist/pairing), request_pairing generates hex code, approve_pairing adds to allowed_ids. SessionPermissionMode enum: INTERACTIVE, AUTONOMOUS, RESTRICTED.

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add security module with memory gating, allowlists, permission modes"
```

---

## Task 13: Integration Wiring

Wire all 12 new modules into Morgan's existing architecture.

**Files:**
- Modify: `morgan-rag/morgan/core/assistant.py`
- Modify: `morgan-server/morgan_server/app.py`
- Modify: `morgan-server/morgan_server/api/routes/chat.py`
- Modify: `morgan-cli/morgan_cli/cli.py`
- Modify: `morgan-rag/pyproject.toml`
- Modify: `morgan-rag/morgan/config/settings.py`

- [ ] **Step 1: Wire tool executor into MorganAssistant.ask()**

Add ToolExecutor initialization in `__init__`, call tools during response generation when LLM requests tool use.

- [ ] **Step 2: Wire workspace loading into server startup**

In `app.py` lifespan, bootstrap workspace, load session context into assistant.

- [ ] **Step 3: Wire auto-compact into chat response loop**

In `chat.py` route, check should_auto_compact after each response, compact if threshold exceeded.

- [ ] **Step 4: Wire daily log append into conversation memory**

After each assistant response, append summary to daily log.

- [ ] **Step 5: Add API routes**

- `POST /api/tools` — list available tools
- `POST /api/tools/{name}/execute` — execute a tool
- `GET /api/skills` — list skills
- `POST /api/skills/{name}/run` — run a skill
- `GET /api/tasks` — list tasks
- `GET /api/tasks/{id}` — get task status
- `GET /api/channels` — list active channels

- [ ] **Step 6: Add CLI commands**

- `morgan tools` — list available tools
- `morgan skills` — list/run skills
- `morgan tasks` — list background tasks
- `morgan channels` — list/start channels
- `morgan schedule` — manage cron jobs
- `morgan workspace` — manage workspace files

- [ ] **Step 7: Update pyproject.toml dependencies**

Add: tiktoken, apscheduler, python-telegram-bot (optional), discord.py (optional), pyyaml

- [ ] **Step 8: Update settings.py**

Add feature flags: `morgan_enable_tools`, `morgan_enable_channels`, `morgan_enable_scheduling`, `morgan_enable_agents`. Add workspace config: `morgan_workspace_path`.

- [ ] **Step 9: Run full test suite**

```bash
cd /home/lazarev/ClaudeCode/Morgan/morgan-rag && python -m pytest tests/ -v --tb=short
```

- [ ] **Step 10: Update CLAUDE.md**

Document new modules, commands, and architecture.

- [ ] **Step 11: Final commit**

```bash
git commit -m "feat: wire all 12 new modules into Morgan core architecture"
```
