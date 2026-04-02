"""
Agent spawner.

Manages the execution of agent definitions by calling either a user-provided
run function or falling back to Morgan's LLM service with full tool support.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional

from morgan.agents.base import AgentDefinition, AgentResult

logger = logging.getLogger(__name__)


class AgentSpawner:
    """
    Spawns and executes agents based on their definitions.

    Agents get the same tool-calling capabilities as the main orchestrator:
    they can call web_search, fetch_url, create_forum_topic, etc.

    Accepts an optional run_fn for custom execution logic. When no run_fn is
    provided, defaults to using the LLM service with a tool execution loop.
    """

    def __init__(
        self,
        run_fn: Optional[
            Callable[
                [str, str, List[str], Optional[str]],
                Coroutine[Any, Any, str],
            ]
        ] = None,
        tool_executor: Optional[Any] = None,
    ):
        self._run_fn = run_fn
        self._tool_executor = tool_executor

    async def spawn(
        self,
        definition: AgentDefinition,
        prompt: str,
        context: Optional[str] = None,
    ) -> AgentResult:
        """Spawn an agent and execute it with full tool support."""
        full_prompt = prompt
        if context:
            full_prompt = f"{prompt}\n\nContext:\n{context}"

        system_prompt = definition.get_system_prompt()

        try:
            if self._run_fn is not None:
                output = await self._run_fn(
                    full_prompt,
                    system_prompt,
                    definition.tools,
                    definition.model,
                )
            else:
                output = await self._default_run(
                    full_prompt, system_prompt, definition
                )

            return AgentResult(
                output=output,
                agent_type=definition.agent_type,
                success=True,
            )
        except Exception as e:
            logger.error(
                "Agent '%s' failed: %s", definition.agent_type, str(e)
            )
            return AgentResult(
                output="",
                agent_type=definition.agent_type,
                success=False,
                error=str(e),
            )

    async def _default_run(
        self,
        prompt: str,
        system_prompt: str,
        definition: AgentDefinition,
    ) -> str:
        """
        Default execution using Morgan's LLM service with tool loop.

        If the agent definition specifies allowed tools and a tool executor
        is available, runs a tool-calling loop (up to 5 rounds). Otherwise
        falls back to a single LLM call.
        """
        from morgan.services import get_llm_service

        llm = get_llm_service()

        # Resolve tool executor — use provided or try to build one
        executor = self._tool_executor
        if executor is None and definition.tools:
            executor = self._build_tool_executor(definition.tools)

        # No tools → simple single call
        if executor is None or not definition.tools:
            response = await llm.agenerate(
                prompt=prompt, system_prompt=system_prompt
            )
            return response.content

        # Build tool schema instructions
        tool_schemas = []
        for tool in executor.list_tools():
            if tool.name in definition.tools or not definition.tools:
                tool_schemas.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema.to_dict(),
                })

        if not tool_schemas:
            response = await llm.agenerate(
                prompt=prompt, system_prompt=system_prompt
            )
            return response.content

        schemas_text = json.dumps(tool_schemas, ensure_ascii=False)
        tool_instructions = (
            "# Tools\n"
            "You have tools available. To use one, respond with ONLY this JSON:\n"
            '{"tool_use":{"name":"<tool_name>","input":{...}}}\n\n'
            f"Available tools:\n{schemas_text}"
        )

        full_system = f"{system_prompt}\n\n{tool_instructions}" if system_prompt else tool_instructions

        messages: list[dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        # Tool loop — up to 5 rounds
        for _ in range(5):
            response = await llm.achat(messages=messages)
            content = (response.content or "").strip()

            tool_call = self._extract_tool_call(content)
            if tool_call is None:
                return content

            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            logger.info("Agent '%s' calling tool: %s", definition.agent_type, tool_name)

            from morgan.tools.base import ToolContext
            ctx = ToolContext(user_id="agent", conversation_id=f"agent:{definition.agent_type}")
            result = await executor.execute(
                tool_name=tool_name,
                input_data=tool_input,
                context=ctx,
            )

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Tool result ({tool_name}):\n{result.output}",
            })

        # Exhausted rounds — ask for final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the tool results above.",
        })
        response = await llm.achat(messages=messages)
        return (response.content or "").strip()

    @staticmethod
    def _extract_tool_call(content: str) -> Optional[Dict[str, Any]]:
        """Extract tool_use JSON from LLM output."""
        if not content:
            return None

        def _try_parse(s: str) -> Optional[Dict[str, Any]]:
            try:
                payload = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(payload, dict) and isinstance(payload.get("tool_use"), dict):
                tu = payload["tool_use"]
                if isinstance(tu.get("name"), str):
                    return {"name": tu["name"], "input": tu.get("input", {})}
            return None

        # Try whole text
        result = _try_parse(content.strip())
        if result:
            return result

        # Try embedded JSON
        match = re.search(r'\{"tool_use"\s*:', content)
        if match:
            depth = 0
            for i in range(match.start(), len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        result = _try_parse(content[match.start():i + 1])
                        if result:
                            return result
                        break

        return None

    @staticmethod
    def _build_tool_executor(allowed_tools: List[str]) -> Optional[Any]:
        """Build a ToolExecutor with the specified tools."""
        try:
            from morgan.tools import ToolExecutor
            from morgan.tools.builtin import ALL_BUILTIN_TOOLS

            executor = ToolExecutor()
            for tool in ALL_BUILTIN_TOOLS:
                if not allowed_tools or tool.name in allowed_tools:
                    executor.register(tool)
            return executor if executor.list_tools() else None
        except Exception as e:
            logger.debug("Failed to build tool executor for agent: %s", e)
            return None
