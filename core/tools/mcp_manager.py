"""
MCP (Model Context Protocol) Tools Manager
Handles tool registration, execution, and management
"""
import asyncio
import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import asyncpg
import aiohttp

from shared.utils.logging import setup_logging
from shared.utils.errors import ErrorHandler, ErrorCode


class MCPTool:
    """MCP Tool definition"""
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        category: str = "utility",
        endpoint_url: Optional[str] = None,
        method: str = "POST",
        parameters_schema: Optional[Dict] = None,
        handler: Optional[Callable] = None,
        enabled: bool = True
    ):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.endpoint_url = endpoint_url
        self.method = method
        self.parameters_schema = parameters_schema or {}
        self.handler = handler  # For built-in tools
        self.enabled = enabled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "endpoint_url": self.endpoint_url,
            "method": self.method,
            "parameters_schema": self.parameters_schema,
            "enabled": self.enabled,
            "has_handler": self.handler is not None
        }


class MCPToolsManager:
    """
    Manages MCP (Model Context Protocol) tools
    Supports both built-in handlers and external API endpoints
    """

    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self.logger = setup_logging("mcp_tools", "INFO", "logs/mcp_tools.log")
        self.error_handler = ErrorHandler(self.logger)

        self.pg_pool: Optional[asyncpg.Pool] = None
        self.tools: Dict[str, MCPTool] = {}  # Cache of tools
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in tools with handlers"""

        # Calculator tool
        self.tools["calculator"] = MCPTool(
            id="calculator",
            name="calculator",
            description="Perform mathematical calculations",
            category="utility",
            parameters_schema={
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            handler=self._calculator_handler,
            enabled=True
        )

        # DateTime tool
        self.tools["datetime"] = MCPTool(
            id="datetime",
            name="datetime",
            description="Get current date and time information",
            category="utility",
            parameters_schema={
                "timezone": {"type": "string", "description": "Timezone (optional)", "default": "UTC"}
            },
            handler=self._datetime_handler,
            enabled=True
        )

        # Memory tool
        self.tools["remember"] = MCPTool(
            id="remember",
            name="remember",
            description="Store information in memory for later recall",
            category="memory",
            parameters_schema={
                "content": {"type": "string", "description": "Information to remember"},
                "category": {"type": "string", "description": "Memory category (optional)"},
                "importance": {"type": "integer", "description": "Importance 1-10", "default": 5}
            },
            handler=None,  # Handled by memory manager
            enabled=True
        )

    async def start(self):
        """Initialize database connection and load tools"""
        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self.logger.info("PostgreSQL connection pool created")

            # Initialize HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Sync built-in tools to database
            await self._sync_tools_to_db()

            # Load external tools from database
            await self._load_tools_from_db()

            self.logger.info(f"MCP Tools Manager started with {len(self.tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to start MCP Tools Manager: {e}", exc_info=True)
            raise

    async def stop(self):
        """Close connections"""
        if self.pg_pool:
            await self.pg_pool.close()
            self.logger.info("PostgreSQL connection pool closed")

        if self.http_session:
            await self.http_session.close()
            self.logger.info("HTTP session closed")

    async def _sync_tools_to_db(self):
        """Sync built-in tools to database"""
        try:
            async with self.pg_pool.acquire() as conn:
                for tool in self.tools.values():
                    await conn.execute(
                        """
                        INSERT INTO mcp_tools (id, name, description, category, parameters_schema, enabled)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (name) DO UPDATE SET
                            description = EXCLUDED.description,
                            category = EXCLUDED.category,
                            parameters_schema = EXCLUDED.parameters_schema,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        tool.id,
                        tool.name,
                        tool.description,
                        tool.category,
                        tool.parameters_schema,
                        tool.enabled
                    )
            self.logger.info("Built-in tools synced to database")

        except Exception as e:
            self.logger.error(f"Failed to sync tools: {e}")

    async def _load_tools_from_db(self):
        """Load external tools from database"""
        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, category, endpoint_url, method,
                           auth_config, parameters_schema, enabled
                    FROM mcp_tools
                    WHERE enabled = true AND endpoint_url IS NOT NULL
                    """
                )

                for row in rows:
                    tool = MCPTool(
                        id=str(row["id"]),
                        name=row["name"],
                        description=row["description"],
                        category=row["category"],
                        endpoint_url=row["endpoint_url"],
                        method=row["method"],
                        parameters_schema=row["parameters_schema"],
                        enabled=row["enabled"]
                    )
                    self.tools[tool.name] = tool

                self.logger.info(f"Loaded {len(rows)} external tools from database")

        except Exception as e:
            self.logger.error(f"Failed to load tools: {e}")

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user_id: User identifier
            conversation_id: Optional conversation ID

        Returns:
            Tool execution result
        """
        start_time = datetime.now()
        execution_id = str(uuid.uuid4())

        try:
            # Get tool
            tool = self.tools.get(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")

            if not tool.enabled:
                raise ValueError(f"Tool is disabled: {tool_name}")

            self.logger.info(f"Executing tool: {tool_name} for user: {user_id}")

            # Execute tool
            if tool.handler:
                # Built-in handler
                result = await tool.handler(parameters)
                status = "success"
                error_message = None
            elif tool.endpoint_url:
                # External API endpoint
                result, status, error_message = await self._call_external_tool(tool, parameters)
            else:
                raise ValueError(f"Tool has no handler or endpoint: {tool_name}")

            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log execution to database
            await self._log_execution(
                tool_id=tool.id,
                user_id=user_id,
                conversation_id=conversation_id,
                parameters=parameters,
                result=result,
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time
            )

            # Update tool usage count
            await self._update_tool_usage(tool.id)

            return {
                "execution_id": execution_id,
                "tool_name": tool_name,
                "status": status,
                "result": result,
                "error": error_message,
                "execution_time_ms": execution_time
            }

        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_message = str(e)

            self.logger.error(f"Tool execution failed: {tool_name}: {error_message}", exc_info=True)

            # Log failed execution
            if tool:
                await self._log_execution(
                    tool_id=tool.id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    parameters=parameters,
                    result=None,
                    status="error",
                    error_message=error_message,
                    execution_time_ms=execution_time
                )

            return {
                "execution_id": execution_id,
                "tool_name": tool_name,
                "status": "error",
                "result": None,
                "error": error_message,
                "execution_time_ms": execution_time
            }

    async def _call_external_tool(
        self,
        tool: MCPTool,
        parameters: Dict[str, Any]
    ) -> tuple[Optional[Dict], str, Optional[str]]:
        """Call an external tool API"""
        try:
            method = tool.method.upper()

            if method == "GET":
                async with self.http_session.get(tool.endpoint_url, params=parameters) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result, "success", None
                    else:
                        error_text = await response.text()
                        return None, "error", f"HTTP {response.status}: {error_text}"

            elif method == "POST":
                async with self.http_session.post(tool.endpoint_url, json=parameters) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result, "success", None
                    else:
                        error_text = await response.text()
                        return None, "error", f"HTTP {response.status}: {error_text}"

            else:
                return None, "error", f"Unsupported HTTP method: {method}"

        except asyncio.TimeoutError:
            return None, "timeout", "Request timeout"
        except Exception as e:
            return None, "error", str(e)

    async def _calculator_handler(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in calculator tool"""
        try:
            expression = parameters.get("expression", "")

            # Safe eval with limited scope
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow
            }

            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return {
                "expression": expression,
                "result": result
            }

        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }

    async def _datetime_handler(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in datetime tool"""
        from datetime import datetime, timezone
        import pytz

        try:
            tz_name = parameters.get("timezone", "UTC")

            # Get current time
            now = datetime.now(timezone.utc)

            # Convert to requested timezone
            if tz_name != "UTC":
                tz = pytz.timezone(tz_name)
                now = now.astimezone(tz)

            return {
                "datetime": now.isoformat(),
                "timezone": tz_name,
                "timestamp": now.timestamp(),
                "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z")
            }

        except Exception as e:
            return {
                "error": str(e)
            }

    async def _log_execution(
        self,
        tool_id: str,
        user_id: str,
        conversation_id: Optional[str],
        parameters: Dict,
        result: Optional[Dict],
        status: str,
        error_message: Optional[str],
        execution_time_ms: int
    ):
        """Log tool execution to database"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO tool_executions
                    (tool_id, user_id, conversation_id, parameters, result, status, error_message, execution_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    tool_id,
                    user_id,
                    conversation_id,
                    parameters,
                    result,
                    status,
                    error_message,
                    execution_time_ms
                )
        except Exception as e:
            self.logger.error(f"Failed to log tool execution: {e}")

    async def _update_tool_usage(self, tool_id: str):
        """Update tool usage statistics"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE mcp_tools
                    SET usage_count = usage_count + 1,
                        last_used_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    tool_id
                )
        except Exception as e:
            self.logger.error(f"Failed to update tool usage: {e}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [tool.to_dict() for tool in self.tools.values() if tool.enabled]

    def get_tool_descriptions_for_llm(self) -> str:
        """Get tool descriptions formatted for LLM"""
        tools_desc = []
        for tool in self.tools.values():
            if tool.enabled:
                params_str = json.dumps(tool.parameters_schema, indent=2)
                tools_desc.append(
                    f"Tool: {tool.name}\n"
                    f"Description: {tool.description}\n"
                    f"Category: {tool.category}\n"
                    f"Parameters: {params_str}\n"
                )

        return "\n".join(tools_desc)
