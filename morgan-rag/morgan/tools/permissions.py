"""
Permission system for the Morgan tool system.

Controls which tools can run, in what mode, and with what restrictions.
Uses fnmatch pattern matching for flexible tool name matching.

Usage:
    from morgan.tools.permissions import (
        PermissionMode, PermissionRule, PermissionContext, check_permission,
    )

    rules = [
        PermissionRule(tool_pattern="bash*", allowed=False),
        PermissionRule(tool_pattern="calculator", allowed=True),
    ]
    ctx = PermissionContext(mode=PermissionMode.DEFAULT, rules=rules)
    result = check_permission("calculator", ctx)  # PermissionResult(allowed=True)
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PermissionMode(str, Enum):
    """
    Permission mode controlling how tool permissions are evaluated.

    DEFAULT: Apply permission rules normally (deny-first).
    PLAN: Only allow read-only / non-destructive tools.
    BYPASS: Skip permission checks entirely (for trusted contexts).
    AUTO: Automatically approve all tool calls (like BYPASS but logged).
    """

    DEFAULT = "default"
    PLAN = "plan"
    BYPASS = "bypass"
    AUTO = "auto"


@dataclass
class PermissionRule:
    """
    A single permission rule that matches tool names via fnmatch patterns.

    Attributes:
        tool_pattern: fnmatch-style pattern to match tool names (e.g., "bash*", "file_*").
        allowed: Whether matching tools are allowed (True) or denied (False).
        reason: Optional human-readable reason for the rule.
    """

    tool_pattern: str
    allowed: bool = True
    reason: Optional[str] = None


@dataclass
class PermissionContext:
    """
    Context for evaluating tool permissions.

    Attributes:
        mode: The permission mode to use.
        rules: List of permission rules, evaluated in order (deny-first).
        allowed_tools: Explicit allowlist of tool names (overrides rules).
        denied_tools: Explicit denylist of tool names (overrides rules).
    """

    mode: PermissionMode = PermissionMode.DEFAULT
    rules: List[PermissionRule] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)


@dataclass
class PermissionResult:
    """
    Result of a permission check.

    Attributes:
        allowed: Whether the tool is permitted to run.
        reason: Human-readable explanation of the decision.
    """

    allowed: bool
    reason: str = ""


def check_permission(tool_name: str, context: PermissionContext) -> PermissionResult:
    """
    Check whether a tool is permitted to run under the given permission context.

    Resolution order:
    1. BYPASS / AUTO modes always allow.
    2. Explicit denied_tools list blocks the tool.
    3. Explicit allowed_tools list permits the tool.
    4. Permission rules are evaluated deny-first:
       - All deny rules are checked first; if any match, the tool is blocked.
       - Then allow rules are checked; if any match, the tool is allowed.
    5. If no rules match, DEFAULT mode denies, PLAN mode denies.

    Args:
        tool_name: The name of the tool to check.
        context: The permission context with mode and rules.

    Returns:
        PermissionResult indicating whether the tool is allowed and why.
    """
    # BYPASS and AUTO modes skip all checks
    if context.mode in (PermissionMode.BYPASS, PermissionMode.AUTO):
        return PermissionResult(
            allowed=True,
            reason=f"Permission mode is {context.mode.value}",
        )

    # Explicit deny list
    if tool_name in context.denied_tools:
        return PermissionResult(
            allowed=False,
            reason=f"Tool '{tool_name}' is in the explicit deny list",
        )

    # Explicit allow list
    if tool_name in context.allowed_tools:
        return PermissionResult(
            allowed=True,
            reason=f"Tool '{tool_name}' is in the explicit allow list",
        )

    # Evaluate rules: deny-first
    deny_rules = [r for r in context.rules if not r.allowed]
    allow_rules = [r for r in context.rules if r.allowed]

    # Check deny rules first
    for rule in deny_rules:
        if fnmatch.fnmatch(tool_name, rule.tool_pattern):
            reason = rule.reason or f"Denied by rule matching '{rule.tool_pattern}'"
            return PermissionResult(allowed=False, reason=reason)

    # Check allow rules
    for rule in allow_rules:
        if fnmatch.fnmatch(tool_name, rule.tool_pattern):
            reason = rule.reason or f"Allowed by rule matching '{rule.tool_pattern}'"
            return PermissionResult(allowed=True, reason=reason)

    # No rules matched — default behavior
    if context.mode == PermissionMode.PLAN:
        return PermissionResult(
            allowed=False,
            reason="PLAN mode: tool not in any allow rule",
        )

    # DEFAULT mode: no rules matched, deny by default
    return PermissionResult(
        allowed=False,
        reason="No matching permission rule found; denied by default",
    )
