"""Safe arithmetic calculator tool.

Uses a restricted AST walk — NO ``eval()`` of arbitrary code.

Supported operators: + - * / // ** % and parentheses, integer and float literals.
All other nodes (function calls, names, attributes, dunder methods, imports, …)
are rejected with a ``ToolResult(ok=False, error=…)``.
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Union

from morgan_brain.interfaces.tools import ToolResult

# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------

# The only AST node types the evaluator will descend into.
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)

_BINOP_MAP: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARYOP_MAP: dict[type[ast.unaryop], Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_Number = Union[int, float]


def _eval_node(node: ast.AST) -> _Number:
    """Recursively evaluate a safe AST node; raise ``ValueError`` for anything else."""
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        raise ValueError(
            f"Unsafe or unsupported expression node: {type(node).__name__!r}. "
            "Only arithmetic operators and numeric literals are allowed."
        )

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric literal: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_fn = _BINOP_MAP.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__!r}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero")
        if isinstance(node.op, ast.FloorDiv) and right == 0:
            raise ValueError("Division by zero")
        return op_fn(left, right)  # type: ignore[no-any-return]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_fn = _UNARYOP_MAP.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__!r}")
        return op_fn(operand)  # type: ignore[no-any-return]

    raise ValueError(f"Unhandled node type: {type(node).__name__!r}")


def safe_eval(expression: str) -> _Number:
    """Parse *expression* and evaluate it safely.

    Raises ``ValueError`` for any expression that is not pure arithmetic.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc
    return _eval_node(tree)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CalculatorTool:
    """Safe arithmetic evaluator (+ - * / // ** %).

    Does NOT use ``eval()``; rejects any expression that contains
    names, function calls, attributes, dunder methods, or imports.
    """

    name = "calculator"
    description = (
        "Evaluate a safe arithmetic expression (supports +, -, *, /, //, **, %)."
    )

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 3 * 4'.",
                }
            },
            "required": ["expression"],
        }

    async def run(self, *, user_id: str, expression: str, **_: Any) -> ToolResult:
        try:
            result = safe_eval(expression)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, output=result)
