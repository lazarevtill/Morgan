"""Safe arithmetic calculator tool.

Uses a restricted AST walk — NO ``eval()`` of arbitrary code.

Supported operators: + - * / // ** % and parentheses, integer and float literals.
All other nodes (function calls, names, attributes, dunder methods, imports, …)
are rejected with a ``ToolResult(ok=False, error=…)``.

Resource-exhaustion guard
-------------------------
* ``**`` (Pow): rejects when the estimated result bit-length would exceed
  ``_MAX_RESULT_BITS`` (default 10 000).  Guards both integer and float exponents.
  The right-hand operand alone must not exceed ``_MAX_EXP`` (1 000).
* Any BinOp intermediate integer result whose ``bit_length()`` exceeds
  ``_MAX_RESULT_BITS`` is rejected immediately, preventing huge-Mult chains.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Union

from morgan_brain.interfaces.tools import ToolResult

# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------

# Maximum estimated bit-length of any intermediate or final integer result.
_MAX_RESULT_BITS: int = 10_000  # ~3 010 decimal digits

# Maximum allowed exponent (right-hand side of **).
_MAX_EXP: int = 1_000

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


def _check_int_magnitude(value: int, op_name: str) -> None:
    """Raise ``ValueError`` if *value* is an integer exceeding ``_MAX_RESULT_BITS``."""
    if value.bit_length() > _MAX_RESULT_BITS:
        raise ValueError(
            f"Result of {op_name} is too large "
            f"(> {_MAX_RESULT_BITS} bits); operation rejected to prevent DoS."
        )


def _guard_pow(left: _Number, right: _Number) -> _Number:
    """Apply exponentiation with resource-exhaustion guards.

    Raises ``ValueError`` if the result would be astronomically large.
    """
    # Guard: exponent must not exceed _MAX_EXP (catches 10**10**10 right side first)
    if isinstance(right, float):
        if not math.isfinite(right):
            raise ValueError("Non-finite exponent")
        abs_right = abs(right)
    else:
        abs_right = abs(right)

    if abs_right > _MAX_EXP:
        raise ValueError(
            f"Exponent {right!r} exceeds the maximum allowed value ({_MAX_EXP}); "
            "operation rejected to prevent DoS."
        )

    # Guard: estimate result bit-length for integer bases
    if isinstance(left, int) and isinstance(right, int) and right > 0:
        # bit_length of left**right ≈ right * bit_length(left)
        estimated_bits = right * max(abs(left).bit_length(), 1)
        if estimated_bits > _MAX_RESULT_BITS:
            raise ValueError(
                f"Result of {left}**{right} would require ~{estimated_bits} bits; "
                "operation rejected to prevent DoS."
            )

    raw = operator.pow(left, right)
    # Post-compute check for integer results (catches float→int edge cases)
    if isinstance(raw, int):
        _check_int_magnitude(raw, f"{left}**{right}")
    result: _Number = raw  # operator.pow returns int|float for int|float inputs
    return result


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

        # --- Resource-exhaustion guards -----------------------------------
        if isinstance(node.op, ast.Pow):
            return _guard_pow(left, right)

        result = op_fn(left, right)

        # Guard intermediate integer magnitude (catches huge-Mult chains)
        if isinstance(result, int):
            _check_int_magnitude(result, type(node.op).__name__)

        return result  # type: ignore[no-any-return]

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
