"""
Safe calculator tool using ast.parse.

Evaluates mathematical expressions safely without ever calling eval() or exec().
Supports basic arithmetic, exponentiation, and common math functions.

Security: Uses ast.parse to build an AST, then walks the tree and evaluates
only known-safe node types. This prevents code injection.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict, Optional

from morgan.tools.base import BaseTool, ToolContext, ToolInputSchema, ToolResult

# Safe binary operators
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

# Safe unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
}

# Safe comparison operators
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Safe math functions and constants
_SAFE_NAMES: Dict[str, Any] = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
    # Functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    # Math module functions
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": pow,
}


def _safe_eval_node(node: ast.AST) -> Any:
    """
    Recursively evaluate an AST node using only safe operations.

    Raises ValueError for any unsupported node type.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Guard against huge exponents
        if op_type == ast.Pow:
            if isinstance(right, (int, float)) and abs(right) > 10000:
                raise ValueError(
                    f"Exponent too large: {right}. Maximum allowed is 10000."
                )
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _UNARY_OPS[op_type](operand)

    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _COMPARE_OPS:
                raise ValueError(f"Unsupported comparison: {op_type.__name__}")
            right = _safe_eval_node(comparator)
            if not _COMPARE_OPS[op_type](left, right):
                result = False
                break
            left = right
        return result

    if isinstance(node, ast.Name):
        name = node.id
        if name not in _SAFE_NAMES:
            raise ValueError(
                f"Unknown name: '{name}'. "
                f"Available: {', '.join(sorted(_SAFE_NAMES.keys()))}"
            )
        return _SAFE_NAMES[name]

    if isinstance(node, ast.Call):
        func = _safe_eval_node(node.func)
        if not callable(func):
            raise ValueError(f"'{func}' is not callable")
        args = [_safe_eval_node(arg) for arg in node.args]
        # No keyword arguments for safety
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        return func(*args)

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(elt) for elt in node.elts)

    if isinstance(node, ast.List):
        return [_safe_eval_node(elt) for elt in node.elts]

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """
    Safely evaluate a mathematical expression.

    Uses ast.parse to parse the expression into an AST, then evaluates
    only known-safe node types. Never uses Python's built-in eval/exec.

    Args:
        expression: A mathematical expression string.

    Returns:
        The computed result.

    Raises:
        ValueError: If the expression contains unsupported operations.
        SyntaxError: If the expression is not valid Python syntax.
    """
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


class CalculatorTool(BaseTool):
    """
    Safe math calculator tool.

    Evaluates mathematical expressions using ast.parse-based safe evaluation.
    Supports arithmetic, comparisons, and common math functions.
    Does NOT use Python's built-in eval() or any form of dynamic code execution.

    Examples:
        {"expression": "2 + 2"}           -> "4"
        {"expression": "sqrt(144)"}        -> "12.0"
        {"expression": "sin(pi / 2)"}      -> "1.0"
        {"expression": "factorial(10)"}    -> "3628800"
    """

    name = "calculator"
    description = "Evaluate mathematical expressions safely"
    aliases = ("calc", "math")
    input_schema = ToolInputSchema(
        properties={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
            },
        },
        required=("expression",),
    )

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Validate calculator input."""
        base_error = super().validate_input(input_data)
        if base_error:
            return base_error

        expr = input_data.get("expression", "")
        if not isinstance(expr, str):
            return "Expression must be a string"
        if not expr.strip():
            return "Expression cannot be empty"
        if len(expr) > 1000:
            return "Expression too long (max 1000 characters)"
        return None

    async def execute(
        self, input_data: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Evaluate the mathematical expression."""
        expression = input_data["expression"].strip()

        try:
            result = safe_eval(expression)
            return ToolResult(
                output=str(result),
                metadata={
                    "expression": expression,
                    "result_type": type(result).__name__,
                },
            )
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
            return ToolResult(
                output=f"Calculation error: {e}",
                is_error=True,
                error_code="CALCULATION_ERROR",
                metadata={"expression": expression},
            )
        except OverflowError:
            return ToolResult(
                output="Calculation error: result too large (overflow)",
                is_error=True,
                error_code="OVERFLOW_ERROR",
                metadata={"expression": expression},
            )
