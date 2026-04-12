"""Compute tools — arithmetic calculation and Python code execution.

WARNING: ``execute_python_code`` runs arbitrary Python in a subprocess with no
sandboxing.  It is intended **only for local benchmark evaluation** and must
never be exposed to untrusted input in a production or multi-tenant
environment.  Use at your own risk.
"""

from __future__ import annotations

import subprocess
import sys
import os
import tempfile
from ast import (
    Add,
    BinOp,
    Constant,
    Div,
    Expression,
    FloorDiv,
    Load,
    Mod,
    Mult,
    Name,
    Pow,
    Sub,
    UAdd,
    USub,
    UnaryOp,
    parse,
    walk,
)
from typing import Any

from langchain_core.tools import tool

from ._http import truncate


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------

ALLOWED_BINOPS = {
    Add: lambda left, right: left + right,
    Sub: lambda left, right: left - right,
    Mult: lambda left, right: left * right,
    Div: lambda left, right: left / right,
    FloorDiv: lambda left, right: left // right,
    Mod: lambda left, right: left % right,
    Pow: lambda left, right: left**right,
}
ALLOWED_UNARYOPS = {
    UAdd: lambda value: +value,
    USub: lambda value: -value,
}


def _safe_eval(node: Any) -> float | int:
    if isinstance(node, Expression):
        return _safe_eval(node.body)
    if isinstance(node, Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, BinOp):
        operator = ALLOWED_BINOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported operator.")
        return operator(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, UnaryOp):
        operator = ALLOWED_UNARYOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported unary operator.")
        return operator(_safe_eval(node.operand))
    if isinstance(node, Name) and node.id in {"pi", "e"}:
        constants = {"pi": 3.141592653589793, "e": 2.718281828459045}
        return constants[node.id]
    raise ValueError("Unsafe expression.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    tree = parse(expression, mode="eval")
    for node in walk(tree):
        if type(node) not in {
            Expression,
            BinOp,
            UnaryOp,
            Constant,
            Load,
            Add,
            Sub,
            Mult,
            Div,
            FloorDiv,
            Mod,
            Pow,
            UAdd,
            USub,
            Name,
        }:
            raise ValueError("Unsafe expression.")

    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


# ---------------------------------------------------------------------------
# Python code execution (UNSAFE — local benchmark only)
# ---------------------------------------------------------------------------


@tool
def execute_python_code(code: str) -> str:
    """Execute generic Python scripts for data calculations or parsing and return stdout and stderr.

    .. warning::
        This tool executes arbitrary Python with **no sandboxing**.  It is
        intended for local benchmark evaluation only.  Never expose it to
        untrusted input.

    Always use ``print()`` to output final answers.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=45,
        )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        os.remove(temp_path)
        if not output.strip():
            return "Execution successful, but no stdout/stderr. Did you forget to print the result?"
        return truncate(output, max_chars=80000)
    except subprocess.TimeoutExpired:
        return "Execution timed out after 45 seconds."
    except Exception as e:
        return f"Execution failed: {e}"
