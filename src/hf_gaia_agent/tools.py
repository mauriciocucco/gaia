"""Tools used by the GAIA LangGraph agent."""

from __future__ import annotations

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
import csv
import json
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from pypdf import PdfReader


def _truncate(value: str, *, max_chars: int = 12000) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "\n...[truncated]"


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            rows.append(", ".join(row))
            if index >= 49:
                break
    return "\n".join(rows)


def _read_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return json.dumps(data, ensure_ascii=True, indent=2)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def read_file_content(path: str) -> str:
    """Read a local task attachment and return plain text content."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".csv":
        content = _read_csv(candidate)
    elif suffix == ".json":
        content = _read_json(candidate)
    elif suffix in {".html", ".htm"}:
        content = _html_to_text(candidate.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".pdf":
        content = _read_pdf(candidate)
    else:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    return _truncate(content)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return short snippets from the top results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."

    lines = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        href = item.get("href") or item.get("url") or ""
        body = item.get("body") or ""
        lines.append(f"{index}. {title}\nURL: {href}\nSnippet: {body}")
    return "\n\n".join(lines)


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return text extracted from the response body."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        text = json.dumps(response.json(), ensure_ascii=True, indent=2)
    elif "text/html" in content_type:
        text = _html_to_text(response.text)
    else:
        text = response.text
    return _truncate(text)


@tool
def read_local_file(path: str) -> str:
    """Read a local text, CSV, JSON, HTML, or PDF file."""
    return read_file_content(path)


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


def build_tools() -> list[Any]:
    return [web_search, fetch_url, read_local_file, calculate]
