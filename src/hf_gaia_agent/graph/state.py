"""Agent state definition for the LangGraph workflow."""

from __future__ import annotations

import re
from typing import Any

from langgraph.graph import MessagesState


COMMON_ENGLISH_HINTS = {
    "the",
    "and",
    "answer",
    "write",
    "word",
    "sentence",
    "understand",
    "opposite",
    "left",
    "right",
    "video",
    "what",
    "how",
    "many",
    "between",
    "published",
}

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
SET_DEFINITION_RE = re.compile(r"set\s+s\s*=\s*\{(?P<body>[^}]+)\}", flags=re.IGNORECASE)


class AgentState(MessagesState):
    task_id: str
    question: str
    file_name: str | None
    local_file_path: str | None
    final_answer: str | None
    tool_trace: list[str]
    error: str | None
    iterations: int
    max_iterations: int
    decision_trace: list[str]
    evidence_used: list[dict[str, Any]]
    reducer_used: str | None
    recovery_reason: str | None
    question_profile: dict[str, Any]
    ranked_candidates: list[dict[str, Any]]
    search_history_normalized: list[str]
    search_history_fingerprints: list[str]
    structured_tool_outputs: list[dict[str, Any]]
    skill_trace: list[str]
    botanical_partial_records: list[dict[str, Any]]
    botanical_item_status: dict[str, dict[str, Any]]
    botanical_search_history: list[str]
