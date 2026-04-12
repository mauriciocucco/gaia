"""Competition nationality fallback resolver."""

from __future__ import annotations

import re
from typing import Any

from ..evidence_solver import solve_answer_from_evidence_records
from ..source_pipeline import (
    EvidenceRecord,
    evidence_records_from_tool_output,
    serialize_evidence,
)
from ..graph.state import AgentState
from .base import FallbackResolver
from .utils import (
    fallback_trace_state,
    invoke_fallback_tool,
    with_fallback_traces,
)


class CompetitionFallback:
    name = "competition"

    def __init__(self, tools_by_name: dict[str, Any]):
        self._tools_by_name = tools_by_name

    def applies(self, state: AgentState, profile: Any) -> bool:
        return _is_competition_nationality_question(state.get("question", ""))

    def run(self, state: AgentState) -> dict[str, Any] | None:
        question = state.get("question", "")
        if not _is_competition_nationality_question(question):
            return None
        if "extract_tables_from_url" not in self._tools_by_name:
            return None

        competition_match = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Competition|Award|Prize))\b",
            question,
        )
        if not competition_match:
            return None

        competition_name = competition_match.group(1).replace(" ", "_")
        wiki_url = f"https://en.wikipedia.org/wiki/{competition_name}"

        context = fallback_trace_state(tools_by_name=self._tools_by_name, state=state)

        table_text = invoke_fallback_tool(
            context=context,
            tool_name="extract_tables_from_url",
            tool_args={"url": wiki_url, "text_filter": "recipient", "max_tables": 5},
            trace_label="competition_nationality_fallback",
        )

        records = evidence_records_from_tool_output("extract_tables_from_url", table_text)
        if not records:
            return None

        answer, reducer = solve_answer_from_evidence_records(question, records)
        if not answer:
            return None

        return with_fallback_traces(
            {
                "final_answer": answer,
                "error": None,
                "reducer_used": reducer or "competition_nationality_fallback",
                "evidence_used": serialize_evidence(records[-3:]),
                "fallback_reason": None,
            },
            context=context,
        )


def _is_competition_nationality_question(question: str) -> bool:
    lowered = question.lower()
    return (
        ("competition" in lowered or "award" in lowered)
        and ("recipient" in lowered or "winner" in lowered)
        and ("nationality" in lowered or "no longer exists" in lowered)
    )
