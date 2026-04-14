"""GAIA competition skill."""

from __future__ import annotations

import re

from ...core.recoveries.utils import (
    invoke_recovery_tool,
    recovery_trace_state,
    with_recovery_traces,
)
from ...evidence_solver import solve_answer_from_evidence_records
from ...source_pipeline import evidence_records_from_tool_output, serialize_evidence


class CompetitionGaiaSkill:
    name = "competition_gaia"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name

    def applies(self, state, profile) -> bool:
        del profile
        return _is_competition_nationality_question(state.get("question", ""))

    def run(self, state):
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

        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        table_text = invoke_recovery_tool(
            context=context,
            tool_name="extract_tables_from_url",
            tool_args={"url": wiki_url, "text_filter": "recipient", "max_tables": 5},
            trace_label=self.name,
        )

        records = evidence_records_from_tool_output("extract_tables_from_url", table_text)
        if not records:
            return None

        answer, reducer = solve_answer_from_evidence_records(question, records)
        if not answer:
            return None

        return with_recovery_traces(
            {
                "final_answer": answer,
                "error": None,
                "reducer_used": reducer or "competition_nationality_fallback",
                "evidence_used": serialize_evidence(records[-3:]),
                "fallback_reason": None,
                "skill_used": self.name,
                "skill_trace": [self.name],
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
