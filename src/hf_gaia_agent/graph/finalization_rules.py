"""Registered finalization rules for benchmark-specific answer handling."""

from __future__ import annotations

from typing import Any

from ..normalize import normalize_submitted_answer
from .answer_policy import is_invalid_final_response
from .contracts import FinalizationRule, FinalizationServices
from .evidence_support import (
    requires_botanical_classification_retry,
    requires_temporal_roster_retry,
)
from .routing import question_profile_from_state
from .state import AgentState


class BotanicalClassificationFinalizationRule:
    name = "botanical_classification"

    def applies(self, state: AgentState, answer_text: str) -> bool:
        return requires_botanical_classification_retry(state, answer_text)

    def finalize(
        self,
        state: AgentState,
        answer_text: str,
        *,
        services: FinalizationServices,
        error: str | None,
        fallback_reason: str | None,
    ) -> dict[str, Any] | None:
        del answer_text
        fallback_answer = services.fallback_tool_answer(state["messages"], state["question"])
        if fallback_answer and not requires_botanical_classification_retry(state, fallback_answer):
            return {
                "final_answer": fallback_answer,
                "error": None,
                "fallback_reason": None,
            }
        return {
            "final_answer": "",
            "error": error or "Botanical classification answer lacked grounded evidence.",
            "fallback_reason": fallback_reason
            or "botanical_classification_evidence_missing",
        }


class TemporalRosterFinalizationRule:
    name = "temporal_roster"

    def applies(self, state: AgentState, answer_text: str) -> bool:
        profile = question_profile_from_state(state)
        if profile.name != "temporal_ordered_list" or not profile.expected_date:
            return False
        normalized = normalize_submitted_answer(answer_text)
        if not normalized or is_invalid_final_response(normalized):
            return True
        return requires_temporal_roster_retry(state, normalized)

    def finalize(
        self,
        state: AgentState,
        answer_text: str,
        *,
        services: FinalizationServices,
        error: str | None,
        fallback_reason: str | None,
    ) -> dict[str, Any] | None:
        del answer_text
        targeted_roster_result = services.run_targeted_resolution("roster", state)
        if targeted_roster_result:
            return targeted_roster_result
        fallback_answer = services.fallback_tool_answer(state["messages"], state["question"])
        if fallback_answer and not requires_temporal_roster_retry(state, fallback_answer):
            return {
                "final_answer": fallback_answer,
                "error": None,
                "fallback_reason": None,
            }
        return {
            "final_answer": "",
            "error": error
            or "Date-sensitive roster answer lacked temporally grounded evidence.",
            "fallback_reason": fallback_reason or "temporal_roster_evidence_missing",
        }


def build_finalization_rules() -> list[FinalizationRule]:
    return [
        TemporalRosterFinalizationRule(),
        BotanicalClassificationFinalizationRule(),
    ]
