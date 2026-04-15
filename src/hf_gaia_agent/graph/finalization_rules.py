"""Registered finalization rules for benchmark-specific answer handling."""

from __future__ import annotations

from typing import Any

from ..normalize import normalize_submitted_answer
from ..source_pipeline import serialize_evidence
from .answer_policy import is_invalid_final_response
from .contracts import FinalizationRule, FinalizationServices
from .evidence_support import (
    botanical_canonical_state_from_state,
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
        recovery_reason: str | None,
    ) -> dict[str, Any] | None:
        del answer_text
        botanical_state = botanical_canonical_state_from_state(state)
        if botanical_state and botanical_state.is_closed:
            return {
                "final_answer": botanical_state.canonical_answer,
                "error": None,
                "reducer_used": "botanical_classification",
                "evidence_used": serialize_evidence(botanical_state.used_records),
                "recovery_reason": None,
            }
        derived_answer = services.tool_derived_answer(state["messages"], state["question"])
        if derived_answer and not requires_botanical_classification_retry(state, derived_answer):
            return {
                "final_answer": derived_answer,
                "error": None,
                "recovery_reason": None,
            }
        return {
            "final_answer": "",
            "error": error or "Botanical classification answer lacked grounded evidence.",
            "recovery_reason": recovery_reason
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
        recovery_reason: str | None,
    ) -> dict[str, Any] | None:
        del answer_text
        derived_answer = services.tool_derived_answer(state["messages"], state["question"])
        if derived_answer and not requires_temporal_roster_retry(state, derived_answer):
            return {
                "final_answer": derived_answer,
                "error": None,
                "recovery_reason": None,
            }
        return {
            "final_answer": "",
            "error": error
            or "Date-sensitive roster answer lacked temporally grounded evidence.",
            "recovery_reason": recovery_reason or "temporal_roster_evidence_missing",
        }


def build_finalization_rules() -> list[FinalizationRule]:
    return [
        TemporalRosterFinalizationRule(),
        BotanicalClassificationFinalizationRule(),
    ]
