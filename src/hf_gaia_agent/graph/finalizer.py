"""Final answer orchestration for the graph workflow."""

from __future__ import annotations

from typing import Any

from ..normalize import normalize_submitted_answer
from ..source_pipeline import serialize_evidence
from .answer_policy import (
    attachment_required_but_missing,
    is_invalid_final_response,
    is_missing_attachment_non_answer,
    looks_like_placeholder_answer,
)
from .contracts import FinalizationServices
from .routing import question_profile_from_state
from .state import AgentState


class WorkflowFinalizer:
    """Collects the final-answer decision tree outside the workflow shell."""

    def __init__(self, services: FinalizationServices):
        self._services = services

    def finalize(self, state: AgentState) -> dict[str, Any]:
        if state.get("final_answer"):
            return {
                "final_answer": state.get("final_answer"),
                "error": state.get("error"),
                "reducer_used": state.get("reducer_used"),
                "evidence_used": state.get("evidence_used", []),
                "recovery_reason": state.get("recovery_reason"),
            }

        last_ai = self._services.last_ai_message(state["messages"])
        raw_answer = last_ai.content if last_ai else ""
        final_answer = normalize_submitted_answer(str(raw_answer))
        error = state.get("error")
        reducer_used = state.get("reducer_used")
        evidence_used = list(state.get("evidence_used") or [])
        recovery_reason = state.get("recovery_reason")

        if attachment_required_but_missing(
            question=state["question"],
            file_name=state.get("file_name"),
            local_file_path=state.get("local_file_path"),
        ):
            derived_answer = self._services.tool_derived_answer(
                state["messages"],
                state["question"],
            )
            if derived_answer:
                return {
                    "final_answer": derived_answer,
                    "error": None,
                    "recovery_reason": None,
                }
            return {
                "final_answer": "",
                "error": error or "Required attachment was not available locally.",
                "recovery_reason": recovery_reason or "attachment_missing",
            }

        preferred_structured_result = self._services.structured_answer_result(
            state,
            preferred_only=True,
        )
        if preferred_structured_result:
            return preferred_structured_result

        run_core_recoveries = getattr(self._services, "run_core_recoveries", None)
        core_recovery = run_core_recoveries(state) if callable(run_core_recoveries) else None
        if core_recovery:
            return core_recovery

        run_skills = getattr(self._services, "run_skills", None)
        skill_result = run_skills(state) if callable(run_skills) else None
        if skill_result:
            return skill_result

        profile = question_profile_from_state(state)
        if profile.name == "temporal_ordered_list":
            run_adapters = getattr(self._services, "run_adapters", None)
            adapter_result = (
                run_adapters("temporal_ordered_list", state)
                if callable(run_adapters)
                else None
            )
            if adapter_result:
                return adapter_result

        for rule in self._services.finalization_rules:
            if not rule.applies(state, final_answer):
                continue
            result = rule.finalize(
                state,
                final_answer,
                services=self._services,
                error=error,
                recovery_reason=recovery_reason,
            )
            if result:
                return result

        if (
            is_invalid_final_response(final_answer)
            or is_missing_attachment_non_answer(
                final_answer,
                file_name=state.get("file_name"),
                local_file_path=state.get("local_file_path"),
            )
            or looks_like_placeholder_answer(state["question"], final_answer)
        ):
            recovered = self._recover_from_evidence(state)
            if recovered:
                return recovered
            final_answer = ""
            error = error or "Model produced an invalid non-answer."
            recovery_reason = recovery_reason or "invalid_model_non_answer"

        if not final_answer:
            recovered = self._recover_from_evidence(state)
            if recovered:
                return recovered
            error = error or "Model did not produce a final answer."
            recovery_reason = recovery_reason or "missing_final_answer"

        return {
            "final_answer": final_answer,
            "error": error,
            "reducer_used": reducer_used,
            "evidence_used": evidence_used,
            "recovery_reason": recovery_reason,
        }

    def _recover_from_evidence(self, state: AgentState) -> dict[str, Any] | None:
        derived_answer = self._services.tool_derived_answer(
            state["messages"],
            state["question"],
        )
        if derived_answer:
            return {
                "final_answer": derived_answer,
                "error": None,
                "recovery_reason": None,
            }

        structured_result = self._services.structured_answer_result(state)
        if structured_result:
            return structured_result

        salvaged_answer = self._services.salvage_answer_from_evidence(state)
        if salvaged_answer:
            return {
                "final_answer": salvaged_answer,
                "error": None,
                "evidence_used": serialize_evidence(
                    self._services.top_grounded_evidence_records(state)
                ),
                "recovery_reason": None,
            }

        verified_answer = self._services.verify_answer_from_evidence(state)
        if verified_answer:
            return {
                "final_answer": verified_answer,
                "error": None,
                "evidence_used": serialize_evidence(
                    self._services.top_grounded_evidence_records(state)
                ),
                "recovery_reason": None,
            }
        return None
