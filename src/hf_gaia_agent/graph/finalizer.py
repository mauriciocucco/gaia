"""Final answer orchestration for the graph workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from ..normalize import normalize_submitted_answer
from ..source_pipeline import serialize_evidence
from .answer_policy import (
    attachment_required_but_missing,
    is_invalid_final_response,
    is_missing_attachment_non_answer,
    looks_like_placeholder_answer,
)
from .contracts import FinalizationServices
from .state import AgentState


class WorkflowFinalizer:
    """Collects the final-answer decision tree outside the workflow shell."""

    def __init__(self, services: FinalizationServices):
        self._services = services

    def finalize(self, state: AgentState) -> dict[str, Any]:
        if state.get("final_answer"):
            return self._with_state_metadata(state, {
                "final_answer": state.get("final_answer"),
                "error": state.get("error"),
                "reducer_used": state.get("reducer_used"),
                "evidence_used": state.get("evidence_used", []),
                "recovery_reason": state.get("recovery_reason"),
            })

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
                return self._with_state_metadata(state, {
                    "final_answer": derived_answer,
                    "error": None,
                    "recovery_reason": None,
                })
            return self._with_state_metadata(state, {
                "final_answer": "",
                "error": error or "Required attachment was not available locally.",
                "recovery_reason": recovery_reason or "attachment_missing",
            })

        preferred_structured_result = self._services.structured_answer_result(
            state,
            preferred_only=True,
        )
        if preferred_structured_result:
            return self._with_state_metadata(state, preferred_structured_result)

        run_resolution_pipeline = getattr(self._services, "run_resolution_pipeline", None)
        has_resolution_inputs = self._has_resolution_inputs(state)
        resolution_result = (
            run_resolution_pipeline(state)
            if callable(run_resolution_pipeline)
            and (
                has_resolution_inputs
                or state.get("iterations", 0) >= state.get("max_iterations", 0)
            )
            else None
        )
        if resolution_result and resolution_result.get("final_answer") is not None:
            return self._with_state_metadata(state, resolution_result)
        if resolution_result:
            if has_resolution_inputs:
                state = {**state, **resolution_result}
            else:
                state = {
                    **state,
                    "decision_trace": list(
                        resolution_result.get("decision_trace") or state.get("decision_trace") or []
                    ),
                }

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
                return self._with_state_metadata(state, result)

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
                return self._with_state_metadata(state, recovered)
            final_answer = ""
            error = error or "Model produced an invalid non-answer."
            recovery_reason = recovery_reason or "invalid_model_non_answer"

        if not final_answer:
            recovered = self._recover_from_evidence(state)
            if recovered:
                return self._with_state_metadata(state, recovered)
            error = error or "Model did not produce a final answer."
            recovery_reason = recovery_reason or "missing_final_answer"

        return self._with_state_metadata(state, {
            "final_answer": final_answer,
            "error": error,
            "reducer_used": reducer_used,
            "evidence_used": evidence_used,
            "recovery_reason": recovery_reason,
        })

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

    @staticmethod
    def _with_state_metadata(state: AgentState, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result.setdefault("tool_trace", list(state.get("tool_trace") or []))
        result.setdefault("decision_trace", list(state.get("decision_trace") or []))
        result.setdefault("skill_trace", list(state.get("skill_trace") or []))
        return result

    @staticmethod
    def _has_resolution_inputs(state: AgentState) -> bool:
        if state.get("structured_tool_outputs"):
            return True
        if state.get("tool_trace"):
            return True
        return any(isinstance(message, ToolMessage) for message in state.get("messages", []))
