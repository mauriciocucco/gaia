"""Concrete implementation of the explicit workflow services interface."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..adapters import SourceAdapter
from ..core.recoveries import RecoveryStrategy
from ..evidence_solver import solve_answer_from_evidence_records
from ..hooks import AgentHook
from ..normalize import normalize_submitted_answer
from ..source_pipeline import QuestionProfile, SourceCandidate, serialize_evidence
from ..skills import Skill
from .. import tools as tools_module
from ..tools import STRUCTURED_TOOL_INVOKERS
from ..tools._payloads import StructuredToolResult, serialize_tool_payloads
from .candidate_support import (
    article_to_paper_auto_links_result,
    execute_python_allowed,
    is_semantically_duplicate_search,
    merge_ranked_candidates,
    normalize_search_query,
    pick_best_unfetched_candidate,
    pick_better_fetch_candidate,
    ranked_candidates_from_state,
    text_span_auto_follow_candidate,
)
from .contracts import ToolInvocationResult
from .evidence_support import (
    format_grounded_evidence_for_llm,
    grounded_temporal_roster_answer,
    last_ai_message,
    should_prefer_structured_answer,
    structured_answer_from_state,
    tool_derived_answer,
    top_grounded_evidence_records,
)
from .answer_policy import is_invalid_final_response
from .finalization_rules import build_finalization_rules
from .routing import question_profile_from_state
from .state import AgentState


class GraphWorkflowServices:
    def __init__(
        self,
        *,
        answer_model: Any,
        tools_by_name: dict[str, Any],
        core_recoveries: list[RecoveryStrategy],
        skills: list[Skill],
        source_adapters: list[SourceAdapter],
        hook: AgentHook,
    ) -> None:
        self._answer_model = answer_model
        self._tools_by_name = tools_by_name
        self._core_recoveries = list(core_recoveries)
        self._skills = list(skills)
        self._source_adapters = list(source_adapters)
        self._hook = hook
        self._finalization_rules = build_finalization_rules()

    @property
    def core_recoveries(self) -> list[RecoveryStrategy]:
        return list(self._core_recoveries)

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)

    @property
    def source_adapters(self) -> list[SourceAdapter]:
        return list(self._source_adapters)

    @property
    def finalization_rules(self):
        return list(self._finalization_rules)

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools_by_name)

    def last_ai_message(self, messages: list[Any]):
        return last_ai_message(messages)

    def tool_derived_answer(self, messages: list[Any], question: str) -> str | None:
        return tool_derived_answer(messages, question)

    def structured_answer_result(
        self, state: AgentState, *, preferred_only: bool = False
    ) -> dict[str, Any] | None:
        structured_answer, reducer_used, used_records = structured_answer_from_state(state)
        if not structured_answer:
            return None
        profile = question_profile_from_state(state)
        labels = profile.classification_labels or {}
        if (
            profile.name == "list_item_classification"
            and labels.get("include") == "vegetable"
            and labels.get("exclude") == "fruit"
            and reducer_used != "botanical_classification"
        ):
            return None
        if preferred_only and not should_prefer_structured_answer(
            profile=profile,
            reducer_used=reducer_used,
        ):
            return None
        if reducer_used == "roster_neighbor" and profile.name == "temporal_ordered_list" and profile.expected_date:
            grounded_answer = grounded_temporal_roster_answer(state)
            if grounded_answer is None:
                return None
            structured_answer = grounded_answer
        return {
            "final_answer": structured_answer,
            "error": None,
            "reducer_used": reducer_used,
            "evidence_used": serialize_evidence(used_records),
            "recovery_reason": None,
        }

    def salvage_answer_from_evidence(self, state: AgentState) -> str | None:
        grounded_records = top_grounded_evidence_records(state)
        if not grounded_records:
            return None
        profile = question_profile_from_state(state)
        evidence_text = format_grounded_evidence_for_llm(grounded_records)[-12000:]
        response = self._answer_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Answer the question using only the provided evidence from previous tool outputs. "
                        "Do not mention uncertainty, access limits, missing pages, or search strategy. "
                        "If the evidence is insufficient, respond exactly with [INSUFFICIENT]. "
                        "If the evidence is sufficient, respond only with [ANSWER]final answer[/ANSWER]. "
                        "Return the shortest grounded answer that satisfies the requested format."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['question']}\n\n"
                        f"Question profile:\n{profile.as_dict()}\n\n"
                        f"Evidence:\n{evidence_text}"
                    )
                ),
            ]
        )
        content = str(getattr(response, "content", "") or "").strip()
        if content == "[INSUFFICIENT]":
            return None
        candidate = normalize_submitted_answer(content)
        if not candidate or is_invalid_final_response(candidate):
            return None
        return candidate

    def verify_answer_from_evidence(self, state: AgentState) -> str | None:
        grounded_records = top_grounded_evidence_records(state)
        if not grounded_records:
            return None
        evidence_text = format_grounded_evidence_for_llm(grounded_records)[-12000:]
        response = self._answer_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Review the evidence one final time. "
                        "If it directly supports a short factual answer, respond only with [ANSWER]final answer[/ANSWER]. "
                        "If not, respond exactly with [INSUFFICIENT]. "
                        "Do not mention uncertainty or propose new searches."
                    )
                ),
                HumanMessage(
                    content=f"Question:\n{state['question']}\n\nEvidence:\n{evidence_text}"
                ),
            ]
        )
        content = str(getattr(response, "content", "") or "").strip()
        if content == "[INSUFFICIENT]":
            return None
        candidate = normalize_submitted_answer(content)
        if not candidate or is_invalid_final_response(candidate):
            return None
        return candidate

    def top_grounded_evidence_records(
        self, state: AgentState, *, limit: int = 6
    ):
        return top_grounded_evidence_records(state, limit=limit)

    def run_core_recoveries(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        for recovery in self._core_recoveries:
            if not recovery.applies(state, profile):
                continue
            result = recovery.run(state)
            if result:
                return result
        return None

    def run_skills(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        current_state = state
        trace_updates: dict[str, Any] | None = None
        for skill in self._skills:
            if not skill.applies(state, profile):
                continue
            result = skill.run(current_state)
            if not result:
                continue
            if result.get("final_answer") is not None:
                return result
            current_state = self._state_with_trace_updates(current_state, result)
            trace_updates = {
                "decision_trace": list(current_state.get("decision_trace") or []),
                "tool_trace": list(current_state.get("tool_trace") or []),
                "ranked_candidates": list(current_state.get("ranked_candidates") or []),
                "search_history_fingerprints": list(
                    current_state.get("search_history_fingerprints") or []
                ),
                "botanical_partial_records": list(
                    current_state.get("botanical_partial_records") or []
                ),
                "botanical_item_status": dict(
                    current_state.get("botanical_item_status") or {}
                ),
                "botanical_search_history": list(
                    current_state.get("botanical_search_history") or []
                ),
            }
        return trace_updates

    def run_skill(self, skill_name: str, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        for skill in self._skills:
            if skill.name != skill_name or not skill.applies(state, profile):
                continue
            return skill.run(state)
        return None

    def run_adapters(
        self, skill_name: str, state: AgentState
    ) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        ranked_candidates = ranked_candidates_from_state(state)
        for adapter in self._source_adapters:
            if adapter.skill_name != skill_name or not adapter.applies(profile, ranked_candidates):
                continue
            records = adapter.fetch_grounded_records(state)
            if not records:
                continue
            answer, reducer = solve_answer_from_evidence_records(state["question"], records)
            if not answer:
                continue
            return {
                "final_answer": answer,
                "error": None,
                "reducer_used": reducer,
                "skill_used": skill_name,
                "skill_trace": [skill_name, adapter.name],
                "evidence_used": serialize_evidence(records[-6:]),
                "recovery_reason": None,
                "tool_trace": list(getattr(adapter, "last_tool_trace", []) or []),
                "decision_trace": list(getattr(adapter, "last_decision_trace", []) or []),
            }
        return None

    def run_resolution_pipeline(self, state: AgentState) -> dict[str, Any] | None:
        structured_result = self.structured_answer_result(state)
        if structured_result:
            return structured_result

        current_state = state
        for resolver in (
            self.run_core_recoveries,
            self.run_skills,
            self._run_applicable_adapters,
        ):
            result = resolver(current_state)
            if not result:
                continue
            if result.get("final_answer") is not None:
                return result
            current_state = self._state_with_trace_updates(current_state, result)
        if current_state is state:
            return None
        return {
            "decision_trace": list(current_state.get("decision_trace") or []),
            "tool_trace": list(current_state.get("tool_trace") or []),
            "ranked_candidates": list(current_state.get("ranked_candidates") or []),
            "search_history_fingerprints": list(
                current_state.get("search_history_fingerprints") or []
            ),
            "botanical_partial_records": list(
                current_state.get("botanical_partial_records") or []
            ),
            "botanical_item_status": dict(current_state.get("botanical_item_status") or {}),
            "botanical_search_history": list(
                current_state.get("botanical_search_history") or []
            ),
        }

    def run_targeted_resolution(
        self, resolution_name: str, state: AgentState
    ) -> dict[str, Any] | None:
        if resolution_name == "roster":
            return self.run_adapters("temporal_ordered_list", state)
        if resolution_name == "competition":
            return self.run_skill("competition_gaia", state)
        if resolution_name == "role_chain":
            return self.run_skill("role_chain_gaia", state)
        if resolution_name == "botanical":
            return self.run_skill("botanical_gaia", state)
        profile = question_profile_from_state(state)
        for recovery in self._core_recoveries:
            if recovery.name != resolution_name or not recovery.applies(state, profile):
                continue
            return recovery.run(state)
        return None

    @staticmethod
    def _state_with_trace_updates(
        state: AgentState, updates: dict[str, Any]
    ) -> AgentState:
        merged = dict(state)
        for key in (
            "decision_trace",
            "tool_trace",
            "ranked_candidates",
            "search_history_fingerprints",
            "botanical_partial_records",
            "botanical_item_status",
            "botanical_search_history",
        ):
            if key in updates:
                raw_value = updates.get(key)
                if key == "botanical_item_status":
                    merged[key] = dict(raw_value or {})
                else:
                    merged[key] = list(raw_value or [])
        return merged  # type: ignore[return-value]

    def _run_applicable_adapters(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        ranked_candidates = ranked_candidates_from_state(state)
        for adapter in self._source_adapters:
            if not adapter.applies(profile, ranked_candidates):
                continue
            records = adapter.fetch_grounded_records(state)
            if not records:
                continue
            answer, reducer = solve_answer_from_evidence_records(state["question"], records)
            if not answer:
                continue
            return {
                "final_answer": answer,
                "error": None,
                "reducer_used": reducer,
                "skill_used": adapter.skill_name,
                "skill_trace": [adapter.skill_name, adapter.name],
                "evidence_used": serialize_evidence(records[-6:]),
                "recovery_reason": None,
                "tool_trace": list(getattr(adapter, "last_tool_trace", []) or []),
                "decision_trace": list(getattr(adapter, "last_decision_trace", []) or []),
            }
        return None

    def ranked_candidates_from_state(self, state: AgentState) -> list[SourceCandidate]:
        return ranked_candidates_from_state(state)

    def merge_ranked_candidates(
        self,
        existing: list[SourceCandidate],
        new_items: list[SourceCandidate],
        *,
        max_items: int = 12,
    ) -> list[SourceCandidate]:
        return merge_ranked_candidates(existing, new_items, max_items=max_items)

    def normalize_search_query(self, query: str) -> str:
        return normalize_search_query(query)

    def is_semantically_duplicate_search(
        self, signature: str, previous_signatures: set[str]
    ) -> bool:
        return is_semantically_duplicate_search(signature, previous_signatures)

    def pick_best_unfetched_candidate(
        self, state: AgentState, *, fetched_urls: set[str]
    ) -> SourceCandidate | None:
        return pick_best_unfetched_candidate(state, fetched_urls=fetched_urls)

    def pick_better_fetch_candidate(
        self,
        *,
        requested_url: str,
        profile: QuestionProfile,
        ranked_candidates: list[SourceCandidate],
        fetched_urls: set[str],
    ) -> SourceCandidate | None:
        return pick_better_fetch_candidate(
            requested_url=requested_url,
            profile=profile,
            ranked_candidates=ranked_candidates,
            fetched_urls=fetched_urls,
        )

    def execute_python_allowed(
        self, state: AgentState
    ) -> tuple[bool, str | None]:
        return execute_python_allowed(state)

    def article_to_paper_auto_links_result(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
        profile: QuestionProfile,
    ) -> tuple[dict[str, Any], str] | None:
        return article_to_paper_auto_links_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result_text=result_text,
            profile=profile,
        )

    def text_span_auto_follow_candidate(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
        profile: QuestionProfile,
        ranked_candidates: list[SourceCandidate],
        fetched_urls: set[str],
    ) -> SourceCandidate | None:
        return text_span_auto_follow_candidate(
            tool_name=tool_name,
            tool_args=tool_args,
            result_text=result_text,
            profile=profile,
            ranked_candidates=ranked_candidates,
            fetched_urls=fetched_urls,
        )

    def invoke_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> ToolInvocationResult:
        tool_ = self._tools_by_name[tool_name]
        self._hook.on_tool_start(tool_name, tool_args)
        try:
            structured_invoke = None
            if tool_ is getattr(tools_module, tool_name, None):
                structured_invoke = STRUCTURED_TOOL_INVOKERS.get(tool_name)
            if callable(structured_invoke):
                result = structured_invoke(**tool_args)
                if isinstance(result, StructuredToolResult):
                    rendered_text = result.text
                    payloads = serialize_tool_payloads(result.payloads)
                else:
                    rendered_text = str(result)
                    payloads = []
            else:
                rendered_text = str(tool_.invoke(tool_args))
                payloads = []
        except Exception as exc:
            rendered_text = f"Tool error: {exc}"
            payloads = []
        self._hook.on_tool_end(tool_name, rendered_text)
        return ToolInvocationResult(text=rendered_text, payloads=payloads)
