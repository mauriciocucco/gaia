"""Concrete implementation of the explicit workflow services interface."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..fallbacks import FallbackResolver
from ..hooks import AgentHook
from ..normalize import normalize_submitted_answer
from ..source_pipeline import QuestionProfile, SourceCandidate, serialize_evidence
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
    fallback_tool_answer,
    format_grounded_evidence_for_llm,
    grounded_temporal_roster_answer,
    last_ai_message,
    should_prefer_structured_answer,
    structured_answer_from_state,
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
        fallback_resolvers: list[FallbackResolver],
        hook: AgentHook,
    ) -> None:
        self._answer_model = answer_model
        self._tools_by_name = tools_by_name
        self._fallback_resolvers = list(fallback_resolvers)
        self._hook = hook
        self._finalization_rules = build_finalization_rules()

    @property
    def fallback_resolvers(self) -> list[FallbackResolver]:
        return list(self._fallback_resolvers)

    @property
    def finalization_rules(self):
        return list(self._finalization_rules)

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools_by_name)

    def last_ai_message(self, messages: list[Any]):
        return last_ai_message(messages)

    def fallback_tool_answer(self, messages: list[Any], question: str) -> str | None:
        return fallback_tool_answer(messages, question)

    def structured_answer_result(
        self, state: AgentState, *, preferred_only: bool = False
    ) -> dict[str, Any] | None:
        structured_answer, reducer_used, used_records = structured_answer_from_state(state)
        if not structured_answer:
            return None
        profile = question_profile_from_state(state)
        if preferred_only and not should_prefer_structured_answer(
            profile=profile,
            reducer_used=reducer_used,
        ):
            return None
        if reducer_used == "roster_neighbor" and profile.name == "roster_neighbor_lookup" and profile.expected_date:
            grounded_answer = grounded_temporal_roster_answer(state)
            if grounded_answer is None:
                return None
            structured_answer = grounded_answer
        return {
            "final_answer": structured_answer,
            "error": None,
            "reducer_used": reducer_used,
            "evidence_used": serialize_evidence(used_records),
            "fallback_reason": None,
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

    def run_fallback_resolvers(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        for resolver in self._fallback_resolvers:
            if resolver.applies(state, profile):
                result = resolver.run(state)
                if result:
                    return result
        return None

    def run_named_fallback(
        self, resolver_name: str, state: AgentState
    ) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        for resolver in self._fallback_resolvers:
            if resolver.name != resolver_name or not resolver.applies(state, profile):
                continue
            return resolver.run(state)
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
