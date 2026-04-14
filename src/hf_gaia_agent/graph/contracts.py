"""Explicit graph service and rule contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage

from ..source_pipeline import EvidenceRecord, QuestionProfile, SourceCandidate
from .state import AgentState


@dataclass(frozen=True)
class ToolInvocationResult:
    text: str
    payloads: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class EvidenceServices(Protocol):
    def last_ai_message(self, messages: list[Any]) -> AIMessage | None: ...
    def fallback_tool_answer(self, messages: list[Any], question: str) -> str | None: ...
    def structured_answer_result(
        self, state: AgentState, *, preferred_only: bool = False
    ) -> dict[str, Any] | None: ...
    def salvage_answer_from_evidence(self, state: AgentState) -> str | None: ...
    def verify_answer_from_evidence(self, state: AgentState) -> str | None: ...
    def top_grounded_evidence_records(
        self, state: AgentState, *, limit: int = 6
    ) -> list[EvidenceRecord]: ...


@runtime_checkable
class FinalizationServices(EvidenceServices, Protocol):
    @property
    def finalization_rules(self) -> list["FinalizationRule"]: ...

    def run_core_recoveries(self, state: AgentState) -> dict[str, Any] | None: ...
    def run_skills(self, state: AgentState) -> dict[str, Any] | None: ...
    def run_skill(self, skill_name: str, state: AgentState) -> dict[str, Any] | None: ...
    def run_adapters(
        self, skill_name: str, state: AgentState
    ) -> dict[str, Any] | None: ...

    def run_fallback_resolvers(self, state: AgentState) -> dict[str, Any] | None: ...
    def run_named_fallback(
        self, resolver_name: str, state: AgentState
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class CandidateRankingServices(Protocol):
    def ranked_candidates_from_state(self, state: AgentState) -> list[SourceCandidate]: ...
    def merge_ranked_candidates(
        self,
        existing: list[SourceCandidate],
        new_items: list[SourceCandidate],
        *,
        max_items: int = 12,
    ) -> list[SourceCandidate]: ...
    def normalize_search_query(self, query: str) -> str: ...
    def is_semantically_duplicate_search(
        self, signature: str, previous_signatures: set[str]
    ) -> bool: ...
    def pick_best_unfetched_candidate(
        self, state: AgentState, *, fetched_urls: set[str]
    ) -> SourceCandidate | None: ...
    def pick_better_fetch_candidate(
        self,
        *,
        requested_url: str,
        profile: QuestionProfile,
        ranked_candidates: list[SourceCandidate],
        fetched_urls: set[str],
    ) -> SourceCandidate | None: ...


@runtime_checkable
class ToolExecutionServices(Protocol):
    @property
    def tool_names(self) -> set[str]: ...

    def execute_python_allowed(
        self, state: AgentState
    ) -> tuple[bool, str | None]: ...
    def article_to_paper_auto_links_result(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
        profile: QuestionProfile,
    ) -> tuple[dict[str, Any], str] | None: ...
    def text_span_auto_follow_candidate(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        result_text: str,
        profile: QuestionProfile,
        ranked_candidates: list[SourceCandidate],
        fetched_urls: set[str],
    ) -> SourceCandidate | None: ...
    def invoke_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> ToolInvocationResult: ...


@runtime_checkable
class AnswerRetryRule(Protocol):
    name: str

    def applies(self, state: AgentState, answer_text: str) -> bool: ...
    def guidance(self, state: AgentState) -> str | None: ...


@runtime_checkable
class FinalizationRule(Protocol):
    name: str

    def applies(self, state: AgentState, answer_text: str) -> bool: ...
    def finalize(
        self,
        state: AgentState,
        answer_text: str,
        *,
        services: FinalizationServices,
        error: str | None,
        fallback_reason: str | None,
    ) -> dict[str, Any] | None: ...
