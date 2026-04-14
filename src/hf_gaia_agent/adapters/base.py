"""Source adapter protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..graph.state import AgentState
from ..source_pipeline import EvidenceRecord, QuestionProfile, SourceCandidate


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    skill_name: str

    def applies(
        self,
        profile: QuestionProfile,
        ranked_candidates: list[SourceCandidate],
    ) -> bool: ...

    def discover_sources(self, state: AgentState) -> list[str]: ...
    def fetch_grounded_records(self, state: AgentState) -> list[EvidenceRecord]: ...
