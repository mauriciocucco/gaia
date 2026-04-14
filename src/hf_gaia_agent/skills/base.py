"""Skill protocols and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..graph.state import AgentState
from ..source_pipeline import QuestionProfile


@dataclass(frozen=True)
class SkillResult:
    final_answer: str
    error: str | None = None
    reducer_used: str | None = None
    skill_used: str | None = None
    evidence_used: list[dict[str, Any]] = field(default_factory=list)
    skill_trace: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    tool_trace: list[str] = field(default_factory=list)
    decision_trace: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "error": self.error,
            "reducer_used": self.reducer_used,
            "skill_used": self.skill_used,
            "evidence_used": self.evidence_used,
            "skill_trace": self.skill_trace,
            "fallback_reason": self.fallback_reason,
            "tool_trace": self.tool_trace,
            "decision_trace": self.decision_trace,
        }


@runtime_checkable
class Skill(Protocol):
    name: str

    def applies(self, state: AgentState, profile: QuestionProfile) -> bool: ...
    def run(self, state: AgentState) -> dict[str, Any] | None: ...
