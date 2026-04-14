"""Base protocol for reusable core recoveries."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ...graph.state import AgentState
from ...source_pipeline import QuestionProfile


@runtime_checkable
class RecoveryStrategy(Protocol):
    name: str

    def applies(self, state: AgentState, profile: QuestionProfile) -> bool: ...
    def run(self, state: AgentState) -> dict[str, Any] | None: ...
