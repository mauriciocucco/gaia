"""Base protocol and registry for fallback resolvers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..source_pipeline import QuestionProfile
from ..graph.state import AgentState


@runtime_checkable
class FallbackResolver(Protocol):
    name: str

    def applies(self, state: AgentState, profile: QuestionProfile) -> bool: ...
    def run(self, state: AgentState) -> dict[str, Any] | None: ...
