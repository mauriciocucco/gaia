"""Graph package for the decomposed LangGraph workflow."""

from __future__ import annotations

from .state import AgentState

# Re-export names that tests and external code monkeypatch on hf_gaia_agent.graph.
from ..tools import build_tools  # noqa: F401

__all__ = [
    "AgentState",
    "GaiaGraphAgent",
    "_build_model",
    "_prepare_context",
    "build_tools",
]


def __getattr__(name: str):
    if name in {"GaiaGraphAgent", "_build_model", "_prepare_context"}:
        from .workflow import GaiaGraphAgent, _build_model, _prepare_context

        exports = {
            "GaiaGraphAgent": GaiaGraphAgent,
            "_build_model": _build_model,
            "_prepare_context": _prepare_context,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
