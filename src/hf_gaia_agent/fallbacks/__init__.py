"""Legacy compatibility registry for historical fallback imports.

The canonical architecture now routes through:
- ``hf_gaia_agent.core.recoveries`` for reusable recoveries
- ``hf_gaia_agent.skills`` for domain/benchmark-specific capabilities
- ``hf_gaia_agent.adapters`` for source-specific grounding

This module remains available so older imports and tests keep working.
"""

from __future__ import annotations

from typing import Any

from .base import FallbackResolver
from .article_to_paper import ArticleToPaperFallback
from .botanical import BotanicalFallback
from .competition import CompetitionFallback
from .role_chain import RoleChainFallback
from .roster import RosterFallback
from .text_span import TextSpanFallback

__all__ = [
    "FallbackResolver",
    "ArticleToPaperFallback",
    "BotanicalFallback",
    "CompetitionFallback",
    "RoleChainFallback",
    "RosterFallback",
    "TextSpanFallback",
    "build_core_fallback_resolvers",
    "build_benchmark_fallback_resolvers",
    "build_fallback_resolvers",
]


def build_core_fallback_resolvers(
    tools_by_name: dict[str, Any],
    answer_model: Any,
) -> list[FallbackResolver]:
    del answer_model
    return [
        ArticleToPaperFallback(tools_by_name),
        TextSpanFallback(tools_by_name),
    ]


def build_benchmark_fallback_resolvers(
    tools_by_name: dict[str, Any],
    answer_model: Any,
) -> list[FallbackResolver]:
    return [
        RoleChainFallback(tools_by_name, answer_model),
        BotanicalFallback(tools_by_name),
        CompetitionFallback(tools_by_name),
        RosterFallback(tools_by_name),
    ]


def build_fallback_resolvers(
    tools_by_name: dict[str, Any],
    answer_model: Any,
    *,
    include_benchmark_specific: bool = True,
) -> list[FallbackResolver]:
    resolvers = build_core_fallback_resolvers(tools_by_name, answer_model)
    if include_benchmark_specific:
        resolvers.extend(build_benchmark_fallback_resolvers(tools_by_name, answer_model))
    return resolvers
