"""Core recoveries registry."""

from __future__ import annotations

from typing import Any

from .article_to_paper import ArticleToPaperRecovery
from .base import RecoveryStrategy
from .text_span import TextSpanRecovery

__all__ = [
    "RecoveryStrategy",
    "ArticleToPaperRecovery",
    "TextSpanRecovery",
    "build_core_recoveries",
]


def build_core_recoveries(
    tools_by_name: dict[str, Any],
    answer_model: Any,
) -> list[RecoveryStrategy]:
    del answer_model
    return [
        ArticleToPaperRecovery(tools_by_name),
        TextSpanRecovery(tools_by_name),
    ]
