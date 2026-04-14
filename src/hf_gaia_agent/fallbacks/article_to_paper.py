"""Legacy compatibility wrapper for :class:`ArticleToPaperRecovery`."""

from __future__ import annotations

from ..core.recoveries.article_to_paper import ArticleToPaperRecovery


class ArticleToPaperFallback(ArticleToPaperRecovery):
    """Backward-compatible alias for the canonical recovery implementation."""

    name = "article_to_paper"
