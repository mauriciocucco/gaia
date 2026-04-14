"""Core recovery for article-to-paper award questions."""

from __future__ import annotations

from ...fallbacks.article_to_paper import ArticleToPaperFallback


class ArticleToPaperRecovery(ArticleToPaperFallback):
    name = "article_to_paper"
