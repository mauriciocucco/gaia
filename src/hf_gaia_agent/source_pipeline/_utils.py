"""Shared utilities for source pipeline and other modules."""

from __future__ import annotations

import re
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)

SEARCH_STOP_WORDS = {
    "the",
    "and",
    "what",
    "which",
    "with",
    "from",
    "that",
    "this",
    "give",
    "only",
    "your",
    "answer",
    "would",
    "could",
    "under",
    "were",
    "was",
    "is",
    "are",
    "for",
    "into",
    "after",
    "before",
    "between",
    "there",
    "linked",
    "article",
    "paper",
    "number",
    "many",
    "much",
    "same",
    "season",
    "regular",
}

_METRIC_ROW_RE = re.compile(
    r"how many\s+.+?\s+did\s+.+?\s+with\s+the\s+"
    r"(?:most|least|fewest|highest|lowest|minimum|maximum|smallest|largest)\s+.+?\s+have",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Extract URLs from *text*, stripping trailing punctuation."""
    urls: list[str] = []
    for match in URL_RE.findall(text):
        cleaned = match.rstrip(").,;:!?")
        if cleaned:
            urls.append(cleaned)
    return urls


def is_youtube_url(url: str) -> bool:
    lowered = url.lower()
    return "youtube.com/" in lowered or "youtu.be/" in lowered


def query_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in SEARCH_STOP_WORDS
    }


def registered_host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def is_metric_row_lookup_question(question: str) -> bool:
    return _METRIC_ROW_RE.search(question) is not None
