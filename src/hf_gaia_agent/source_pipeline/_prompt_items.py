"""Shared prompt-item extraction helpers for self-contained list questions."""

from __future__ import annotations

import re


LABELED_LIST_PATTERNS: tuple[str, ...] = (
    r"here's the list i have so far:\s*(?P<body>.+?)(?:\n\s*\n|$)",
    r"comma separated list[:\s]+(?P<body>.+?)(?:\n\s*\n|$)",
    r"list of items[:\s]+(?P<body>.+?)(?:\n\s*\n|$)",
)

INSTRUCTIONAL_CUES: tuple[str, ...] = (
    "please ",
    "could you",
    "i need ",
    "alphabetize",
    "which of these",
    "which are",
    "just the vegetables",
    "from this list",
    "from my list",
    "comma separated list",
    "professor of botany",
    "stickler",
    "classify",
    "categorize",
)


def _normalize_item_text(body: str) -> list[str]:
    compact = re.sub(r"\s+", " ", body).strip()
    if not compact:
        return []
    return [item.strip(" .,:;!?") for item in compact.split(",") if item.strip(" .,:;!?")]


def _looks_instructional_paragraph(paragraph: str) -> bool:
    lowered = paragraph.lower()
    return any(cue in lowered for cue in INSTRUCTIONAL_CUES)


def extract_prompt_list_items(question: str) -> list[str]:
    for pattern in LABELED_LIST_PATTERNS:
        match = re.search(pattern, question, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        items = _normalize_item_text(match.group("body"))
        if items:
            return items

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", question) if chunk.strip()]
    fallback_candidates: list[tuple[int, list[str]]] = []
    for paragraph in paragraphs:
        if paragraph.count(",") < 2 or _looks_instructional_paragraph(paragraph):
            continue
        items = _normalize_item_text(paragraph)
        if len(items) < 3:
            continue
        fallback_candidates.append((paragraph.count(","), items))
    if not fallback_candidates:
        return []
    fallback_candidates.sort(key=lambda item: item[0], reverse=True)
    return fallback_candidates[0][1]
