"""Answer validation, canonicalization, and extraction policies."""

from __future__ import annotations

import re

from ..normalize import normalize_submitted_answer
from .prompts import INVALID_FINAL_PATTERNS, INVALID_TOOL_OUTPUT_PATTERNS, NUMERIC_QUESTION_PREFIXES


def is_invalid_final_response(text: str) -> bool:
    value = text.strip()
    if "[ANSWER]" in value and "[/ANSWER]" in value:
        return False

    normalized = normalize_submitted_answer(text).strip().lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in INVALID_FINAL_PATTERNS)


def is_invalid_tool_output(text: str) -> bool:
    normalized = normalize_submitted_answer(text).strip().lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in INVALID_TOOL_OUTPUT_PATTERNS)


def looks_like_placeholder_answer(question: str, answer: str) -> bool:
    normalized = normalize_submitted_answer(answer).strip().lower()
    lowered = question.lower()
    if "award number" in lowered or "supported by" in lowered:
        if normalized in {"n/a", "na", "none", "unknown", "not available"}:
            return True
        for candidate in re.findall(r"\b[A-Z0-9]{8,}\b", normalize_submitted_answer(answer).upper()):
            if sum(ch.isalpha() for ch in candidate) >= 2 and sum(ch.isdigit() for ch in candidate) >= 2:
                return False
        return True
    return False


def question_expects_numeric_answer(question: str) -> bool:
    lowered = question.strip().lower()
    return any(prefix in lowered for prefix in NUMERIC_QUESTION_PREFIXES)


def extract_numeric_answer(text: str) -> str | None:
    patterns = (
        r"\b(?:highest|maximum|minimum|lowest|total|count|answer)\b[^\d]{0,50}?\bis\s+(-?\d+(?:\.\d+)?)\b",
        r"\bthere\s+(?:are|were|is|was)\s+(-?\d+(?:\.\d+)?)\b",
        r"\b(?:i counted|counted)\s+(-?\d+(?:\.\d+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?\b", text)
    if len(numbers) == 1:
        return numbers[0]
    return None


def is_missing_attachment_non_answer(
    text: str,
    *,
    file_name: str | None,
    local_file_path: str | None,
) -> bool:
    if not file_name or local_file_path:
        return False
    normalized = normalize_submitted_answer(text).strip().lower()
    if not normalized:
        return True
    missing_cues = (
        "not available",
        "no image",
        "no file",
        "no data provided",
        "cannot analyze",
        "unable to analyze",
        "cannot determine",
        "cannot provide",
        "no chess position",
    )
    return any(cue in normalized for cue in missing_cues)


def attachment_required_but_missing(
    *,
    question: str,
    file_name: str | None,
    local_file_path: str | None,
) -> bool:
    if not file_name or local_file_path:
        return False
    lowered = question.lower()
    attachment_cues = (
        "attached",
        "attachment",
        "image",
        "audio",
        "voice memo",
        "listen to",
        "provided in the image",
        "recipe as",
    )
    return any(cue in lowered for cue in attachment_cues)


def canonicalize_final_answer(question: str, answer: str) -> str:
    normalized = normalize_submitted_answer(answer).strip()
    if not normalized:
        return ""
    lowered = question.lower()
    if question_expects_numeric_answer(question):
        numeric = extract_numeric_answer(normalized)
        if numeric:
            return numeric
    if "number before and after" in lowered:
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) == 2:
            compact_parts: list[str] = []
            for part in parts:
                fragment = part.split(":")[-1].strip()
                tokens = fragment.split()
                if tokens:
                    compact_parts.append(tokens[-1])
            if len(compact_parts) == 2:
                return ", ".join(compact_parts)
    if "award number" in lowered:
        for candidate in re.findall(r"\b[A-Z0-9]{10,}\b", normalized.upper()):
            if sum(ch.isalpha() for ch in candidate) >= 2 and sum(ch.isdigit() for ch in candidate) >= 2:
                return candidate
    if "without abbreviations" in lowered:
        normalized = re.sub(r"\bSt\.?\s+", "Saint ", normalized)
    return normalized


def extract_question_shaped_answer(*, question: str, text: str) -> str | None:
    normalized = normalize_submitted_answer(text)
    if not normalized:
        return None
    if question_expects_numeric_answer(question):
        numeric = extract_numeric_answer(normalized)
        if numeric:
            return numeric
    return None
