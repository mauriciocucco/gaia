"""Text-span attribute reducer — extracts a named entity attribute from prose."""

from __future__ import annotations

import re
from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._constants import PERSON_NAME_RE, TEXT_SPAN_ATTRIBUTE_HINTS
from ._parsing import (
    clean_person_name,
    extract_requested_name_part,
    tokenize,
)


class TextSpanReducer:
    name = "text_span_attribute"
    priority = 4

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        answer = _solve_text_span_attribute_from_records(question, evidence_records)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _solve_text_span_attribute_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "mentioned" not in lowered or not any(hint in lowered for hint in TEXT_SPAN_ATTRIBUTE_HINTS):
        return None

    target_terms = _target_entity_terms(question)
    if not target_terms:
        return None

    best: tuple[int, str] | None = None
    for record in evidence_records:
        if record.kind not in {"text", "transcript"}:
            continue
        for passage in _candidate_passages(record.content):
            score = _passage_target_score(passage, target_terms)
            if score <= 0:
                continue
            candidate = _extract_requested_attribute_from_passage(question, passage)
            if not candidate:
                continue
            if best is None or score > best[0]:
                best = (score, candidate)

    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# Target entity extraction
# ---------------------------------------------------------------------------


def _target_entity_terms(question: str) -> set[str]:
    patterns = (
        r"(?:surname|last name|first name|city name)\s+of\s+the\s+(?P<target>.+?)(?:\s+mentioned|\s+from|\s+in\b)",
        r"only\s+(?P<target>.+?)\s+mentioned",
    )
    target = None
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            target = match.group("target").strip()
            break
    if not target:
        return set()

    terms = set(tokenize(target))
    lowered = target.lower()
    if "veterinarian" in lowered:
        terms.update({"doctor", "veterinarian"})
    if "equine" in lowered:
        terms.update({"horse", "equine"})
    return terms


# ---------------------------------------------------------------------------
# Passage scoring and extraction
# ---------------------------------------------------------------------------


def _candidate_passages(content: str) -> list[str]:
    pieces: list[str] = []
    for block in re.split(r"\n{2,}", content):
        block = block.strip()
        if not block:
            continue
        pieces.append(block)
        pieces.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block)
            if sentence.strip()
        )
    return list(dict.fromkeys(pieces))


def _passage_target_score(passage: str, target_terms: set[str]) -> int:
    passage_tokens = tokenize(passage)
    score = sum(token in passage_tokens for token in target_terms)
    if "named " in passage.lower():
        score += 1
    if "dr." in passage.lower():
        score += 1
    return score


def _extract_requested_attribute_from_passage(question: str, passage: str) -> str | None:
    name = _extract_person_name_from_passage(passage)
    if not name:
        return None
    lowered = question.lower()
    if "city name" in lowered:
        return None
    return extract_requested_name_part(question, name)


def _extract_person_name_from_passage(passage: str) -> str | None:
    patterns = (
        rf"\bnamed\s+Dr\.?\s+(?P<name>{PERSON_NAME_RE})",
        rf"\bis\s+Dr\.?\s+(?P<name>{PERSON_NAME_RE})",
        rf"\bwas\s+Dr\.?\s+(?P<name>{PERSON_NAME_RE})",
        rf"\bnamed\s+(?P<name>{PERSON_NAME_RE})",
        rf"\bis\s+(?P<name>{PERSON_NAME_RE})",
        rf"\bwas\s+(?P<name>{PERSON_NAME_RE})",
    )
    for pattern in patterns:
        match = re.search(pattern, passage)
        if match:
            candidate = clean_person_name(match.group("name"))
            if candidate.lower() in {"dr", "dr."}:
                continue
            return candidate
    return None
