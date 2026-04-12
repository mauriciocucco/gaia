"""Award-number reducer — extracts NASA award identifiers from evidence."""

from __future__ import annotations

import re
from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._constants import NASA_AWARD_RE
from ._parsing import clean_tool_content, normalize_text


class AwardReducer:
    name = "award_number"
    priority = 5

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        answer = _solve_award_number_from_records(question, evidence_records)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _solve_award_number_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "award number" not in lowered and "supported by" not in lowered:
        return None
    subject_name = _extract_award_subject_name(question)
    candidates: list[tuple[int, int, float, str]] = []
    for record in evidence_records:
        if record.kind not in {"text", "links"}:
            continue
        text = clean_tool_content(record.content)
        for match in NASA_AWARD_RE.finditer(text):
            candidate = match.group(1).strip().rstrip(".,;:")
            if (
                len(candidate) >= 8
                and sum(ch.isalpha() for ch in candidate) >= 2
                and sum(ch.isdigit() for ch in candidate) >= 2
            ):
                context = _award_candidate_context(text, match.start(), match.end())
                score, subject_score = _score_award_candidate(
                    record=record,
                    text=text,
                    context=context,
                    candidate=candidate,
                    subject_name=subject_name,
                )
                candidates.append((score, subject_score, record.confidence, candidate))
    if not candidates:
        return None
    if subject_name and any(subject_score > 0 for _score, subject_score, _confidence, _candidate in candidates):
        candidates = [
            candidate_info
            for candidate_info in candidates
            if candidate_info[1] > 0
        ]
    score, _subject_score, _confidence, candidate = max(candidates)
    if score < 4:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------


def _extract_award_subject_name(question: str) -> str | None:
    patterns = (
        r"performed by\s+(?P<name>.+?)\s+supported by",
        r"was the work performed by\s+(?P<name>.+?)\s+supported by",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("name")).strip(" ?.")
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_award_candidate(
    *,
    record: EvidenceRecord,
    text: str,
    context: str,
    candidate: str,
    subject_name: str | None,
) -> tuple[int, int]:
    del candidate
    normalized_context = normalize_text(context)
    combined_text = "\n".join(
        part for part in (record.title_or_caption, record.source_url, text) if part
    )
    normalized_combined = normalize_text(combined_text)

    score = 0
    if "nasa" in normalized_context:
        score += 2
    if "supported by" in normalized_context:
        score += 2
    if "award" in normalized_context:
        score += 1
    if any(
        phrase in normalized_context
        for phrase in ("award no", "award number", "under award", "nasa award")
    ):
        score += 2
    if record.kind == "text":
        score += 2
    if record.extraction_method in {"fetch_url", "find_text_in_url"}:
        score += 2
    if any(marker in normalized_combined for marker in (" csv ", "chorus", "dataset")):
        score -= 2

    local_subject_score = _award_subject_match_score(context, subject_name)
    global_subject_score = 0
    if local_subject_score == 0:
        global_subject_score = min(2, _award_subject_match_score(combined_text, subject_name))
    score += (local_subject_score * 2) + global_subject_score
    return score, local_subject_score


def _award_subject_match_score(text: str, subject_name: str | None) -> int:
    if not subject_name:
        return 0
    normalized_subject = normalize_text(subject_name)
    subject_parts = [part for part in normalized_subject.split() if part]
    if not subject_parts:
        return 0

    normalized_text_value = normalize_text(text)
    compact_text = re.sub(r"[^a-z0-9]+", "", text.lower())
    score = 0

    surname = subject_parts[-1]
    if len(surname) >= 3 and surname in normalized_text_value.split():
        score += 6

    significant_tokens = [token for token in subject_parts if len(token) >= 3]
    if significant_tokens and all(token in normalized_text_value.split() for token in significant_tokens):
        score += 2

    initials = "".join(part[0] for part in subject_parts if part)
    if len(initials) >= 2 and initials in compact_text:
        score += 5

    return score


def _award_candidate_context(text: str, start: int, end: int) -> str:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("!", 0, start),
        text.rfind("?", 0, start),
        text.rfind("\n", 0, start),
    )
    right_candidates = [idx for idx in (
        text.find(".", end),
        text.find("!", end),
        text.find("?", end),
        text.find("\n", end),
    ) if idx != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right].strip()
