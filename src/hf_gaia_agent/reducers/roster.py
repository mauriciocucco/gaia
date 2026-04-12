"""Roster neighbor reducer — finds players before/after a given jersey number."""

from __future__ import annotations

import re
from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._constants import ROSTER_NUMBERED_NAME_RE
from ._parsing import (
    clean_label,
    clean_tool_content,
    extract_pipe_tables,
    last_name,
    normalize_text,
    parse_number,
)


class RosterReducer:
    name = "roster_neighbor"
    priority = 2

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        answer = _solve_roster_neighbor_from_records(question, evidence_records)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _solve_roster_neighbor_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "number before and after" not in lowered:
        return None
    subject_name = _extract_roster_subject_name(question)
    if subject_name:
        candidate = _solve_roster_neighbor_for_subject(
            question=question,
            evidence_records=evidence_records,
            subject_name=subject_name,
        )
        if candidate:
            return candidate
    subject_match = re.search(
        r"before and after\s+(?P<name>[A-Z][A-Za-z''.\-]+(?:\s+[A-Z][A-Za-z''.\-]+){0,3})'s number",
        question,
    )
    if not subject_match:
        return None
    subject_tokens = {
        token for token in normalize_text(subject_match.group("name")).split() if token
    }
    if not subject_tokens:
        return None

    for record in evidence_records:
        if not _record_matches_roster_temporal_request(record, question):
            continue
        parsed_rows = _extract_roster_rows(record)
        candidate = _solve_roster_neighbor_from_parsed_rows(
            parsed_rows=parsed_rows,
            subject_tokens=subject_tokens,
        )
        if candidate:
            return candidate
    return None


def _extract_roster_subject_name(question: str) -> str | None:
    broad_match = re.search(
        r"before and after\s+(?P<name>.+?)'s number",
        question,
    )
    if broad_match:
        return broad_match.group("name").strip()
    unicode_match = re.search(
        r"before and after\s+(?P<name>[^\W\d_][\w''.\-]+(?:\s+[^\W\d_][\w''.\-]+){0,3})'s number",
        question,
    )
    if unicode_match:
        return unicode_match.group("name").strip()
    return None


def _solve_roster_neighbor_for_subject(
    *,
    question: str,
    evidence_records: Sequence[EvidenceRecord],
    subject_name: str,
) -> str | None:
    subject_tokens = {
        token for token in normalize_text(subject_name).split() if token
    }
    if not subject_tokens:
        return None

    for record in evidence_records:
        if not _record_matches_roster_temporal_request(record, question):
            continue
        parsed_rows = _extract_roster_rows(record)
        candidate = _solve_roster_neighbor_from_parsed_rows(
            parsed_rows=parsed_rows,
            subject_tokens=subject_tokens,
        )
        if candidate:
            return candidate
    return None


def _solve_roster_neighbor_from_parsed_rows(
    *,
    parsed_rows: Sequence[tuple[int, str]],
    subject_tokens: set[str],
) -> str | None:
    if len(parsed_rows) < 3 or not subject_tokens:
        return None

    target_number = None
    for number, raw_name in parsed_rows:
        name_tokens = set(normalize_text(raw_name).split())
        if _subject_tokens_match_name_tokens(subject_tokens, name_tokens):
            target_number = number
            break
    if target_number is None:
        return None

    before_name = next(
        (name for number, name in parsed_rows if number == target_number - 1),
        None,
    )
    after_name = next(
        (name for number, name in parsed_rows if number == target_number + 1),
        None,
    )
    if before_name and after_name:
        return f"{last_name(before_name)}, {last_name(after_name)}"
    return None


def _subject_tokens_match_name_tokens(
    subject_tokens: set[str],
    name_tokens: set[str],
) -> bool:
    if not subject_tokens or not name_tokens:
        return False
    if subject_tokens <= name_tokens or subject_tokens & name_tokens == subject_tokens:
        return True
    for subject_token in subject_tokens:
        if not any(
            subject_token == name_token
            or (len(subject_token) >= 4 and name_token.startswith(subject_token))
            or (len(name_token) >= 4 and subject_token.startswith(name_token))
            for name_token in name_tokens
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Roster row extraction
# ---------------------------------------------------------------------------


def _extract_roster_rows(record: EvidenceRecord) -> list[tuple[int, str]]:
    if record.kind == "table":
        return _extract_roster_rows_from_tables(record.content)
    if record.kind == "text":
        return _extract_roster_rows_from_text(record.content)
    return []


def _extract_roster_rows_from_tables(content: str) -> list[tuple[int, str]]:
    tables = extract_pipe_tables(clean_tool_content(content))
    for rows in tables:
        if len(rows) < 2:
            continue
        headers = rows[0]
        data_rows = rows[1:]
        column_count = min(len(row) for row in rows)
        if column_count < 2:
            continue

        numeric_columns = [
            index
            for index in range(column_count)
            if sum(parse_number(row[index]) is not None for row in data_rows) >= 2
        ]
        if not numeric_columns:
            continue
        number_column = numeric_columns[0]
        name_column = next(
            (
                index
                for index in range(column_count)
                if index != number_column
                and any(
                    hint in normalize_text(headers[index])
                    for hint in ("name", "pitcher", "player")
                )
            ),
            None,
        )
        if name_column is None:
            continue

        parsed_rows: list[tuple[int, str]] = []
        for row in data_rows:
            if len(row) <= max(number_column, name_column):
                continue
            raw_number = parse_number(row[number_column])
            raw_name = clean_label(row[name_column])
            if raw_number is None or not raw_name:
                continue
            parsed_rows.append((int(raw_number), raw_name))
        if len(parsed_rows) >= 3:
            return parsed_rows
    return []


def _extract_roster_rows_from_text(content: str) -> list[tuple[int, str]]:
    text = clean_tool_content(content)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed_rows: list[tuple[int, str]] = []
    seen_numbers: set[int] = set()

    index = 0
    while index < len(lines):
        line = clean_label(lines[index])
        inline_match = ROSTER_NUMBERED_NAME_RE.match(line)
        if inline_match:
            number = int(inline_match.group("number"))
            name = clean_label(inline_match.group("name"))
            if 0 < number < 200 and number not in seen_numbers:
                parsed_rows.append((number, name))
                seen_numbers.add(number)
            index += 1
            continue

        if line.isdigit() and index + 1 < len(lines):
            number = int(line)
            next_line = clean_label(lines[index + 1])
            if 0 < number < 200 and ROSTER_NUMBERED_NAME_RE.match(f"{number} {next_line}"):
                if number not in seen_numbers:
                    parsed_rows.append((number, next_line))
                    seen_numbers.add(number)
                index += 2
                continue

        index += 1

    return parsed_rows


def _record_matches_roster_temporal_request(record: EvidenceRecord, question: str) -> bool:
    lowered = question.lower()
    if "as of " not in lowered:
        return True
    haystack = f"{record.source_url}\n{record.title_or_caption}\n{record.content}".lower()
    if "list_of_current" in haystack or "current roster" in haystack:
        return False
    expected_bits = re.findall(
        r"january|february|march|april|may|june|july|august|september|october|november|december|\b\d{4}\b",
        lowered,
    )
    if expected_bits and all(bit in haystack for bit in expected_bits):
        return True
    year_bits = [bit for bit in expected_bits if re.fullmatch(r"\d{4}", bit)]
    if year_bits and any(bit in haystack for bit in year_bits):
        if any(
            token in haystack
            for token in (
                "archive",
                "oldid",
                "season",
                "media guide",
                "player directory",
                "player list",
                "show other players",
                "pitchers",
                "roster",
            )
        ):
            return True
    return not expected_bits
