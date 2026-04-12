"""Temporal row-filter reducer — finds rows matching century/year constraints."""

from __future__ import annotations

import re
from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._constants import EXTINCT_COUNTRY_NAMES
from ._parsing import (
    clean_label,
    clean_tool_content,
    extract_pipe_tables,
    extract_requested_name_part,
    find_column_index,
    looks_like_nationality_line,
    looks_like_person_name,
    normalize_text,
    parse_year,
)


class TemporalReducer:
    name = "temporal_row_filter"
    priority = 3

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        answer = _solve_temporal_row_filter_from_records(question, evidence_records)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _solve_temporal_row_filter_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "country that no longer exists" not in lowered:
        return None

    matching_rows: list[tuple[int, str, str]] = []
    for year, name, nationality in _iter_named_rows(evidence_records):
        if not _year_matches_question(year, question):
            continue
        if normalize_text(nationality) not in EXTINCT_COUNTRY_NAMES:
            continue
        matching_rows.append((year, name, nationality))

    if len(matching_rows) != 1:
        return None

    _year, name, _nationality = matching_rows[0]
    return extract_requested_name_part(question, name)


# ---------------------------------------------------------------------------
# Named-row iteration
# ---------------------------------------------------------------------------


def _iter_named_rows(
    evidence_records: Sequence[EvidenceRecord],
) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for record in evidence_records:
        cleaned = clean_tool_content(record.content)
        if not cleaned:
            continue
        if record.kind == "table":
            rows.extend(_named_rows_from_pipe_tables(cleaned))
            continue
        if record.kind == "text":
            rows.extend(_named_rows_from_text(cleaned))
    return rows


def _named_rows_from_pipe_tables(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for table in extract_pipe_tables(content):
        if len(table) < 2:
            continue
        headers = [normalize_text(cell) for cell in table[0]]
        year_idx = find_column_index(headers, {"year"})
        name_idx = find_column_index(headers, {"recipient", "winner", "name", "conductor"})
        nationality_idx = find_column_index(headers, {"nationality", "country", "nation"})
        if year_idx is None or name_idx is None or nationality_idx is None:
            continue
        for row in table[1:]:
            if len(row) <= max(year_idx, name_idx, nationality_idx):
                continue
            year = parse_year(row[year_idx])
            name = clean_label(row[name_idx])
            nationality = clean_label(row[nationality_idx])
            if year is None or not name or not nationality:
                continue
            rows.append((year, name, nationality))
    return rows


def _named_rows_from_text(content: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    current_year: int | None = None
    current_name: str | None = None
    for line in lines:
        year = parse_year(line)
        if year is not None and re.fullmatch(r"\d{4}", line):
            current_year = year
            current_name = None
            continue
        if current_year is None:
            continue
        if current_name is None and looks_like_person_name(line):
            current_name = clean_label(line)
            continue
        if current_name is not None and looks_like_nationality_line(line):
            rows.append((current_year, current_name, clean_label(line)))
            current_year = None
            current_name = None
    return rows


# ---------------------------------------------------------------------------
# Year constraint matching
# ---------------------------------------------------------------------------


def _year_matches_question(year: int, question: str) -> bool:
    lowered = question.lower()
    lower_bound = None
    upper_bound = None
    if "20th century" in lowered:
        lower_bound, upper_bound = 1901, 2000
    if "21st century" in lowered:
        lower_bound, upper_bound = 2001, 2100

    after_match = re.search(r"after\s+(\d{4})", lowered)
    if after_match:
        lower_bound = max(lower_bound or 0, int(after_match.group(1)) + 1)
    before_match = re.search(r"before\s+(\d{4})", lowered)
    if before_match:
        upper_bound = min(upper_bound or 9999, int(before_match.group(1)) - 1)

    if lower_bound is not None and year < lower_bound:
        return False
    if upper_bound is not None and year > upper_bound:
        return False
    return True
