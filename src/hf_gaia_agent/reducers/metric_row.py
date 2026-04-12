"""Metric-row lookup reducer — selects a row by one metric and returns another."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._parsing import (
    CandidateAnswer,
    clean_label,
    clean_tool_content,
    extract_pipe_tables,
    format_number,
    looks_like_abbreviated_person_name,
    looks_like_person_name,
    metric_tokens,
    normalize_text,
    parse_metric_row_lookup_question,
    parse_number,
    pick_metric_column_for_tokens,
    token_overlap_score,
    tokenize,
    tool_priority,
)


class MetricRowReducer:
    name = "metric_row_lookup"
    priority = 0

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        answer = _solve_metric_row_lookup_from_records(question, evidence_records)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


@dataclass(frozen=True)
class MetricLookupContext:
    question: str
    tool_name: str
    answer_metric: str
    comparison_mode: str
    select_metric: str
    question_tokens: set[str]
    answer_metric_tokens: set[str]
    select_metric_tokens: set[str]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _solve_metric_row_lookup_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    parsed = parse_metric_row_lookup_question(question)
    if parsed is None:
        return None
    answer_metric, comparison_mode, select_metric = parsed
    question_tokens = tokenize(question)
    answer_metric_tokens = metric_tokens(answer_metric)
    select_metric_tokens = metric_tokens(select_metric)

    best: CandidateAnswer | None = None
    for record in evidence_records:
        if record.kind not in {"table", "text"}:
            continue
        content = clean_tool_content(record.content)
        if not content:
            continue
        context = MetricLookupContext(
            question=question,
            tool_name=record.extraction_method,
            answer_metric=answer_metric,
            comparison_mode=comparison_mode,
            select_metric=select_metric,
            question_tokens=question_tokens,
            answer_metric_tokens=answer_metric_tokens,
            select_metric_tokens=select_metric_tokens,
        )
        candidate = _solve_metric_row_lookup_from_table(
            context=context,
            content=content,
        )
        if candidate is None:
            candidate = _solve_metric_row_lookup_from_linear_text(
                context=context,
                content=content,
            )
        if candidate is None:
            candidate = _solve_metric_row_lookup_from_ranked_leaderboard_text(
                context=context,
                content=content,
            )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate

    return best.answer if best is not None else None


# ---------------------------------------------------------------------------
# Table-based metric lookup
# ---------------------------------------------------------------------------


def _solve_metric_row_lookup_from_table(
    *,
    context: MetricLookupContext,
    content: str,
) -> CandidateAnswer | None:
    best: CandidateAnswer | None = None

    for rows in extract_pipe_tables(content):
        if len(rows) < 2:
            continue
        headers = rows[0]
        data_rows = rows[1:]
        column_count = min(len(row) for row in rows)
        if column_count < 2:
            continue

        numeric_counts = [
            sum(parse_number(row[index]) is not None for row in data_rows)
            for index in range(column_count)
        ]
        answer_column = pick_metric_column_for_tokens(
            headers,
            numeric_counts,
            context.answer_metric_tokens,
        )
        select_column = pick_metric_column_for_tokens(
            headers,
            numeric_counts,
            context.select_metric_tokens,
        )
        if answer_column is None or select_column is None or answer_column == select_column:
            continue

        best_row: list[str] | None = None
        best_selector_value: float | None = None
        for row in data_rows:
            if len(row) <= max(answer_column, select_column):
                continue
            selector_value = parse_number(row[select_column])
            answer_value = parse_number(row[answer_column])
            if selector_value is None or answer_value is None:
                continue
            if best_row is None or best_selector_value is None:
                best_row = row
                best_selector_value = selector_value
                continue
            if context.comparison_mode == "max" and selector_value > best_selector_value:
                best_row = row
                best_selector_value = selector_value
            elif context.comparison_mode == "min" and selector_value < best_selector_value:
                best_row = row
                best_selector_value = selector_value

        if best_row is None:
            continue

        answer_value = parse_number(best_row[answer_column])
        if answer_value is None:
            continue

        score = (
            tool_priority(context.tool_name)
            + 18
            + 6
            * token_overlap_score(
                headers[answer_column],
                context.question_tokens | context.answer_metric_tokens,
            )
            + 6
            * token_overlap_score(
                headers[select_column],
                context.question_tokens | context.select_metric_tokens,
            )
        )
        candidate = CandidateAnswer(answer=format_number(answer_value), score=score)
        if best is None or candidate.score > best.score:
            best = candidate

    return best


# ---------------------------------------------------------------------------
# Linear-text stat sections
# ---------------------------------------------------------------------------


def _solve_metric_row_lookup_from_linear_text(
    *,
    context: MetricLookupContext,
    content: str,
) -> CandidateAnswer | None:
    best: CandidateAnswer | None = None

    for section_title, headers, rows in _extract_linear_stat_sections(
        content=content,
        answer_metric_tokens=context.answer_metric_tokens,
        select_metric_tokens=context.select_metric_tokens,
    ):
        headers_with_label = ["Player", *headers]
        numeric_counts = [
            sum(parse_number(row[index]) is not None for row in rows)
            for index in range(len(headers_with_label))
        ]
        answer_column = pick_metric_column_for_tokens(
            headers_with_label,
            numeric_counts,
            context.answer_metric_tokens,
        )
        select_column = pick_metric_column_for_tokens(
            headers_with_label,
            numeric_counts,
            context.select_metric_tokens,
        )
        if answer_column is None or select_column is None or answer_column == select_column:
            continue

        best_row: list[str] | None = None
        best_selector_value: float | None = None
        for row in rows:
            if len(row) <= max(answer_column, select_column):
                continue
            selector_value = parse_number(row[select_column])
            answer_value = parse_number(row[answer_column])
            if selector_value is None or answer_value is None:
                continue
            if best_row is None or best_selector_value is None:
                best_row = row
                best_selector_value = selector_value
                continue
            if context.comparison_mode == "max" and selector_value > best_selector_value:
                best_row = row
                best_selector_value = selector_value
            elif context.comparison_mode == "min" and selector_value < best_selector_value:
                best_row = row
                best_selector_value = selector_value

        if best_row is None:
            continue

        answer_value = parse_number(best_row[answer_column])
        if answer_value is None:
            continue

        score = (
            tool_priority(context.tool_name)
            + 16
            + 8 * token_overlap_score(section_title, context.question_tokens)
            + 6
            * token_overlap_score(
                headers_with_label[answer_column],
                context.question_tokens | context.answer_metric_tokens,
            )
            + 6
            * token_overlap_score(
                headers_with_label[select_column],
                context.question_tokens | context.select_metric_tokens,
            )
        )
        candidate = CandidateAnswer(answer=format_number(answer_value), score=score)
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def _extract_linear_stat_sections(
    *,
    content: str,
    answer_metric_tokens: set[str],
    select_metric_tokens: set[str],
) -> list[tuple[str, list[str], list[list[str]]]]:
    lines = [clean_label(line) for line in content.splitlines() if clean_label(line)]
    sections: list[tuple[str, list[str], list[list[str]]]] = []

    index = 0
    while index < len(lines):
        section_title = lines[index]
        if "stats" not in section_title.lower():
            index += 1
            continue

        header_index = index + 1
        headers: list[str] = []
        while header_index < len(lines):
            line = lines[header_index]
            if "stats" in line.lower() and headers:
                break
            if re.fullmatch(r"[A-Z0-9]{1,5}(?:[+-])?", line):
                headers.append(line)
                header_index += 1
                continue
            if looks_like_person_name(line):
                break
            if len(normalize_text(line).split()) > 4 and not line.isupper():
                break
            headers.append(line)
            header_index += 1
            if len(headers) >= 24:
                break

        if (
            len(headers) < 2
            or not _headers_cover_metric_tokens(headers, answer_metric_tokens)
            or not _headers_cover_metric_tokens(headers, select_metric_tokens)
        ):
            index += 1
            continue

        rows: list[list[str]] = []
        row_index = header_index
        while row_index < len(lines):
            line = lines[row_index]
            if "stats" in line.lower():
                break
            if not looks_like_person_name(line):
                if rows:
                    break
                row_index += 1
                continue

            values_start = row_index + 1
            if values_start < len(lines) and looks_like_abbreviated_person_name(lines[values_start]):
                values_start += 1
            values = lines[values_start : values_start + len(headers)]
            if len(values) < len(headers):
                break

            rows.append([line, *values])
            row_index = values_start + len(headers)

        if len(rows) >= 2:
            sections.append((section_title, headers, rows))

        index = max(index + 1, row_index)

    return sections


# ---------------------------------------------------------------------------
# Ranked leaderboard text
# ---------------------------------------------------------------------------


def _solve_metric_row_lookup_from_ranked_leaderboard_text(
    *,
    context: MetricLookupContext,
    content: str,
) -> CandidateAnswer | None:
    lines = [clean_label(line) for line in content.splitlines() if clean_label(line)]
    best: CandidateAnswer | None = None

    index = 0
    while index < len(lines):
        if lines[index] != "PLAYER":
            index += 1
            continue

        header_index = index
        raw_headers: list[str] = []
        while header_index < len(lines):
            line = lines[header_index]
            if re.fullmatch(r"\d{1,3}", line):
                break
            lowered = line.lower()
            if lowered.startswith("caret-") or lowered in {"standard", "expanded", "statcast", "down", "up"}:
                header_index += 1
                continue
            if len(normalize_text(line).split()) > 4 and not line.isupper():
                header_index += 1
                continue
            raw_headers.append(line)
            header_index += 1

        headers: list[str] = []
        for header in raw_headers:
            if header in {"PLAYER", "TEAM"} or re.fullmatch(r"[A-Za-z0-9]{1,5}(?:[+-])?", header):
                if not headers or headers[-1] != header:
                    headers.append(header)

        numeric_headers = [header for header in headers if header not in {"PLAYER", "TEAM"}]
        if (
            len(numeric_headers) < 2
            or not _headers_cover_metric_tokens(numeric_headers, context.answer_metric_tokens)
            or not _headers_cover_metric_tokens(numeric_headers, context.select_metric_tokens)
        ):
            index += 1
            continue

        numeric_counts = [1] * len(numeric_headers)
        answer_column = pick_metric_column_for_tokens(
            numeric_headers,
            numeric_counts,
            context.answer_metric_tokens,
        )
        select_column = pick_metric_column_for_tokens(
            numeric_headers,
            numeric_counts,
            context.select_metric_tokens,
        )
        if answer_column is None or select_column is None or answer_column == select_column:
            index += 1
            continue

        rows: list[list[str]] = []
        row_index = header_index
        while row_index < len(lines):
            if lines[row_index] == "PLAYER" and rows:
                break
            if not re.fullmatch(r"\d{1,3}", lines[row_index]):
                row_index += 1
                continue

            team_index: int | None = None
            cursor = row_index + 1
            while cursor < len(lines):
                token = lines[cursor]
                if token == "PLAYER" and rows:
                    break
                if re.fullmatch(r"[A-Z]{3}", token):
                    team_index = cursor
                    break
                cursor += 1
                if cursor - row_index > 12:
                    break

            if team_index is None:
                row_index += 1
                continue

            values: list[str] = []
            cursor = team_index + 1
            while cursor < len(lines) and len(values) < len(numeric_headers):
                token = lines[cursor]
                if parse_number(token) is not None:
                    values.append(token)
                cursor += 1

            if len(values) < len(numeric_headers):
                break

            rows.append(values)
            row_index = cursor

        if not rows:
            index += 1
            continue

        best_row: list[str] | None = None
        best_selector_value: float | None = None
        for row in rows:
            selector_value = parse_number(row[select_column])
            answer_value = parse_number(row[answer_column])
            if selector_value is None or answer_value is None:
                continue
            if best_row is None or best_selector_value is None:
                best_row = row
                best_selector_value = selector_value
                continue
            if context.comparison_mode == "max" and selector_value > best_selector_value:
                best_row = row
                best_selector_value = selector_value
            elif context.comparison_mode == "min" and selector_value < best_selector_value:
                best_row = row
                best_selector_value = selector_value

        if best_row is None:
            index = max(index + 1, row_index)
            continue

        answer_value = parse_number(best_row[answer_column])
        if answer_value is None:
            index = max(index + 1, row_index)
            continue

        score = (
            tool_priority(context.tool_name)
            + 14
            + 6
            * token_overlap_score(
                numeric_headers[answer_column],
                context.question_tokens | context.answer_metric_tokens,
            )
            + 6
            * token_overlap_score(
                numeric_headers[select_column],
                context.question_tokens | context.select_metric_tokens,
            )
        )
        candidate = CandidateAnswer(answer=format_number(answer_value), score=score)
        if best is None or candidate.score > best.score:
            best = candidate

        index = max(index + 1, row_index)

    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _headers_cover_metric_tokens(headers: list[str], metric_toks: set[str]) -> bool:
    if not metric_toks:
        return False
    for header in headers:
        normalized = normalize_text(header)
        if normalized in metric_toks:
            return True
        if any(token in normalized.split() for token in metric_toks):
            return True
    return False
