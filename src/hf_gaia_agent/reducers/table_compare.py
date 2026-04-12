"""Table comparison reducer — min/max from pipe tables and parenthetical rows."""

from __future__ import annotations

from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import ReducerResult
from ._constants import PARENTHETICAL_ROW_RE
from ._parsing import (
    CandidateAnswer,
    ToolEvidence,
    clean_label,
    clean_tool_content,
    comparison_mode,
    extract_pipe_tables,
    format_answer,
    parse_metric_row_lookup_question,
    parse_number,
    pick_better_row,
    pick_label_column,
    pick_metric_column,
    token_overlap_score,
    tokenize,
    tool_priority,
)


class TableCompareReducer:
    name = "table_comparison"
    priority = 1

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None:
        table_outputs = [
            ToolEvidence(tool_name=record.extraction_method, content=record.content)
            for record in evidence_records
            if record.kind == "table"
        ]
        answer = solve_answer_from_tool_evidence(question, table_outputs)
        if answer is None:
            return None
        return ReducerResult(answer=answer, reducer_name=self.name)


# ---------------------------------------------------------------------------
# Standalone public function (used by workflow.py directly)
# ---------------------------------------------------------------------------


def solve_answer_from_tool_evidence(
    question: str, tool_outputs: Sequence[ToolEvidence]
) -> str | None:
    if parse_metric_row_lookup_question(question) is not None:
        return None
    comp_mode = comparison_mode(question)
    if comp_mode is None:
        return None

    best: CandidateAnswer | None = None
    for evidence in tool_outputs:
        content = clean_tool_content(evidence.content)
        if not content:
            continue

        candidate = _solve_pipe_tables(question, evidence.tool_name, content, comp_mode)
        if candidate is None:
            candidate = _solve_parenthetical_rows(
                question, evidence.tool_name, content, comp_mode
            )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate

    return best.answer if best is not None else None


# ---------------------------------------------------------------------------
# Pipe-table solver
# ---------------------------------------------------------------------------


def _solve_pipe_tables(
    question: str,
    tool_name: str,
    content: str,
    comp_mode: str,
) -> CandidateAnswer | None:
    tables = extract_pipe_tables(content)
    if not tables:
        return None

    question_tokens = tokenize(question)
    best: CandidateAnswer | None = None
    for rows in tables:
        if len(rows) < 2:
            continue

        headers = rows[0]
        data_rows = rows[1:]
        column_count = min(len(row) for row in rows)
        if column_count < 2:
            continue

        numeric_counts = []
        for index in range(column_count):
            count = sum(parse_number(row[index]) is not None for row in data_rows)
            numeric_counts.append(count)
        numeric_columns = [index for index, count in enumerate(numeric_counts) if count >= 1]
        if not numeric_columns:
            continue

        metric_column = pick_metric_column(headers, numeric_counts, question_tokens)
        if metric_column is None:
            continue

        label_column = pick_label_column(headers, data_rows, metric_column, question_tokens)
        if label_column is None:
            continue

        best_row = None
        for row in data_rows:
            if len(row) <= max(label_column, metric_column):
                continue
            label = row[label_column].strip()
            value = parse_number(row[metric_column])
            if not label or value is None:
                continue
            best_row = pick_better_row(question, comp_mode, best_row, (label, value))

        if best_row is None:
            continue

        answer = format_answer(best_row[0], best_row[1], question)
        if answer is None:
            continue

        score = (
            tool_priority(tool_name)
            + 20
            + 6 * token_overlap_score(headers[metric_column], question_tokens)
            + 2 * token_overlap_score(content[:800], question_tokens)
        )
        candidate = CandidateAnswer(answer=answer, score=score)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


# ---------------------------------------------------------------------------
# Parenthetical-row solver
# ---------------------------------------------------------------------------


def _solve_parenthetical_rows(
    question: str,
    tool_name: str,
    content: str,
    comp_mode: str,
) -> CandidateAnswer | None:
    matches: list[tuple[str, float]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = PARENTHETICAL_ROW_RE.match(line)
        if match is None:
            continue
        label = clean_label(match.group("label"))
        value = parse_number(match.group("number"))
        if not label or value is None:
            continue
        matches.append((label, value))

    if len(matches) < 2:
        return None

    best_row = None
    for row in matches:
        best_row = pick_better_row(question, comp_mode, best_row, row)
    if best_row is None:
        return None

    answer = format_answer(best_row[0], best_row[1], question)
    if answer is None:
        return None

    score = tool_priority(tool_name) + 10 + 2 * token_overlap_score(
        content[:800], tokenize(question)
    )
    return CandidateAnswer(answer=answer, score=score)
