"""Shared parsing and scoring utilities used across evidence reducers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..normalize import normalize_submitted_answer
from ._constants import (
    EXTINCT_COUNTRY_NAMES,
    GENERIC_NUMERIC_HEADERS,
    IOC_COUNTRY_CODES,
    LABEL_COLUMN_HINTS,
    MAX_COMPARISON_HINTS,
    METRIC_TOKEN_ALIASES,
    MIN_COMPARISON_HINTS,
    NUMBER_RE,
    NUMERIC_QUESTION_HINTS,
    PARENTHETICAL_ROW_RE,
    PERSON_NAME_RE,
    STOP_WORDS,
    YEAR_RE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolEvidence:
    tool_name: str
    content: str


@dataclass(frozen=True)
class CandidateAnswer:
    answer: str
    score: int


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def clean_tool_content(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if "[ANSWER]" in text.upper():
        return normalize_submitted_answer(text)
    return text


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s{2,}", " ", normalized).strip()


def tokenize(value: str) -> set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in STOP_WORDS
    }


def token_overlap_score(text: str, question_tokens: set[str]) -> int:
    if not question_tokens:
        return 0
    text_tokens = tokenize(text)
    return sum(token in text_tokens for token in question_tokens)


# ---------------------------------------------------------------------------
# Number / label helpers
# ---------------------------------------------------------------------------


def parse_number(value: str) -> float | None:
    match = NUMBER_RE.search(value.replace("\u2212", "-"))
    if match is None:
        return None
    raw = match.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def clean_label(value: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", "", value).strip(" ,;:-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def last_name(value: str) -> str:
    parts = [part for part in clean_label(value).split() if part]
    if not parts:
        return ""
    return parts[-1]


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip("0").rstrip(".")


def clean_person_name(value: str) -> str:
    cleaned = clean_label(value)
    cleaned = re.sub(r"^(?:Dr\.?\s+)", "", cleaned)
    return cleaned.strip(" .")


# ---------------------------------------------------------------------------
# Pipe-table extraction
# ---------------------------------------------------------------------------


def extract_pipe_tables(content: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            if len(current) >= 2:
                tables.extend(finalize_pipe_table_rows(current))
            current = []
            continue

        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 2:
            if len(current) >= 2:
                tables.extend(finalize_pipe_table_rows(current))
            current = []
            continue
        current.append(cells)

    if len(current) >= 2:
        tables.extend(finalize_pipe_table_rows(current))
    return tables


def finalize_pipe_table_rows(rows: list[list[str]]) -> list[list[list[str]]]:
    normalized = normalize_pipe_table_rows(rows)
    return [
        segment
        for segment in split_pipe_table_segments(normalized)
        if len(segment) >= 2
    ]


def normalize_pipe_table_rows(rows: list[list[str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        trimmed = list(row)
        while trimmed and not trimmed[0].strip():
            trimmed = trimmed[1:]
        while trimmed and not trimmed[-1].strip():
            trimmed = trimmed[:-1]
        if not trimmed:
            continue
        if _is_markdown_separator_row(trimmed):
            continue
        normalized.append(trimmed)
    while len(normalized) >= 2 and len(normalized[0]) < 2:
        normalized = normalized[1:]
    return normalized


def _is_markdown_separator_row(row: list[str]) -> bool:
    return all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) is not None
        for cell in row
    )


def split_pipe_table_segments(rows: list[list[str]]) -> list[list[list[str]]]:
    segments: list[list[list[str]]] = []
    current: list[list[str]] = []
    expected_width: int | None = None
    for row in rows:
        width = len(row)
        if width < 2:
            if len(current) >= 2:
                segments.append(current)
            current = []
            expected_width = None
            continue
        if not current:
            current = [row]
            expected_width = width
            continue
        if expected_width is not None and width != expected_width:
            if len(current) >= 2:
                segments.append(current)
            current = [row]
            expected_width = width
            continue
        current.append(row)
    if len(current) >= 2:
        segments.append(current)
    return segments


# ---------------------------------------------------------------------------
# Tool / comparison helpers
# ---------------------------------------------------------------------------


def tool_priority(tool_name: str) -> int:
    priorities = {
        "extract_tables_from_url": 40,
        "find_text_in_url": 28,
        "fetch_url": 20,
        "fetch_wikipedia_page": 18,
    }
    return priorities.get(tool_name, 10)


def comparison_mode(question: str) -> str | None:
    lowered = question.lower()
    if any(hint in lowered for hint in MIN_COMPARISON_HINTS):
        return "min"
    if any(hint in lowered for hint in MAX_COMPARISON_HINTS):
        return "max"
    return None


def tie_break_mode(question: str) -> str | None:
    lowered = question.lower()
    if "last in alphabetical order" in lowered or "alphabetically last" in lowered:
        return "alpha_desc"
    if "alphabetical order" in lowered or "alphabetically first" in lowered:
        return "alpha_asc"
    return None


def expects_numeric_answer(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in NUMERIC_QUESTION_HINTS)


# ---------------------------------------------------------------------------
# Metric column picking
# ---------------------------------------------------------------------------


def metric_tokens(value: str) -> set[str]:
    tokens = tokenize(value)
    lowered = value.lower()
    for phrase, aliases in METRIC_TOKEN_ALIASES.items():
        if phrase in lowered:
            tokens.update(aliases)
    return tokens


def pick_metric_column(
    headers: list[str], numeric_counts: list[int], question_tokens: set[str]
) -> int | None:
    best_index = None
    best_score: tuple[int, int, int] | None = None
    numeric_columns = [index for index, count in enumerate(numeric_counts) if count >= 1]
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    for index in numeric_columns:
        header = headers[index] if index < len(headers) else ""
        normalized_header = normalize_text(header)
        header_score = token_overlap_score(header, question_tokens)
        if normalized_header in GENERIC_NUMERIC_HEADERS:
            header_score -= 1
        score = (header_score, numeric_counts[index], -index)
        if best_score is None or score > best_score:
            best_index = index
            best_score = score

    if best_index is None:
        return None
    if best_score is not None and best_score[0] <= 0:
        return None
    return best_index


def pick_metric_column_for_tokens(
    headers: list[str],
    numeric_counts: list[int],
    metric_toks: set[str],
) -> int | None:
    if not metric_toks:
        return None
    best_index = None
    best_score: tuple[int, int, int] | None = None
    for index, count in enumerate(numeric_counts):
        if count < 1:
            continue
        header = headers[index] if index < len(headers) else ""
        normalized_header = normalize_text(header)
        overlap = token_overlap_score(header, metric_toks)
        if normalized_header in metric_toks:
            overlap += 3
        if normalized_header in GENERIC_NUMERIC_HEADERS:
            overlap -= 1
        score = (overlap, count, -index)
        if best_score is None or score > best_score:
            best_index = index
            best_score = score
    if best_index is None:
        return None
    if best_score is not None and best_score[0] <= 0:
        return None
    return best_index


# ---------------------------------------------------------------------------
# Label column / row selection / answer formatting
# ---------------------------------------------------------------------------


def pick_label_column(
    headers: list[str],
    data_rows: list[list[str]],
    metric_column: int,
    question_tokens: set[str],
) -> int | None:
    best_index = None
    best_score: tuple[int, int, int] | None = None
    column_count = min(len(row) for row in data_rows + [headers])
    for index in range(column_count):
        if index == metric_column:
            continue
        non_numeric_count = sum(
            1
            for row in data_rows
            if parse_number(row[index]) is None and row[index].strip()
        )
        if non_numeric_count == 0:
            continue
        header = headers[index] if index < len(headers) else ""
        normalized_header = normalize_text(header)
        header_hint_score = int(
            any(hint in normalized_header for hint in LABEL_COLUMN_HINTS)
        )
        overlap_score = token_overlap_score(header, question_tokens)
        score = (header_hint_score + overlap_score, non_numeric_count, -index)
        if best_score is None or score > best_score:
            best_index = index
            best_score = score
    return best_index


def pick_better_row(
    question: str,
    comp_mode: str,
    current: tuple[str, float] | None,
    candidate: tuple[str, float],
) -> tuple[str, float]:
    if current is None:
        return candidate

    current_label, current_value = current
    candidate_label, candidate_value = candidate
    if comp_mode == "min":
        if candidate_value < current_value:
            return candidate
        if candidate_value > current_value:
            return current
    else:
        if candidate_value > current_value:
            return candidate
        if candidate_value < current_value:
            return current

    tb = tie_break_mode(question)
    if tb == "alpha_desc":
        return (
            candidate
            if normalize_text(candidate_label) > normalize_text(current_label)
            else current
        )
    if tb == "alpha_asc":
        return (
            candidate
            if normalize_text(candidate_label) < normalize_text(current_label)
            else current
        )
    return current


def format_answer(label: str, value: float, question: str) -> str | None:
    if expects_numeric_answer(question):
        return format_number(value)

    cleaned_label = clean_label(label)
    if not cleaned_label:
        return None

    if question_requests_ioc_code(question):
        code = to_ioc_country_code(cleaned_label)
        if code is None:
            return None
        return code
    return cleaned_label


def question_requests_ioc_code(question: str) -> bool:
    lowered = question.lower()
    return "ioc country code" in lowered or "ioc code" in lowered


def to_ioc_country_code(label: str) -> str | None:
    if re.fullmatch(r"[A-Z]{3}", label.strip()):
        return label.strip().upper()
    return IOC_COUNTRY_CODES.get(normalize_text(label))


# ---------------------------------------------------------------------------
# Named-row helpers (year / name / nationality)
# ---------------------------------------------------------------------------


def find_column_index(headers: list[str], expected_tokens: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if any(token in header for token in expected_tokens):
            return index
    return None


def parse_year(value: str) -> int | None:
    match = YEAR_RE.search(value)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_requested_name_part(question: str, name: str) -> str:
    lowered = question.lower()
    cleaned_name = clean_label(name)
    if "first name" in lowered:
        parts = [part for part in cleaned_name.split() if part and not part.endswith(".")]
        return parts[0] if parts else cleaned_name
    if "surname" in lowered or "last name" in lowered:
        return last_name(cleaned_name)
    return cleaned_name


# ---------------------------------------------------------------------------
# Person / nationality detection helpers
# ---------------------------------------------------------------------------


def looks_like_person_name(value: str) -> bool:
    cleaned = clean_label(value)
    if not cleaned or re.search(r"\d", cleaned):
        return False
    return bool(re.fullmatch(PERSON_NAME_RE, cleaned))


def looks_like_nationality_line(value: str) -> bool:
    normalized = normalize_text(value)
    if not normalized or re.search(r"\d", value):
        return False
    if normalized in EXTINCT_COUNTRY_NAMES:
        return True
    return len(normalized.split()) <= 4 and normalized not in {"competition", "winners", "participants"}


def looks_like_abbreviated_person_name(value: str) -> bool:
    cleaned = clean_label(value)
    if not cleaned:
        return False
    return bool(re.fullmatch(r"[A-Z]\.\s+[A-Z][A-Za-z'\u2019.\-]+(?:\s+[A-Z][A-Za-z'\u2019.\-]+){0,2}", cleaned))


# ---------------------------------------------------------------------------
# Metric-row question detection (shared between metric_row and table_compare)
# ---------------------------------------------------------------------------


def parse_metric_row_lookup_question(
    question: str,
) -> tuple[str, str, str] | None:
    match = re.search(
        r"how many\s+(?P<answer_metric>.+?)\s+did\s+.+?\s+with\s+the\s+"
        r"(?P<comparison>most|least|fewest|highest|lowest|minimum|maximum|smallest|largest)\s+"
        r"(?P<select_metric>.+?)\s+have",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    comparison_word = match.group("comparison").lower()
    comp_mode = "max" if comparison_word in MAX_COMPARISON_HINTS else "min"
    answer_metric = match.group("answer_metric").strip()
    select_metric = match.group("select_metric").strip()
    if not answer_metric or not select_metric:
        return None
    return answer_metric, comp_mode, select_metric
