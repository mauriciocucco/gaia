"""Structured answer extraction from previously collected tool evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence

from .normalize import normalize_submitted_answer

MIN_COMPARISON_HINTS = (
    "least",
    "fewest",
    "lowest",
    "minimum",
    "smallest",
    "earliest",
    "shortest",
)
MAX_COMPARISON_HINTS = (
    "most",
    "highest",
    "maximum",
    "largest",
    "greatest",
    "latest",
    "longest",
)
NUMERIC_QUESTION_HINTS = (
    "how many",
    "what number",
    "what is the number",
    "what was the number",
    "what is the highest number",
    "what was the highest number",
    "what is the maximum number",
    "what was the maximum number",
    "what is the least number",
    "what was the least number",
    "what is the minimum number",
    "what was the minimum number",
)
LABEL_COLUMN_HINTS = (
    "country",
    "nation",
    "noc",
    "committee",
    "team",
    "name",
    "city",
    "state",
    "player",
    "athlete",
    "school",
    "club",
    "title",
)
GENERIC_NUMERIC_HEADERS = {
    "rank",
    "place",
    "pos",
    "position",
    "year",
    "no",
    "no.",
    "#",
}
STOP_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "what",
    "which",
    "give",
    "your",
    "answer",
    "there",
    "their",
    "have",
    "had",
    "into",
    "than",
    "then",
    "for",
    "was",
    "were",
    "are",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "least",
    "most",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "smallest",
    "largest",
    "first",
    "last",
    "alphabetical",
    "order",
    "return",
}
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
PARENTHETICAL_ROW_RE = re.compile(
    r"^(?P<label>[^()]+?)\s*\((?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?)\)\s*$"
)


@dataclass(frozen=True)
class ToolEvidence:
    tool_name: str
    content: str


@dataclass(frozen=True)
class _CandidateAnswer:
    answer: str
    score: int


IOC_COUNTRY_CODES = {
    "argentina": "ARG",
    "armenia": "ARM",
    "australia": "AUS",
    "austria": "AUT",
    "azerbaijan": "AZE",
    "bahamas": "BAH",
    "bahrain": "BRN",
    "barbados": "BAR",
    "belarus": "BLR",
    "belgium": "BEL",
    "bermuda": "BER",
    "bolivia": "BOL",
    "bosnia and herzegovina": "BIH",
    "botswana": "BOT",
    "brazil": "BRA",
    "bulgaria": "BUL",
    "cambodia": "CAM",
    "cameroon": "CMR",
    "canada": "CAN",
    "chile": "CHI",
    "china": "CHN",
    "chinese taipei": "TPE",
    "colombia": "COL",
    "costa rica": "CRC",
    "croatia": "CRO",
    "cuba": "CUB",
    "cyprus": "CYP",
    "czech republic": "CZE",
    "czechia": "CZE",
    "denmark": "DEN",
    "dominican republic": "DOM",
    "ecuador": "ECU",
    "egypt": "EGY",
    "estonia": "EST",
    "ethiopia": "ETH",
    "finland": "FIN",
    "france": "FRA",
    "georgia": "GEO",
    "germany": "GER",
    "ghana": "GHA",
    "great britain": "GBR",
    "greece": "GRE",
    "guatemala": "GUA",
    "haiti": "HAI",
    "hong kong": "HKG",
    "hungary": "HUN",
    "iceland": "ISL",
    "india": "IND",
    "indonesia": "INA",
    "iran": "IRI",
    "iraq": "IRQ",
    "ireland": "IRL",
    "israel": "ISR",
    "italy": "ITA",
    "ivory coast": "CIV",
    "jamaica": "JAM",
    "japan": "JPN",
    "kazakhstan": "KAZ",
    "kenya": "KEN",
    "kosovo": "KOS",
    "kyrgyzstan": "KGZ",
    "latvia": "LAT",
    "lebanon": "LBN",
    "lithuania": "LTU",
    "luxembourg": "LUX",
    "malaysia": "MAS",
    "mexico": "MEX",
    "moldova": "MDA",
    "mongolia": "MGL",
    "montenegro": "MNE",
    "morocco": "MAR",
    "namibia": "NAM",
    "netherlands": "NED",
    "new zealand": "NZL",
    "nigeria": "NGR",
    "north korea": "PRK",
    "north macedonia": "MKD",
    "norway": "NOR",
    "pakistan": "PAK",
    "panama": "PAN",
    "paraguay": "PAR",
    "peru": "PER",
    "philippines": "PHI",
    "poland": "POL",
    "portugal": "POR",
    "puerto rico": "PUR",
    "qatar": "QAT",
    "romania": "ROU",
    "russia": "RUS",
    "saint lucia": "LCA",
    "saudi arabia": "KSA",
    "serbia": "SRB",
    "singapore": "SGP",
    "slovakia": "SVK",
    "slovenia": "SLO",
    "south africa": "RSA",
    "south korea": "KOR",
    "soviet union": "URS",
    "spain": "ESP",
    "sweden": "SWE",
    "switzerland": "SUI",
    "syria": "SYR",
    "tajikistan": "TJK",
    "thailand": "THA",
    "trinidad and tobago": "TTO",
    "tunisia": "TUN",
    "turkey": "TUR",
    "ukraine": "UKR",
    "united arab emirates": "UAE",
    "united kingdom": "GBR",
    "united states": "USA",
    "united states of america": "USA",
    "uruguay": "URU",
    "uzbekistan": "UZB",
    "venezuela": "VEN",
    "vietnam": "VIE",
}


def solve_answer_from_tool_evidence(
    question: str, tool_outputs: Sequence[ToolEvidence]
) -> str | None:
    comparison_mode = _comparison_mode(question)
    if comparison_mode is None:
        return None

    best: _CandidateAnswer | None = None
    for evidence in tool_outputs:
        content = _clean_tool_content(evidence.content)
        if not content:
            continue

        candidate = _solve_pipe_tables(question, evidence.tool_name, content, comparison_mode)
        if candidate is None:
            candidate = _solve_parenthetical_rows(
                question, evidence.tool_name, content, comparison_mode
            )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate

    return best.answer if best is not None else None


def _clean_tool_content(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if "[ANSWER]" in text.upper():
        return normalize_submitted_answer(text)
    return text


def _solve_pipe_tables(
    question: str,
    tool_name: str,
    content: str,
    comparison_mode: str,
) -> _CandidateAnswer | None:
    tables = _extract_pipe_tables(content)
    if not tables:
        return None

    question_tokens = _tokenize(question)
    best: _CandidateAnswer | None = None
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
            count = sum(_parse_number(row[index]) is not None for row in data_rows)
            numeric_counts.append(count)
        numeric_columns = [index for index, count in enumerate(numeric_counts) if count >= 1]
        if not numeric_columns:
            continue

        metric_column = _pick_metric_column(headers, numeric_counts, question_tokens)
        if metric_column is None:
            continue

        label_column = _pick_label_column(headers, data_rows, metric_column, question_tokens)
        if label_column is None:
            continue

        best_row = None
        for row in data_rows:
            if len(row) <= max(label_column, metric_column):
                continue
            label = row[label_column].strip()
            value = _parse_number(row[metric_column])
            if not label or value is None:
                continue
            best_row = _pick_better_row(question, comparison_mode, best_row, (label, value))

        if best_row is None:
            continue

        answer = _format_answer(best_row[0], best_row[1], question)
        if answer is None:
            continue

        score = (
            _tool_priority(tool_name)
            + 20
            + 6 * _token_overlap_score(headers[metric_column], question_tokens)
            + 2 * _token_overlap_score(content[:800], question_tokens)
        )
        candidate = _CandidateAnswer(answer=answer, score=score)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _solve_parenthetical_rows(
    question: str,
    tool_name: str,
    content: str,
    comparison_mode: str,
) -> _CandidateAnswer | None:
    matches: list[tuple[str, float]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = PARENTHETICAL_ROW_RE.match(line)
        if match is None:
            continue
        label = _clean_label(match.group("label"))
        value = _parse_number(match.group("number"))
        if not label or value is None:
            continue
        matches.append((label, value))

    if len(matches) < 2:
        return None

    best_row = None
    for row in matches:
        best_row = _pick_better_row(question, comparison_mode, best_row, row)
    if best_row is None:
        return None

    answer = _format_answer(best_row[0], best_row[1], question)
    if answer is None:
        return None

    score = _tool_priority(tool_name) + 10 + 2 * _token_overlap_score(
        content[:800], _tokenize(question)
    )
    return _CandidateAnswer(answer=answer, score=score)


def _extract_pipe_tables(content: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            if len(current) >= 2:
                tables.append(current)
            current = []
            continue

        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 2:
            if len(current) >= 2:
                tables.append(current)
            current = []
            continue
        current.append(cells)

    if len(current) >= 2:
        tables.append(current)
    return tables


def _pick_metric_column(
    headers: list[str], numeric_counts: list[int], question_tokens: set[str]
) -> int | None:
    best_index = None
    best_score: tuple[int, int, int] | None = None
    numeric_columns = [index for index, count in enumerate(numeric_counts) if count >= 1]
    if len(numeric_columns) == 1:
        return numeric_columns[0]

    for index in numeric_columns:
        header = headers[index] if index < len(headers) else ""
        normalized_header = _normalize_text(header)
        header_score = _token_overlap_score(header, question_tokens)
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


def _pick_label_column(
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
            if _parse_number(row[index]) is None and row[index].strip()
        )
        if non_numeric_count == 0:
            continue
        header = headers[index] if index < len(headers) else ""
        normalized_header = _normalize_text(header)
        header_hint_score = int(
            any(hint in normalized_header for hint in LABEL_COLUMN_HINTS)
        )
        overlap_score = _token_overlap_score(header, question_tokens)
        score = (header_hint_score + overlap_score, non_numeric_count, -index)
        if best_score is None or score > best_score:
            best_index = index
            best_score = score
    return best_index


def _pick_better_row(
    question: str,
    comparison_mode: str,
    current: tuple[str, float] | None,
    candidate: tuple[str, float],
) -> tuple[str, float]:
    if current is None:
        return candidate

    current_label, current_value = current
    candidate_label, candidate_value = candidate
    if comparison_mode == "min":
        if candidate_value < current_value:
            return candidate
        if candidate_value > current_value:
            return current
    else:
        if candidate_value > current_value:
            return candidate
        if candidate_value < current_value:
            return current

    tie_break = _tie_break_mode(question)
    if tie_break == "alpha_desc":
        return (
            candidate
            if _normalize_text(candidate_label) > _normalize_text(current_label)
            else current
        )
    if tie_break == "alpha_asc":
        return (
            candidate
            if _normalize_text(candidate_label) < _normalize_text(current_label)
            else current
        )
    return current


def _format_answer(label: str, value: float, question: str) -> str | None:
    if _expects_numeric_answer(question):
        return _format_number(value)

    cleaned_label = _clean_label(label)
    if not cleaned_label:
        return None

    if _question_requests_ioc_code(question):
        code = _to_ioc_country_code(cleaned_label)
        if code is None:
            return None
        return code
    return cleaned_label


def _question_requests_ioc_code(question: str) -> bool:
    lowered = question.lower()
    return "ioc country code" in lowered or "ioc code" in lowered


def _expects_numeric_answer(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in NUMERIC_QUESTION_HINTS)


def _to_ioc_country_code(label: str) -> str | None:
    if re.fullmatch(r"[A-Z]{3}", label.strip()):
        return label.strip().upper()
    return IOC_COUNTRY_CODES.get(_normalize_text(label))


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip("0").rstrip(".")


def _comparison_mode(question: str) -> str | None:
    lowered = question.lower()
    if any(hint in lowered for hint in MIN_COMPARISON_HINTS):
        return "min"
    if any(hint in lowered for hint in MAX_COMPARISON_HINTS):
        return "max"
    return None


def _tie_break_mode(question: str) -> str | None:
    lowered = question.lower()
    if "last in alphabetical order" in lowered or "alphabetically last" in lowered:
        return "alpha_desc"
    if "alphabetical order" in lowered or "alphabetically first" in lowered:
        return "alpha_asc"
    return None


def _parse_number(value: str) -> float | None:
    match = NUMBER_RE.search(value.replace("\u2212", "-"))
    if match is None:
        return None
    raw = match.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _clean_label(value: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", "", value).strip(" ,;:-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s{2,}", " ", normalized).strip()


def _tokenize(value: str) -> set[str]:
    return {
        token
        for token in _normalize_text(value).split()
        if len(token) >= 3 and token not in STOP_WORDS
    }


def _token_overlap_score(text: str, question_tokens: set[str]) -> int:
    if not question_tokens:
        return 0
    text_tokens = _tokenize(text)
    return sum(token in text_tokens for token in question_tokens)


def _tool_priority(tool_name: str) -> int:
    priorities = {
        "extract_tables_from_url": 40,
        "find_text_in_url": 28,
        "fetch_url": 20,
        "fetch_wikipedia_page": 18,
    }
    return priorities.get(tool_name, 10)
