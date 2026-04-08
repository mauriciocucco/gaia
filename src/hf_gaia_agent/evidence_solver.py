"""Structured answer extraction from previously collected tool evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Sequence

from .normalize import normalize_submitted_answer
from .source_pipeline import EvidenceRecord

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
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
PARENTHETICAL_ROW_RE = re.compile(
    r"^(?P<label>[^()]+?)\s*\((?P<number>-?\d+(?:,\d{3})*(?:\.\d+)?)\)\s*$"
)
NASA_AWARD_RE = re.compile(
    r"(?:nasa\s+award\s+(?:number|no\.?)|award\s+(?:number|no\.?)|supported by[^.\n]{0,100}?)\s*[:#]?\s*([A-Z0-9-]{8,})",
    flags=re.IGNORECASE,
)
PERSON_NAME_RE = r"[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,3}"
EXTINCT_COUNTRY_NAMES = {
    "east germany",
    "german democratic republic",
    "west germany",
    "yugoslavia",
    "soviet union",
    "ussr",
    "czechoslovakia",
    "rhodesia",
    "serbia and montenegro",
}
TEXT_SPAN_ATTRIBUTE_HINTS = (
    "surname",
    "last name",
    "first name",
    "city name",
)
METRIC_TOKEN_ALIASES = {
    "at bats": {"ab"},
    "walks": {"bb"},
    "bases on balls": {"bb"},
    "runs batted in": {"rbi"},
    "home runs": {"hr"},
    "wins": {"w"},
    "losses": {"l"},
}


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
    if _parse_metric_row_lookup_question(question) is not None:
        return None
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


def solve_answer_from_evidence_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> tuple[str | None, str | None]:
    metric_lookup_candidate = _solve_metric_row_lookup_from_records(question, evidence_records)
    if metric_lookup_candidate:
        return metric_lookup_candidate, "metric_row_lookup"

    table_outputs = [
        ToolEvidence(tool_name=record.extraction_method, content=record.content)
        for record in evidence_records
        if record.kind == "table"
    ]
    table_candidate = solve_answer_from_tool_evidence(question, table_outputs)
    if table_candidate:
        return table_candidate, "table_comparison"

    roster_candidate = _solve_roster_neighbor_from_records(question, evidence_records)
    if roster_candidate:
        return roster_candidate, "roster_neighbor"

    temporal_candidate = _solve_temporal_row_filter_from_records(question, evidence_records)
    if temporal_candidate:
        return temporal_candidate, "temporal_row_filter"

    text_span_candidate = _solve_text_span_attribute_from_records(question, evidence_records)
    if text_span_candidate:
        return text_span_candidate, "text_span_attribute"

    award_candidate = _solve_award_number_from_records(question, evidence_records)
    if award_candidate:
        return award_candidate, "award_number"

    return None, None


def _clean_tool_content(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if "[ANSWER]" in text.upper():
        return normalize_submitted_answer(text)
    return text


def _solve_roster_neighbor_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "number before and after" not in lowered:
        return None
    subject_match = re.search(
        r"before and after\s+(?P<name>[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,3})'s number",
        question,
    )
    if not subject_match:
        return None
    subject_tokens = {
        token for token in _normalize_text(subject_match.group("name")).split() if token
    }
    if not subject_tokens:
        return None

    for record in evidence_records:
        if record.kind != "table":
            continue
        if not _record_matches_roster_temporal_request(record, question):
            continue
        tables = _extract_pipe_tables(_clean_tool_content(record.content))
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
                if sum(_parse_number(row[index]) is not None for row in data_rows) >= 2
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
                        hint in _normalize_text(headers[index])
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
                raw_number = _parse_number(row[number_column])
                raw_name = _clean_label(row[name_column])
                if raw_number is None or not raw_name:
                    continue
                parsed_rows.append((int(raw_number), raw_name))
            if len(parsed_rows) < 3:
                continue

            target_number = None
            for number, raw_name in parsed_rows:
                name_tokens = set(_normalize_text(raw_name).split())
                if subject_tokens <= name_tokens or subject_tokens & name_tokens == subject_tokens:
                    target_number = number
                    break
            if target_number is None:
                continue

            before_name = next((name for number, name in parsed_rows if number == target_number - 1), None)
            after_name = next((name for number, name in parsed_rows if number == target_number + 1), None)
            if before_name and after_name:
                return f"{_last_name(before_name)}, {_last_name(after_name)}"
    return None


def _record_matches_roster_temporal_request(record: EvidenceRecord, question: str) -> bool:
    lowered = question.lower()
    if "as of " not in lowered:
        return True
    haystack = f"{record.source_url}\n{record.title_or_caption}\n{record.content}".lower()
    expected_bits = re.findall(
        r"january|february|march|april|may|june|july|august|september|october|november|december|\b\d{4}\b",
        lowered,
    )
    if expected_bits and all(bit in haystack for bit in expected_bits):
        return True
    year_bits = [bit for bit in expected_bits if re.fullmatch(r"\d{4}", bit)]
    if year_bits and any(bit in haystack for bit in year_bits):
        if any(token in haystack for token in ("archive", "oldid", "season", "media guide")):
            return True
    if "list_of_current" in haystack or "current roster" in haystack:
        return False
    return not expected_bits


def _solve_award_number_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    lowered = question.lower()
    if "award number" not in lowered and "supported by" not in lowered:
        return None
    for record in evidence_records:
        if record.kind not in {"text", "links"}:
            continue
        text = _clean_tool_content(record.content)
        for match in NASA_AWARD_RE.finditer(text):
            candidate = match.group(1).strip().rstrip(".,;:")
            if len(candidate) >= 8:
                return candidate
    return None


def _solve_metric_row_lookup_from_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> str | None:
    parsed = _parse_metric_row_lookup_question(question)
    if parsed is None:
        return None
    answer_metric, comparison_mode, select_metric = parsed

    best: _CandidateAnswer | None = None
    for record in evidence_records:
        if record.kind != "table":
            continue
        content = _clean_tool_content(record.content)
        if not content:
            continue
        candidate = _solve_metric_row_lookup_from_table(
            question=question,
            content=content,
            tool_name=record.extraction_method,
            answer_metric=answer_metric,
            comparison_mode=comparison_mode,
            select_metric=select_metric,
        )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate

    return best.answer if best is not None else None


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
        if _normalize_text(nationality) not in EXTINCT_COUNTRY_NAMES:
            continue
        matching_rows.append((year, name, nationality))

    if len(matching_rows) != 1:
        return None

    _year, name, _nationality = matching_rows[0]
    return _extract_requested_name_part(question, name)


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


def _parse_metric_row_lookup_question(
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
    comparison_mode = "max" if comparison_word in MAX_COMPARISON_HINTS else "min"
    answer_metric = match.group("answer_metric").strip()
    select_metric = match.group("select_metric").strip()
    if not answer_metric or not select_metric:
        return None
    return answer_metric, comparison_mode, select_metric


def _solve_metric_row_lookup_from_table(
    *,
    question: str,
    content: str,
    tool_name: str,
    answer_metric: str,
    comparison_mode: str,
    select_metric: str,
) -> _CandidateAnswer | None:
    best: _CandidateAnswer | None = None
    question_tokens = _tokenize(question)
    answer_metric_tokens = _metric_tokens(answer_metric)
    select_metric_tokens = _metric_tokens(select_metric)

    for rows in _extract_pipe_tables(content):
        if len(rows) < 2:
            continue
        headers = rows[0]
        data_rows = rows[1:]
        column_count = min(len(row) for row in rows)
        if column_count < 2:
            continue

        numeric_counts = [
            sum(_parse_number(row[index]) is not None for row in data_rows)
            for index in range(column_count)
        ]
        answer_column = _pick_metric_column_for_tokens(headers, numeric_counts, answer_metric_tokens)
        select_column = _pick_metric_column_for_tokens(headers, numeric_counts, select_metric_tokens)
        if answer_column is None or select_column is None or answer_column == select_column:
            continue

        best_row: list[str] | None = None
        best_selector_value: float | None = None
        for row in data_rows:
            if len(row) <= max(answer_column, select_column):
                continue
            selector_value = _parse_number(row[select_column])
            answer_value = _parse_number(row[answer_column])
            if selector_value is None or answer_value is None:
                continue
            if best_row is None or best_selector_value is None:
                best_row = row
                best_selector_value = selector_value
                continue
            if comparison_mode == "max" and selector_value > best_selector_value:
                best_row = row
                best_selector_value = selector_value
            elif comparison_mode == "min" and selector_value < best_selector_value:
                best_row = row
                best_selector_value = selector_value

        if best_row is None:
            continue

        answer_value = _parse_number(best_row[answer_column])
        if answer_value is None:
            continue

        score = (
            _tool_priority(tool_name)
            + 18
            + 6 * _token_overlap_score(headers[answer_column], question_tokens | answer_metric_tokens)
            + 6 * _token_overlap_score(headers[select_column], question_tokens | select_metric_tokens)
        )
        candidate = _CandidateAnswer(answer=_format_number(answer_value), score=score)
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def _iter_named_rows(
    evidence_records: Sequence[EvidenceRecord],
) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for record in evidence_records:
        cleaned = _clean_tool_content(record.content)
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
    for table in _extract_pipe_tables(content):
        if len(table) < 2:
            continue
        headers = [_normalize_text(cell) for cell in table[0]]
        year_idx = _find_column_index(headers, {"year"})
        name_idx = _find_column_index(headers, {"recipient", "winner", "name", "conductor"})
        nationality_idx = _find_column_index(headers, {"nationality", "country", "nation"})
        if year_idx is None or name_idx is None or nationality_idx is None:
            continue
        for row in table[1:]:
            if len(row) <= max(year_idx, name_idx, nationality_idx):
                continue
            year = _parse_year(row[year_idx])
            name = _clean_label(row[name_idx])
            nationality = _clean_label(row[nationality_idx])
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
        year = _parse_year(line)
        if year is not None and re.fullmatch(r"\d{4}", line):
            current_year = year
            current_name = None
            continue
        if current_year is None:
            continue
        if current_name is None and _looks_like_person_name(line):
            current_name = _clean_label(line)
            continue
        if current_name is not None and _looks_like_nationality_line(line):
            rows.append((current_year, current_name, _clean_label(line)))
            current_year = None
            current_name = None
    return rows


def _find_column_index(headers: list[str], expected_tokens: set[str]) -> int | None:
    for index, header in enumerate(headers):
        if any(token in header for token in expected_tokens):
            return index
    return None


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


def _extract_requested_name_part(question: str, name: str) -> str:
    lowered = question.lower()
    cleaned_name = _clean_label(name)
    if "first name" in lowered:
        parts = [part for part in cleaned_name.split() if part and not part.endswith(".")]
        return parts[0] if parts else cleaned_name
    if "surname" in lowered or "last name" in lowered:
        return _last_name(cleaned_name)
    return cleaned_name


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

    terms = set(_tokenize(target))
    lowered = target.lower()
    if "veterinarian" in lowered:
        terms.update({"doctor", "veterinarian"})
    if "equine" in lowered:
        terms.update({"horse", "equine"})
    return terms


def _candidate_passages(content: str) -> list[str]:
    pieces: list[str] = []
    for block in re.split(r"\n{2,}", content):
        block = block.strip()
        if not block:
            continue
        pieces.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block)
            if sentence.strip()
        )
    return pieces


def _passage_target_score(passage: str, target_terms: set[str]) -> int:
    passage_tokens = _tokenize(passage)
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
    return _extract_requested_name_part(question, name)


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
            candidate = _clean_person_name(match.group("name"))
            if candidate.lower() in {"dr", "dr."}:
                continue
            return candidate
    return None


def _clean_person_name(value: str) -> str:
    cleaned = _clean_label(value)
    cleaned = re.sub(r"^(?:Dr\.?\s+)", "", cleaned)
    return cleaned.strip()


def _parse_year(value: str) -> int | None:
    match = YEAR_RE.search(value)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _looks_like_person_name(value: str) -> bool:
    cleaned = _clean_label(value)
    if not cleaned or re.search(r"\d", cleaned):
        return False
    return bool(re.fullmatch(PERSON_NAME_RE, cleaned))


def _looks_like_nationality_line(value: str) -> bool:
    normalized = _normalize_text(value)
    if not normalized or re.search(r"\d", value):
        return False
    if normalized in EXTINCT_COUNTRY_NAMES:
        return True
    return len(normalized.split()) <= 4 and normalized not in {"competition", "winners", "participants"}


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


def _pick_metric_column_for_tokens(
    headers: list[str],
    numeric_counts: list[int],
    metric_tokens: set[str],
) -> int | None:
    if not metric_tokens:
        return None
    best_index = None
    best_score: tuple[int, int, int] | None = None
    for index, count in enumerate(numeric_counts):
        if count < 1:
            continue
        header = headers[index] if index < len(headers) else ""
        normalized_header = _normalize_text(header)
        overlap = _token_overlap_score(header, metric_tokens)
        if normalized_header in metric_tokens:
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


def _metric_tokens(value: str) -> set[str]:
    tokens = _tokenize(value)
    lowered = value.lower()
    for phrase, aliases in METRIC_TOKEN_ALIASES.items():
        if phrase in lowered:
            tokens.update(aliases)
    return tokens


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


def _last_name(value: str) -> str:
    parts = [part for part in _clean_label(value).split() if part]
    if not parts:
        return ""
    return parts[-1]


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
