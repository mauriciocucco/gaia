"""Evidence collection, structured answers, and benchmark grounding helpers."""

from __future__ import annotations

import re
from typing import Any, Iterable

from langchain_core.messages import AIMessage, ToolMessage

from ..evidence_solver import (
    ToolEvidence,
    solve_answer_from_evidence_records,
    solve_answer_from_tool_evidence,
)
from ..normalize import normalize_submitted_answer
from ..source_pipeline import (
    EvidenceRecord,
    QuestionProfile,
    evidence_records_from_tool_output,
)
from .answer_policy import (
    extract_question_shaped_answer,
    is_invalid_final_response,
    is_invalid_tool_output,
)
from .prompts import FALLBACK_ANSWER_TOOL_NAMES, PREFERRED_STRUCTURED_REDUCERS
from .routing import question_profile_from_state
from .state import AgentState


def _structured_tool_outputs(state: AgentState) -> list[dict[str, Any]]:
    raw_outputs = state.get("structured_tool_outputs")
    return raw_outputs if isinstance(raw_outputs, list) else []


def _tool_evidence_from_messages(messages: list[Any]) -> list[ToolEvidence]:
    tool_outputs: list[ToolEvidence] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        raw_content = str(message.content).strip()
        normalized = normalize_submitted_answer(raw_content)
        if not normalized or is_invalid_tool_output(normalized):
            continue
        tool_outputs.append(
            ToolEvidence(
                tool_name=(getattr(message, "name", "") or "tool").strip(),
                content=raw_content,
            )
        )
    return tool_outputs


def _tool_evidence_from_state(state: AgentState) -> list[ToolEvidence]:
    structured_outputs = _structured_tool_outputs(state)
    if not structured_outputs:
        return _tool_evidence_from_messages(state["messages"])

    tool_outputs: list[ToolEvidence] = []
    for item in structured_outputs:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name", "")).strip() or "tool"
        content = str(item.get("content", "")).strip()
        normalized = normalize_submitted_answer(content)
        if not normalized or is_invalid_tool_output(normalized):
            continue
        tool_outputs.append(ToolEvidence(tool_name=tool_name, content=content))
    return tool_outputs


def collect_evidence_records_from_messages(messages: list[Any]) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        raw_content = str(message.content).strip()
        normalized = normalize_submitted_answer(raw_content)
        if not normalized or is_invalid_tool_output(normalized):
            continue
        records.extend(
            evidence_records_from_tool_output(
                (getattr(message, "name", "") or "tool").strip(),
                raw_content,
            )
        )
    return records


def collect_evidence_records_from_state(state: AgentState) -> list[EvidenceRecord]:
    structured_outputs = _structured_tool_outputs(state)
    if not structured_outputs:
        return collect_evidence_records_from_messages(state["messages"])

    records: list[EvidenceRecord] = []
    for item in structured_outputs:
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name", "")).strip() or "tool"
        payloads = item.get("payloads")
        content = str(item.get("content", "")).strip()
        source: Any = payloads if payloads else content
        normalized = normalize_submitted_answer(content)
        if not normalized or is_invalid_tool_output(normalized):
            continue
        records.extend(evidence_records_from_tool_output(tool_name, source))
    return records


def structured_answer_from_state(
    state: AgentState,
) -> tuple[str | None, str | None, list[EvidenceRecord]]:
    tool_candidate = solve_answer_from_tool_evidence(
        state["question"], _tool_evidence_from_state(state)
    )
    records = collect_evidence_records_from_state(state)
    if tool_candidate:
        used_records = [record for record in records if record.kind == "table"] or records[-6:]
        return tool_candidate, "table_comparison", used_records[:6]
    record_candidate, reducer = solve_answer_from_evidence_records(
        state["question"], records
    )
    if record_candidate:
        return record_candidate, reducer, records[-6:]
    return None, None, []


def should_prefer_structured_answer(
    *, profile: QuestionProfile, reducer_used: str | None
) -> bool:
    if reducer_used not in PREFERRED_STRUCTURED_REDUCERS:
        return False
    if reducer_used == "roster_neighbor":
        return profile.name == "roster_neighbor_lookup"
    if reducer_used == "text_span_attribute":
        return profile.name == "text_span_lookup"
    return True


def record_looks_current_only(record: EvidenceRecord) -> bool:
    haystack = f"{record.source_url}\n{record.title_or_caption}\n{record.content}".lower()
    return (
        "list_of_current" in haystack
        or "current roster" in haystack
        or bool(re.search(r"\bcurrent\b.{0,40}\broster\b", haystack))
        or "current season roster" in haystack
        or ("/wiki/template:" in haystack and "roster" in haystack)
        or (
            "wikipedia.org/wiki/" in haystack
            and "oldid=" not in haystack
            and "roster view talk edit" in haystack
        )
    )


def record_has_roster_context(record: EvidenceRecord) -> bool:
    scope = f"{record.source_url}\n{record.title_or_caption}".lower()
    haystack = f"{scope}\n{record.content}".lower()
    if any(fragment in haystack for fragment in ("/player/", "/players/", "player directory")):
        return any(token in haystack for token in ("show other players", "pitchers")) and bool(
            re.search(r"\b20\d{2}\b", haystack)
        )
    explicit_roster_markers = (
        "roster",
        "rosters",
        "pitchers",
        "staff",
        "depth chart",
        "roster listing",
        "team roster",
        "template:",
    )
    if any(fragment in scope for fragment in ("/stats/", "individual pitching", "individual batting")) and not any(
        token in scope for token in explicit_roster_markers
    ):
        return False
    if any(token in scope for token in explicit_roster_markers):
        return True
    if record.kind == "table":
        return any(token in haystack for token in ("pitchers", "roster", "staff", "depth chart"))
    return False


def record_matches_roster_subject(record: EvidenceRecord, profile: QuestionProfile) -> bool:
    if profile.name != "roster_neighbor_lookup" or not profile.subject_name:
        return True
    haystack = normalize_submitted_answer(
        f"{record.source_url}\n{record.title_or_caption}\n{record.content}"
    ).lower()
    subject_tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            normalize_submitted_answer(profile.subject_name).lower(),
        )
        if token
    ]
    if not subject_tokens:
        return True
    return all(token in haystack for token in subject_tokens)


def record_has_temporal_support(record: EvidenceRecord, profile: QuestionProfile) -> bool:
    if not profile.expected_date:
        return True
    if record_looks_current_only(record):
        return False
    if not record_has_roster_context(record):
        return False
    if not record_matches_roster_subject(record, profile):
        return False
    haystack = f"{record.source_url}\n{record.title_or_caption}\n{record.content}".lower()
    if profile.expected_date.lower() in haystack:
        return True
    year_tokens = set(re.findall(r"\b\d{4}\b", profile.expected_date))
    month_tokens = set(
        re.findall(
            r"january|february|march|april|may|june|july|august|september|october|november|december",
            profile.expected_date.lower(),
        )
    )
    has_year = not year_tokens or any(token in haystack for token in year_tokens)
    has_month = not month_tokens or any(token in haystack for token in month_tokens)
    if has_year and has_month:
        return True
    if has_year and any(
        token in haystack
        for token in (
            "archive",
            "oldid",
            "season",
            "media guide",
            "player directory",
            "show other players",
        )
    ):
        return True
    return False


def has_temporally_grounded_roster_evidence(state: AgentState) -> bool:
    profile = question_profile_from_state(state)
    if profile.name != "roster_neighbor_lookup" or not profile.expected_date:
        return False
    records = collect_evidence_records_from_state(state)
    relevant_records = [
        record
        for record in records
        if record.kind in {"table", "text"} and record_has_roster_context(record)
    ]
    return any(record_has_temporal_support(record, profile) for record in relevant_records)


def grounded_temporal_roster_answer(state: AgentState) -> str | None:
    profile = question_profile_from_state(state)
    if profile.name != "roster_neighbor_lookup" or not profile.expected_date:
        return None
    records = collect_evidence_records_from_state(state)
    temporal_records = [
        record
        for record in records
        if record.kind in {"table", "text"}
        and record_has_roster_context(record)
        and record_has_temporal_support(record, profile)
    ]
    if not temporal_records:
        return None
    answer, reducer = solve_answer_from_evidence_records(state["question"], temporal_records)
    return answer if reducer == "roster_neighbor" else None


def has_temporal_roster_grounding_gap(state: AgentState) -> bool:
    profile = question_profile_from_state(state)
    if profile.name != "roster_neighbor_lookup" or not profile.expected_date:
        return False
    records = collect_evidence_records_from_state(state)
    if not records:
        return False
    relevant_records = [
        record
        for record in records
        if record.kind in {"table", "text"} and record_has_roster_context(record)
    ]
    if not relevant_records:
        return False
    return not any(record_has_temporal_support(record, profile) for record in relevant_records)


def requires_temporal_roster_retry(state: AgentState, answer_text: str) -> bool:
    candidate = normalize_submitted_answer(answer_text)
    if not candidate or is_invalid_final_response(candidate):
        return False
    profile = question_profile_from_state(state)
    if profile.name != "roster_neighbor_lookup" or not profile.expected_date:
        return False
    grounded_answer = grounded_temporal_roster_answer(state)
    if grounded_answer is None:
        return True
    return normalize_submitted_answer(grounded_answer) != candidate


def requires_botanical_classification_retry(state: AgentState, answer_text: str) -> bool:
    candidate = normalize_submitted_answer(answer_text)
    if not candidate or is_invalid_final_response(candidate):
        return False
    profile = question_profile_from_state(state)
    if profile.name != "botanical_classification":
        return False
    return not any(
        record.kind in {"text", "table", "transcript"}
        for record in collect_evidence_records_from_state(state)
    )


def top_grounded_evidence_records(
    state: AgentState, *, limit: int = 6
) -> list[EvidenceRecord]:
    from .candidate_support import ranked_candidates_from_state

    profile = question_profile_from_state(state)
    ranked_candidates = ranked_candidates_from_state(state)
    candidate_bonus = {candidate.url: max(0, 24 - 4 * i) for i, candidate in enumerate(ranked_candidates[:6])}
    scored_records: list[tuple[int, EvidenceRecord]] = []
    for index, record in enumerate(collect_evidence_records_from_state(state)):
        if record.kind == "links":
            continue
        score = int(record.confidence * 100) + index
        if record.kind == "table":
            score += 35
        elif record.kind == "text":
            score += 28
        elif record.kind == "transcript":
            score += 20
        score += candidate_bonus.get(record.source_url, 0)
        haystack = f"{record.title_or_caption}\n{record.content}".lower()
        if profile.text_filter and profile.text_filter.lower() in haystack:
            score += 12
        if profile.subject_name and any(
            token.lower() in haystack for token in profile.subject_name.split()
        ):
            score += 10
        if profile.expected_author and profile.expected_author.lower() in haystack:
            score += 8
        if profile.expected_date and profile.expected_date.lower() in haystack:
            score += 8
        if profile.name == "roster_neighbor_lookup" and profile.expected_date:
            if record_has_temporal_support(record, profile):
                score += 20
            if record_looks_current_only(record):
                score -= 30
        scored_records.append((score, record))
    scored_records.sort(key=lambda item: item[0], reverse=True)
    return [record for _score, record in scored_records[:limit]]


def format_grounded_evidence_for_llm(records: Iterable[EvidenceRecord]) -> str:
    blocks: list[str] = []
    for record in records:
        source_bits = [f"Kind: {record.kind}", f"Tool: {record.extraction_method}"]
        if record.source_url:
            source_bits.append(f"URL: {record.source_url}")
        if record.title_or_caption:
            source_bits.append(f"Title: {record.title_or_caption}")
        blocks.append(f"{' | '.join(source_bits)}\n{record.content}")
    return "\n\n".join(blocks)


def last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def extract_answer_from_tool_output(
    *, tool_name: str, question: str, content: str
) -> str | None:
    normalized = normalize_submitted_answer(content)
    if not normalized or is_invalid_tool_output(normalized):
        return None
    if tool_name in {"calculate", "count_wikipedia_studio_albums"}:
        return normalized
    if tool_name == "analyze_youtube_video":
        extracted = extract_question_shaped_answer(question=question, text=normalized)
        return extracted or normalized
    return None


def fallback_tool_answer(messages: list[Any], question: str) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in FALLBACK_ANSWER_TOOL_NAMES:
            continue
        candidate = extract_answer_from_tool_output(
            tool_name=tool_name,
            question=question,
            content=str(message.content),
        )
        if candidate and not is_invalid_final_response(candidate):
            return candidate
    return None
