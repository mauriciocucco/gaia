"""Shared fallback utilities used by multiple fallback resolvers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import ToolMessage

from ..evidence_solver import solve_answer_from_evidence_records
from ..normalize import normalize_submitted_answer
from ..source_pipeline import (
    EvidenceRecord,
    SourceCandidate,
    evidence_records_from_tool_output,
    parse_result_blocks,
    score_candidates,
    serialize_candidates,
    serialize_evidence,
)
from ..graph.candidate_support import merge_ranked_candidates
from ..graph.state import AgentState
from ..graph.routing import question_profile_from_state


@dataclass
class FallbackExecutionContext:
    tools_by_name: dict[str, Any]
    state: AgentState
    question_profile: Any
    tool_trace: list[str]
    decision_trace: list[str]
    ranked_candidates: list[SourceCandidate]


def fallback_trace_state(
    *,
    tools_by_name: dict[str, Any],
    state: AgentState,
) -> FallbackExecutionContext:
    ranked_candidates: list[SourceCandidate] = []
    for raw in state.get("ranked_candidates") or []:
        if isinstance(raw, SourceCandidate):
            ranked_candidates.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        ranked_candidates.append(
            SourceCandidate(
                title=str(raw.get("title", "")),
                url=str(raw.get("url", "")),
                snippet=str(raw.get("snippet", "")),
                origin_tool=str(raw.get("origin_tool", "")),
                score=int(raw.get("score", 0)),
                reasons=tuple(raw.get("reasons") or ()),
            )
        )
    return FallbackExecutionContext(
        tools_by_name=tools_by_name,
        state=state,
        question_profile=question_profile_from_state(state),
        tool_trace=list(state.get("tool_trace") or []),
        decision_trace=list(state.get("decision_trace") or []),
        ranked_candidates=ranked_candidates,
    )


def with_fallback_traces(
    result: dict[str, Any] | None,
    *,
    context: FallbackExecutionContext,
) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        **result,
        "tool_trace": context.tool_trace,
        "decision_trace": context.decision_trace,
        "ranked_candidates": serialize_candidates(context.ranked_candidates),
    }


def invoke_fallback_tool(
    *,
    context: FallbackExecutionContext,
    tool_name: str,
    tool_args: dict[str, Any],
    trace_label: str = "finalize_fallback",
) -> str:
    tool = context.tools_by_name[tool_name]
    context.tool_trace.append(f"{tool_name}({tool_args}) [{trace_label}]")
    context.decision_trace.append(f"tool:{tool_name}:{trace_label}")
    try:
        result = tool.invoke(tool_args)
    except Exception as exc:
        result = f"Tool error: {exc}"
    result_text = str(result)
    if tool_name in {"web_search", "search_wikipedia", "extract_links_from_url"}:
        parsed_candidates = parse_result_blocks(result_text, origin_tool=tool_name)
        if parsed_candidates:
            scored_candidates = score_candidates(
                parsed_candidates,
                question=context.state["question"],
                profile=context.question_profile,
            )
            context.ranked_candidates[:] = merge_ranked_candidates(
                context.ranked_candidates,
                scored_candidates,
            )
    return result_text


def fallback_result_from_records(
    question: str,
    records: list[EvidenceRecord],
    *,
    expected_reducer: str,
) -> dict[str, Any] | None:
    if not records:
        return None
    answer, reducer = solve_answer_from_evidence_records(question, records)
    if not answer or reducer != expected_reducer:
        return None
    return {
        "final_answer": answer,
        "error": None,
        "reducer_used": reducer,
        "evidence_used": serialize_evidence(records[-3:]),
        "fallback_reason": None,
    }


def try_find_text_fallback(
    *,
    context: FallbackExecutionContext,
    candidate_urls: list[str],
    queries: list[str],
    title_hint: str,
    expected_reducer: str,
) -> dict[str, Any] | None:
    if "find_text_in_url" not in context.tools_by_name:
        return None

    for candidate_url in candidate_urls:
        for query in dict.fromkeys(query for query in queries if query):
            found_text = invoke_fallback_tool(
                context=context,
                tool_name="find_text_in_url",
                tool_args={"url": candidate_url, "query": query, "max_matches": 8},
            )
            normalized = normalize_submitted_answer(found_text).strip().lower()
            if normalized in {"", "no matches found.", "page not found"}:
                continue
            if "warning: target url returned error 404" in normalized:
                continue
            record = EvidenceRecord(
                kind="text",
                source_url=candidate_url,
                source_type="page_text",
                adapter_name="reference_text_source",
                content=found_text,
                title_or_caption=title_hint,
                confidence=0.85,
                extraction_method="find_text_in_url",
                derived_from=("find_text_in_url",),
            )
            result = fallback_result_from_records(
                context.state["question"],
                [record],
                expected_reducer=expected_reducer,
            )
            if result:
                return with_fallback_traces(result, context=context)
    return None


def try_fetch_fallback(
    *,
    context: FallbackExecutionContext,
    candidate_urls: list[str],
    expected_reducer: str,
) -> dict[str, Any] | None:
    if "fetch_url" not in context.tools_by_name:
        return None

    for candidate_url in candidate_urls:
        fetched = invoke_fallback_tool(
            context=context,
            tool_name="fetch_url",
            tool_args={"url": candidate_url},
        )
        normalized = normalize_submitted_answer(fetched).strip().lower()
        if not normalized or "warning: target url returned error 404" in normalized:
            continue
        records = evidence_records_from_tool_output("fetch_url", fetched)
        result = fallback_result_from_records(
            context.state["question"],
            records,
            expected_reducer=expected_reducer,
        )
        if result:
            return with_fallback_traces(result, context=context)
    return None


def candidate_urls_from_state(
    state: AgentState,
    ranked_candidates: list[SourceCandidate],
    *,
    predicate: Any = None,
    prefer_expected_domains: bool = False,
) -> list[str]:
    profile = question_profile_from_state(state)
    url_filter = predicate or (lambda url: True)
    urls: list[str] = []

    for candidate in ranked_candidates:
        if candidate.url and url_filter(candidate.url):
            urls.append(candidate.url)

    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        content = str(message.content or "")
        if tool_name in {"web_search", "search_wikipedia", "extract_links_from_url"}:
            for candidate in parse_result_blocks(content, origin_tool=tool_name):
                if url_filter(candidate.url):
                    urls.append(candidate.url)
        else:
            metadata = evidence_records_from_tool_output(tool_name, content)
            for record in metadata:
                if url_filter(record.source_url):
                    urls.append(record.source_url)

    for url in profile.target_urls:
        if url and url_filter(url):
            urls.append(url)

    deduped = list(dict.fromkeys(urls))
    if not prefer_expected_domains or not profile.expected_domains:
        return deduped

    matching = [
        url for url in deduped
        if any(
            (urlparse(url).hostname or "").lower().endswith(expected.lower())
            for expected in profile.expected_domains
        )
    ]
    non_matching = [url for url in deduped if url not in matching]
    return [*matching, *non_matching]
