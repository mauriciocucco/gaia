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
from ..graph.candidate_support import (
    is_low_quality_ranked_candidate,
    is_obviously_bad_candidate_url,
    is_semantically_duplicate_search,
    merge_ranked_candidates,
    normalize_search_query,
)
from ..graph.state import AgentState
from ..graph.routing import question_profile_from_state

_SEARCH_TOOL_NAMES = {"web_search", "search_wikipedia"}
_RANKING_TOOL_NAMES = _SEARCH_TOOL_NAMES | {"extract_links_from_url"}
_FETCH_TRACKED_TOOL_NAMES = {"fetch_url", "fetch_wikipedia_page"}


@dataclass
class FallbackAttemptBudget:
    remaining_searches: int | None = None
    remaining_fetches: int | None = None

    def consume_search(self) -> bool:
        if self.remaining_searches is None:
            return True
        if self.remaining_searches <= 0:
            return False
        self.remaining_searches -= 1
        return True

    def consume_fetch(self) -> bool:
        if self.remaining_fetches is None:
            return True
        if self.remaining_fetches <= 0:
            return False
        self.remaining_fetches -= 1
        return True


@dataclass
class FallbackExecutionContext:
    tools_by_name: dict[str, Any]
    state: AgentState
    question_profile: Any
    tool_trace: list[str]
    decision_trace: list[str]
    ranked_candidates: list[SourceCandidate]
    search_history: list[str]
    previous_search_signatures: set[str]
    fetched_urls: set[str]


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
    search_history = list(state.get("search_history_normalized") or [])
    return FallbackExecutionContext(
        tools_by_name=tools_by_name,
        state=state,
        question_profile=question_profile_from_state(state),
        tool_trace=list(state.get("tool_trace") or []),
        decision_trace=list(state.get("decision_trace") or []),
        ranked_candidates=ranked_candidates,
        search_history=search_history,
        previous_search_signatures=set(search_history),
        fetched_urls=_fetched_urls_from_tool_trace(state.get("tool_trace") or []),
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

    query = str(tool_args.get("query", "")).strip()
    if tool_name in _SEARCH_TOOL_NAMES and query:
        signature = normalize_search_query(query)
        if signature:
            context.search_history.append(signature)
            context.previous_search_signatures.add(signature)

    fetched_url = str(tool_args.get("url", "")).strip()
    if tool_name in _FETCH_TRACKED_TOOL_NAMES and fetched_url:
        context.fetched_urls.add(fetched_url)

    try:
        result = tool.invoke(tool_args)
    except Exception as exc:
        result = f"Tool error: {exc}"
    result_text = str(result)

    if tool_name in _RANKING_TOOL_NAMES:
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
                max_items=24,
            )
    return result_text


def try_search_fallback(
    *,
    context: FallbackExecutionContext,
    query: str,
    max_results: int = 5,
    budget: FallbackAttemptBudget | None = None,
    tool_name: str = "web_search",
    trace_label: str = "finalize_fallback",
) -> str | None:
    normalized_query = str(query).strip()
    if not normalized_query or tool_name not in context.tools_by_name:
        return None
    signature = normalize_search_query(normalized_query)
    if signature and is_semantically_duplicate_search(
        signature,
        context.previous_search_signatures,
    ):
        return None
    if budget is not None and not budget.consume_search():
        return None
    return invoke_fallback_tool(
        context=context,
        tool_name=tool_name,
        tool_args={"query": normalized_query, "max_results": max_results},
        trace_label=trace_label,
    )


def ranked_candidates_from_result_text(
    *,
    context: FallbackExecutionContext,
    result_text: str,
    origin_tool: str,
    predicate: Any = None,
    include_low_quality: bool = False,
) -> list[SourceCandidate]:
    candidate_filter = predicate or (lambda candidate: True)
    parsed_candidates = [
        candidate
        for candidate in parse_result_blocks(result_text, origin_tool=origin_tool)
        if candidate.url and candidate_filter(candidate)
    ]
    if not parsed_candidates:
        return []
    scored_candidates = score_candidates(
        parsed_candidates,
        question=context.state["question"],
        profile=context.question_profile,
    )
    if include_low_quality:
        return scored_candidates
    return [
        candidate
        for candidate in scored_candidates
        if not is_low_quality_ranked_candidate(candidate)
    ]


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
    budget: FallbackAttemptBudget | None = None,
    max_candidate_urls: int = 2,
    max_matches: int = 8,
) -> dict[str, Any] | None:
    if "find_text_in_url" not in context.tools_by_name:
        return None

    filtered_queries = list(dict.fromkeys(query for query in queries if str(query).strip()))
    if not filtered_queries:
        return None

    ordered_candidate_urls = quality_filtered_candidate_urls(
        context=context,
        candidate_urls=candidate_urls,
        max_urls=max_candidate_urls,
    )
    for candidate_url in ordered_candidate_urls:
        for query in filtered_queries:
            if budget is not None and not budget.consume_fetch():
                return None
            found_text = invoke_fallback_tool(
                context=context,
                tool_name="find_text_in_url",
                tool_args={"url": candidate_url, "query": query, "max_matches": max_matches},
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
    budget: FallbackAttemptBudget | None = None,
    max_candidate_urls: int = 2,
) -> dict[str, Any] | None:
    if "fetch_url" not in context.tools_by_name:
        return None

    ordered_candidate_urls = fetch_candidate_urls(
        context=context,
        candidate_urls=candidate_urls,
        max_urls=max_candidate_urls,
    )
    for candidate_url in ordered_candidate_urls:
        if budget is not None and not budget.consume_fetch():
            return None
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
    include_low_quality: bool = False,
) -> list[str]:
    profile = question_profile_from_state(state)
    url_filter = predicate or (lambda url: True)

    seeded_candidates = [
        candidate
        for candidate in ranked_candidates
        if candidate.url and url_filter(candidate.url)
    ]
    supplemental_candidates: list[SourceCandidate] = []
    raw_urls: list[str] = []

    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        content = str(message.content or "")
        if tool_name in _RANKING_TOOL_NAMES:
            for candidate in parse_result_blocks(content, origin_tool=tool_name):
                if candidate.url and url_filter(candidate.url):
                    supplemental_candidates.append(candidate)
                    raw_urls.append(candidate.url)
            continue
        for record in evidence_records_from_tool_output(tool_name, content):
            source_url = str(record.source_url).strip()
            if source_url and url_filter(source_url):
                supplemental_candidates.append(
                    SourceCandidate(
                        title=record.title_or_caption or source_url,
                        url=source_url,
                        snippet=record.content[:240],
                        origin_tool=tool_name,
                    )
                )
                raw_urls.append(source_url)

    for url in profile.target_urls:
        if url and url_filter(url):
            supplemental_candidates.append(
                SourceCandidate(
                    title=url,
                    url=url,
                    snippet="",
                    origin_tool="question_target",
                )
            )
            raw_urls.append(url)

    scored_supplemental = score_candidates(
        supplemental_candidates,
        question=state["question"],
        profile=profile,
    )
    merged_candidates = merge_ranked_candidates(
        seeded_candidates,
        scored_supplemental,
        max_items=max(24, len(seeded_candidates) + len(scored_supplemental) + 4),
    )

    ordered_urls: list[str] = []
    for candidate in merged_candidates:
        if not include_low_quality and is_low_quality_ranked_candidate(candidate):
            continue
        ordered_urls.append(candidate.url)

    for raw_url in raw_urls:
        if raw_url in ordered_urls:
            continue
        if not include_low_quality and is_obviously_bad_candidate_url(raw_url):
            continue
        ordered_urls.append(raw_url)

    ordered_urls = list(dict.fromkeys(url for url in ordered_urls if url))
    if not prefer_expected_domains or not profile.expected_domains:
        return ordered_urls

    matching = [
        url
        for url in ordered_urls
        if any(
            (urlparse(url).hostname or "").lower().endswith(expected.lower())
            for expected in profile.expected_domains
        )
    ]
    non_matching = [url for url in ordered_urls if url not in matching]
    return [*matching, *non_matching]


def quality_filtered_candidate_urls(
    *,
    context: FallbackExecutionContext,
    candidate_urls: list[str],
    include_low_quality: bool = False,
    max_urls: int | None = None,
) -> list[str]:
    ranked_by_url = {candidate.url: candidate for candidate in context.ranked_candidates}
    filtered: list[str] = []
    for candidate_url in dict.fromkeys(url for url in candidate_urls if url):
        ranked_candidate = ranked_by_url.get(candidate_url)
        if ranked_candidate is not None:
            if not include_low_quality and is_low_quality_ranked_candidate(ranked_candidate):
                continue
        elif not include_low_quality and is_obviously_bad_candidate_url(candidate_url):
            continue
        filtered.append(candidate_url)
        if max_urls is not None and len(filtered) >= max_urls:
            break
    return filtered


def fetch_candidate_urls(
    *,
    context: FallbackExecutionContext,
    candidate_urls: list[str],
    include_low_quality: bool = False,
    max_urls: int | None = None,
) -> list[str]:
    filtered = quality_filtered_candidate_urls(
        context=context,
        candidate_urls=candidate_urls,
        include_low_quality=include_low_quality,
    )
    fetchable = [
        candidate_url
        for candidate_url in filtered
        if candidate_url not in context.fetched_urls
    ]
    if max_urls is None:
        return fetchable
    return fetchable[:max_urls]


def unfetched_first_candidate_urls(
    candidate_urls: list[str],
    *,
    fetched_urls: set[str],
) -> list[str]:
    deduped = list(dict.fromkeys(url for url in candidate_urls if url))
    unfetched = [url for url in deduped if url not in fetched_urls]
    exhausted = [url for url in deduped if url in fetched_urls]
    return [*unfetched, *exhausted]


def _fetched_urls_from_tool_trace(tool_trace: list[str]) -> set[str]:
    fetched_urls: set[str] = set()
    for entry in tool_trace:
        if not any(entry.startswith(f"{tool_name}(") for tool_name in _FETCH_TRACKED_TOOL_NAMES):
            continue
        url_match = re.search(r"'url':\s*'([^']+)'", entry)
        if url_match:
            fetched_urls.add(url_match.group(1))
    return fetched_urls
