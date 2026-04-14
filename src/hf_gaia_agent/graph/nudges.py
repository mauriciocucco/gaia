"""Prompt nudges for search/candidate steering."""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

from .candidate_support import bucket_ranked_candidates, ranked_candidates_from_state
from .routing import question_is_metric_row_lookup, question_profile_from_state
from .state import AgentState, URL_RE


_SEARCH_TOOL_NAMES = {"web_search", "search_wikipedia"}
_FETCH_TOOL_NAMES = {
    "fetch_url",
    "find_text_in_url",
    "extract_tables_from_url",
    "extract_links_from_url",
}
_STRATEGY_SHIFT_GUIDANCE = (
    "Change strategy now: use search_wikipedia with 1-3 named entities, "
    "use fetch_wikipedia_page if the title is clear, fetch a likely official URL directly, "
    "or answer from the best evidence already collected. Do not call web_search again with the same idea."
)


def _last_tool_name(state: AgentState) -> str | None:
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            return (getattr(message, "name", "") or "").strip()
    return None


def _fetched_urls_from_state(state: AgentState) -> set[str]:
    fetched_urls: set[str] = set()
    for entry in state.get("tool_trace") or []:
        if not any(entry.startswith(f"{tool_name}(") for tool_name in _FETCH_TOOL_NAMES):
            continue
        url_match = re.search(r"'url':\s*'([^']+)'", entry)
        if url_match:
            fetched_urls.add(url_match.group(1))
    return fetched_urls


def build_ranked_candidate_nudge(state: AgentState) -> str | None:
    ranked_candidates = ranked_candidates_from_state(state)
    if not ranked_candidates:
        return None
    last_tool_name = _last_tool_name(state)
    if last_tool_name not in {"web_search", "search_wikipedia", "extract_links_from_url"}:
        return None
    buckets = bucket_ranked_candidates(
        ranked_candidates,
        fetched_urls=_fetched_urls_from_state(state),
    )
    if not buckets.useful_unfetched:
        return None
    profile = question_profile_from_state(state)
    lines: list[str] = []
    for candidate in buckets.useful_unfetched[:3]:
        reasons = []
        for reason in candidate.reasons[:3]:
            if reason == "expected_domain":
                reasons.append("expected domain")
            elif reason == "preferred_source":
                reasons.append("preferred source")
            elif reason == "article_path":
                reasons.append("article-like URL")
            elif reason == "paper_mention":
                reasons.append("paper mention")
            elif reason.startswith("token_overlap:"):
                reasons.append(f"token overlap {reason.split(':', 1)[1]}")
            elif reason:
                reasons.append(reason.replace("_", " "))
        reason_text = ", ".join(reasons) if reasons else "best textual match"
        lines.append(f"- {candidate.title}\n  URL: {candidate.url}\n  Why: {reason_text}")
    guidance = [
        "Most promising sources right now:",
        *lines,
        "Pick the best candidate and READ it before doing more search.",
    ]
    if profile.name == "wikipedia_lookup":
        guidance.append(
            "If the answer depends on a table or participant counts, prefer extract_tables_from_url on the best Wikipedia candidate."
        )
    if profile.name == "text_span_lookup":
        guidance.append(
            "For text-span lookups, prefer the exact exercise/page candidate over generic course mirrors or bulk PDFs."
        )
    if question_is_metric_row_lookup(state["question"]):
        guidance.append(
            "For stat lookups, prefer batting, hitting, or stats pages over roster pages; once a fetched table contains both relevant metrics, stop and answer from that table."
        )
    if profile.name == "temporal_ordered_list":
        guidance.append(
            "Prefer a dated team roster, season-specific player directory, or season-specific player page over a current player biography or current roster template."
        )
    if profile.name == "list_item_classification":
        guidance.append(
            "For botanical classification, search first, then read a source page. Do not finalize from common-usage intuition or from search snippets alone."
        )
    return "\n".join(guidance)


def build_stuck_search_nudge(state: AgentState) -> str | None:
    if _last_tool_name(state) not in _SEARCH_TOOL_NAMES:
        return None

    search_history = [
        item
        for item in (
            state.get("search_history_fingerprints")
            or state.get("search_history_normalized")
            or []
        )
        if item
    ]
    if not search_history:
        return None

    latest_signature = search_history[-1]
    latest_count = sum(signature == latest_signature for signature in search_history)
    buckets = bucket_ranked_candidates(
        ranked_candidates_from_state(state),
        fetched_urls=_fetched_urls_from_state(state),
    )

    if latest_count < 3 and not (
        latest_count >= 2 and not buckets.useful_unfetched
    ):
        return None

    signature_hint = latest_signature.replace(" ", ", ")
    guidance = [
        "SEARCH LOOP DETECTED.",
        (
            "The same normalized search has already repeated without producing new useful evidence. "
            f"Repeated search signature: {signature_hint or latest_signature}."
        ),
    ]
    if buckets.useful_unfetched:
        candidate_lines = "\n".join(
            f"  - {candidate.url}" for candidate in buckets.useful_unfetched[:3]
        )
        guidance.append(
            "Do not repeat this query. Read one of these unfetched candidates first:"
        )
        guidance.append(candidate_lines)
        guidance.append(
            "Use fetch_url, find_text_in_url, or extract_tables_from_url on one of them before any new search."
        )
        return "\n".join(guidance)

    if buckets.exhausted_useful:
        guidance.append(
            "The useful ranked candidates are already exhausted, so rereading them will not help."
        )
    elif buckets.low_quality_unfetched:
        guidance.append(
            "The remaining ranked candidates are low-quality or off-topic, so fetching them is not a productive next step."
        )
    guidance.append(_STRATEGY_SHIFT_GUIDANCE)
    return "\n".join(guidance)


def build_search_nudge(state: AgentState) -> str | None:
    decision_trace = state.get("decision_trace") or []
    recent = decision_trace[-2:]
    if len(recent) < 2 or not all(
        entry.removeprefix("tool:") in _SEARCH_TOOL_NAMES for entry in recent
    ):
        return None
    buckets = bucket_ranked_candidates(
        ranked_candidates_from_state(state),
        fetched_urls=_fetched_urls_from_state(state),
    )
    if buckets.useful_unfetched:
        url_list = "\n".join(
            f"  - {candidate.url}" for candidate in buckets.useful_unfetched[:5]
        )
        return (
            f"STOP SEARCHING. You have done {len(recent)} consecutive searches without reading any page. "
            "You MUST now use fetch_url, find_text_in_url, or extract_tables_from_url on one of these URLs "
            f"from your ranked search candidates:\n{url_list}\n"
            "Pick the most relevant one and READ it. Do NOT call web_search or search_wikipedia again."
        )
    if buckets.exhausted_useful or buckets.low_quality_unfetched:
        return (
            "STOP SEARCHING. The ranked candidates are already exhausted or low-quality. "
            f"{_STRATEGY_SHIFT_GUIDANCE}"
        )
    urls: list[str] = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in _SEARCH_TOOL_NAMES:
            break
        found = URL_RE.findall(str(message.content))
        urls.extend(found)
        if len(urls) >= 5:
            break
    unique_urls = list(dict.fromkeys(urls))[:5]
    if unique_urls:
        url_list = "\n".join(f"  - {url}" for url in unique_urls)
        return (
            f"STOP SEARCHING. You have done {len(recent)} consecutive searches without reading any page. "
            "You MUST now use fetch_url, find_text_in_url, or extract_tables_from_url on one of these URLs "
            f"from your search results:\n{url_list}\n"
            "Pick the most relevant one and READ it. Do NOT call web_search or search_wikipedia again."
        )
    return (
        "STOP SEARCHING. You have done multiple searches with no useful results. "
        "Try a completely different approach: "
        "1) Construct a likely URL directly and fetch it (e.g. for a known website), or "
        "2) Use very different search keywords, or "
        "3) Break the problem into smaller sub-questions."
    )
