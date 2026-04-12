"""Prompt nudges for search/candidate steering."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from .candidate_support import ranked_candidates_from_state
from .routing import question_is_metric_row_lookup, question_profile_from_state
from .state import AgentState, URL_RE


def build_ranked_candidate_nudge(state: AgentState) -> str | None:
    ranked_candidates = ranked_candidates_from_state(state)
    if not ranked_candidates:
        return None
    last_tool_name = None
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            last_tool_name = (getattr(message, "name", "") or "").strip()
            break
    if last_tool_name not in {"web_search", "search_wikipedia", "extract_links_from_url"}:
        return None
    profile = question_profile_from_state(state)
    lines: list[str] = []
    for candidate in ranked_candidates[:3]:
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
    if profile.name == "roster_neighbor_lookup":
        guidance.append(
            "Prefer a dated team roster, season-specific player directory, or season-specific player page over a current player biography or current roster template."
        )
    if profile.name == "botanical_classification":
        guidance.append(
            "For botanical classification, search first, then read a source page. Do not finalize from common-usage intuition or from search snippets alone."
        )
    return "\n".join(guidance)


def build_search_nudge(state: AgentState) -> str | None:
    search_tool_names = {"web_search", "search_wikipedia"}
    decision_trace = state.get("decision_trace") or []
    recent = decision_trace[-2:]
    if len(recent) < 2 or not all(
        entry.removeprefix("tool:") in search_tool_names for entry in recent
    ):
        return None
    ranked_candidates = ranked_candidates_from_state(state)
    if ranked_candidates:
        url_list = "\n".join(f"  - {candidate.url}" for candidate in ranked_candidates[:5])
        return (
            f"STOP SEARCHING. You have done {len(recent)} consecutive searches without reading any page. "
            "You MUST now use fetch_url, find_text_in_url, or extract_tables_from_url on one of these URLs "
            f"from your ranked search candidates:\n{url_list}\n"
            "Pick the most relevant one and READ it. Do NOT call web_search or search_wikipedia again."
        )
    urls: list[str] = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in search_tool_names:
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
