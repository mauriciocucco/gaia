"""Candidate ranking, search-normalization, and tool-followup helpers."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

from langchain_core.messages import ToolMessage

from ..normalize import normalize_submitted_answer
from ..source_pipeline import QuestionProfile, SourceCandidate
from .routing import question_supports_direct_python
from .state import AgentState, URL_RE


def ranked_candidates_from_state(state: AgentState) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for raw in state.get("ranked_candidates") or []:
        if isinstance(raw, SourceCandidate):
            candidates.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        candidates.append(
            SourceCandidate(
                title=str(raw.get("title", "")),
                url=str(raw.get("url", "")),
                snippet=str(raw.get("snippet", "")),
                origin_tool=str(raw.get("origin_tool", "")),
                score=int(raw.get("score", 0)),
                reasons=tuple(raw.get("reasons") or ()),
            )
        )
    return candidates


def merge_ranked_candidates(
    existing: list[SourceCandidate],
    new_items: list[SourceCandidate],
    *,
    max_items: int = 12,
) -> list[SourceCandidate]:
    by_url: dict[str, SourceCandidate] = {}
    for candidate in [*existing, *new_items]:
        url = candidate.url.strip()
        if not url:
            continue
        previous = by_url.get(url)
        if previous is None or candidate.score > previous.score:
            by_url[url] = candidate
    merged = sorted(by_url.values(), key=lambda item: (-item.score, len(item.url)))
    return merged[:max_items]


def normalize_search_query(query: str) -> str:
    tokens = sorted(
        {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 3}
    )
    return " ".join(tokens)


def is_semantically_duplicate_search(
    signature: str, previous_signatures: set[str]
) -> bool:
    if not signature:
        return False
    current_tokens = set(signature.split())
    for previous in previous_signatures:
        previous_tokens = set(previous.split())
        if current_tokens == previous_tokens:
            return True
        union = current_tokens | previous_tokens
        if union and len(current_tokens & previous_tokens) / len(union) >= 0.8:
            return True
    return False


def execute_python_allowed(state: AgentState) -> tuple[bool, str | None]:
    if state.get("local_file_path"):
        return True, "attachment"
    if question_supports_direct_python(state["question"]):
        return True, "prompt"
    from .evidence_support import collect_evidence_records_from_state

    records = collect_evidence_records_from_state(state)
    if any(record.kind in {"table", "text", "transcript"} for record in records):
        return True, "fetched_evidence"
    return False, None


def is_obviously_bad_candidate_url(url: str) -> bool:
    lowered = url.lower()
    bad_fragments = (
        "forum.",
        "forums.",
        "reddit.com",
        "redd.it",
        "quora.com",
        "zhihu.com",
        "news.google",
        "grokipedia",
        "instagram.com",
        "facebook.com",
        "fandom.com",
        "pinterest.com",
        "tiktok.com",
        "lowyat.net",
        "naca.com",
        "/search?",
    )
    return any(fragment in lowered for fragment in bad_fragments)


def preferred_ranked_fetch_candidate(
    *,
    requested_url: str,
    profile: QuestionProfile,
    ranked_candidates: list[SourceCandidate],
    fetched_urls: set[str],
) -> SourceCandidate | None:
    if not ranked_candidates:
        return None
    requested_lower = requested_url.lower()
    best_candidate = ranked_candidates[0]
    if best_candidate.url in fetched_urls or best_candidate.url == requested_url:
        return None

    if profile.name == "text_span_lookup":
        best_reasons = set(best_candidate.reasons)
        requested_is_generic_mirror = any(
            fragment in requested_lower
            for fragment in (
                "/courses/",
                "/ancillary_materials/",
                "/bookshelves/introductory_chemistry/introductory_chemistry_(libretexts)",
            )
        )
        requested_is_exact_exercise = any(
            token in requested_lower for token in ("1.e", "1.e%3a", "1.0e", "exercise")
        )
        if (
            best_candidate.score >= 45
            and {"exercise_page", "canonical_textbook_path"} & best_reasons
            and (requested_is_generic_mirror or not requested_is_exact_exercise)
        ):
            return best_candidate

    if profile.name == "article_to_paper":
        best_reasons = set(best_candidate.reasons)
        requested_is_article = "/articles/" in requested_lower or re.search(
            r"/\d{5,}/", requested_lower
        )
        if (
            requested_is_article
            and best_candidate.score >= 40
            and {"linked_source", "primary_source_hint", "paper_mention"} & best_reasons
        ):
            return best_candidate

    if profile.name == "table_lookup":
        best_reasons = set(best_candidate.reasons)
        requested_is_discussion = any(
            fragment in requested_lower
            for fragment in ("reddit.com", "redd.it", "forum", "forums.", "quora.com")
        )
        if (
            requested_is_discussion
            and best_candidate.score >= 35
            and {"expected_domain", "tableish_title"} & best_reasons
        ):
            return best_candidate

    if profile.name == "roster_neighbor_lookup" and profile.expected_date:
        decoded_requested_lower = unquote(requested_url).lower()
        requested_is_current_roster = (
            "list_of_current" in requested_lower
            or "current" in requested_lower
            or ("/wiki/template:" in requested_lower and "roster" in requested_lower)
        )
        requested_is_minor_or_player_page = any(
            fragment in decoded_requested_lower
            for fragment in (
                "/player/",
                "/players/",
                "minorbaseball",
                "minor-league",
                "minor_league",
                "milb",
            )
        )
        requested_is_subject_profile = bool(
            profile.subject_name
            and any(
                token.lower() in decoded_requested_lower
                for token in profile.subject_name.split()
            )
            and "roster" not in decoded_requested_lower
        )
        best_reasons = set(best_candidate.reasons)
        if (
            requested_is_current_roster
            and best_candidate.score >= 45
            and {
                "dated_roster_hint",
                "expected_date",
                "expected_date_partial",
                "expected_year",
            }
            & best_reasons
        ):
            return best_candidate
        if (
            requested_is_minor_or_player_page or requested_is_subject_profile
        ) and best_candidate.score >= 25 and {
            "roster_page_hint",
            "dated_roster_hint",
            "tableish_title",
        } & best_reasons:
            return best_candidate
    return None


def pick_better_fetch_candidate(
    *,
    requested_url: str,
    profile: QuestionProfile,
    ranked_candidates: list[SourceCandidate],
    fetched_urls: set[str],
) -> SourceCandidate | None:
    if not requested_url or not is_obviously_bad_candidate_url(requested_url):
        return preferred_ranked_fetch_candidate(
            requested_url=requested_url,
            profile=profile,
            ranked_candidates=ranked_candidates,
            fetched_urls=fetched_urls,
        )

    for candidate in ranked_candidates:
        if candidate.url in fetched_urls or candidate.url == requested_url:
            continue
        if is_obviously_bad_candidate_url(candidate.url):
            continue
        if candidate.score >= 20:
            return candidate
    return None


def pick_best_unfetched_candidate(
    state: AgentState, *, fetched_urls: set[str]
) -> SourceCandidate | None:
    for candidate in ranked_candidates_from_state(state):
        if candidate.url in fetched_urls or is_obviously_bad_candidate_url(candidate.url):
            continue
        return candidate

    candidates: list[str] = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in ("web_search", "search_wikipedia"):
            continue
        urls = URL_RE.findall(str(message.content))
        for url in urls:
            cleaned_url = url.rstrip(".,;:)")
            if cleaned_url in fetched_urls:
                continue
            if any(
                skip in cleaned_url
                for skip in ("google.com", "zhihu.com", "spotify.com", "news.google")
            ):
                continue
            candidates.append(cleaned_url)
        if len(candidates) >= 10:
            break

    for url in candidates:
        if "wikipedia.org" in url or "libretexts.org" in url:
            return SourceCandidate(
                title=url,
                url=url,
                snippet="",
                origin_tool="fallback",
                score=1,
                reasons=("fallback_url",),
            )
    if not candidates:
        return None
    return SourceCandidate(
        title=candidates[0],
        url=candidates[0],
        snippet="",
        origin_tool="fallback",
        score=1,
        reasons=("fallback_url",),
    )


def text_span_auto_follow_candidate(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    result_text: str,
    profile: QuestionProfile,
    ranked_candidates: list[SourceCandidate],
    fetched_urls: set[str],
) -> SourceCandidate | None:
    if profile.name != "text_span_lookup" or tool_name != "find_text_in_url":
        return None
    if normalize_submitted_answer(result_text).strip().lower() != "no matches found.":
        return None
    return preferred_ranked_fetch_candidate(
        requested_url=str(tool_args.get("url", "")).strip(),
        profile=profile,
        ranked_candidates=ranked_candidates,
        fetched_urls=fetched_urls,
    )


def article_to_paper_auto_links_result(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    result_text: str,
    profile: QuestionProfile,
) -> tuple[dict[str, Any], str] | None:
    if profile.name != "article_to_paper" or tool_name != "extract_links_from_url":
        return None
    if normalize_submitted_answer(result_text).strip().lower() != "no matching links found.":
        return None
    if not str(tool_args.get("text_filter", "")).strip():
        return None
    return (
        {"url": str(tool_args.get("url", "")).strip(), "text_filter": ""},
        "extract_links_from_url",
    )
