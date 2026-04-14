"""Core recovery for article-to-paper award identifier questions."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from ...graph.routing import question_profile_from_state
from ...graph.state import AgentState
from ...source_pipeline import SourceCandidate
from .utils import (
    RecoveryAttemptBudget,
    candidate_urls_from_state,
    recovery_trace_state,
    try_fetch_recovery,
    try_find_text_recovery,
    try_search_recovery,
    unfetched_first_candidate_urls,
)


class ArticleToPaperRecovery:
    name = "article_to_paper"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name

    def applies(self, state: AgentState, profile) -> bool:
        if profile.name != "article_to_paper":
            return False
        question_lower = state["question"].lower()
        return "award number" in question_lower or "supported by" in question_lower

    def run(self, state: AgentState) -> dict[str, object] | None:
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        budget = RecoveryAttemptBudget(remaining_searches=3, remaining_fetches=6)
        candidate_urls = unfetched_first_candidate_urls(
            self._identifier_candidate_urls(state, context.ranked_candidates),
            fetched_urls=context.fetched_urls,
        )
        if not candidate_urls and "web_search" in self._tools_by_name:
            for query in self._search_queries(state, context.ranked_candidates):
                try_search_recovery(
                    context=context,
                    query=query,
                    max_results=5,
                    budget=budget,
                )
            candidate_urls = unfetched_first_candidate_urls(
                self._identifier_candidate_urls(state, context.ranked_candidates),
                fetched_urls=context.fetched_urls,
            )

        result = try_find_text_recovery(
            context=context,
            candidate_urls=candidate_urls,
            queries=["NASA award number", "supported by NASA"],
            title_hint="Linked primary source",
            expected_reducer="award_number",
            budget=budget,
            max_candidate_urls=1,
            max_matches=1,
        )
        if result:
            return result

        result = try_fetch_recovery(
            context=context,
            candidate_urls=candidate_urls,
            expected_reducer="award_number",
            budget=budget,
            max_candidate_urls=1,
        )
        if result:
            return result

        if "web_search" in self._tools_by_name:
            for query in self._search_queries(state, context.ranked_candidates):
                try_search_recovery(
                    context=context,
                    query=query,
                    max_results=5,
                    budget=budget,
                )

        expanded_urls = unfetched_first_candidate_urls(
            self._identifier_candidate_urls(state, context.ranked_candidates),
            fetched_urls=context.fetched_urls,
        )
        if expanded_urls == candidate_urls:
            return None

        result = try_find_text_recovery(
            context=context,
            candidate_urls=expanded_urls,
            queries=["NASA award number", "supported by NASA"],
            title_hint="Linked primary source",
            expected_reducer="award_number",
            budget=budget,
            max_candidate_urls=1,
            max_matches=1,
        )
        if result:
            return result
        return try_fetch_recovery(
            context=context,
            candidate_urls=expanded_urls,
            expected_reducer="award_number",
            budget=budget,
            max_candidate_urls=1,
        )

    def _identifier_candidate_urls(
        self, state: AgentState, ranked_candidates: list[SourceCandidate]
    ) -> list[str]:
        profile = question_profile_from_state(state)
        publisher_domains = {domain.lower() for domain in profile.expected_domains if domain}
        publisher_domains.update(
            (urlparse(url).hostname or "").lower()
            for url in profile.target_urls
            if url
        )
        candidates = candidate_urls_from_state(
            state, ranked_candidates, prefer_expected_domains=False
        )
        urls: list[str] = []
        for url in candidates:
            hostname = (urlparse(url).hostname or "").lower()
            if not hostname:
                continue
            if publisher_domains and any(hostname.endswith(domain) for domain in publisher_domains):
                continue
            urls.append(url)
        return list(dict.fromkeys(urls))

    def _search_queries(
        self, state: AgentState, ranked_candidates: list[SourceCandidate]
    ) -> list[str]:
        subject = _award_subject_name(state["question"])
        titles: list[str] = []
        for candidate in ranked_candidates:
            title = re.sub(r"\s+", " ", candidate.title).strip()
            if len(title.split()) < 4:
                continue
            if title not in titles:
                titles.append(title)

        queries: list[str] = []
        for title in titles[:2]:
            if subject:
                queries.append(f'"{title}" "{subject}" NASA award number')
            queries.append(f'"{title}" "NASA award number"')
            if subject:
                queries.append(f'"{title}" "{subject}" supported by NASA')
        if subject:
            queries.extend(
                (
                    f"{subject} NASA award number",
                    f"{subject} supported by NASA award number",
                )
            )
        return list(dict.fromkeys(query for query in queries if query.strip()))


def _award_subject_name(question: str) -> str | None:
    patterns = (
        r"performed by\s+(?P<name>.+?)\s+supported by",
        r"was the work performed by\s+(?P<name>.+?)\s+supported by",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("name")).strip(" ?.")
    return None
