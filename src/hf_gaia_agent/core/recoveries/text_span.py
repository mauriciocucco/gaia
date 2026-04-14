"""Core recovery for referenced text-span lookups."""

from __future__ import annotations

from ...graph.routing import question_profile_from_state
from ...graph.state import AgentState
from .utils import (
    RecoveryAttemptBudget,
    candidate_urls_from_state,
    recovery_trace_state,
    try_fetch_recovery,
    try_find_text_recovery,
)


class TextSpanRecovery:
    name = "text_span"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name

    def applies(self, state: AgentState, profile) -> bool:
        return profile.name == "text_span_lookup"

    def run(self, state: AgentState) -> dict[str, object] | None:
        profile = question_profile_from_state(state)
        if profile.name != "text_span_lookup":
            return None
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        budget = RecoveryAttemptBudget(remaining_fetches=4)
        candidate_urls = candidate_urls_from_state(
            state,
            context.ranked_candidates,
            predicate=lambda url: any(
                token in url.lower()
                for token in ("exercise", "1.e", "1.e%3a", "1.e:_")
            ),
            prefer_expected_domains=True,
        )
        queries = [profile.text_filter] if profile.text_filter else []
        result = try_find_text_recovery(
            context=context,
            candidate_urls=candidate_urls,
            queries=queries,
            title_hint="Referenced text span",
            expected_reducer="text_span_attribute",
            budget=budget,
            max_candidate_urls=2,
        )
        if result:
            return result
        return try_fetch_recovery(
            context=context,
            candidate_urls=candidate_urls,
            expected_reducer="text_span_attribute",
            budget=budget,
            max_candidate_urls=2,
        )
