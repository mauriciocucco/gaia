"""Registered answer-retry rules kept outside the workflow shell."""

from __future__ import annotations

from .contracts import AnswerRetryRule
from .evidence_support import (
    requires_botanical_classification_retry,
    requires_temporal_roster_retry,
)
from .routing import question_profile_from_state
from .state import AgentState


class TemporalRosterRetryRule:
    name = "temporal_roster"

    def applies(self, state: AgentState, answer_text: str) -> bool:
        return requires_temporal_roster_retry(state, answer_text)

    def guidance(self, state: AgentState) -> str | None:
        profile = question_profile_from_state(state)
        return (
            f" This question is date-sensitive ({profile.expected_date}). "
            "Your current evidence looks like a current or undated roster, so do not answer yet. "
            "Do not treat current player profiles, roster templates, or season stat leaderboards as substitutes for dated roster evidence. "
            "A season-specific official player directory or season-specific player detail page is acceptable if it explicitly matches the requested season and shows the numbered neighboring players. "
            "Fetch a roster/archive/oldid/team page or season-specific player directory that is explicitly grounded to the requested date or season."
        )


class BotanicalClassificationRetryRule:
    name = "botanical_classification"

    def applies(self, state: AgentState, answer_text: str) -> bool:
        return requires_botanical_classification_retry(state, answer_text)

    def guidance(self, state: AgentState) -> str | None:
        return (
            " This is a botanical classification task. Do not answer from common culinary usage, prior knowledge, or search snippets alone. "
            "Search for relevant sources, read at least one source page with fetch_url or find_text_in_url, and only then answer from grounded evidence."
        )


def build_answer_retry_rules() -> list[AnswerRetryRule]:
    return [TemporalRosterRetryRule(), BotanicalClassificationRetryRule()]


def should_retry_answer(state: AgentState, answer_text: str) -> bool:
    return any(rule.applies(state, answer_text) for rule in build_answer_retry_rules())


def build_retry_guidance(state: AgentState, answer_text: str) -> list[str]:
    guidance: list[str] = []
    for rule in build_answer_retry_rules():
        if not rule.applies(state, answer_text):
            continue
        message = rule.guidance(state)
        if message:
            guidance.append(message)
    return guidance
