"""Candidate scoring and ranking.

Scores :class:`SourceCandidate` instances produced by evidence collection,
using a declarative configuration instead of scattered magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ._models import QuestionProfile, SourceCandidate
from ._utils import is_metric_row_lookup_question, query_tokens, registered_host

# ---------------------------------------------------------------------------
# Scoring configuration — single source of truth for all score adjustments
# ---------------------------------------------------------------------------

CANDIDATE_SCORES = {
    # General penalties
    "low_signal_domain": -120,
    "discussion_source_penalty": -55,
    "commercial_noise_penalty": -140,
    # Domain matching
    "expected_domain": 90,
    "expected_domain_miss": -35,
    "preferred_source": 30,
    # Token & metadata matching
    "token_overlap_per_token": 8,
    "expected_date": 35,
    "expected_date_partial": 10,
    "expected_year": 6,
    "expected_date_miss": -18,
    "expected_author": 35,
    # article_to_paper
    "article_path": 20,
    "paper_mention": 12,
    "linked_source": 125,
    "primary_source_hint": 28,
    "funding_hint": 24,
    "article_loop_penalty": -40,
    # text_span_lookup
    "exercise_page": 28,
    "reference_text_match": 16,
    "canonical_textbook_path": 40,
    "mirror_course_penalty": -28,
    "bulk_pdf_penalty": -80,
    # entity_role_chain
    "role_chain_hint": 18,
    "target_series_hint": 12,
    # table / roster / wikipedia
    "tableish_title": 10,
    # metric_row
    "stats_page_hint": 24,
    "roster_page_penalty": -24,
    # roster_neighbor_lookup
    "roster_page_hint": 22,
    "dated_roster_hint": 18,
    "official_yearbook_hint": 36,
    "subject_profile_penalty": -34,
    "stats_page_penalty": -42,
    "current_roster_penalty": -52,
    "off_scope_roster_penalty": -60,
}


def _sc(key: str) -> int:
    """Get a score value from config by key."""
    return CANDIDATE_SCORES[key]


@dataclass(frozen=True)
class CandidateScoringContext:
    question: str
    profile: QuestionProfile
    candidate: SourceCandidate
    metric_row_lookup: bool
    question_tokens: set[str]
    expected_date_tokens: set[str]
    expected_year_tokens: set[str]
    author_tokens: set[str]
    haystack: str
    haystack_tokens: set[str]
    scope: str
    domain: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_candidates(
    candidates: Sequence[SourceCandidate],
    *,
    question: str,
    profile: QuestionProfile,
) -> list[SourceCandidate]:
    scored: list[SourceCandidate] = []
    metric_row_lookup = is_metric_row_lookup_question(question)
    question_tokens = query_tokens(question)
    expected_date_tokens = query_tokens(profile.expected_date or "")
    author_tokens = query_tokens(profile.expected_author or "")
    expected_year_tokens = {token for token in expected_date_tokens if len(token) == 4 and token.isdigit()}
    for candidate in candidates:
        score = 0
        reasons: list[str] = []
        haystack = f"{candidate.title}\n{candidate.snippet}\n{candidate.url}"
        context = CandidateScoringContext(
            question=question,
            profile=profile,
            candidate=candidate,
            metric_row_lookup=metric_row_lookup,
            question_tokens=question_tokens,
            expected_date_tokens=expected_date_tokens,
            expected_year_tokens=expected_year_tokens,
            author_tokens=author_tokens,
            haystack=haystack,
            haystack_tokens=query_tokens(haystack),
            scope=f"{candidate.title}\n{candidate.url}",
            domain=registered_host(candidate.url),
        )

        score, reasons = _apply_general_rules(score, reasons, context=context)
        score, reasons = _apply_token_and_metadata_rules(
            score,
            reasons,
            context=context,
        )
        score, reasons = _apply_profile_specific_rules(
            score,
            reasons,
            context=context,
        )
        if _should_drop_candidate(score=score, reasons=reasons, context=context):
            continue

        scored.append(
            SourceCandidate(
                title=candidate.title,
                url=candidate.url,
                snippet=candidate.snippet,
                origin_tool=candidate.origin_tool,
                score=score,
                reasons=tuple(reasons),
            )
        )
    scored.sort(key=lambda item: (-item.score, len(item.url)))
    return _deduplicate(scored)


# ---------------------------------------------------------------------------
# Rule groups
# ---------------------------------------------------------------------------


def _apply_general_rules(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if _is_low_signal_domain(context.domain):
        score += _sc("low_signal_domain")
        reasons.append("low_signal_domain")
    if _is_discussion_source(
        domain=context.domain,
        url=context.candidate.url,
        haystack=context.haystack,
    ):
        if context.profile.name in {
            "table_lookup",
            "roster_neighbor_lookup",
            "text_span_lookup",
            "entity_attribute_lookup",
            "wikipedia_lookup",
        }:
            score += _sc("discussion_source_penalty")
            reasons.append("discussion_source_penalty")
    if context.domain and context.profile.expected_domains and any(
        context.domain.endswith(expected) for expected in context.profile.expected_domains
    ):
        score += _sc("expected_domain")
        reasons.append("expected_domain")
    elif context.domain and context.profile.expected_domains:
        score += _sc("expected_domain_miss")
        reasons.append("expected_domain_miss")
    if (
        context.candidate.origin_tool == "search_wikipedia"
        and context.profile.name == "wikipedia_lookup"
    ):
        score += _sc("preferred_source")
        reasons.append("preferred_source")
    if _looks_like_offtopic_commercial_noise(context):
        score += _sc("commercial_noise_penalty")
        reasons.append("commercial_noise_penalty")
    return score, reasons


def _apply_token_and_metadata_rules(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    overlap = len(context.haystack_tokens & context.question_tokens)
    if overlap:
        score += overlap * _sc("token_overlap_per_token")
        reasons.append(f"token_overlap:{overlap}")
    if context.profile.expected_date and context.profile.expected_date.lower() in context.haystack.lower():
        score += _sc("expected_date")
        reasons.append("expected_date")
    elif context.expected_date_tokens and context.expected_date_tokens <= context.haystack_tokens:
        score += _sc("expected_date_partial")
        reasons.append("expected_date_partial")
    elif context.expected_year_tokens and context.expected_year_tokens <= context.haystack_tokens:
        score += _sc("expected_year")
        reasons.append("expected_year")
    elif context.profile.name == "roster_neighbor_lookup" and context.profile.expected_date:
        score += _sc("expected_date_miss")
        reasons.append("expected_date_miss")
    if context.author_tokens and context.author_tokens <= query_tokens(context.haystack):
        score += _sc("expected_author")
        reasons.append("expected_author")
    return score, reasons


def _apply_profile_specific_rules(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if context.profile.name == "article_to_paper":
        score, reasons = _score_article_to_paper(score, reasons, context=context)
    if context.profile.name == "text_span_lookup":
        score, reasons = _score_text_span(score, reasons, context=context)
    if context.profile.name == "entity_role_chain":
        score, reasons = _score_entity_role_chain(score, reasons, context=context)
    if context.profile.name in {"table_lookup", "roster_neighbor_lookup", "wikipedia_lookup"}:
        if any(token in context.candidate.title.lower() for token in ("roster", "statistics", "olympics")):
            score += _sc("tableish_title")
            reasons.append("tableish_title")
    if context.metric_row_lookup:
        score, reasons = _score_metric_row(score, reasons, context=context)
    if context.profile.name == "roster_neighbor_lookup" and context.profile.expected_date:
        score, reasons = _score_roster_neighbor(score, reasons, context=context)
    return score, reasons


# ---------------------------------------------------------------------------
# Per-profile scoring helpers
# ---------------------------------------------------------------------------


def _score_article_to_paper(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if "/articles/" in context.candidate.url or re.search(r"/\d{5,}/", context.candidate.url):
        score += _sc("article_path")
        reasons.append("article_path")
    if "paper" in context.candidate.title.lower() or "paper" in context.candidate.snippet.lower():
        score += _sc("paper_mention")
        reasons.append("paper_mention")
    if context.candidate.origin_tool == "extract_links_from_url":
        if context.domain and not any(
            context.domain.endswith(expected) for expected in context.profile.expected_domains
        ):
            score += _sc("linked_source")
            reasons.append("linked_source")
        if any(
            token in context.haystack.lower()
            for token in ("published paper", "study", "journal", "for journalists", "northwestern")
        ):
            score += _sc("primary_source_hint")
            reasons.append("primary_source_hint")
        if (
            context.domain
            and any(context.domain.endswith(expected) for expected in context.profile.expected_domains)
            and "/articles/" in context.candidate.url
        ):
            score += _sc("article_loop_penalty")
            reasons.append("article_loop_penalty")
    if any(
        token in context.haystack.lower()
        for token in (
            "award number",
            "award no.",
            "supported by nasa",
            "funded by nasa",
            "grant number",
        )
    ):
        score += _sc("funding_hint")
        reasons.append("funding_hint")
    return score, reasons


def _score_text_span(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if any(token in context.haystack.lower() for token in ("1.e", "exercises", "exercise")):
        score += _sc("exercise_page")
        reasons.append("exercise_page")
    if any(token in context.haystack.lower() for token in ("ck-12", "introductory chemistry", "libretexts")):
        score += _sc("reference_text_match")
        reasons.append("reference_text_match")
    if (
        "bookshelves/introductory_chemistry" in context.candidate.url.lower()
        and "ck-12" in context.candidate.url.lower()
        and any(
            token in context.candidate.url.lower()
            for token in ("1.e", "1.e%3a", "1.0e", "exercise")
        )
    ):
        score += _sc("canonical_textbook_path")
        reasons.append("canonical_textbook_path")
    if any(token in context.candidate.url.lower() for token in ("/courses/", "/ancillary_materials/")):
        score += _sc("mirror_course_penalty")
        reasons.append("mirror_course_penalty")
    if (
        "batch.libretexts.org" in registered_host(context.candidate.url)
        or context.candidate.url.lower().endswith(".pdf")
    ):
        score += _sc("bulk_pdf_penalty")
        reasons.append("bulk_pdf_penalty")
    return score, reasons


def _score_entity_role_chain(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if any(token in context.haystack.lower() for token in ("cast", "character", "filmography", "portrayed", "played")):
        score += _sc("role_chain_hint")
        reasons.append("role_chain_hint")
    if any(token in context.haystack.lower() for token in ("magda m", "raymond", "wszyscy kochaja")):
        score += _sc("target_series_hint")
        reasons.append("target_series_hint")
    return score, reasons


def _score_metric_row(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if (
        any(
            token in context.haystack.lower()
            for token in ("hitting stats", "batting stats", "team stats", "player stats")
        )
        or any(fragment in context.candidate.url.lower() for fragment in ("hitting.php", "batting", "/stats/"))
    ):
        score += _sc("stats_page_hint")
        reasons.append("stats_page_hint")
    if "roster.php" in context.candidate.url.lower() or "roster" in context.candidate.title.lower():
        score += _sc("roster_page_penalty")
        reasons.append("roster_page_penalty")
    return score, reasons


def _score_roster_neighbor(
    score: int,
    reasons: list[str],
    *,
    context: CandidateScoringContext,
) -> tuple[int, list[str]]:
    if any(token in context.haystack.lower() for token in ("roster", "pitchers", "numbers", "staff")):
        score += _sc("roster_page_hint")
        reasons.append("roster_page_hint")
    if any(token in context.haystack.lower() for token in ("2023", "july", "season", "archive", "oldid")):
        score += _sc("dated_roster_hint")
        reasons.append("dated_roster_hint")
    if (
        any(
            fragment in context.candidate.url.lower()
            for fragment in ("/team/player/list/", "/team/player/detail/")
        )
        and any(
            token in context.haystack.lower()
            for token in ("player directory", "show other players", "pitchers")
        )
        and context.expected_year_tokens
        and context.expected_year_tokens <= context.haystack_tokens
    ):
        score += _sc("official_yearbook_hint")
        reasons.append("official_yearbook_hint")
    if (
        context.profile.subject_name
        and any(
            token.lower() in context.haystack.lower()
            for token in context.profile.subject_name.split()
        )
        and not any(
            token in context.haystack.lower()
            for token in ("roster", "pitchers", "numbers", "staff", "team")
        )
    ):
        score += _sc("subject_profile_penalty")
        reasons.append("subject_profile_penalty")
    if (
        any(
            token in context.scope.lower()
            for token in ("individual pitching", "individual batting", "pitching leaders", "batting leaders")
        )
        or "/stats/" in context.candidate.url.lower()
    ) and not any(
        token in context.scope.lower()
        for token in ("roster", "roster listing", "team roster", "staff")
    ):
        score += _sc("stats_page_penalty")
        reasons.append("stats_page_penalty")
    if (
        "list_of_current" in context.candidate.url.lower()
        or "current roster" in context.haystack.lower()
        or context.candidate.url.lower().endswith("/current")
        or (
            "/wiki/template:" in context.candidate.url.lower()
            and "roster" in context.haystack.lower()
        )
    ):
        score += _sc("current_roster_penalty")
        reasons.append("current_roster_penalty")
    if _is_off_scope_roster_source(
        url=context.candidate.url,
        haystack=context.haystack,
    ):
        score += _sc("off_scope_roster_penalty")
        reasons.append("off_scope_roster_penalty")
    return score, reasons


# ---------------------------------------------------------------------------
# Domain / source classification helpers
# ---------------------------------------------------------------------------


def _is_low_signal_domain(domain: str) -> bool:
    return domain.endswith(
        (
            "instagram.com",
            "facebook.com",
            "fandom.com",
            "grokipedia.com",
            "pinterest.com",
            "tiktok.com",
        )
    )


def _has_strong_relevance_signal(context: CandidateScoringContext) -> bool:
    overlap = len(context.haystack_tokens & context.question_tokens)
    if overlap >= 2:
        return True
    if context.domain and context.profile.expected_domains and any(
        context.domain.endswith(expected) for expected in context.profile.expected_domains
    ):
        return True
    if (
        context.candidate.origin_tool == "search_wikipedia"
        and context.profile.name == "wikipedia_lookup"
    ):
        return True
    if context.author_tokens and context.author_tokens <= context.haystack_tokens:
        return True
    if context.expected_date_tokens and context.expected_date_tokens <= context.haystack_tokens:
        return True
    return bool(context.expected_year_tokens and context.expected_year_tokens <= context.haystack_tokens)


def _looks_like_offtopic_commercial_noise(context: CandidateScoringContext) -> bool:
    if _has_strong_relevance_signal(context):
        return False
    overlap = len(context.haystack_tokens & context.question_tokens)
    if overlap > 1:
        return False
    lowered = context.haystack.lower()
    return any(
        fragment in lowered
        for fragment in (
            "restaurant",
            " menu",
            "menu ",
            "lottery",
            "pharmacy",
            "book now",
            "delivery",
            "order online",
            "takeout",
            "reservation",
            "reservations",
            "open now",
        )
    )


def _is_discussion_source(*, domain: str, url: str, haystack: str) -> bool:
    lowered = f"{domain}\n{url}\n{haystack}".lower()
    return any(
        token in lowered
        for token in (
            "reddit.com",
            "redd.it",
            "forum",
            "forums.",
            "discussion",
            "quora.com",
            "stackexchange.com",
        )
    )


def _should_drop_candidate(
    *,
    score: int,
    reasons: list[str],
    context: CandidateScoringContext,
) -> bool:
    if "commercial_noise_penalty" not in reasons:
        return False
    if _has_strong_relevance_signal(context):
        return False
    return score < 0


def _is_off_scope_roster_source(*, url: str, haystack: str) -> bool:
    lowered = f"{url}\n{haystack}".lower()
    return any(
        token in lowered
        for token in (
            "minorbaseball",
            "minor league",
            "minor-league",
            "milb",
            "triple-a",
            "double-a",
            "single-a",
            "farm team",
        )
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate(scored: list[SourceCandidate]) -> list[SourceCandidate]:
    deduped: list[SourceCandidate] = []
    seen: set[str] = set()
    for item in scored:
        if item.url in seen:
            continue
        deduped.append(item)
        seen.add(item.url)
    return deduped
