"""Ordered registry of question classifiers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from ._models import QuestionProfile
from ._prompt_items import extract_prompt_list_items
from ._question_detectors import (
    is_article_to_paper_question,
    is_botanical_classification_question,
    is_entity_role_chain_question,
    is_olympics_country_code_question,
    is_roster_neighbor_question,
    is_text_span_lookup_question,
    looks_like_table_question,
)
from ._question_extractors import expected_domains
from ._utils import is_youtube_url


@dataclass(frozen=True, slots=True)
class QuestionClassificationContext:
    question: str
    lowered_question: str
    urls: tuple[str, ...]
    generic_urls: tuple[str, ...]
    file_name: str | None
    local_file_path: str | None
    expected_date: str | None
    expected_author: str | None
    subject_name: str | None
    text_filter: str | None


@dataclass(frozen=True, slots=True)
class QuestionClassifier:
    name: str
    applies: Callable[[QuestionClassificationContext], bool]
    build_profile: Callable[[QuestionClassificationContext], QuestionProfile]

    def classify(
        self, context: QuestionClassificationContext
    ) -> QuestionProfile | None:
        if not self.applies(context):
            return None
        return self.build_profile(context)


def _no_urls(_context: QuestionClassificationContext) -> tuple[str, ...]:
    return ()


def _profile(
    context: QuestionClassificationContext,
    *,
    name: str,
    profile_family: str | None,
    target_urls: Callable[[QuestionClassificationContext], tuple[str, ...]],
    expected_domains_resolver: Callable[[QuestionClassificationContext], tuple[str, ...]],
    preferred_tools: tuple[str, ...],
    text_filter: Callable[[QuestionClassificationContext], str | None],
    prompt_items: Callable[[QuestionClassificationContext], tuple[str, ...]] = _no_urls,
    classification_labels: Callable[[QuestionClassificationContext], dict[str, str] | None] = lambda _context: None,
    ordering_key: Callable[[QuestionClassificationContext], str | None] = lambda _context: None,
    entity_name: Callable[[QuestionClassificationContext], str | None] = lambda _context: None,
    scope: Callable[[QuestionClassificationContext], str | None] = lambda _context: None,
) -> QuestionProfile:
    return QuestionProfile(
        name=name,
        target_urls=target_urls(context),
        expected_domains=expected_domains_resolver(context),
        preferred_tools=preferred_tools,
        expected_date=context.expected_date,
        expected_author=context.expected_author,
        subject_name=context.subject_name,
        text_filter=text_filter(context),
        profile_family=profile_family or name,
        prompt_items=prompt_items(context),
        classification_labels=classification_labels(context),
        ordering_key=ordering_key(context),
        entity_name=entity_name(context),
        scope=scope(context),
    )


def _generic_urls(context: QuestionClassificationContext) -> tuple[str, ...]:
    return context.generic_urls


def _prompt_items(context: QuestionClassificationContext) -> tuple[str, ...]:
    return tuple(extract_prompt_list_items(context.question))


def _youtube_urls(context: QuestionClassificationContext) -> tuple[str, ...]:
    return tuple(url for url in context.urls if is_youtube_url(url))


def _empty_domains(_context: QuestionClassificationContext) -> tuple[str, ...]:
    return ()


def _fixed_domains(
    domains: tuple[str, ...],
) -> Callable[[QuestionClassificationContext], tuple[str, ...]]:
    return lambda _context: domains


def _default_domains(
    defaults: tuple[str, ...],
) -> Callable[[QuestionClassificationContext], tuple[str, ...]]:
    return lambda context: expected_domains(context.question, default=defaults)


def _inferred_text_filter(context: QuestionClassificationContext) -> str | None:
    return context.text_filter


def _roster_text_filter(context: QuestionClassificationContext) -> str | None:
    return context.text_filter or (context.subject_name or "")


def _olympics_text_filter(context: QuestionClassificationContext) -> str | None:
    return context.text_filter or "athletes"


def _botanical_text_filter(context: QuestionClassificationContext) -> str | None:
    return context.text_filter or "botanical classification"


def _entity_role_chain_text_filter(
    context: QuestionClassificationContext,
) -> str | None:
    return context.text_filter or "cast character"


def _botanical_labels(_context: QuestionClassificationContext) -> dict[str, str]:
    return {"include": "vegetable", "exclude": "fruit"}


def _roster_ordering_key(_context: QuestionClassificationContext) -> str:
    return "jersey_number"


def _roster_entity_name(context: QuestionClassificationContext) -> str | None:
    if not context.subject_name:
        return None
    match = context.question.split(context.subject_name)[0]
    del match
    return None


def _roster_scope(context: QuestionClassificationContext) -> str | None:
    lowered = context.lowered_question
    if "pitcher" in lowered:
        return "pitchers"
    if "hitter" in lowered or "batter" in lowered:
        return "hitters"
    return None


QUESTION_CLASSIFIERS: tuple[QuestionClassifier, ...] = (
    QuestionClassifier(
        name="attachment_required",
        applies=lambda context: bool(context.file_name and not context.local_file_path),
        build_profile=lambda context: _profile(
            context,
            name="attachment_required",
            profile_family="attachment_required",
            target_urls=_no_urls,
            expected_domains_resolver=_empty_domains,
            preferred_tools=("read_local_file",),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="transcript_or_video",
        applies=lambda context: any(is_youtube_url(url) for url in context.urls),
        build_profile=lambda context: _profile(
            context,
            name="transcript_or_video",
            profile_family="transcript_or_video",
            target_urls=_youtube_urls,
            expected_domains_resolver=_fixed_domains(("youtube.com", "youtu.be")),
            preferred_tools=("get_youtube_transcript", "analyze_youtube_video"),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="article_to_paper",
        applies=lambda context: is_article_to_paper_question(context.lowered_question),
        build_profile=lambda context: _profile(
            context,
            name="article_to_paper",
            profile_family="article_to_paper",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(("universetoday.com",)),
            preferred_tools=("web_search", "fetch_url", "extract_links_from_url"),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="text_span_lookup",
        applies=lambda context: is_text_span_lookup_question(context.lowered_question),
        build_profile=lambda context: _profile(
            context,
            name="text_span_lookup",
            profile_family="text_span_lookup",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("web_search", "find_text_in_url", "fetch_url"),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="olympics_country_code",
        applies=lambda context: is_olympics_country_code_question(
            context.lowered_question
        ),
        build_profile=lambda context: _profile(
            context,
            name="wikipedia_lookup",
            profile_family="wikipedia_lookup",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(("wikipedia.org",)),
            preferred_tools=(
                "search_wikipedia",
                "fetch_wikipedia_page",
                "extract_tables_from_url",
            ),
            text_filter=_olympics_text_filter,
        ),
    ),
    QuestionClassifier(
        name="temporal_ordered_list",
        applies=lambda context: is_roster_neighbor_question(context.lowered_question),
        build_profile=lambda context: _profile(
            context,
            name="temporal_ordered_list",
            profile_family="temporal_ordered_list",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("web_search", "extract_tables_from_url", "fetch_url"),
            text_filter=_roster_text_filter,
            ordering_key=_roster_ordering_key,
            entity_name=_roster_entity_name,
            scope=_roster_scope,
        ),
    ),
    QuestionClassifier(
        name="list_item_classification",
        applies=lambda context: is_botanical_classification_question(
            context.lowered_question
        ),
        build_profile=lambda context: _profile(
            context,
            name="list_item_classification",
            profile_family="list_item_classification",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("web_search", "fetch_url", "find_text_in_url"),
            text_filter=_botanical_text_filter,
            prompt_items=_prompt_items,
            classification_labels=_botanical_labels,
        ),
    ),
    QuestionClassifier(
        name="entity_role_chain",
        applies=lambda context: is_entity_role_chain_question(context.lowered_question),
        build_profile=lambda context: _profile(
            context,
            name="entity_role_chain",
            profile_family="entity_role_chain",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(("wikipedia.org",)),
            preferred_tools=("web_search", "fetch_url", "find_text_in_url"),
            text_filter=_entity_role_chain_text_filter,
        ),
    ),
    QuestionClassifier(
        name="direct_url",
        applies=lambda context: bool(context.generic_urls),
        build_profile=lambda context: _profile(
            context,
            name="direct_url",
            profile_family="direct_url",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("fetch_url", "extract_tables_from_url", "extract_links_from_url"),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="wikipedia_keyword",
        applies=lambda context: "wikipedia" in context.lowered_question,
        build_profile=lambda context: _profile(
            context,
            name="wikipedia_lookup",
            profile_family="wikipedia_lookup",
            target_urls=_generic_urls,
            expected_domains_resolver=_fixed_domains(("wikipedia.org",)),
            preferred_tools=(
                "search_wikipedia",
                "fetch_wikipedia_page",
                "extract_tables_from_url",
            ),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="table_lookup",
        applies=lambda context: looks_like_table_question(context.lowered_question),
        build_profile=lambda context: _profile(
            context,
            name="table_lookup",
            profile_family="table_lookup",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("web_search", "extract_tables_from_url", "find_text_in_url"),
            text_filter=_inferred_text_filter,
        ),
    ),
    QuestionClassifier(
        name="entity_attribute_lookup",
        applies=lambda _context: True,
        build_profile=lambda context: _profile(
            context,
            name="entity_attribute_lookup",
            profile_family="entity_attribute_lookup",
            target_urls=_generic_urls,
            expected_domains_resolver=_default_domains(()),
            preferred_tools=("web_search", "fetch_url", "find_text_in_url"),
            text_filter=_inferred_text_filter,
        ),
    ),
)
