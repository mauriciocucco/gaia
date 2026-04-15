"""GAIA skill for grounded botanical set classification."""

from __future__ import annotations

import re
from urllib.parse import unquote

from langchain_core.messages import ToolMessage

from ...botanical_classification import build_botanical_canonical_state
from ...core.recoveries.utils import (
    RecoveryAttemptBudget,
    fetch_candidate_urls,
    invoke_recovery_tool,
    ranked_candidates_from_result_text,
    recovery_trace_state,
    try_search_recovery,
)
from ...graph.routing import (
    extract_prompt_list_items,
    normalize_botanical_text,
    question_profile_from_state,
)
from ...source_pipeline import EvidenceRecord, SourceCandidate, evidence_records_from_tool_output
from ..set_classification import build_set_classification_result


class BotanicalGaiaSkill:
    name = "botanical_gaia"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name

    def applies(self, state, profile) -> bool:
        labels = profile.classification_labels or {}
        return (
            profile.name == "list_item_classification"
            and labels.get("include") == "vegetable"
            and labels.get("exclude") == "fruit"
        )

    def run(self, state):
        profile = question_profile_from_state(state)
        if not self.applies(state, profile):
            return None

        decision_trace = list(state.get("decision_trace") or [])
        decision_trace.append("skill:botanical_gaia:profile_match")

        has_wikipedia_tools = all(
            tool_name in self._tools_by_name
            for tool_name in ("search_wikipedia", "fetch_wikipedia_page")
        )
        has_web_tools = all(
            tool_name in self._tools_by_name for tool_name in ("web_search", "fetch_url")
        )
        if not has_wikipedia_tools and not has_web_tools:
            decision_trace.append("skill:botanical_gaia:missing_tools")
            return {"decision_trace": decision_trace}

        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        context.decision_trace.append("skill:botanical_gaia:profile_match")

        items = list(profile.prompt_items or extract_prompt_list_items(state["question"]))
        context.decision_trace.append(f"skill:botanical_gaia:items_extracted={len(items)}")
        if not items:
            context.decision_trace.append("skill:botanical_gaia:no_items")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        existing_records = _botanical_existing_records(state)
        has_existing_leads = (
            any(isinstance(message, ToolMessage) for message in state.get("messages", []))
            or bool(state.get("ranked_candidates"))
            or bool(existing_records)
        )
        if not has_existing_leads and len(items) > 8:
            resolution = build_botanical_canonical_state(items, existing_records)
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        all_records = existing_records
        resolution = build_botanical_canonical_state(items, all_records)
        for item in resolution.unresolved_items:
            item_budget = RecoveryAttemptBudget(remaining_searches=3, remaining_fetches=3)
            if has_wikipedia_tools and _should_try_wikipedia_first(item):
                wiki_search = try_search_recovery(
                    context=context,
                    query=_wikipedia_query_for_item(item),
                    max_results=5,
                    budget=item_budget,
                    tool_name="search_wikipedia",
                )
                if wiki_search:
                    wiki_candidates = ranked_candidates_from_result_text(
                        context=context,
                        result_text=wiki_search,
                        origin_tool="search_wikipedia",
                        predicate=lambda candidate: _wikipedia_candidate_matches_item(
                            item, candidate
                        ),
                    )
                    best_wiki_candidate = (
                        max(
                            wiki_candidates,
                            key=lambda candidate: _wikipedia_candidate_selection_key(
                                item, candidate
                            ),
                        )
                        if wiki_candidates
                        else None
                    )
                    wiki_title = _candidate_title(best_wiki_candidate)
                    if best_wiki_candidate and wiki_title and item_budget.consume_fetch():
                        fetched = invoke_recovery_tool(
                            context=context,
                            tool_name="fetch_wikipedia_page",
                            tool_args={"title": wiki_title},
                        )
                        all_records.extend(
                            evidence_records_from_tool_output("fetch_wikipedia_page", fetched)
                        )
                        resolution = build_botanical_canonical_state(items, all_records)
                        if item not in resolution.unresolved_items:
                            continue

            if not has_web_tools:
                continue
            for query in (
                f"{item} botanical fruit or vegetable",
                f"{item} botany fruit vegetable",
            ):
                search_text = try_search_recovery(
                    context=context,
                    query=query,
                    max_results=5,
                    budget=item_budget,
                )
                if not search_text:
                    continue
                candidate_urls = [
                    candidate.url
                    for candidate in ranked_candidates_from_result_text(
                        context=context,
                        result_text=search_text,
                        origin_tool="web_search",
                        predicate=lambda candidate: _search_candidate_matches_item(
                            item, candidate
                        ),
                    )
                ]
                for candidate_url in fetch_candidate_urls(
                    context=context,
                    candidate_urls=candidate_urls,
                    max_urls=2,
                ):
                    if not item_budget.consume_fetch():
                        break
                    fetched = invoke_recovery_tool(
                        context=context,
                        tool_name="fetch_url",
                        tool_args={"url": candidate_url},
                    )
                    all_records.extend(evidence_records_from_tool_output("fetch_url", fetched))
                    resolution = build_botanical_canonical_state(items, all_records)
                    if item not in resolution.unresolved_items:
                        break
                if item not in resolution.unresolved_items:
                    break

        if not resolution.is_closed:
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        context.decision_trace.append("skill:botanical_gaia:resolved")
        return build_set_classification_result(
            skill_name=self.name,
            included_items=resolution.included_items,
            records=resolution.used_records,
            reducer_used="botanical_classification",
            tool_trace=context.tool_trace,
            decision_trace=context.decision_trace,
        )


def _botanical_existing_records(state) -> list[EvidenceRecord]:
    supporting_records: list[EvidenceRecord] = []
    seen_urls: set[str] = set()
    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in {
            "fetch_url",
            "find_text_in_url",
            "fetch_wikipedia_page",
        }:
            continue
        for record in evidence_records_from_tool_output(tool_name, str(message.content or "")):
            if record.source_url and record.source_url in seen_urls:
                continue
            supporting_records.append(record)
            if record.source_url:
                seen_urls.add(record.source_url)
    return supporting_records


def _should_try_wikipedia_first(item: str) -> bool:
    normalized = _normalized_botanical_query_tokens(item)
    if not normalized:
        return False
    if len(normalized.split()) > 3:
        return False
    return bool(re.fullmatch(r"[a-z ]+", normalized))


def _wikipedia_query_for_item(item: str) -> str:
    return _normalized_botanical_query_tokens(item)


def _normalized_botanical_query_tokens(item: str) -> str:
    ignored = {"fresh", "whole", "raw", "ripe", "dried"}
    tokens = [
        token for token in normalize_botanical_text(item).split() if token and token not in ignored
    ]
    return " ".join(tokens)


def _wikipedia_candidate_matches_item(item: str, candidate: SourceCandidate) -> bool:
    title = _candidate_title(candidate)
    return bool(title and _item_text_matches(item, title))


def _search_candidate_matches_item(item: str, candidate: SourceCandidate) -> bool:
    haystack = "\n".join((candidate.title, candidate.snippet, unquote(candidate.url))).strip()
    return _item_text_matches(item, haystack)


def _candidate_title(candidate: SourceCandidate | None) -> str | None:
    if candidate is None:
        return None
    title = str(candidate.title or "").strip()
    return title or None


def _wikipedia_candidate_selection_key(item: str, candidate: SourceCandidate) -> tuple[int, int, int]:
    title = _candidate_title(candidate)
    if not title:
        return (-1, -999, -999)
    normalized_title = normalize_botanical_text(title)
    title_tokens = [token for token in normalized_title.split() if token != "disambiguation"]
    if not title_tokens:
        return (-1, -999, -999)
    token_groups = _item_token_groups(item)
    if not token_groups or not all(
        any(token in variants for token in title_tokens) for variants in token_groups
    ):
        return (-1, -999, -999)
    canonical_query = _normalized_botanical_query_tokens(item)
    singular_query = " ".join(sorted(variants, key=len)[0] for variants in token_groups)
    exactness = 2 if normalized_title in {canonical_query, singular_query} else 1
    matched_tokens = {
        token
        for token in title_tokens
        if any(token in variants for variants in token_groups)
    }
    extra_penalty = -len([token for token in title_tokens if token not in matched_tokens])
    return (exactness, extra_penalty, -len(title_tokens))


def _item_text_matches(item: str, text: str) -> bool:
    normalized = normalize_botanical_text(text)
    token_groups = _item_token_groups(item)
    return bool(token_groups) and all(
        any(variant in normalized for variant in variants)
        for variants in token_groups
    )


def _item_token_groups(item: str) -> list[set[str]]:
    ignored = {"fresh", "whole", "raw", "ripe", "dried"}
    groups: list[set[str]] = []
    for token in normalize_botanical_text(item).split():
        if token in ignored:
            continue
        variants = {token}
        if token.endswith("ies") and len(token) > 4:
            variants.add(token[:-3] + "y")
        if token.endswith("oes") and len(token) > 4:
            variants.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            variants.add(token[:-1])
        groups.append(variants)
    return groups
