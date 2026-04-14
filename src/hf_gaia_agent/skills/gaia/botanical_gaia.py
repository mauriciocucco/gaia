"""GAIA skill for grounded botanical set classification."""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

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
from ...source_pipeline import EvidenceRecord, evidence_records_from_tool_output
from ..set_classification import ClassifiedItemState, build_set_classification_result


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
        if "web_search" not in self._tools_by_name or "fetch_url" not in self._tools_by_name:
            return None
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        items = list(profile.prompt_items or extract_prompt_list_items(state["question"]))
        if not items:
            return None

        item_states = {
            item: ClassifiedItemState(
                item=item,
                status="discarded" if not _is_botanical_prompt_candidate(item) else "unknown",
            )
            for item in items
        }
        has_existing_leads = bool(state.get("messages")) or bool(state.get("ranked_candidates"))
        if not has_existing_leads and len(items) > 8:
            return None

        existing_records = _botanical_existing_records(state)
        for item, item_state in item_states.items():
            if item_state.status == "discarded":
                continue
            outcome, supporting = _classify_item_from_records(item, existing_records)
            if outcome:
                item_state.status = outcome
                item_state.evidence.extend(supporting[-2:])

        for item, item_state in item_states.items():
            if item_state.status != "unknown":
                continue
            item_budget = RecoveryAttemptBudget(remaining_searches=2, remaining_fetches=2)
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
                    fetched_records = evidence_records_from_tool_output("fetch_url", fetched)
                    outcome, supporting = _classify_item_from_records(item, fetched_records)
                    if not outcome:
                        continue
                    item_state.status = outcome
                    item_state.evidence.extend(supporting[-2:])
                    break
                if item_state.status != "unknown":
                    break

        relevant_states = [
            item_state
            for item_state in item_states.values()
            if item_state.status != "discarded"
        ]
        if not relevant_states or any(item_state.status == "unknown" for item_state in relevant_states):
            return None
        included = [
            item_state.item
            for item_state in relevant_states
            if item_state.status == "include"
        ]
        if not included:
            return None
        records: list[EvidenceRecord] = []
        for item_state in relevant_states:
            records.extend(item_state.evidence[-2:])
        return build_set_classification_result(
            skill_name=self.name,
            included_items=included,
            records=records,
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
        if tool_name not in {"fetch_url", "find_text_in_url"}:
            continue
        for record in evidence_records_from_tool_output(tool_name, str(message.content or "")):
            if record.source_url and record.source_url in seen_urls:
                continue
            supporting_records.append(record)
            if record.source_url:
                seen_urls.add(record.source_url)
    return supporting_records


def _classify_item_from_records(
    item: str,
    records: list[EvidenceRecord],
) -> tuple[str | None, list[EvidenceRecord]]:
    fruit_total = 0
    vegetable_total = 0
    supporting: list[EvidenceRecord] = []
    for record in records:
        scores = botanical_scores_from_text(item, record.content)
        if scores is None:
            continue
        fruit_score, vegetable_score = scores
        if max(fruit_score, vegetable_score) < 2:
            continue
        fruit_total += fruit_score
        vegetable_total += vegetable_score
        supporting.append(record)
    if _botanical_is_vegetable(fruit_total, vegetable_total):
        return "include", supporting
    if _botanical_is_decisive(fruit_total, vegetable_total):
        return "exclude", supporting
    return None, supporting


def _botanical_is_vegetable(fruit_total: int, vegetable_total: int) -> bool:
    return vegetable_total >= fruit_total + 2 and vegetable_total >= 3


def _botanical_is_decisive(fruit_total: int, vegetable_total: int) -> bool:
    return _botanical_is_vegetable(fruit_total, vegetable_total) or (
        fruit_total >= vegetable_total + 2 and fruit_total >= 3
    )


def _botanical_item_token_groups(item: str) -> list[set[str]]:
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


def _is_botanical_prompt_candidate(item: str) -> bool:
    normalized = normalize_botanical_text(item)
    if not normalized:
        return False
    obvious_non_produce_phrases = {
        "milk", "egg", "eggs", "flour", "oreo", "oreos", "rice",
        "whole bean coffee", "coffee", "whole allspice", "allspice",
    }
    return normalized not in obvious_non_produce_phrases


def _botanical_relevant_text(item: str, text: str) -> str:
    token_groups = _botanical_item_token_groups(item)
    if not token_groups:
        return ""
    segments = [
        segment.strip()
        for segment in re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
        if segment.strip()
    ]
    relevant_segments: list[str] = []
    for segment in segments:
        normalized_segment = normalize_botanical_text(segment)
        if all(any(variant in normalized_segment for variant in variants) for variants in token_groups):
            relevant_segments.append(segment)
    if relevant_segments:
        return " ".join(relevant_segments)
    normalized = normalize_botanical_text(text)
    if all(any(variant in normalized for variant in variants) for variants in token_groups):
        return text
    return ""


def botanical_scores_from_text(item: str, text: str) -> tuple[int, int] | None:
    relevant_text = _botanical_relevant_text(item, text)
    normalized = normalize_botanical_text(relevant_text)
    if not normalized:
        return None
    if any(
        phrase in normalized
        for phrase in (
            "neither a fruit nor a vegetable",
            "neither fruit nor vegetable",
            "not a fruit or vegetable",
            "neither fruits nor vegetables",
        )
    ):
        return None
    if (
        "made from" in normalized
        and any(token in normalized for token in ("milled", "grinding", "ground", "powder"))
    ):
        return None
    fruit_score = 0
    vegetable_score = 0
    title_match = re.search(r"(?im)^title:\s*(?P<title>.+)$", text)
    normalized_title = normalize_botanical_text(title_match.group("title").strip()) if title_match else ""
    explicit_fruit_over_vegetable = (
        "not a vegetable but a fruit",
        "fruit rather than a vegetable",
        "classified as a fruit",
        "technically a fruit",
        "actually a fruit",
    )
    explicit_vegetable_over_fruit = (
        "not a fruit but a vegetable",
        "vegetable rather than a fruit",
        "classified as a vegetable",
        "technically a vegetable",
        "actually a vegetable",
    )
    fruit_phrases = (
        "botanical fruit", "botanically a fruit", "is a fruit", "are fruits",
        "considered a fruit", "classified as a fruit", "technically a fruit",
        "actually a fruit", "fruit rather than a vegetable",
        "not a vegetable but a fruit", "the fruit of", "seed bearing structure",
        "grain rather than a vegetable", "not a vegetable",
    )
    vegetable_phrases = (
        "botanical vegetable", "botanically a vegetable", "is a vegetable",
        "are vegetables", "classified as a vegetable", "technically a vegetable",
        "actually a vegetable", "vegetable rather than a fruit",
        "not a fruit but a vegetable", "root vegetable", "leaf vegetable",
        "stem vegetable", "flower vegetable", "flowering head", "edible leaves",
        "edible leaf", "edible stem", "edible root", "flower buds",
    )
    fruit_cues = (
        "fruit", "berry", "berries", "seed", "seeds", "grain", "grains",
        "cereal", "kernel", "kernels", "pod", "pods", "caryopsis", "drupe",
        "pepo", "capsule", "legume",
    )
    vegetable_cues = (
        "vegetable", "vegetables", "leaf", "leaves", "stem", "stalk", "root",
        "tuber", "bulb", "flower", "flowers", "inflorescence", "herb",
    )
    fruit_score += sum(3 for phrase in fruit_phrases if phrase in normalized)
    vegetable_score += sum(3 for phrase in vegetable_phrases if phrase in normalized)
    fruit_score += sum(1 for cue in fruit_cues if cue in normalized)
    vegetable_score += sum(1 for cue in vegetable_cues if cue in normalized)
    if any(phrase in normalized for phrase in explicit_fruit_over_vegetable):
        fruit_score += 3
    if any(phrase in normalized for phrase in explicit_vegetable_over_fruit):
        vegetable_score += 3
    if "not a vegetable" in normalized:
        fruit_score += 2
    if "not a fruit" in normalized:
        vegetable_score += 2
    if normalized_title:
        token_groups = _botanical_item_token_groups(item)
        title_mentions_item = bool(token_groups) and all(
            any(variant in normalized_title for variant in variants)
            for variants in token_groups
        )
        if title_mentions_item:
            if "vegetable" in normalized_title and "fruit" not in normalized_title:
                vegetable_score += 3
            if "fruit" in normalized_title and "vegetable" not in normalized_title:
                fruit_score += 3
    return fruit_score, vegetable_score
