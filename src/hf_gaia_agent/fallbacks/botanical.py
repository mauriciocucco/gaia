"""Botanical classification fallback resolver."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from ..normalize import normalize_submitted_answer
from ..source_pipeline import (
    EvidenceRecord,
    evidence_records_from_tool_output,
    serialize_evidence,
)
from ..graph.state import AgentState
from ..graph.routing import (
    extract_prompt_list_items,
    normalize_botanical_text,
    question_profile_from_state,
)
from .base import FallbackResolver
from .utils import (
    FallbackAttemptBudget,
    fallback_trace_state,
    fetch_candidate_urls,
    invoke_fallback_tool,
    ranked_candidates_from_result_text,
    try_search_fallback,
    with_fallback_traces,
)


class BotanicalFallback:
    name = "botanical"

    def __init__(self, tools_by_name: dict[str, Any]):
        self._tools_by_name = tools_by_name

    def applies(self, state: AgentState, profile: Any) -> bool:
        return profile.name == "botanical_classification"

    def run(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        if profile.name != "botanical_classification":
            return None
        if "web_search" not in self._tools_by_name or "fetch_url" not in self._tools_by_name:
            return None
        context = fallback_trace_state(tools_by_name=self._tools_by_name, state=state)

        items = _botanical_candidate_items(state)
        if not items:
            return None
        has_existing_leads = bool(state.get("messages")) or bool(
            state.get("ranked_candidates")
        )
        if not has_existing_leads and len(items) > 6:
            return None

        vegetables: list[str] = []
        supporting_records: list[EvidenceRecord] = []
        for item in items:
            fruit_total = 0
            vegetable_total = 0
            item_supporting_records: list[EvidenceRecord] = []
            for record in _botanical_existing_records(state):
                scores = botanical_scores_from_text(item, record.content)
                if scores is None:
                    continue
                fruit_score, vegetable_score = scores
                if max(fruit_score, vegetable_score) < 2:
                    continue
                fruit_total += fruit_score
                vegetable_total += vegetable_score
                item_supporting_records.append(record)
            if _botanical_is_vegetable(fruit_total, vegetable_total):
                vegetables.append(item)
                supporting_records.extend(item_supporting_records[-2:])
                continue
            if _botanical_is_decisive(fruit_total, vegetable_total):
                supporting_records.extend(item_supporting_records[-2:])
                continue

            item_budget = FallbackAttemptBudget(remaining_searches=2, remaining_fetches=2)
            for query in (
                f"{item} botanical fruit or vegetable",
                f"{item} botany fruit vegetable",
            ):
                search_text = try_search_fallback(
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
                    fetched = invoke_fallback_tool(
                        context=context,
                        tool_name="fetch_url",
                        tool_args={"url": candidate_url},
                    )
                    fetched_records = evidence_records_from_tool_output("fetch_url", fetched)
                    scores = botanical_scores_from_text(item, fetched)
                    if scores is None:
                        continue
                    fruit_score, vegetable_score = scores
                    if max(fruit_score, vegetable_score) < 2:
                        continue
                    fruit_total += fruit_score
                    vegetable_total += vegetable_score
                    item_supporting_records.extend(fetched_records)
                if _botanical_is_decisive(fruit_total, vegetable_total):
                    break
            if _botanical_is_vegetable(fruit_total, vegetable_total):
                vegetables.append(item)
            if item_supporting_records:
                supporting_records.extend(item_supporting_records[-2:])

        if not vegetables:
            return None
        ordered = sorted(dict.fromkeys(vegetables), key=lambda value: value.lower())
        return with_fallback_traces(
            {
                "final_answer": ", ".join(ordered),
                "error": None,
                "reducer_used": "botanical_classification",
                "evidence_used": serialize_evidence(supporting_records[-6:]),
                "fallback_reason": None,
            },
            context=context,
        )


def _botanical_existing_records(state: AgentState) -> list[EvidenceRecord]:
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


def _botanical_is_vegetable(fruit_total: int, vegetable_total: int) -> bool:
    return vegetable_total >= fruit_total + 2 and vegetable_total >= 3


def _botanical_is_decisive(fruit_total: int, vegetable_total: int) -> bool:
    return _botanical_is_vegetable(fruit_total, vegetable_total) or (
        fruit_total >= vegetable_total + 2 and fruit_total >= 3
    )


def _botanical_candidate_items(state: AgentState) -> list[str]:
    prompt_items = extract_prompt_list_items(state["question"])
    if not prompt_items:
        return []

    haystacks: list[str] = []
    last_ai = _last_ai_message(state["messages"])
    if last_ai is not None:
        haystacks.append(normalize_submitted_answer(str(last_ai.content or "")).lower())
    for message in state["messages"]:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", []) or []:
            if tool_call.get("name") != "web_search":
                continue
            query = str(tool_call.get("args", {}).get("query", "") or "")
            if query:
                haystacks.append(normalize_submitted_answer(query).lower())

    candidates: list[str] = []
    for item in prompt_items:
        token_groups = _botanical_item_token_groups(item)
        if not token_groups:
            continue
        referenced_in_existing_reasoning = any(
            all(
                any(variant in haystack for variant in variants)
                for variants in token_groups
            )
            for haystack in haystacks
        )
        if referenced_in_existing_reasoning or _is_botanical_prompt_candidate(item):
            candidates.append(item)
    return candidates


def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


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
        if all(
            any(variant in normalized_segment for variant in variants)
            for variants in token_groups
        ):
            relevant_segments.append(segment)

    if relevant_segments:
        return " ".join(relevant_segments)

    normalized = normalize_botanical_text(text)
    if all(
        any(variant in normalized for variant in variants)
        for variants in token_groups
    ):
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
    ambiguous_dual_classification = (
        "both a fruit and a vegetable",
        "both fruit and vegetable",
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
    if any(phrase in normalized for phrase in ambiguous_dual_classification):
        if any(phrase in normalized for phrase in explicit_fruit_over_vegetable + ("botanical fruit", "botanically a fruit")):
            fruit_score += 1
            vegetable_score = max(0, vegetable_score - 3)
        if any(phrase in normalized for phrase in explicit_vegetable_over_fruit + ("botanical vegetable", "botanically a vegetable")):
            vegetable_score += 1
            fruit_score = max(0, fruit_score - 3)

    return fruit_score, vegetable_score


def classify_botanical_item_from_text(item: str, text: str) -> str | None:
    scores = botanical_scores_from_text(item, text)
    if scores is None:
        return None
    fruit_score, vegetable_score = scores
    if fruit_score >= vegetable_score + 2 and fruit_score >= 3:
        return "fruit"
    if vegetable_score >= fruit_score + 2 and vegetable_score >= 3:
        return "vegetable"
    return None
