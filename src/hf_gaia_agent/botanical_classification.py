"""Shared helpers for botanical vegetable-vs-fruit classification."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .graph.routing import normalize_botanical_text
from .source_pipeline import EvidenceRecord


@dataclass(frozen=True)
class BotanicalCanonicalState:
    items: list[str]
    relevant_items: list[str]
    included_items: list[str]
    excluded_items: list[str]
    unresolved_items: list[str]
    used_records: list[EvidenceRecord]
    canonical_answer: str
    is_closed: bool
    has_grounded_records: bool


def sort_canonical_items(items: list[str]) -> list[str]:
    return sorted(dict.fromkeys(items), key=lambda value: value.lower())


def canonical_botanical_answer(items: list[str]) -> str:
    return ", ".join(sort_canonical_items(items))


def is_botanical_prompt_candidate(item: str) -> bool:
    normalized = normalize_botanical_text(item)
    if not normalized:
        return False
    obvious_non_produce_phrases = {
        "milk",
        "egg",
        "eggs",
        "flour",
        "oreo",
        "oreos",
        "rice",
        "whole bean coffee",
        "coffee",
        "whole allspice",
        "allspice",
    }
    return normalized not in obvious_non_produce_phrases


def botanical_item_token_groups(item: str) -> list[set[str]]:
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


def botanical_relevant_text(item: str, text: str) -> str:
    token_groups = botanical_item_token_groups(item)
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
    relevant_text = botanical_relevant_text(item, text)
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
        "botanical fruit",
        "botanically a fruit",
        "is a fruit",
        "are fruits",
        "considered a fruit",
        "classified as a fruit",
        "technically a fruit",
        "actually a fruit",
        "fruit rather than a vegetable",
        "not a vegetable but a fruit",
        "the fruit of",
        "seed bearing structure",
        "grain rather than a vegetable",
        "not a vegetable",
    )
    vegetable_phrases = (
        "botanical vegetable",
        "botanically a vegetable",
        "is a vegetable",
        "are vegetables",
        "classified as a vegetable",
        "technically a vegetable",
        "actually a vegetable",
        "vegetable rather than a fruit",
        "not a fruit but a vegetable",
        "root vegetable",
        "leaf vegetable",
        "stem vegetable",
        "flower vegetable",
        "flowering head",
        "edible leaves",
        "edible leaf",
        "edible stem",
        "edible root",
        "flower buds",
    )
    fruit_cues = (
        "fruit",
        "berry",
        "berries",
        "seed",
        "seeds",
        "grain",
        "grains",
        "cereal",
        "kernel",
        "kernels",
        "pod",
        "pods",
        "caryopsis",
        "drupe",
        "pepo",
        "capsule",
        "legume",
    )
    vegetable_cues = (
        "vegetable",
        "vegetables",
        "leaf",
        "leaves",
        "stem",
        "stalk",
        "root",
        "tuber",
        "bulb",
        "flower",
        "flowers",
        "inflorescence",
        "herb",
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
        if "vegetable" in normalized_title:
            vegetable_score += 2
        if "fruit" in normalized_title:
            fruit_score += 2
    return fruit_score, vegetable_score


def botanical_is_vegetable(fruit_total: int, vegetable_total: int) -> bool:
    return vegetable_total >= fruit_total + 2 and vegetable_total >= 3


def botanical_is_decisive(fruit_total: int, vegetable_total: int) -> bool:
    return botanical_is_vegetable(fruit_total, vegetable_total) or (
        fruit_total >= vegetable_total + 2 and fruit_total >= 3
    )


def classify_botanical_item_from_records(
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
    if botanical_is_vegetable(fruit_total, vegetable_total):
        return "include", supporting
    if botanical_is_decisive(fruit_total, vegetable_total):
        return "exclude", supporting
    return None, supporting


def build_botanical_canonical_state(
    items: list[str],
    records: list[EvidenceRecord],
) -> BotanicalCanonicalState:
    normalized_items = [item for item in items if item]
    relevant_items = [item for item in normalized_items if is_botanical_prompt_candidate(item)]
    included_items: list[str] = []
    excluded_items: list[str] = []
    unresolved_items: list[str] = []
    used_records: list[EvidenceRecord] = []
    for item in relevant_items:
        outcome, supporting = classify_botanical_item_from_records(item, records)
        used_records.extend(supporting[-2:])
        if outcome == "include":
            included_items.append(item)
            continue
        if outcome == "exclude":
            excluded_items.append(item)
            continue
        unresolved_items.append(item)
    included_items = sort_canonical_items(included_items)
    excluded_items = sort_canonical_items(excluded_items)
    unresolved_items = sort_canonical_items(unresolved_items)
    return BotanicalCanonicalState(
        items=normalized_items,
        relevant_items=sort_canonical_items(relevant_items),
        included_items=included_items,
        excluded_items=excluded_items,
        unresolved_items=unresolved_items,
        used_records=used_records[-6:],
        canonical_answer=canonical_botanical_answer(included_items),
        is_closed=bool(relevant_items) and not unresolved_items,
        has_grounded_records=any(record.kind in {"text", "table", "transcript"} for record in records),
    )
