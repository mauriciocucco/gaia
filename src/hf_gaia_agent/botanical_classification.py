"""Shared helpers for botanical vegetable-vs-fruit classification."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .botanical_aliases import botanical_aliases_for_item, botanical_token_groups
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
    return botanical_token_groups(normalize_botanical_text(item))


def botanical_alias_token_groups(item: str) -> list[list[set[str]]]:
    return [botanical_token_groups(alias) for alias in botanical_aliases_for_item(item)]


def botanical_relevant_text(item: str, text: str) -> str:
    common_groups = botanical_item_token_groups(item)
    alias_groups = botanical_alias_token_groups(item)
    all_group_sets = [groups for groups in [common_groups, *alias_groups] if groups]
    if not all_group_sets:
        return ""
    segments = [
        segment.strip()
        for segment in re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
        if segment.strip()
    ]
    direct_match_indices: list[int] = []
    for index, segment in enumerate(segments):
        normalized_segment = normalize_botanical_text(segment)
        if _segment_matches_any_token_groups(normalized_segment, all_group_sets):
            direct_match_indices.append(index)
    relevant_segments: list[str] = []
    if direct_match_indices:
        selected_indices: list[int] = []
        for index in direct_match_indices:
            selected_indices.append(index)
            for neighbor in (index - 1, index + 1):
                if (
                    0 <= neighbor < len(segments)
                    and (
                        _is_botanical_adjacent_context(segments[neighbor])
                        or _segment_matches_any_token_groups(
                            normalize_botanical_text(segments[neighbor]),
                            all_group_sets,
                        )
                    )
                ):
                    selected_indices.append(neighbor)
        seen_indices: set[int] = set()
        for index in selected_indices:
            if index in seen_indices:
                continue
            seen_indices.add(index)
            relevant_segments.append(segments[index])
    if relevant_segments:
        return " ".join(relevant_segments)
    normalized = normalize_botanical_text(text)
    if _record_metadata_matches_any_token_groups(text, all_group_sets) and any(
        all(any(variant in normalized for variant in variants) for variants in token_groups)
        for token_groups in all_group_sets
    ):
        return text
    return ""


def _segment_matches_any_token_groups(
    normalized_segment: str,
    all_group_sets: list[list[set[str]]],
) -> bool:
    return any(
        all(
            any(variant in normalized_segment for variant in variants)
            for variants in token_groups
        )
        for token_groups in all_group_sets
    )


def _record_metadata_matches_any_token_groups(
    text: str,
    all_group_sets: list[list[set[str]]],
) -> bool:
    metadata_haystacks: list[str] = []
    title_match = re.search(r"(?im)^title:\s*(?P<title>.+)$", text)
    if title_match:
        metadata_haystacks.append(title_match.group("title").strip())
    url_match = re.search(r"(?im)^(?:url|url source):\s*(?P<url>\S+)\s*$", text)
    if url_match:
        metadata_haystacks.append(url_match.group("url").strip())
    return any(
        _segment_matches_any_token_groups(
            normalize_botanical_text(metadata_value),
            all_group_sets,
        )
        for metadata_value in metadata_haystacks
        if metadata_value.strip()
    )


def _is_botanical_adjacent_context(segment: str) -> bool:
    normalized = normalize_botanical_text(segment)
    if not normalized:
        return False
    if not normalized.startswith(("its ", "their ", "this ", "these ", "those ", "they ", "it ")):
        return False
    return any(
        cue in normalized
        for cue in (
            "fruit",
            "vegetable",
            "root",
            "roots",
            "leaf",
            "leaves",
            "stem",
            "stalk",
            "flower",
            "flowers",
            "tuber",
            "tubers",
            "seed",
            "seeds",
            "berry",
            "berries",
            "herb",
        )
    )




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
    if _is_processed_food_only_match(item=item, normalized_text=normalized):
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
        "fruit of the plant",
        "seed bearing structure",
        "seed bearing",
        "seed-bearing",
        "grain rather than a vegetable",
        "not a vegetable",
        "seed pod",
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
    botanical_seed_structure_phrases = (
        "edible seeds",
        "contained in underground pods",
        "fruit develops underground",
        "fruits develop underground",
        "develop underground",
        "grain legume",
        "legume crop",
        "geocarpy",
        "technically called legumes",
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
    fruit_score += sum(2 for phrase in botanical_seed_structure_phrases if phrase in normalized)
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
    normalized_item = normalize_botanical_text(item)
    if "zucchini" in normalized_item and "pepo" in normalized:
        fruit_score += 4
    if "bell pepper" in normalized_item and "fruit" in normalized and (
        "capsicum" in normalized or "pepper" in normalized
    ):
        fruit_score += 3
    if "peanut" in normalized_item and any(
        cue in normalized for cue in ("legume", "pod", "seed", "arachis")
    ):
        fruit_score += 4
    if "peanut" in normalized_item:
        peanut_seed_structure_matches = sum(
            cue in normalized
            for cue in (
                "legume crop",
                "edible seeds",
                "underground pods",
                "grain legume",
                "geocarpy",
                "technically called legumes",
                "arachis",
            )
        )
        if peanut_seed_structure_matches >= 2:
            fruit_score += 5
    return fruit_score, vegetable_score


def _is_processed_food_only_match(*, item: str, normalized_text: str) -> bool:
    if not re.search(
        r"made from[\w\s,-]{0,60}\b(?:milled|grinding|ground|powder)\b",
        normalized_text,
    ):
        return False
    strong_botanical_cues = (
        "botanical fruit",
        "fruit of the plant",
        "seed bearing",
        "seed-bearing",
        "seed pod",
        "legume crop",
        "edible seeds",
        "contained in underground pods",
        "develop underground",
        "fruits develop underground",
        "fruit develops underground",
        "grain legume",
        "geocarpy",
        "technically called legumes",
    )
    if any(cue in normalized_text for cue in strong_botanical_cues):
        return False
    normalized_item = normalize_botanical_text(item)
    if "peanut" in normalized_item and any(
        cue in normalized_text for cue in ("legume", "pod", "seed", "arachis")
    ):
        return False
    return True


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
