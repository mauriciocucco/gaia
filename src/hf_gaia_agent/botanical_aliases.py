"""Shared botanical alias helpers for common-name vs scientific-name matching."""

from __future__ import annotations

import re


BOTANICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "bell pepper": ("capsicum annuum", "capsicum"),
    "zucchini": ("cucurbita pepo", "courgette"),
    "peanuts": ("peanut", "arachis hypogaea"),
}

_IGNORED_DESCRIPTORS = {"fresh", "whole", "raw", "ripe", "dried"}


def normalize_botanical_alias_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def botanical_aliases_for_item(item: str) -> tuple[str, ...]:
    normalized_item = normalize_botanical_alias_text(item)
    tokens = [
        token
        for token in normalized_item.split()
        if token and token not in _IGNORED_DESCRIPTORS
    ]
    if not tokens:
        return ()
    canonical_item = " ".join(tokens)
    return BOTANICAL_ALIASES.get(canonical_item, ())


def botanical_token_groups(text: str) -> list[set[str]]:
    groups: list[set[str]] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _IGNORED_DESCRIPTORS:
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
