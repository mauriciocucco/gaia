"""Question-type detectors for profiling."""

from __future__ import annotations


def is_article_to_paper_question(lowered_question: str) -> bool:
    article_cues = ("article", "news story", "write-up")
    link_cues = ("link", "linked", "links", "source", "reference")
    target_cues = ("paper", "study", "journal", "publication")
    position_cues = ("bottom", "below", "end of the article", "at the end")
    return (
        any(cue in lowered_question for cue in article_cues)
        and any(cue in lowered_question for cue in link_cues)
        and any(cue in lowered_question for cue in target_cues)
        and any(cue in lowered_question for cue in position_cues)
    )


def is_text_span_lookup_question(lowered_question: str) -> bool:
    requested_attribute_cues = (
        "surname of",
        "last name of",
        "first name of",
        "city name of",
    )
    reference_cues = ("mentioned in", "found in", "named in", "appears in")
    material_cues = (
        "exercise",
        "chapter",
        "materials",
        "compiled",
        "licensed by",
        "textbook",
        "page",
    )
    return any(token in lowered_question for token in requested_attribute_cues) and (
        any(cue in lowered_question for cue in reference_cues)
        or any(cue in lowered_question for cue in material_cues)
    )


def is_entity_role_chain_question(lowered_question: str) -> bool:
    actor_cues = ("actor who played", "actress who played", "performer who played")
    second_hop_cues = ("play in", "played in", "role in", "character in")
    return any(cue in lowered_question for cue in actor_cues) and any(
        cue in lowered_question for cue in second_hop_cues
    )


def is_roster_neighbor_question(lowered_question: str) -> bool:
    ordering_cues = ("before and after", "adjacent to", "next to")
    number_cues = ("number", "jersey", "shirt number")
    return any(cue in lowered_question for cue in ordering_cues) and any(
        cue in lowered_question for cue in number_cues
    )


def is_botanical_classification_question(lowered_question: str) -> bool:
    if "vegetable" not in lowered_question and "fruit" not in lowered_question:
        return False
    botanical_cues = ("professor of botany", "botanical", "botany", "stickler")
    listing_cues = (
        "here's the list i have so far",
        "comma separated list",
        "fruits and vegetables",
    )
    return any(cue in lowered_question for cue in botanical_cues) and any(
        cue in lowered_question for cue in listing_cues
    )


def is_olympics_country_code_question(lowered_question: str) -> bool:
    return (
        "olympics" in lowered_question
        and "athletes" in lowered_question
        and "ioc country code" in lowered_question
    )


def looks_like_table_question(lowered_question: str) -> bool:
    table_cues = (
        "least number of athletes",
        "number of athletes",
        "most walks",
        "at bats",
        "roster",
        "pitchers",
        "table",
    )
    return any(cue in lowered_question for cue in table_cues)
