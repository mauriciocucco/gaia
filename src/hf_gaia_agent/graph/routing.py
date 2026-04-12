"""Routing logic, question analysis helpers, and research hints."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..source_pipeline import QuestionProfile, profile_question
from ..source_pipeline._utils import (
    extract_urls,
    is_metric_row_lookup_question,
    is_youtube_url,
)
from .state import AgentState, COMMON_ENGLISH_HINTS, SET_DEFINITION_RE, URL_RE


def question_profile_from_state(state: AgentState) -> QuestionProfile:
    raw = state.get("question_profile")
    if isinstance(raw, QuestionProfile):
        return raw
    if isinstance(raw, dict):
        return QuestionProfile(
            name=str(raw.get("name", "")),
            target_urls=tuple(raw.get("target_urls") or ()),
            expected_domains=tuple(raw.get("expected_domains") or ()),
            preferred_tools=tuple(raw.get("preferred_tools") or ()),
            expected_date=raw.get("expected_date"),
            expected_author=raw.get("expected_author"),
            subject_name=raw.get("subject_name"),
            text_filter=raw.get("text_filter"),
        )
    return profile_question(
        state["question"],
        file_name=state.get("file_name"),
        local_file_path=state.get("local_file_path"),
    )


def question_is_self_contained(question: str) -> bool:
    lowered = question.lower()
    if "|---|" in question or "given this table" in lowered:
        return True
    if "here's the list i have so far:" in lowered:
        return True
    if "full table" in lowered or "attached python code" in lowered:
        return True
    if "comma separated list" in lowered and question.count(",") >= 6:
        return True
    return False


def question_supports_direct_python(question: str) -> bool:
    if question_is_self_contained(question):
        return True
    arithmetic_patterns = (
        r"\bcalculate\b",
        r"\bsum\b",
        r"\bdifference\b",
        r"\bproduct\b",
        r"\btimes\b",
        r"\bmultiply\b",
        r"\bdivid(?:e|ed)\b",
        r"\bplus\b",
        r"\bminus\b",
    )
    return bool(
        re.search(r"\d", question)
        and any(re.search(pattern, question.lower()) for pattern in arithmetic_patterns)
    )


def question_is_metric_row_lookup(question: str) -> bool:
    return is_metric_row_lookup_question(question)


def build_research_hint_block(question: str) -> str:
    lowered = question.lower()
    hints: list[str] = []

    if (
        "|---|" in question
        or "here's the list i have so far:" in lowered
        or "comma separated list" in lowered and "," in question
        or "attached python code" in lowered
    ):
        hints.append(
            "This task may be solvable from the information already present in the prompt. "
            "Before searching the web, check whether direct reasoning over the provided table, list, or code is enough."
        )

    if "wikipedia" in lowered:
        hints.append(
            "This task explicitly references Wikipedia. Prefer search_wikipedia and fetch_wikipedia_page before broad web search."
        )

    if (
        "linked at the bottom of the article" in lowered
        or "linked at the bottom" in lowered
        or "links to a paper at the bottom" in lowered
        or "link to a paper at the bottom" in lowered
    ):
        hints.append(
            "This task refers to an article that links to a source. Find the article page, then inspect its outgoing links "
            "with extract_links_from_url instead of relying only on search snippets."
        )

    generic_urls = [url for url in extract_urls(question) if not is_youtube_url(url)]
    if generic_urls:
        hints.append(
            f"This task already names a URL: {', '.join(generic_urls)}. "
            "Inspect that page directly with fetch_url, find_text_in_url, extract_tables_from_url, or extract_links_from_url."
        )

    classification_cues = ("categoriz", "classify", "botanical", "stickler", "professor of botany")
    if any(cue in lowered for cue in classification_cues):
        hints.append(
            "This task involves classification or categorization. "
            "If unsure about a category (e.g. botanical fruit vs vegetable), research it with web_search "
            "or execute_python_code with explicit reasoning rather than relying on assumptions."
        )

    if not hints:
        return ""
    hint_body = "\n".join(f"- {hint}" for hint in hints)
    return f"\n\nResearch hints:\n{hint_body}"


def build_profile_guidance_block(*, question: str, profile: QuestionProfile) -> str:
    hints: list[str] = [f"Question profile: {profile.name}."]
    lowered = question.lower()

    if question_is_self_contained(question):
        hints.append(
            "The prompt appears self-contained. Solve from the prompt first and avoid web tools unless the prompt clearly lacks the needed facts."
        )
        hints.append(
            "Do not use execute_python_code for open-ended classification or filtering when the prompt already contains the full list/table."
        )
    if question_is_metric_row_lookup(question):
        hints.append(
            "This is a stat-table lookup. Once one fetched table contains both the selector metric and the requested metric, stop searching and answer from that table."
        )
        hints.append(
            "Prefer structured stats sources over discussion threads, recap articles, or player-specific pages."
        )
        hints.append(
            "Prefer batting, hitting, or stats pages over roster pages when the question asks for one stat of the player with the most or least of another stat."
        )
    if profile.name == "wikipedia_lookup":
        hints.append(
            "Prefer the canonical Wikipedia page or the directly relevant list page. If the answer depends on counts, rosters, or participants, use extract_tables_from_url before broader search."
        )
    if profile.name == "roster_neighbor_lookup":
        hints.append(
            "Fetch a roster or pitchers table directly and answer from that table. Do not reconstruct roster data from memory or with invented Python lists."
        )
        if profile.expected_date:
            hints.append(
                f"The question is date-sensitive ({profile.expected_date}). Prefer sources that explicitly mention that date/season rather than a generic current roster page."
            )
            hints.append(
                "Treat current or undated roster pages as exploratory only. Finalize only from dated, seasonal, archive, oldid, or otherwise temporally grounded evidence."
            )
            hints.append(
                "Official team player directories or player-detail pages for the requested season are valid evidence if they show numbered neighboring players."
            )
            hints.append(
                "If you learn the subject's jersey number first, search for a season-specific team player list or season-specific player detail pages that show the adjacent numbered players."
            )
    if profile.name == "article_to_paper":
        hints.append(
            "Find the exact article page, inspect its outgoing links, then read the linked paper or source. Answer only from fetched evidence."
        )
    if profile.name == "botanical_classification":
        hints.append(
            "This is a botanical classification task. Do not answer from culinary/common usage or from memory."
        )
        hints.append(
            "Search snippets alone are not sufficient. Ground the classification in fetched page text before finalizing."
        )
        hints.append(
            "Use web_search to find relevant sources, then fetch_url or find_text_in_url to verify ambiguous items before filtering and alphabetizing the final list."
        )
    if profile.name == "text_span_lookup":
        hints.append(
            "This is a targeted text-span lookup. Prefer the exact exercise/page and use find_text_in_url with the key noun phrase before browsing broadly."
        )
        hints.append(
            "Avoid mirrors, bulk PDFs, and book-level landing pages when an exact exercise page is available."
        )
    if profile.name == "entity_role_chain":
        hints.append(
            "This is a two-hop entity resolution task: identify the actor first, then read a cast/character source for the target show or series."
        )
        hints.append(
            "Avoid social posts, fandom pages, and generic popularity pages when cast or filmography sources are available."
        )
    if "libretexts.org" in profile.expected_domains:
        hints.append(
            "This looks like a LibreTexts text-span lookup. Find the exact exercise page, localize the matching passage with find_text_in_url, and answer from retrieved text only."
        )
    if "competition" in lowered and ("recipient" in lowered or "nationality" in lowered):
        hints.append(
            "Prefer a winners or recipients list with years and nationalities before reformulating the search."
        )
    if profile.expected_domains:
        hints.append(
            f"Prefer these domains when several results look plausible: {', '.join(profile.expected_domains)}."
        )
    if profile.target_urls:
        hints.append(f"Known target URL(s): {', '.join(profile.target_urls)}.")
    if profile.text_filter:
        hints.append(
            f"When extracting tables or links, a useful filter is: {profile.text_filter}."
        )

    hint_body = "\n".join(f"- {hint}" for hint in hints)
    return f"\n\nProfile guidance:\n{hint_body}"


def english_hint_score(text: str) -> int:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(token in COMMON_ENGLISH_HINTS for token in tokens)


def maybe_decode_reversed_question(text: str) -> str | None:
    candidate = text[::-1].strip()
    if not candidate:
        return None
    original_score = english_hint_score(text)
    candidate_score = english_hint_score(candidate)
    if candidate_score >= original_score + 2:
        return candidate
    return None


def try_prompt_reducer(question: str) -> tuple[str | None, str | None]:
    non_commutative_subset = find_non_commutative_subset(question)
    if non_commutative_subset:
        return ", ".join(non_commutative_subset), "non_commutative_subset"
    return None, None


def find_non_commutative_subset(question: str) -> list[str] | None:
    lowered = question.lower()
    if "not commutative" not in lowered or "table defining" not in lowered:
        return None

    set_match = SET_DEFINITION_RE.search(question)
    if not set_match:
        return None
    set_elements = [item.strip() for item in set_match.group("body").split(",") if item.strip()]
    if not set_elements:
        return None

    table = _parse_markdown_operation_table(question)
    if not table:
        return None

    involved: set[str] = set()
    for left in set_elements:
        for right in set_elements:
            left_result = table.get((left, right))
            right_result = table.get((right, left))
            if left_result is None or right_result is None:
                return None
            if left_result != right_result:
                involved.update((left, right))
    if not involved:
        return []
    return sorted(involved)


def _parse_markdown_operation_table(question: str) -> dict[tuple[str, str], str] | None:
    lines = [line.strip() for line in question.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None

    header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
    column_labels = header_cells[1:]
    if not column_labels:
        return None

    table: dict[tuple[str, str], str] = {}
    for raw_line in lines[2:]:
        cells = [cell.strip() for cell in raw_line.strip("|").split("|")]
        if len(cells) != len(column_labels) + 1:
            return None
        row_label = cells[0]
        for column_label, value in zip(column_labels, cells[1:], strict=False):
            table[(row_label, column_label)] = value
    return table


def extract_prompt_list_items(question: str) -> list[str]:
    match = re.search(
        r"here's the list i have so far:\s*(?P<body>.+?)(?:\n\s*\n|$)",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    body = re.sub(r"\s+", " ", match.group("body")).strip()
    return [item.strip() for item in body.split(",") if item.strip()]


def normalize_botanical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s{2,}", " ", normalized).strip()
