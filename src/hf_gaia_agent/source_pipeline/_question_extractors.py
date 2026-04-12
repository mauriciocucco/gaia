"""Extraction helpers for question profiling."""

from __future__ import annotations

import re

MONTH_DATE_RE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?P<day>\d{1,2}),\s+(?P<year>\d{4})\b",
    flags=re.IGNORECASE,
)
SLASH_DATE_RE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\b")
MONTH_YEAR_RE = re.compile(
    r"\b(?:as of\s+)?(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?P<year>\d{4})\b",
    flags=re.IGNORECASE,
)
AUTHOR_PUBLISHED_RE = re.compile(
    r"\bby\s+(?P<author>[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,3})\s+was published\b"
)


def extract_expected_date(question: str) -> str | None:
    match = MONTH_DATE_RE.search(question)
    if match:
        return re.sub(r"\s+", " ", match.group(0).strip())
    slash_match = SLASH_DATE_RE.search(question)
    if slash_match:
        return slash_match.group(0)
    month_year_match = MONTH_YEAR_RE.search(question)
    if month_year_match:
        return re.sub(r"\s+", " ", month_year_match.group(0).strip())
    return None


def extract_expected_author(question: str) -> str | None:
    match = AUTHOR_PUBLISHED_RE.search(question)
    if not match:
        return None
    return match.group("author").strip()


def extract_subject_name(question: str) -> str | None:
    match = re.search(
        r"before and after\s+(?P<name>[A-Z][A-Za-z''.\-]+(?:\s+[A-Z][A-Za-z''.\-]+){0,3})'s number",
        question,
    )
    if match:
        return match.group("name").strip()
    return None


def extract_subject_name_unicode(question: str) -> str | None:
    match = re.search(r"before and after\s+(?P<name>.+?)'s number", question)
    if match:
        return match.group("name").strip()
    return None


def infer_text_filter(question: str) -> str | None:
    lowered = question.lower()
    mentioned_match = re.search(
        r"(?:surname|last name|first name)\s+of\s+the\s+(?P<target>.+?)\s+mentioned",
        question,
        flags=re.IGNORECASE,
    )
    if mentioned_match:
        return mentioned_match.group("target").strip()
    if "athletes" in lowered:
        return "athletes"
    if "walks" in lowered and "at bats" in lowered:
        return "walks at bats"
    if "pitcher" in lowered or "roster" in lowered:
        return "pitcher roster"
    if "actor who played" in lowered and "play in" in lowered:
        return "cast character"
    if "award number" in lowered:
        return "award number support"
    return None


def expected_domains(question: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    lowered = question.lower()
    domains = list(default)
    if "wikipedia" in lowered and "wikipedia.org" not in domains:
        domains.append("wikipedia.org")
    if "universe today" in lowered and "universetoday.com" not in domains:
        domains.append("universetoday.com")
    if "libretext" in lowered and "libretexts.org" not in domains:
        domains.append("libretexts.org")
    if "yankee" in lowered and "baseball-reference.com" not in domains:
        domains.append("baseball-reference.com")
    return tuple(dict.fromkeys(domains))
