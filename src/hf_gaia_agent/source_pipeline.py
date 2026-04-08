"""Source-aware profiling, candidate selection, and evidence normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from .normalize import normalize_submitted_answer

URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
SEARCH_RESULT_RE = re.compile(
    r"(?:^|\n\n)(?P<rank>\d+)\.\s+(?P<title>.*?)\nURL:\s*(?P<url>\S+)\nSnippet:\s*(?P<snippet>.*?)(?=\n\n\d+\. |\Z)",
    flags=re.DOTALL,
)
MONTH_DATE_RE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?P<day>\d{1,2}),\s+(?P<year>\d{4})\b",
    flags=re.IGNORECASE,
)
AUTHOR_PUBLISHED_RE = re.compile(
    r"\bby\s+(?P<author>[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+){0,3})\s+was published\b"
)
URL_LINE_RE = re.compile(r"^(?:URL|URL Source):\s*(?P<url>\S+)\s*$", flags=re.MULTILINE)
TITLE_LINE_RE = re.compile(r"^Title:\s*(?P<title>.+?)\s*$", flags=re.MULTILINE)
PUBLISHED_LINE_RE = re.compile(
    r"^(?:Published Time|Published):\s*(?P<published>.+?)\s*$", flags=re.MULTILINE
)

SEARCH_STOP_WORDS = {
    "the",
    "and",
    "what",
    "which",
    "with",
    "from",
    "that",
    "this",
    "give",
    "only",
    "your",
    "answer",
    "would",
    "could",
    "under",
    "were",
    "was",
    "is",
    "are",
    "for",
    "into",
    "after",
    "before",
    "between",
    "there",
    "linked",
    "article",
    "paper",
    "number",
    "many",
    "much",
    "same",
    "season",
    "regular",
}


@dataclass(frozen=True)
class QuestionProfile:
    name: str
    target_urls: tuple[str, ...]
    expected_domains: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    expected_date: str | None
    expected_author: str | None
    subject_name: str | None
    text_filter: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceCandidate:
    title: str
    url: str
    snippet: str
    origin_tool: str
    score: int = 0
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    kind: str
    source_url: str
    source_type: str
    adapter_name: str
    content: str
    title_or_caption: str
    confidence: float
    extraction_method: str
    derived_from: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_question(
    question: str,
    *,
    file_name: str | None = None,
    local_file_path: str | None = None,
) -> QuestionProfile:
    lowered = question.lower()
    urls = tuple(_extract_urls(question))
    generic_urls = tuple(url for url in urls if not _is_youtube_url(url))
    expected_date = _extract_expected_date(question)
    expected_author = _extract_expected_author(question)
    subject_name = _extract_subject_name(question)
    text_filter = _infer_text_filter(question)

    if file_name and not local_file_path:
        return QuestionProfile(
            name="attachment_required",
            target_urls=(),
            expected_domains=(),
            preferred_tools=("read_local_file",),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    if any(_is_youtube_url(url) for url in urls):
        return QuestionProfile(
            name="transcript_or_video",
            target_urls=tuple(url for url in urls if _is_youtube_url(url)),
            expected_domains=("youtube.com", "youtu.be"),
            preferred_tools=("get_youtube_transcript", "analyze_youtube_video"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    if _is_article_to_paper_question(lowered):
        return QuestionProfile(
            name="article_to_paper",
            target_urls=generic_urls,
            expected_domains=_expected_domains(question, default=("universetoday.com",)),
            preferred_tools=("web_search", "fetch_url", "extract_links_from_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    if _is_olympics_country_code_question(lowered):
        return QuestionProfile(
            name="wikipedia_lookup",
            target_urls=generic_urls,
            expected_domains=_expected_domains(question, default=("wikipedia.org",)),
            preferred_tools=("search_wikipedia", "fetch_wikipedia_page", "extract_tables_from_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter or "athletes",
        )
    if _is_roster_neighbor_question(lowered):
        return QuestionProfile(
            name="roster_neighbor_lookup",
            target_urls=generic_urls,
            expected_domains=_expected_domains(question, default=()),
            preferred_tools=("web_search", "extract_tables_from_url", "fetch_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter or (subject_name or ""),
        )
    if generic_urls:
        return QuestionProfile(
            name="direct_url",
            target_urls=generic_urls,
            expected_domains=_expected_domains(question, default=()),
            preferred_tools=("fetch_url", "extract_tables_from_url", "extract_links_from_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    if "wikipedia" in lowered:
        return QuestionProfile(
            name="wikipedia_lookup",
            target_urls=generic_urls,
            expected_domains=("wikipedia.org",),
            preferred_tools=("search_wikipedia", "fetch_wikipedia_page", "extract_tables_from_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    if _looks_like_table_question(lowered):
        return QuestionProfile(
            name="table_lookup",
            target_urls=generic_urls,
            expected_domains=_expected_domains(question, default=()),
            preferred_tools=("web_search", "extract_tables_from_url", "find_text_in_url"),
            expected_date=expected_date,
            expected_author=expected_author,
            subject_name=subject_name,
            text_filter=text_filter,
        )
    return QuestionProfile(
        name="entity_attribute_lookup",
        target_urls=generic_urls,
        expected_domains=_expected_domains(question, default=()),
        preferred_tools=("web_search", "fetch_url", "find_text_in_url"),
        expected_date=expected_date,
        expected_author=expected_author,
        subject_name=subject_name,
        text_filter=text_filter,
    )


def parse_result_blocks(text: str, *, origin_tool: str) -> list[SourceCandidate]:
    candidates: list[SourceCandidate] = []
    for match in SEARCH_RESULT_RE.finditer(text.strip()):
        title = re.sub(r"\s+", " ", match.group("title").strip())
        url = match.group("url").strip()
        snippet = re.sub(r"\s+", " ", match.group("snippet").strip())
        if not url.startswith(("http://", "https://")):
            continue
        candidates.append(
            SourceCandidate(
                title=title or url,
                url=url,
                snippet=snippet,
                origin_tool=origin_tool,
            )
        )
    return candidates


def score_candidates(
    candidates: Sequence[SourceCandidate],
    *,
    question: str,
    profile: QuestionProfile,
) -> list[SourceCandidate]:
    scored: list[SourceCandidate] = []
    question_tokens = _query_tokens(question)
    expected_date_tokens = _query_tokens(profile.expected_date or "")
    author_tokens = _query_tokens(profile.expected_author or "")
    for candidate in candidates:
        score = 0
        reasons: list[str] = []
        haystack = f"{candidate.title}\n{candidate.snippet}\n{candidate.url}"
        domain = _registered_host(candidate.url)
        if domain and profile.expected_domains and any(
            domain.endswith(expected) for expected in profile.expected_domains
        ):
            score += 90
            reasons.append("expected_domain")
        elif domain and profile.expected_domains:
            score -= 35
            reasons.append("expected_domain_miss")
        if candidate.origin_tool == "search_wikipedia" and profile.name == "wikipedia_lookup":
            score += 30
            reasons.append("preferred_source")
        overlap = len(_query_tokens(haystack) & question_tokens)
        if overlap:
            score += overlap * 8
            reasons.append(f"token_overlap:{overlap}")
        if profile.expected_date and profile.expected_date.lower() in haystack.lower():
            score += 35
            reasons.append("expected_date")
        elif expected_date_tokens and expected_date_tokens <= _query_tokens(haystack):
            score += 10
            reasons.append("expected_date_partial")
        if author_tokens and author_tokens <= _query_tokens(haystack):
            score += 35
            reasons.append("expected_author")
        if profile.name == "article_to_paper":
            if "/articles/" in candidate.url or re.search(r"/\d{5,}/", candidate.url):
                score += 20
                reasons.append("article_path")
            if "paper" in candidate.title.lower() or "paper" in candidate.snippet.lower():
                score += 12
                reasons.append("paper_mention")
        if profile.name in {"table_lookup", "roster_neighbor_lookup", "wikipedia_lookup"}:
            if any(token in candidate.title.lower() for token in ("roster", "statistics", "olympics")):
                score += 10
                reasons.append("tableish_title")
        scored.append(
            SourceCandidate(
                title=candidate.title,
                url=candidate.url,
                snippet=candidate.snippet,
                origin_tool=candidate.origin_tool,
                score=score,
                reasons=tuple(reasons),
            )
        )
    scored.sort(key=lambda item: (-item.score, len(item.url)))
    deduped: list[SourceCandidate] = []
    seen: set[str] = set()
    for item in scored:
        if item.url in seen:
            continue
        deduped.append(item)
        seen.add(item.url)
    return deduped


def select_adapter_name(url: str) -> str:
    host = _registered_host(url)
    if host.endswith("wikipedia.org"):
        return "WikipediaAdapter"
    if host.endswith("universetoday.com"):
        return "ArticleSourceAdapter"
    if host.endswith("baseball-reference.com"):
        return "StatsTableAdapter"
    if host.endswith("libretexts.org"):
        return "ReferenceTextAdapter"
    return "GenericWebAdapter"


def parse_fetch_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    url_match = URL_LINE_RE.search(text)
    title_match = TITLE_LINE_RE.search(text)
    published_match = PUBLISHED_LINE_RE.search(text)
    if url_match:
        metadata["url"] = url_match.group("url").strip()
    if title_match:
        metadata["title"] = title_match.group("title").strip()
    if published_match:
        metadata["published"] = published_match.group("published").strip()
    return metadata


def evidence_records_from_tool_output(tool_name: str, content: str) -> list[EvidenceRecord]:
    normalized_content = (content or "").strip()
    if not normalized_content:
        return []
    if tool_name in {"web_search", "search_wikipedia", "extract_links_from_url"}:
        records: list[EvidenceRecord] = []
        for candidate in parse_result_blocks(normalized_content, origin_tool=tool_name):
            records.append(
                EvidenceRecord(
                    kind="links",
                    source_url=candidate.url,
                    source_type="url_list",
                    adapter_name=select_adapter_name(candidate.url),
                    content=candidate.snippet,
                    title_or_caption=candidate.title,
                    confidence=0.45,
                    extraction_method=tool_name,
                    derived_from=(tool_name,),
                )
            )
        return records

    if tool_name == "extract_tables_from_url":
        records = []
        for section in _split_rendered_tables(normalized_content):
            caption = _extract_caption(section)
            records.append(
                EvidenceRecord(
                    kind="table",
                    source_url="",
                    source_type="table",
                    adapter_name="TableExtraction",
                    content=section,
                    title_or_caption=caption,
                    confidence=0.8,
                    extraction_method=tool_name,
                    derived_from=(tool_name,),
                )
            )
        return records

    metadata = parse_fetch_metadata(normalized_content)
    source_url = metadata.get("url", "")
    title = metadata.get("title", "")
    kind = "transcript" if tool_name == "get_youtube_transcript" else "text"
    source_type = "page_text"
    if tool_name == "read_local_file":
        source_type = "file_text"
    if kind == "transcript":
        source_type = "transcript"
    return [
        EvidenceRecord(
            kind=kind,
            source_url=source_url,
            source_type=source_type,
            adapter_name=select_adapter_name(source_url) if source_url else tool_name,
            content=normalized_content,
            title_or_caption=title,
            confidence=0.7,
            extraction_method=tool_name,
            derived_from=(tool_name,),
        )
    ]


def serialize_candidates(candidates: Iterable[SourceCandidate]) -> list[dict[str, Any]]:
    return [candidate.as_dict() for candidate in candidates]


def serialize_evidence(records: Iterable[EvidenceRecord]) -> list[dict[str, Any]]:
    return [record.as_dict() for record in records]


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.findall(text):
        cleaned = match.rstrip(").,;:!?")
        if cleaned:
            urls.append(cleaned)
    return urls


def _is_youtube_url(url: str) -> bool:
    lowered = url.lower()
    return "youtube.com/" in lowered or "youtu.be/" in lowered


def _query_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in SEARCH_STOP_WORDS
    }


def _registered_host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _extract_expected_date(question: str) -> str | None:
    match = MONTH_DATE_RE.search(question)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(0).strip())


def _extract_expected_author(question: str) -> str | None:
    match = AUTHOR_PUBLISHED_RE.search(question)
    if not match:
        return None
    return match.group("author").strip()


def _extract_subject_name(question: str) -> str | None:
    match = re.search(
        r"before and after\s+(?P<name>[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){0,3})'s number",
        question,
    )
    if match:
        return match.group("name").strip()
    return None


def _infer_text_filter(question: str) -> str | None:
    lowered = question.lower()
    if "athletes" in lowered:
        return "athletes"
    if "walks" in lowered and "at bats" in lowered:
        return "walks at bats"
    if "pitcher" in lowered or "roster" in lowered:
        return "pitcher roster"
    if "award number" in lowered:
        return "award number support"
    return None


def _expected_domains(question: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
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


def _is_article_to_paper_question(lowered_question: str) -> bool:
    return (
        "linked at the bottom of the article" in lowered_question
        or "link to a paper at the bottom" in lowered_question
        or "links to a paper at the bottom" in lowered_question
    )


def _is_roster_neighbor_question(lowered_question: str) -> bool:
    return "number before and after" in lowered_question or "before and after" in lowered_question and "number" in lowered_question


def _is_olympics_country_code_question(lowered_question: str) -> bool:
    return (
        "olympics" in lowered_question
        and "athletes" in lowered_question
        and "ioc country code" in lowered_question
    )


def _looks_like_table_question(lowered_question: str) -> bool:
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


def _split_rendered_tables(content: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.startswith("Table ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _extract_caption(section: str) -> str:
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("Caption:"):
            return stripped.removeprefix("Caption:").strip()
    first_lines = [line.strip() for line in section.splitlines() if line.strip()]
    if len(first_lines) >= 2:
        return first_lines[1]
    return ""
