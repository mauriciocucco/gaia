"""Evidence normalization: parse tool outputs into structured EvidenceRecords."""

from __future__ import annotations

import re
from typing import Any, Iterable

from ._models import EvidenceRecord, SourceCandidate
from .source_labels import source_family_for_url
from ..tools._payloads import (
    SearchResultPayload,
    TableExtractPayload,
    TextDocumentPayload,
    deserialize_tool_payloads,
)

SEARCH_RESULT_RE = re.compile(
    r"(?:^|\n\n)(?P<rank>\d+)\.\s+(?P<title>.*?)\nURL:\s*(?P<url>\S+)\nSnippet:\s*(?P<snippet>.*?)(?=\n\n\d+\. |\Z)",
    flags=re.DOTALL,
)
KIND_LINE_RE = re.compile(r"^Kind:\s*(?P<kind>.+?)\s*$", flags=re.MULTILINE)
URL_LINE_RE = re.compile(r"^(?:URL|URL Source):\s*(?P<url>\S+)\s*$", flags=re.MULTILINE)
TITLE_LINE_RE = re.compile(r"^Title:\s*(?P<title>.+?)\s*$", flags=re.MULTILINE)
PUBLISHED_LINE_RE = re.compile(
    r"^(?:Published Time|Published):\s*(?P<published>.+?)\s*$", flags=re.MULTILINE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_result_blocks(text: str, *, origin_tool: str) -> list[SourceCandidate]:
    payloads = deserialize_tool_payloads(text)
    if payloads:
        candidates: list[SourceCandidate] = []
        for index, payload in enumerate(payloads, start=1):
            if not isinstance(payload, SearchResultPayload):
                continue
            if not payload.url.startswith(("http://", "https://")):
                continue
            candidates.append(
                SourceCandidate(
                    title=payload.title or payload.url,
                    url=payload.url,
                    snippet=payload.snippet,
                    origin_tool=origin_tool,
                    score=0,
                    reasons=(),
                )
            )
        return candidates

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


def parse_fetch_metadata(text: str) -> dict[str, str]:
    payloads = deserialize_tool_payloads(text)
    if payloads:
        for payload in payloads:
            if isinstance(payload, TextDocumentPayload):
                metadata = {"kind": payload.kind}
                if payload.url:
                    metadata["url"] = payload.url
                if payload.title:
                    metadata["title"] = payload.title
                if payload.published:
                    metadata["published"] = payload.published
                return metadata
            if isinstance(payload, TableExtractPayload):
                metadata: dict[str, str] = {}
                if payload.url:
                    metadata["url"] = payload.url
                if payload.title:
                    metadata["title"] = payload.title
                return metadata

    metadata: dict[str, str] = {}
    kind_match = KIND_LINE_RE.search(text)
    url_match = URL_LINE_RE.search(text)
    title_match = TITLE_LINE_RE.search(text)
    published_match = PUBLISHED_LINE_RE.search(text)
    if kind_match:
        metadata["kind"] = kind_match.group("kind").strip()
    if url_match:
        metadata["url"] = url_match.group("url").strip()
    if title_match:
        metadata["title"] = title_match.group("title").strip()
    if published_match:
        metadata["published"] = published_match.group("published").strip()
    return metadata


def evidence_records_from_tool_output(tool_name: str, content: str) -> list[EvidenceRecord]:
    payloads = deserialize_tool_payloads(content)
    if payloads:
        if tool_name in {"web_search", "search_wikipedia", "extract_links_from_url"}:
            records: list[EvidenceRecord] = []
            for payload in payloads:
                if not isinstance(payload, SearchResultPayload):
                    continue
                records.append(
                    EvidenceRecord(
                        kind="links",
                        source_url=payload.url,
                        source_type="url_list",
                        adapter_name=source_family_for_url(payload.url),
                        content=payload.snippet,
                        title_or_caption=payload.title,
                        confidence=0.45,
                        extraction_method=tool_name,
                        derived_from=(tool_name,),
                    )
                )
            return records

        if tool_name == "extract_tables_from_url":
            records: list[EvidenceRecord] = []
            for payload in payloads:
                if not isinstance(payload, TableExtractPayload):
                    continue
                for table in payload.tables:
                    records.append(
                        EvidenceRecord(
                            kind="table",
                            source_url=payload.url,
                            source_type="table",
                            adapter_name="table_extraction_source",
                            content=table.content,
                            title_or_caption=table.caption or payload.title,
                            confidence=0.8,
                            extraction_method=tool_name,
                            derived_from=(tool_name,),
                        )
                    )
            return records

        for payload in payloads:
            if not isinstance(payload, TextDocumentPayload):
                continue
            source_type = "page_text"
            kind = "transcript" if tool_name == "get_youtube_transcript" else "text"
            if tool_name == "read_local_file":
                source_type = "file_text"
            if kind == "transcript":
                source_type = "transcript"
            return [
                EvidenceRecord(
                    kind=kind,
                    source_url=payload.url,
                    source_type=source_type,
                    adapter_name=source_family_for_url(payload.url)
                    if payload.url
                    else tool_name,
                    content=payload.content,
                    title_or_caption=payload.title,
                    confidence=0.7,
                    extraction_method=tool_name,
                    derived_from=(tool_name,),
                )
            ]
        return []

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
                    adapter_name=source_family_for_url(candidate.url),
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
        metadata = parse_fetch_metadata(normalized_content)
        source_url = metadata.get("url", "")
        source_title = metadata.get("title", "")
        for section in _split_rendered_tables(normalized_content):
            caption = _extract_caption(section)
            records.append(
                EvidenceRecord(
                    kind="table",
                    source_url=source_url,
                    source_type="table",
                    adapter_name="table_extraction_source",
                    content=section,
                    title_or_caption=caption or source_title,
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
            adapter_name=source_family_for_url(source_url) if source_url else tool_name,
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_rendered_tables(content: str) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.startswith(("URL:", "URL Source:", "Title:", "Published:", "Published Time:")):
            continue
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
