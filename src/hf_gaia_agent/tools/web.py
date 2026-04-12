"""Web fetching and extraction tools — fetch URLs, extract links, find text, extract tables."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from langchain_core.tools import tool

from ._formatting import render_search_results, render_table_extract, render_text_document
from ._http import HTTP_HEADERS, make_client, truncate
from ._parsing import (
    html_to_text,
    extract_text_section,
    iter_html_tables,
    read_pdf_bytes,
    score_text_match,
    query_terms,
)
from ._web_helpers import (
    r_jina_ai_url,
    same_registered_host,
    wikipedia_page_url,
    wikipedia_title_from_url,
)
from ._payloads import SearchResultPayload, StructuredToolResult, TextDocumentPayload
from ._web_tables import (
    render_markdown_tables as _render_markdown_tables,
    table_payload_from_rendered as _table_payload_from_rendered,
)
from ._wikipedia_discography import count_wikipedia_studio_album_count_for_artist
from .search import _normalize_search_result, _format_search_results


# ---------------------------------------------------------------------------
# Internal fetchers
# ---------------------------------------------------------------------------


def _fetch_html_text(url: str) -> tuple[str, str]:
    with make_client() as client:
        direct_error: Exception | None = None
        response: httpx.Response | None = None
        try:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                raise RuntimeError(f"URL did not return HTML content: {url}")
            response_url = str(getattr(response, "url", "") or url)
            return response.text, response_url
        except Exception as exc:
            direct_error = exc

        page_title = wikipedia_title_from_url(url)
        if page_title:
            response = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "format": "json",
                    "page": page_title,
                    "prop": "text",
                    "redirects": "1",
                    "formatversion": "2",
                },
            )
            response.raise_for_status()
            payload = response.json()
            html = str(payload.get("parse", {}).get("text") or "").strip()
            resolved_title = str(payload.get("parse", {}).get("title") or page_title).strip()
            if html:
                return html, wikipedia_page_url(resolved_title)

        if direct_error is not None:
            raise direct_error
        raise RuntimeError(f"Unable to fetch HTML content from URL: {url}")


def _fetch_r_jina_markdown(url: str) -> str:
    with make_client() as client:
        response = client.get(r_jina_ai_url(url))
        response.raise_for_status()
    return response.text


def _render_text_payload(payload: TextDocumentPayload, *, max_chars: int = 80_000) -> StructuredToolResult:
    return StructuredToolResult(
        text=render_text_document(
            kind=payload.kind,
            content=payload.content,
            url=payload.url,
            title=payload.title,
            published=payload.published,
            max_chars=max_chars,
        ),
        payloads=(payload,),
    )


def _fetch_url_result(url: str) -> StructuredToolResult:
    response: httpx.Response | None = None
    direct_error: Exception | None = None
    with make_client() as client:
        try:
            response = client.get(url)
            response.raise_for_status()
        except Exception as exc:
            direct_error = exc
            response = None
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                raise
            fallback_response = client.get(r_jina_ai_url(url))
            fallback_response.raise_for_status()
            response = fallback_response

    if response is None:
        if direct_error is not None:
            raise direct_error
        raise RuntimeError(f"Unable to fetch URL: {url}")

    content_type = response.headers.get("content-type", "")
    response_url = str(response.url)
    display_url = url if response_url.startswith("https://r.jina.ai/") else response_url
    if "application/json" in content_type:
        text = json.dumps(response.json(), ensure_ascii=True, indent=2)
        return _render_text_payload(
            TextDocumentPayload(
                kind="json_document",
                url=display_url,
                content=text,
            )
        )
    elif "application/pdf" in content_type or response_url.lower().endswith(".pdf"):
        text = read_pdf_bytes(response.content)
        return _render_text_payload(
            TextDocumentPayload(
                kind="document_text",
                url=display_url,
                content=text,
            )
        )
    elif "text/html" in content_type:
        title = ""
        soup = BeautifulSoup(response.text, "html.parser")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)
        body_text = html_to_text(response.text)
        return _render_text_payload(
            TextDocumentPayload(
                kind="page_text",
                url=display_url,
                title=title,
                content=body_text,
            )
        )
    return _render_text_payload(
        TextDocumentPayload(
            kind="page_text",
            url=display_url,
            content=response.text,
        )
    )


def _fetch_url_text(url: str) -> str:
    return _fetch_url_result(url).text




# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return text extracted from the response body."""
    return _fetch_url_result(url).text


@tool
def fetch_wikipedia_page(title: str) -> str:
    """Fetch the plain-text extract for an English Wikipedia page by title."""
    return fetch_wikipedia_page_result(title=title).text


def fetch_wikipedia_page_result(title: str) -> StructuredToolResult:
    """Structured variant of ``fetch_wikipedia_page`` for internal workflow use."""
    with make_client(timeout=20.0) as client:
        response = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts|info",
                "explaintext": "1",
                "redirects": "1",
                "inprop": "url",
                "titles": title,
                "formatversion": "2",
            },
        )
        response.raise_for_status()

    pages = response.json().get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError(f"Wikipedia returned no page data for title '{title}'.")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"Wikipedia page not found for title '{title}'.")

    page_title = str(page.get("title") or title).strip()
    full_url = str(page.get("fullurl") or wikipedia_page_url(page_title)).strip()
    extract = str(page.get("extract") or "").strip()
    if not extract:
        raise RuntimeError(f"Wikipedia page '{page_title}' returned no extract text.")
    return _render_text_payload(
        TextDocumentPayload(
            kind="page_text",
            url=full_url,
            title=page_title,
            content=extract,
        )
    )


@tool
def extract_links_from_url(
    url: str,
    text_filter: str = "",
    max_results: int = 20,
    same_domain_only: bool = False,
) -> str:
    """Fetch an HTML page and list links, optionally filtered by anchor text or restricted to the same domain."""
    return extract_links_from_url_result(
        url=url,
        text_filter=text_filter,
        max_results=max_results,
        same_domain_only=same_domain_only,
    ).text


def extract_links_from_url_result(
    url: str,
    text_filter: str = "",
    max_results: int = 20,
    same_domain_only: bool = False,
) -> StructuredToolResult:
    """Structured variant of ``extract_links_from_url`` for internal workflow use."""
    html, base_url = _fetch_html_text(url)
    base_host = urlparse(base_url).netloc
    lowered_filter = text_filter.strip().lower()
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        resolved = urljoin(base_url, href)
        resolved_host = urlparse(resolved).netloc
        if same_domain_only and not same_registered_host(base_host, resolved_host):
            continue

        text = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        title_attr = re.sub(r"\s+", " ", str(anchor.get("title") or "").strip())
        haystack = " ".join(part for part in (text, title_attr, resolved) if part).lower()
        if lowered_filter and lowered_filter not in haystack:
            continue
        if resolved in seen:
            continue

        normalized = _normalize_search_result(
            title=text or title_attr or resolved,
            url=resolved,
            snippet=title_attr,
        )
        if normalized is None:
            continue
        results.append(normalized)
        seen.add(resolved)
        if len(results) >= max_results:
            break

    if not results:
        return StructuredToolResult(text="No matching links found.")
    payloads = tuple(
        SearchResultPayload(
            title=str(item.get("title", "")),
            url=str(item.get("href", "")),
            snippet=str(item.get("body", "")),
            rank=index,
        )
        for index, item in enumerate(results, start=1)
    )
    return StructuredToolResult(
        text=render_search_results(results),
        payloads=payloads,
    )


@tool
def find_text_in_url(url: str, query: str, max_matches: int = 8) -> str:
    """Fetch a URL and return the most relevant lines or snippets containing the query text."""
    return find_text_in_url_result(
        url=url,
        query=query,
        max_matches=max_matches,
    ).text


def find_text_in_url_result(
    url: str, query: str, max_matches: int = 8
) -> StructuredToolResult:
    """Structured variant of ``find_text_in_url`` for internal workflow use."""
    text = _fetch_url_text(url)
    normalized_query = query.strip().lower()
    if not normalized_query:
        raise ValueError("query must not be empty.")

    matches: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if normalized_query in line.lower():
            matches.append(line)
            if len(matches) >= max_matches:
                break

    if matches:
        rendered = "\n".join(matches)
        return StructuredToolResult(
            text=rendered,
            payloads=(
                TextDocumentPayload(
                    kind="text_matches",
                    url=url,
                    content=rendered,
                    metadata={"query": query},
                ),
            ),
        )

    scored_lines = [
        (line, score_text_match(line, normalized_query))
        for line in lines
    ]
    scored_lines = [(line, score) for line, score in scored_lines if score > 0]
    if scored_lines:
        scored_lines.sort(key=lambda item: (-item[1], len(item[0])))
        rendered = "\n".join(line for line, _score in scored_lines[:max_matches])
        return StructuredToolResult(
            text=rendered,
            payloads=(
                TextDocumentPayload(
                    kind="text_matches",
                    url=url,
                    content=rendered,
                    metadata={"query": query},
                ),
            ),
        )

    window_matches: list[str] = []
    lowered_text = text.lower()
    start = 0
    while len(window_matches) < max_matches:
        index = lowered_text.find(normalized_query, start)
        if index == -1:
            break
        snippet_start = max(0, index - 140)
        snippet_end = min(len(text), index + len(normalized_query) + 140)
        snippet = re.sub(r"\s+", " ", text[snippet_start:snippet_end]).strip()
        if snippet and snippet not in window_matches:
            window_matches.append(snippet)
        start = index + len(normalized_query)

    if window_matches:
        rendered = "\n".join(window_matches)
        return StructuredToolResult(
            text=rendered,
            payloads=(
                TextDocumentPayload(
                    kind="text_matches",
                    url=url,
                    content=rendered,
                    metadata={"query": query},
                ),
            ),
        )

    words = [word for word in query_terms(normalized_query) if len(word) >= 3]
    if not words:
        return "No matches found."

    snippets: list[tuple[str, int]] = []
    for line in lines:
        score = score_text_match(line, normalized_query)
        if score <= 0:
            continue
        snippets.append((line, score))
    if snippets:
        snippets.sort(key=lambda item: (-item[1], len(item[0])))
        rendered = "\n".join(text for text, _score in snippets[:max_matches])
        return StructuredToolResult(
            text=rendered,
            payloads=(
                TextDocumentPayload(
                    kind="text_matches",
                    url=url,
                    content=rendered,
                    metadata={"query": query},
                ),
            ),
        )
    return StructuredToolResult(text="No matches found.")


@tool
def extract_tables_from_url(
    url: str,
    text_filter: str = "",
    max_tables: int = 10,
    max_rows_per_table: int = 400,
) -> str:
    """Fetch an HTML page and extract readable text from its tables, optionally filtering to relevant tables."""
    return extract_tables_from_url_result(
        url=url,
        text_filter=text_filter,
        max_tables=max_tables,
        max_rows_per_table=max_rows_per_table,
    ).text


def extract_tables_from_url_result(
    url: str,
    text_filter: str = "",
    max_tables: int = 10,
    max_rows_per_table: int = 400,
) -> StructuredToolResult:
    """Structured variant of ``extract_tables_from_url`` for internal workflow use."""
    try:
        html, resolved_url = _fetch_html_text(url)
    except Exception:
        markdown_text = _fetch_r_jina_markdown(url)
        rendered = _render_markdown_tables(
            markdown_text,
            text_filter=text_filter,
            max_tables=max_tables,
            max_rows_per_table=max_rows_per_table,
        )
        return _table_payload_from_rendered(
            rendered_content=rendered,
            url=url,
        )

    soup = BeautifulSoup(html, "html.parser")
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    lowered_filter = text_filter.strip().lower()
    tables = iter_html_tables(soup)
    rendered_tables: list[str] = []
    rendered_candidates: list[tuple[int, str]] = []
    for table in tables:
        rows: list[str] = []
        caption_el = table.find("caption")
        if caption_el:
            caption_text = caption_el.get_text(" ", strip=True)
            if caption_text:
                rows.append(f"Caption: {caption_text}")
        for row in table.find_all("tr")[:max_rows_per_table]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            values = [re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() for cell in cells]
            values = [value for value in values if value]
            if values:
                rows.append(" | ".join(values))
        if not rows:
            continue

        rendered = "\n".join(rows)
        score = score_text_match(rendered, lowered_filter) if lowered_filter else 1
        if lowered_filter and score <= 0:
            continue
        rendered_candidates.append((score, rendered))

    rendered_candidates.sort(key=lambda item: -item[0])
    for table_index, (_score, rendered) in enumerate(rendered_candidates[:max_tables], start=1):
        rendered_tables.append(f"Table {table_index}\n{rendered}")

    if not rendered_tables:
        return _table_payload_from_rendered(
            rendered_content="No readable HTML tables found.",
            url=resolved_url,
            title=page_title,
        )
    return _table_payload_from_rendered(
        rendered_content="\n\n".join(rendered_tables),
        url=resolved_url,
        title=page_title,
    )


@tool
def count_wikipedia_studio_albums(
    artist_name: str,
    start_year: int,
    end_year: int,
) -> str:
    """Count studio albums listed on the artist's English Wikipedia page between two years inclusive."""
    return str(
        count_wikipedia_studio_album_count_for_artist(
            artist_name=artist_name,
            start_year=start_year,
            end_year=end_year,
            extract_tables_from_url_tool=extract_tables_from_url,
        )
    )
