"""Web search tools — multi-provider search with fallback."""

from __future__ import annotations

import contextlib
import io
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool

from ._formatting import render_search_results
from ._http import SEARCH_HEADERS, SEARCH_TIMEOUT, make_client
from ._payloads import SearchResultPayload, StructuredToolResult


# ---------------------------------------------------------------------------
# Result normalisation & formatting
# ---------------------------------------------------------------------------


def _normalize_search_result(
    *, title: str | None, url: str | None, snippet: str | None
) -> dict[str, str] | None:
    normalized_title = re.sub(r"\s+", " ", str(title or "").strip())
    normalized_url = str(url or "").strip()
    normalized_snippet = re.sub(r"\s+", " ", str(snippet or "").strip())
    if not normalized_title or not normalized_url.startswith(("http://", "https://")):
        return None
    return {
        "title": normalized_title,
        "href": normalized_url,
        "body": normalized_snippet,
    }


def _format_search_results(results: list[dict[str, str]]) -> str:
    return render_search_results(results)


def _merge_search_results(
    existing: list[dict[str, str]],
    incoming: list[dict[str, str]],
    *,
    max_results: int,
) -> list[dict[str, str]]:
    merged = list(existing)
    seen = {item.get("href", "") for item in existing}
    for item in incoming:
        href = item.get("href", "")
        if not href or href in seen:
            continue
        merged.append(item)
        seen.add(href)
        if len(merged) >= max_results:
            break
    return merged


def _search_payloads(results: list[dict[str, str]]) -> tuple[SearchResultPayload, ...]:
    payloads: list[SearchResultPayload] = []
    for index, item in enumerate(results, start=1):
        payloads.append(
            SearchResultPayload(
                title=str(item.get("title", "")),
                url=str(item.get("href", "")),
                snippet=str(item.get("body", "")),
                rank=index,
            )
        )
    return tuple(payloads)


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------


def _search_tavily(query: str, *, max_results: int) -> list[dict[str, str]]:
    from tavily import TavilyClient  # lazy import — optional dependency
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    results: list[dict[str, str]] = []
    for item in response.get("results", []):
        normalized = _normalize_search_result(
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("content"),
        )
        if normalized is not None:
            results.append(normalized)
    return results


def _search_brave_html(query: str, *, max_results: int) -> list[dict[str, str]]:
    with make_client(
        timeout=SEARCH_TIMEOUT,
        headers=SEARCH_HEADERS,
        retries=1,
    ) as client:
        response = client.get(
            "https://search.brave.com/search",
            params={"q": query, "source": "web"},
        )
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.sub(r"\s+", " ", page_title).strip()
    if "pow captcha" in title.lower():
        raise RuntimeError("Brave search returned a captcha challenge.")

    results: list[dict[str, str]] = []
    for container in soup.select("div.result-content"):
        anchor = container.select_one("a[href]")
        title_el = container.select_one(".search-snippet-title")
        snippet_el = container.select_one(".generic-snippet .content")
        item = _normalize_search_result(
            title=title_el.get_text(" ", strip=True) if title_el else None,
            url=anchor.get("href") if anchor else None,
            snippet=snippet_el.get_text(" ", strip=True) if snippet_el else "",
        )
        if item is None:
            continue
        results.append(item)
        if len(results) >= max_results:
            break
    return results


def _search_duckduckgo(query: str, *, max_results: int) -> list[dict[str, str]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

    results: list[dict[str, str]] = []
    for item in raw_results:
        normalized = _normalize_search_result(
            title=item.get("title"),
            url=item.get("href") or item.get("url"),
            snippet=item.get("body"),
        )
        if normalized is not None:
            results.append(normalized)
    return results


def _search_bing_rss(query: str, *, max_results: int) -> list[dict[str, str]]:
    with make_client(
        timeout=SEARCH_TIMEOUT,
        headers=SEARCH_HEADERS,
        retries=1,
    ) as client:
        response = client.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss", "count": str(max_results)},
        )
        response.raise_for_status()

    root = ET.fromstring(response.text)
    results: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        normalized = _normalize_search_result(
            title=item.findtext("title"),
            url=item.findtext("link"),
            snippet=item.findtext("description"),
        )
        if normalized is not None:
            results.append(normalized)
        if len(results) >= max_results:
            break
    return results


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return short snippets from the top results."""
    return web_search_result(query=query, max_results=max_results).text


def web_search_result(query: str, max_results: int = 5) -> StructuredToolResult:
    """Structured variant of ``web_search`` for internal workflow use."""
    providers = (
        ("tavily", _search_tavily),
        ("brave", _search_brave_html),
        ("duckduckgo", _search_duckduckgo),
        ("bing", _search_bing_rss),
    )
    results: list[dict[str, str]] = []
    errors: list[str] = []

    for provider_name, provider in providers:
        if len(results) >= max_results:
            break
        try:
            provider_results = provider(query, max_results=max_results)
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")
            continue
        results = _merge_search_results(
            results,
            provider_results,
            max_results=max_results,
        )

    if not results:
        if errors:
            raise RuntimeError("All search providers failed: " + " | ".join(errors))
        return StructuredToolResult(text="No results found.")
    return StructuredToolResult(
        text=_format_search_results(results),
        payloads=_search_payloads(results),
    )


@tool
def search_wikipedia(query: str, max_results: int = 5) -> str:
    """Search English Wikipedia and return candidate pages with snippets."""
    return search_wikipedia_result(query=query, max_results=max_results).text


def search_wikipedia_result(query: str, max_results: int = 5) -> StructuredToolResult:
    """Structured variant of ``search_wikipedia`` for internal workflow use."""
    from ._web_helpers import wikipedia_page_url

    with make_client(timeout=SEARCH_TIMEOUT, retries=1) as client:
        response = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": str(max_results),
                "utf8": "1",
            },
        )
        response.raise_for_status()

    payload = response.json()
    raw_results = payload.get("query", {}).get("search", [])
    results: list[dict[str, str]] = []
    for item in raw_results:
        title = str(item.get("title") or "").strip()
        snippet_html = str(item.get("snippet") or "").strip()
        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)
        normalized = _normalize_search_result(
            title=title,
            url=wikipedia_page_url(title),
            snippet=snippet,
        )
        if normalized is not None:
            results.append(normalized)

    if not results:
        return StructuredToolResult(text="No Wikipedia results found.")
    return StructuredToolResult(
        text=_format_search_results(results),
        payloads=_search_payloads(results),
    )
