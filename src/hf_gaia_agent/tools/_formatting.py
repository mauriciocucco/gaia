"""Shared rendering helpers for tool outputs."""

from __future__ import annotations

from ._http import truncate


def render_search_results(results: list[dict[str, str]]) -> str:
    lines = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        href = item.get("href") or item.get("url") or ""
        body = item.get("body") or ""
        lines.append(f"{index}. {title}\nURL: {href}\nSnippet: {body}")
    return "\n\n".join(lines)


def render_text_document(
    *,
    kind: str,
    content: str,
    url: str = "",
    title: str = "",
    published: str = "",
    max_chars: int = 80_000,
) -> str:
    metadata: list[str] = [f"Kind: {kind}"]
    if url:
        metadata.append(f"URL: {url}")
    if title:
        metadata.append(f"Title: {title}")
    if published:
        metadata.append(f"Published: {published}")
    body = content.strip()
    if body:
        return truncate("\n".join(metadata) + f"\n\n{body}", max_chars=max_chars)
    return truncate("\n".join(metadata), max_chars=max_chars)


def render_table_extract(
    *,
    content: str,
    url: str = "",
    title: str = "",
    max_chars: int = 80_000,
) -> str:
    metadata: list[str] = []
    if url:
        metadata.append(f"URL: {url}")
    if title:
        metadata.append(f"Title: {title}")
    body = content.strip()
    if metadata and body:
        return truncate("\n".join(metadata) + f"\n{body}", max_chars=max_chars)
    if metadata:
        return truncate("\n".join(metadata), max_chars=max_chars)
    return truncate(body, max_chars=max_chars)
