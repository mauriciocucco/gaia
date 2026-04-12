"""Shared URL/Wikipedia helpers used by search and web tools."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, quote, unquote, urlparse


def wikipedia_page_url(title: str) -> str:
    normalized_title = title.strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(normalized_title, safe='()_')}"


def wikipedia_title_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    if not host.endswith("wikipedia.org"):
        return None
    if not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path.removeprefix("/wiki/")).strip()
    if not title or ":" in title:
        return None
    return title.replace("_", " ")


def r_jina_ai_url(url: str) -> str:
    return f"https://r.jina.ai/{url}"


def extract_youtube_video_id(url: str) -> str:
    """Extract a YouTube video ID from a standard URL."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        match = re.match(r"^/(?:embed|shorts)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)
    raise ValueError(f"Unsupported YouTube URL: {url}")


def same_registered_host(left: str, right: str) -> bool:
    def _normalize(host: str) -> str:
        return host.lower().removeprefix("www.")
    return _normalize(left) == _normalize(right)
