"""Tools used by the GAIA LangGraph agent."""

from __future__ import annotations

import base64
from ast import (
    Add,
    BinOp,
    Constant,
    Div,
    Expression,
    FloorDiv,
    Load,
    Mod,
    Mult,
    Name,
    Pow,
    Sub,
    UAdd,
    USub,
    UnaryOp,
    parse,
    walk,
)
import contextlib
import csv
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)

import httpx
from bs4 import BeautifulSoup, Comment, Tag
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from pypdf import PdfReader


_HTTP_HEADERS = {
    "User-Agent": "GAIABot/1.0 (contact@example.com) python-httpx/0.27.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_SEARCH_HEADERS = {
    **_HTTP_HEADERS,
}
_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".ogg", ".wav", ".webm"}


def _truncate(value: str, *, max_chars: int = 12000) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "\n...[truncated]"


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def _read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            rows.append(", ".join(row))
            if index >= 49:
                break
    return "\n".join(rows)


def _read_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return json.dumps(data, ensure_ascii=True, indent=2)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _read_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


_XLSX_MAIN_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_XLSX_REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
_XLSX_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _read_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw_xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(raw_xml)
    values: list[str] = []
    for item in root.findall("main:si", _XLSX_MAIN_NS):
        text_parts = [part.text or "" for part in item.findall(".//main:t", _XLSX_MAIN_NS)]
        values.append("".join(text_parts).strip())
    return values


def _read_xlsx_sheet_entries(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("rel:Relationship", _XLSX_REL_NS)
        if rel.attrib.get("Id") and rel.attrib.get("Target")
    }

    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("main:sheets/main:sheet", _XLSX_MAIN_NS):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{_XLSX_DOC_REL_NS}id")
        target = rel_map.get(rel_id or "", "")
        if not target:
            continue
        normalized_target = target.lstrip("/")
        if not normalized_target.startswith("xl/"):
            normalized_target = f"xl/{normalized_target}"
        sheets.append((name, normalized_target))
    return sheets


def _read_xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value = cell.findtext("main:v", default="", namespaces=_XLSX_MAIN_NS).strip()
    if cell_type == "s":
        if value.isdigit():
            index = int(value)
            if 0 <= index < len(shared_strings):
                return shared_strings[index]
        return value
    if cell_type == "inlineStr":
        text_parts = [part.text or "" for part in cell.findall(".//main:t", _XLSX_MAIN_NS)]
        return "".join(text_parts).strip()
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    formula = cell.findtext("main:f", default="", namespaces=_XLSX_MAIN_NS).strip()
    if formula and not value:
        return f"={formula}"
    return value


def _read_xlsx(path: Path, *, max_rows: int = 80) -> str:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_xlsx_shared_strings(archive)
        sheet_entries = _read_xlsx_sheet_entries(archive)
        if not sheet_entries:
            raise ValueError(f"No worksheets found in {path.name}.")

        lines: list[str] = []
        rows_emitted = 0
        for sheet_name, member_path in sheet_entries:
            if rows_emitted >= max_rows:
                break
            try:
                root = ET.fromstring(archive.read(member_path))
            except KeyError:
                continue

            sheet_rows: list[str] = []
            for row in root.findall(".//main:sheetData/main:row", _XLSX_MAIN_NS):
                values = [
                    cell_value
                    for cell in row.findall("main:c", _XLSX_MAIN_NS)
                    if (cell_value := _read_xlsx_cell_value(cell, shared_strings))
                ]
                if not values:
                    continue
                sheet_rows.append(", ".join(values))
                rows_emitted += 1
                if rows_emitted >= max_rows:
                    break

            if sheet_rows:
                lines.append(f"Sheet: {sheet_name}")
                lines.extend(sheet_rows)

        if not lines:
            raise ValueError(f"No readable worksheet rows found in {path.name}.")
        return "\n".join(lines)


def _audio_api_config() -> tuple[str, str, str]:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        model = os.getenv("AUDIO_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe").strip()
    elif provider == "huggingface":
        api_key = os.getenv("HF_TOKEN", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1").strip()
        model = os.getenv("AUDIO_TRANSCRIPTION_MODEL", "whisper-1").strip()
    else:
        raise RuntimeError(
            f"Audio transcription is not configured for MODEL_PROVIDER '{provider}'."
        )

    if not api_key:
        raise RuntimeError("Missing API key for audio transcription.")
    if not base_url:
        raise RuntimeError("Missing base URL for audio transcription.")
    if not model:
        raise RuntimeError("Missing audio transcription model.")
    return api_key, base_url.rstrip("/"), model


def _transcribe_audio(path: Path) -> str:
    api_key, base_url, model = _audio_api_config()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data: dict[str, Any] = {
        "model": model,
        "response_format": "json",
    }
    language = os.getenv("AUDIO_TRANSCRIPTION_LANGUAGE", "").strip()
    prompt = os.getenv("AUDIO_TRANSCRIPTION_PROMPT", "").strip()
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt

    with path.open("rb") as handle:
        files = {"file": (path.name, handle, content_type)}
        with httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {api_key}"}) as client:
            response = client.post(f"{base_url}/audio/transcriptions", data=data, files=files)
            response.raise_for_status()

    payload = response.json()
    transcript = str(payload.get("text") or "").strip()
    if not transcript:
        raise RuntimeError(f"Audio transcription returned no text for {path.name}.")
    return _truncate(transcript, max_chars=20000)


def read_file_content(path: str) -> str:
    """Read a local task attachment and return plain text content."""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"File not found: {candidate}")

    suffix = candidate.suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".csv":
        content = _read_csv(candidate)
    elif suffix == ".json":
        content = _read_json(candidate)
    elif suffix in {".html", ".htm"}:
        content = _html_to_text(candidate.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".pdf":
        content = _read_pdf(candidate)
    elif suffix == ".xlsx":
        content = _read_xlsx(candidate)
    elif suffix in _AUDIO_SUFFIXES:
        content = _transcribe_audio(candidate)
    else:
        content = candidate.read_text(encoding="utf-8", errors="replace")
    return _truncate(content)


def _extract_text_section(full_text: str, start_marker: str, end_markers: list[str]) -> str:
    lower = full_text.lower()
    start_index = lower.find(start_marker.lower())
    if start_index == -1:
        return ""

    end_index = len(full_text)
    for marker in end_markers:
        candidate = lower.find(marker.lower(), start_index + len(start_marker))
        if candidate != -1:
            end_index = min(end_index, candidate)
    return full_text[start_index:end_index]


def _find_discography_table(html: str, section_name: str = "studio albums") -> Tag | None:
    """Find the first table following a section heading on a Wikipedia page."""
    soup = BeautifulSoup(html, "html.parser")
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text().strip().lower()
        if section_name not in text:
            continue
        table = heading.find_next("table")
        if table is not None:
            return table
    return None


def _extract_albums_from_table(
    table: Tag, start_year: int, end_year: int
) -> set[str]:
    """Extract album titles from a Wikipedia discography table within a year range."""
    albums: set[str] = set()
    rows = table.find_all("tr")
    if len(rows) < 2:
        return albums

    current_year: int | None = None
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        cell_texts = [c.get_text().strip() for c in cells]

        # Detect year from any cell
        row_year: int | None = None
        for ct in cell_texts:
            m = re.search(r"\b((?:19|20)\d{2})\b", ct)
            if m:
                row_year = int(m.group(1))
                break

        if row_year is not None:
            current_year = row_year
        else:
            row_year = current_year

        if row_year is None or not (start_year <= row_year <= end_year):
            continue

        # Title: first non-empty cell that is not just a 4-digit year
        title = ""
        for ct in cell_texts:
            cleaned = re.sub(r"\[.*?\]", "", ct).strip()
            if cleaned and not re.match(r"^\d{4}$", cleaned):
                title = cleaned
                break

        if title:
            albums.add(title.lower())

    return albums


def _extract_albums_from_rendered_tables(
    rendered_tables: str,
    start_year: int,
    end_year: int,
) -> set[str]:
    albums: set[str] = set()
    for raw_line in rendered_tables.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Table ") or line.startswith("Caption:"):
            continue
        if line.lower() in {"year | album details", "year album details"}:
            continue
        match = re.match(r"^(?P<year>(?:19|20)\d{2})\s*\|\s*(?P<title>.+)$", line)
        if not match:
            match = re.match(r"^(?P<year>(?:19|20)\d{2})\s+(?P<title>.+)$", line)
        if not match:
            continue
        year = int(match.group("year"))
        if start_year <= year <= end_year:
            albums.add(match.group("title").strip().lower())
    return albums


def count_wikipedia_studio_album_count_for_artist(
    artist_name: str,
    start_year: int,
    end_year: int,
) -> int:
    artist_slug = quote(artist_name.replace(" ", "_"), safe="()_")
    urls = [
        f"https://en.wikipedia.org/wiki/{artist_slug}_discography",
        f"https://en.wikipedia.org/wiki/{artist_slug}",
    ]

    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        for url in urls:
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            table = _find_discography_table(response.text)
            if table is not None:
                albums = _extract_albums_from_table(table, start_year, end_year)
                if albums:
                    return len(albums)

    for url in urls:
        try:
            rendered_tables = extract_tables_from_url(
                url=url,
                text_filter="studio albums",
                max_tables=3,
                max_rows_per_table=80,
            )
        except Exception:
            continue
        albums = _extract_albums_from_rendered_tables(rendered_tables, start_year, end_year)
        if albums:
            return len(albums)

    # Fallback: text-based extraction from main page
    url = f"https://en.wikipedia.org/wiki/{artist_slug}"
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()

    page_text = _html_to_text(response.text)
    section_text = _extract_text_section(
        page_text,
        "Studio albums",
        ["EPs", "Live albums", "Compilation albums", "Filmography", "References"],
    )
    if not section_text:
        raise ValueError(f"Studio albums section not found for {artist_name}.")

    albums_fallback: set[tuple[int, str]] = set()
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line or line.lower() in {"studio albums", "year album details"}:
            continue
        match = re.match(r"^(?P<year>(?:19|20)\d{2})\s+(?P<title>.+)$", line)
        if not match:
            continue
        year = int(match.group("year"))
        if start_year <= year <= end_year:
            albums_fallback.add((year, match.group("title").strip().lower()))
    return len(albums_fallback)


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
    lines = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        href = item.get("href") or item.get("url") or ""
        body = item.get("body") or ""
        lines.append(f"{index}. {title}\nURL: {href}\nSnippet: {body}")
    return "\n\n".join(lines)


def _wikipedia_page_url(title: str) -> str:
    normalized_title = title.strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(normalized_title, safe='()_')}"


def _wikipedia_title_from_url(url: str) -> str | None:
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


def _r_jina_ai_url(url: str) -> str:
    return f"https://r.jina.ai/{url}"


def _query_terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _score_text_match(text: str, query: str) -> int:
    normalized_text = text.lower()
    normalized_query = query.strip().lower()
    if not normalized_query:
        return 0

    score = 0
    if normalized_query in normalized_text:
        score += 100

    terms = [term for term in _query_terms(normalized_query) if len(term) >= 3]
    unique_terms = list(dict.fromkeys(terms))
    for term in unique_terms:
        if term in normalized_text:
            score += 10
    return score


def _iter_html_tables(soup: BeautifulSoup) -> list[Tag]:
    tables = list(soup.find_all("table"))
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment_html = str(comment).strip()
        if "<table" not in comment_html.lower():
            continue
        comment_soup = BeautifulSoup(comment_html, "html.parser")
        tables.extend(comment_soup.find_all("table"))
    return tables


def _fetch_html_text(url: str) -> tuple[str, str]:
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
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

        page_title = _wikipedia_title_from_url(url)
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
                return html, _wikipedia_page_url(resolved_title)

        if direct_error is not None:
            raise direct_error
        raise RuntimeError(f"Unable to fetch HTML content from URL: {url}")


def _fetch_r_jina_markdown(url: str) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        response = client.get(_r_jina_ai_url(url))
        response.raise_for_status()
    return response.text


def _clean_markdown_cell(value: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[\[edit\]\([^)]+\)\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" |")


def _render_markdown_tables(
    markdown_text: str,
    *,
    text_filter: str,
    max_tables: int,
    max_rows_per_table: int,
) -> str:
    lowered_filter = text_filter.strip().lower()
    lines = markdown_text.splitlines()
    rendered_candidates: list[tuple[int, str]] = []
    current_table: list[str] = []
    pending_heading = ""

    def _flush_current_table() -> None:
        nonlocal current_table, pending_heading
        if len(current_table) < 2:
            current_table = []
            return

        rows: list[str] = []
        if pending_heading:
            rows.append(f"Caption: {pending_heading}")

        for raw_line in current_table:
            stripped = raw_line.strip()
            if re.fullmatch(r"\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", stripped):
                continue
            cells = [_clean_markdown_cell(cell) for cell in stripped.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
            if len(rows) >= max_rows_per_table + (1 if pending_heading else 0):
                break

        current_table = []
        if not rows:
            return

        rendered = "\n".join(rows)
        score = _score_text_match(rendered, lowered_filter) if lowered_filter else 1
        if lowered_filter and score <= 0:
            return
        rendered_candidates.append((score, rendered))

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            pending_heading = stripped.lstrip("#").strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            current_table.append(stripped)
            continue
        if current_table:
            _flush_current_table()

    if current_table:
        _flush_current_table()

    rendered_candidates.sort(key=lambda item: -item[0])
    rendered_tables = [
        f"Table {index}\n{rendered}"
        for index, (_score, rendered) in enumerate(rendered_candidates[:max_tables], start=1)
    ]
    if not rendered_tables:
        return "No readable HTML tables found."
    return _truncate("\n\n".join(rendered_tables), max_chars=20000)


def _fetch_url_text(url: str) -> str:
    response: httpx.Response | None = None
    direct_error: Exception | None = None
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
        try:
            response = client.get(url)
            response.raise_for_status()
        except Exception as exc:
            direct_error = exc
            response = None
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                raise
            fallback_response = client.get(_r_jina_ai_url(url))
            fallback_response.raise_for_status()
            response = fallback_response

    if response is None:
        if direct_error is not None:
            raise direct_error
        raise RuntimeError(f"Unable to fetch URL: {url}")

    content_type = response.headers.get("content-type", "")
    response_url = str(response.url)
    if "application/json" in content_type:
        text = json.dumps(response.json(), ensure_ascii=True, indent=2)
    elif "application/pdf" in content_type or response_url.lower().endswith(".pdf"):
        text = _read_pdf_bytes(response.content)
    elif "text/html" in content_type:
        title = ""
        soup = BeautifulSoup(response.text, "html.parser")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)
        body_text = _html_to_text(response.text)
        if title:
            text = f"Title: {title}\nURL: {response_url}\n\n{body_text}"
        else:
            text = body_text
    else:
        text = response.text
    return _truncate(text)


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


def _search_brave_html(query: str, *, max_results: int) -> list[dict[str, str]]:
    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers=_SEARCH_HEADERS,
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
    with httpx.Client(
        timeout=20.0,
        follow_redirects=True,
        headers=_SEARCH_HEADERS,
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


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return short snippets from the top results."""
    providers = (
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
        return "No results found."
    return _format_search_results(results)


@tool
def search_wikipedia(query: str, max_results: int = 5) -> str:
    """Search English Wikipedia and return candidate pages with snippets."""
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
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
            url=_wikipedia_page_url(title),
            snippet=snippet,
        )
        if normalized is not None:
            results.append(normalized)

    if not results:
        return "No Wikipedia results found."
    return _format_search_results(results)


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return text extracted from the response body."""
    return _fetch_url_text(url)


@tool
def fetch_wikipedia_page(title: str) -> str:
    """Fetch the plain-text extract for an English Wikipedia page by title."""
    with httpx.Client(timeout=20.0, follow_redirects=True, headers=_HTTP_HEADERS) as client:
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
    full_url = str(page.get("fullurl") or _wikipedia_page_url(page_title)).strip()
    extract = str(page.get("extract") or "").strip()
    if not extract:
        raise RuntimeError(f"Wikipedia page '{page_title}' returned no extract text.")
    return _truncate(f"Title: {page_title}\nURL: {full_url}\n\n{extract}", max_chars=20000)


def _same_registered_host(left: str, right: str) -> bool:
    def _normalize(host: str) -> str:
        return host.lower().removeprefix("www.")

    return _normalize(left) == _normalize(right)


@tool
def extract_links_from_url(
    url: str,
    text_filter: str = "",
    max_results: int = 20,
    same_domain_only: bool = False,
) -> str:
    """Fetch an HTML page and list links, optionally filtered by anchor text or restricted to the same domain."""
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
        if same_domain_only and not _same_registered_host(base_host, resolved_host):
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
        return "No matching links found."
    return _format_search_results(results)


@tool
def find_text_in_url(url: str, query: str, max_matches: int = 8) -> str:
    """Fetch a URL and return the most relevant lines or snippets containing the query text."""
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
        return "\n".join(matches)

    scored_lines = [
        (line, _score_text_match(line, normalized_query))
        for line in lines
    ]
    scored_lines = [(line, score) for line, score in scored_lines if score > 0]
    if scored_lines:
        scored_lines.sort(key=lambda item: (-item[1], len(item[0])))
        return "\n".join(line for line, _score in scored_lines[:max_matches])

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
        return "\n".join(window_matches)

    words = [word for word in _query_terms(normalized_query) if len(word) >= 3]
    if not words:
        return "No matches found."

    snippets: list[tuple[str, int]] = []
    for line in lines:
        score = _score_text_match(line, normalized_query)
        if score <= 0:
            continue
        snippets.append((line, score))
    if snippets:
        snippets.sort(key=lambda item: (-item[1], len(item[0])))
        return "\n".join(text for text, _score in snippets[:max_matches])
    return "No matches found."


@tool
def extract_tables_from_url(
    url: str,
    text_filter: str = "",
    max_tables: int = 5,
    max_rows_per_table: int = 15,
) -> str:
    """Fetch an HTML page and extract readable text from its tables, optionally filtering to relevant tables."""
    try:
        html, _resolved_url = _fetch_html_text(url)
    except Exception:
        markdown_text = _fetch_r_jina_markdown(url)
        return _render_markdown_tables(
            markdown_text,
            text_filter=text_filter,
            max_tables=max_tables,
            max_rows_per_table=max_rows_per_table,
        )

    soup = BeautifulSoup(html, "html.parser")
    lowered_filter = text_filter.strip().lower()
    tables = _iter_html_tables(soup)
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
        score = _score_text_match(rendered, lowered_filter) if lowered_filter else 1
        if lowered_filter and score <= 0:
            continue
        rendered_candidates.append((score, rendered))

    rendered_candidates.sort(key=lambda item: -item[0])
    for table_index, (_score, rendered) in enumerate(rendered_candidates[:max_tables], start=1):
        rendered_tables.append(f"Table {table_index}\n{rendered}")

    if not rendered_tables:
        return "No readable HTML tables found."
    return _truncate("\n\n".join(rendered_tables), max_chars=20000)


@tool
def get_youtube_transcript(url: str, languages_csv: str = "en,en-US") -> str:
    """Fetch a YouTube transcript for a video URL using prioritized languages."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import CouldNotRetrieveTranscript
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Reinstall project dependencies."
        ) from exc

    video_id = extract_youtube_video_id(url)
    languages = [item.strip() for item in languages_csv.split(",") if item.strip()]
    client = YouTubeTranscriptApi()
    try:
        transcript = client.fetch(video_id, languages=languages or ["en"])
    except CouldNotRetrieveTranscript as exc:
        return f"Transcript unavailable for {video_id}: {exc}"

    lines = [
        f"[{snippet.start:.2f}s] {snippet.text}"
        for snippet in transcript
    ]
    return _truncate("\n".join(lines), max_chars=20000)


_FRAME_INTERVAL_SECONDS = 5
_MAX_FRAMES = 20
_COUNTING_VISUAL_CUES = (
    "how many",
    "highest number",
    "maximum number",
    "minimum number",
    "lowest number",
    "simultaneously",
    "at the same time",
    "on camera simultaneously",
)


def _check_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' not found on PATH. Install it first.")
    return path


def _download_video(url: str, output_dir: Path) -> Path:
    yt_dlp = _check_binary("yt-dlp")
    output_path = output_dir / "video.%(ext)s"
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
        "--quiet",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True, timeout=120)
    videos = list(output_dir.glob("video.*"))
    if not videos:
        raise FileNotFoundError("yt-dlp did not produce an output file.")
    return videos[0]


def _extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
    ffmpeg = _check_binary("ffmpeg")
    pattern = output_dir / "frame_%04d.jpg"
    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vf", f"fps=1/{_FRAME_INTERVAL_SECONDS},scale=768:-1",
        "-q:v", "2",
        "-frames:v", str(_MAX_FRAMES),
        str(pattern),
        "-y",
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True, timeout=120)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    return frames


def _encode_frame_base64(frame_path: Path) -> str:
    return base64.b64encode(frame_path.read_bytes()).decode("ascii")


def _is_counting_visual_question(question: str) -> bool:
    lowered = question.strip().lower()
    return any(cue in lowered for cue in _COUNTING_VISUAL_CUES)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [str(text or "").strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced if item.strip())

    match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _extract_max_count_from_payload(payload: dict[str, Any]) -> int | None:
    for key in ("max_count", "max_species_count", "highest_count", "highest_species_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())

    frames = payload.get("frames")
    if not isinstance(frames, list):
        return None

    counts: list[int] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        for key in ("count", "species_count", "visible_count"):
            value = item.get(key)
            if isinstance(value, int):
                counts.append(value)
                break
            if isinstance(value, str) and value.strip().isdigit():
                counts.append(int(value.strip()))
                break
    if counts:
        return max(counts)
    return None


def _build_video_analysis_prompt(
    *,
    question: str,
    video_id: str,
    frame_count: int,
    frame_interval_seconds: int,
    counting_mode: bool,
) -> str:
    if counting_mode:
        return (
            f"These are {frame_count} frames sampled every {frame_interval_seconds} seconds "
            f"from a YouTube video (ID: {video_id}).\n\n"
            f"Question: {question}\n\n"
            "Analyze each frame independently and count only what the question asks for.\n"
            "For species questions, count biological species, not individuals, ages, or sexes.\n"
            "Do not count chicks and adults of the same species separately.\n"
            "Look carefully for small or distant subjects in the background before deciding.\n"
            "Return JSON only using this schema:\n"
            '{\"frames\":[{\"timestamp_s\":0,\"count\":0,\"notes\":\"short note\"}],\"max_count\":0}\n'
            "Use integer counts."
        )
    return (
        f"These are {frame_count} frames sampled every {frame_interval_seconds} seconds "
        f"from a YouTube video (ID: {video_id}).\n\n"
        f"Question: {question}\n\n"
        "Analyze the frames carefully and answer the question. "
        "Be specific and precise."
    )


@tool
def analyze_youtube_video(url: str, question: str) -> str:
    """Download a YouTube video, extract frames, and analyze them with a vision model to answer the question."""
    from langchain_core.messages import HumanMessage as HM
    from langchain_openai import ChatOpenAI
    import os

    video_id = extract_youtube_video_id(url)

    with tempfile.TemporaryDirectory(prefix="gaia_video_") as tmp:
        tmp_path = Path(tmp)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        try:
            video_file = _download_video(url, tmp_path)
        except Exception as exc:
            return f"Failed to download video {video_id}: {exc}"

        try:
            frames = _extract_frames(video_file, frames_dir)
        except Exception as exc:
            return f"Failed to extract frames from video {video_id}: {exc}"

        if not frames:
            return f"No frames extracted from video {video_id}."

        counting_mode = _is_counting_visual_question(question)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _build_video_analysis_prompt(
                    question=question,
                    video_id=video_id,
                    frame_count=len(frames),
                    frame_interval_seconds=_FRAME_INTERVAL_SECONDS,
                    counting_mode=counting_mode,
                ),
            }
        ]
        for i, frame in enumerate(frames):
            timestamp = i * _FRAME_INTERVAL_SECONDS
            content.append({
                "type": "text",
                "text": f"[Frame at {timestamp}s]",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_frame_base64(frame)}",
                    "detail": "high" if counting_mode else "low",
                },
            })

        provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
        model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": 0,
            "timeout": 120,
        }
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
        elif provider == "huggingface":
            token = os.getenv("HF_TOKEN")
            base_url = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
            if token:
                kwargs["api_key"] = token
            kwargs["base_url"] = base_url

        vision_model = ChatOpenAI(**kwargs)
        response = vision_model.invoke([HM(content=content)])
        response_text = str(response.content)
        if counting_mode:
            payload = _extract_json_object(response_text)
            if payload is not None:
                max_count = _extract_max_count_from_payload(payload)
                if max_count is not None:
                    return str(max_count)
        return _truncate(response_text, max_chars=8000)


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
        )
    )


@tool
def read_local_file(path: str) -> str:
    """Read a local text, CSV, JSON, HTML, PDF, XLSX, or supported audio file."""
    return read_file_content(path)


ALLOWED_BINOPS = {
    Add: lambda left, right: left + right,
    Sub: lambda left, right: left - right,
    Mult: lambda left, right: left * right,
    Div: lambda left, right: left / right,
    FloorDiv: lambda left, right: left // right,
    Mod: lambda left, right: left % right,
    Pow: lambda left, right: left**right,
}
ALLOWED_UNARYOPS = {
    UAdd: lambda value: +value,
    USub: lambda value: -value,
}


def _safe_eval(node: Any) -> float | int:
    if isinstance(node, Expression):
        return _safe_eval(node.body)
    if isinstance(node, Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, BinOp):
        operator = ALLOWED_BINOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported operator.")
        return operator(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, UnaryOp):
        operator = ALLOWED_UNARYOPS.get(type(node.op))
        if not operator:
            raise ValueError("Unsupported unary operator.")
        return operator(_safe_eval(node.operand))
    if isinstance(node, Name) and node.id in {"pi", "e"}:
        constants = {"pi": 3.141592653589793, "e": 2.718281828459045}
        return constants[node.id]
    raise ValueError("Unsafe expression.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    tree = parse(expression, mode="eval")
    for node in walk(tree):
        if type(node) not in {
            Expression,
            BinOp,
            UnaryOp,
            Constant,
            Load,
            Add,
            Sub,
            Mult,
            Div,
            FloorDiv,
            Mod,
            Pow,
            UAdd,
            USub,
            Name,
        }:
            raise ValueError("Unsafe expression.")

    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def build_tools() -> list[Any]:
    return [
        web_search,
        search_wikipedia,
        fetch_url,
        fetch_wikipedia_page,
        extract_links_from_url,
        find_text_in_url,
        extract_tables_from_url,
        get_youtube_transcript,
        analyze_youtube_video,
        count_wikipedia_studio_albums,
        read_local_file,
        calculate,
    ]
