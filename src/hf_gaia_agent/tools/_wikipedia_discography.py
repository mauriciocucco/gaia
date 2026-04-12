"""Wikipedia discography helpers used by album-count tools."""

from __future__ import annotations

import re
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup, Tag

from ._http import make_client
from ._parsing import extract_text_section, html_to_text


def find_discography_table(html: str, section_name: str = "studio albums") -> Tag | None:
    soup = BeautifulSoup(html, "html.parser")
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text().strip().lower()
        if section_name not in text:
            continue
        table = heading.find_next("table")
        if table is not None:
            return table
    return None


def extract_albums_from_table(table: Tag, start_year: int, end_year: int) -> set[str]:
    albums: set[str] = set()
    rows = table.find_all("tr")
    if len(rows) < 2:
        return albums

    current_year: int | None = None
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        cell_texts = [cell.get_text().strip() for cell in cells]

        row_year: int | None = None
        for cell_text in cell_texts:
            match = re.search(r"\b((?:19|20)\d{2})\b", cell_text)
            if match:
                row_year = int(match.group(1))
                break

        current_year = row_year if row_year is not None else current_year
        row_year = current_year

        if row_year is None or not (start_year <= row_year <= end_year):
            continue

        title = ""
        for cell_text in cell_texts:
            cleaned = re.sub(r"\[.*?\]", "", cell_text).strip()
            if cleaned and not re.match(r"^\d{4}$", cleaned):
                title = cleaned
                break

        if title:
            albums.add(title.lower())
    return albums


def extract_albums_from_rendered_tables(
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
    *,
    extract_tables_from_url_tool,
) -> int:
    artist_slug = quote(artist_name.replace(" ", "_"), safe="()_")
    urls = [
        f"https://en.wikipedia.org/wiki/{artist_slug}_discography",
        f"https://en.wikipedia.org/wiki/{artist_slug}",
    ]

    with make_client() as client:
        for url in urls:
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            table = find_discography_table(response.text)
            if table is not None:
                albums = extract_albums_from_table(table, start_year, end_year)
                if albums:
                    return len(albums)

    for url in urls:
        try:
            rendered_tables = extract_tables_from_url_tool(
                url=url,
                text_filter="studio albums",
                max_tables=3,
                max_rows_per_table=80,
            )
        except Exception:
            continue
        albums = extract_albums_from_rendered_tables(
            rendered_tables,
            start_year,
            end_year,
        )
        if albums:
            return len(albums)

    url = f"https://en.wikipedia.org/wiki/{artist_slug}"
    with make_client() as client:
        response = client.get(url)
        response.raise_for_status()

    page_text = html_to_text(response.text)
    section_text = extract_text_section(
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
