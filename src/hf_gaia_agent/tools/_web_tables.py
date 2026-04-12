"""Table rendering helpers for web extraction tools."""

from __future__ import annotations

import re

from ._formatting import render_table_extract
from ._http import truncate
from ._parsing import score_text_match
from ._payloads import StructuredToolResult, TableExtractPayload, TableSectionPayload


def clean_markdown_cell(value: str) -> str:
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", value)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[\[edit\]\([^)]+\)\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" |")


def render_markdown_tables(
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

    def flush_current_table() -> None:
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
            cells = [clean_markdown_cell(cell) for cell in stripped.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
            if len(rows) >= max_rows_per_table + (1 if pending_heading else 0):
                break

        current_table = []
        if not rows:
            return

        rendered = "\n".join(rows)
        score = score_text_match(rendered, lowered_filter) if lowered_filter else 1
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
            flush_current_table()

    if current_table:
        flush_current_table()

    rendered_candidates.sort(key=lambda item: -item[0])
    rendered_tables = [
        f"Table {index}\n{rendered}"
        for index, (_score, rendered) in enumerate(rendered_candidates[:max_tables], start=1)
    ]
    if not rendered_tables:
        return "No readable HTML tables found."
    return truncate("\n\n".join(rendered_tables), max_chars=80000)


def table_payload_from_rendered(
    *,
    rendered_content: str,
    url: str = "",
    title: str = "",
) -> StructuredToolResult:
    if rendered_content.strip() == "No readable HTML tables found.":
        return StructuredToolResult(
            text=render_table_extract(content=rendered_content, url=url, title=title)
        )

    sections: list[TableSectionPayload] = []
    current_lines: list[str] = []
    current_caption = ""
    current_index: int | None = None

    def flush_current() -> None:
        nonlocal current_lines, current_caption, current_index
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                TableSectionPayload(
                    content=content,
                    caption=current_caption,
                    index=current_index,
                )
            )
        current_lines = []
        current_caption = ""
        current_index = None

    for raw_line in rendered_content.splitlines():
        line = raw_line.rstrip()
        if line.startswith("Table "):
            if current_lines:
                flush_current()
            try:
                current_index = int(line.split(" ", 1)[1])
            except Exception:
                current_index = None
            continue
        if line.startswith("Caption:"):
            current_caption = line.removeprefix("Caption:").strip()
        current_lines.append(line)
    if current_lines:
        flush_current()

    payload = TableExtractPayload(url=url, title=title, tables=tuple(sections))
    return StructuredToolResult(
        text=render_table_extract(content=rendered_content, url=url, title=title),
        payloads=(payload,),
    )
