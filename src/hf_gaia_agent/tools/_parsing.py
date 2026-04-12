"""Parsing helpers for HTML, PDF, CSV, JSON, and XLSX files."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag
from pypdf import PdfReader

from ._http import truncate

# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


# ---------------------------------------------------------------------------
# File-format readers
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            rows.append(", ".join(row))
            if index >= 49:
                break
    return "\n".join(rows)


def read_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return json.dumps(data, ensure_ascii=True, indent=2)


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def read_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    chunks: list[str] = []
    for page in reader.pages[:20]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# XLSX reader (pure-XML, no openpyxl dependency)
# ---------------------------------------------------------------------------

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


def read_xlsx(path: Path, *, max_rows: int = 80) -> str:
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


# ---------------------------------------------------------------------------
# HTML table helpers (used by web tools)
# ---------------------------------------------------------------------------


def iter_html_tables(soup: BeautifulSoup) -> list[Tag]:
    tables = list(soup.find_all("table"))
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment_html = str(comment).strip()
        if "<table" not in comment_html.lower():
            continue
        comment_soup = BeautifulSoup(comment_html, "html.parser")
        tables.extend(comment_soup.find_all("table"))
    return tables


def extract_text_section(full_text: str, start_marker: str, end_markers: list[str]) -> str:
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


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------


def query_terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def score_text_match(text: str, query: str) -> int:
    normalized_text = text.lower()
    normalized_query = query.strip().lower()
    if not normalized_query:
        return 0

    score = 0
    if normalized_query in normalized_text:
        score += 100

    terms = [term for term in query_terms(normalized_query) if len(term) >= 3]
    unique_terms = list(dict.fromkeys(terms))
    for term in unique_terms:
        if term in normalized_text:
            score += 10
    return score
