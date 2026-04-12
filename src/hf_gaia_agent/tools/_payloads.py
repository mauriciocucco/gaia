"""Structured payloads for internal tool execution.

The public LangChain tools still render plain text for the model, but the
workflow can now retain typed payloads alongside that text so downstream
normalization does not need to re-parse everything from regexes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SearchResultPayload:
    title: str
    url: str
    snippet: str = ""
    rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "search_result"
        return payload


@dataclass(frozen=True)
class TextDocumentPayload:
    kind: str
    content: str
    url: str = ""
    title: str = ""
    published: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "text_document"
        return payload


@dataclass(frozen=True)
class TableSectionPayload:
    content: str
    caption: str = ""
    index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "table_section"
        return payload


@dataclass(frozen=True)
class TableExtractPayload:
    url: str = ""
    title: str = ""
    tables: tuple[TableSectionPayload, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "table_extract"
        payload["tables"] = [table.as_dict() for table in self.tables]
        return payload


ToolPayload = (
    SearchResultPayload
    | TextDocumentPayload
    | TableSectionPayload
    | TableExtractPayload
)


@dataclass(frozen=True)
class StructuredToolResult:
    text: str
    payloads: tuple[ToolPayload, ...] = ()


def serialize_tool_payloads(payloads: Iterable[ToolPayload]) -> list[dict[str, Any]]:
    return [payload.as_dict() for payload in payloads]


def deserialize_tool_payloads(raw_payloads: Any) -> list[ToolPayload]:
    if not isinstance(raw_payloads, list):
        return []

    parsed: list[ToolPayload] = []
    for item in raw_payloads:
        if not isinstance(item, dict):
            continue
        payload_type = str(item.get("type", "")).strip()
        if payload_type == "search_result":
            parsed.append(
                SearchResultPayload(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("snippet", "")),
                    rank=(
                        int(item["rank"])
                        if isinstance(item.get("rank"), int)
                        else None
                    ),
                )
            )
            continue
        if payload_type == "text_document":
            metadata = item.get("metadata")
            parsed.append(
                TextDocumentPayload(
                    kind=str(item.get("kind", "")),
                    content=str(item.get("content", "")),
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    published=str(item.get("published", "")),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
            continue
        if payload_type == "table_section":
            parsed.append(
                TableSectionPayload(
                    content=str(item.get("content", "")),
                    caption=str(item.get("caption", "")),
                    index=(
                        int(item["index"])
                        if isinstance(item.get("index"), int)
                        else None
                    ),
                )
            )
            continue
        if payload_type == "table_extract":
            raw_tables = item.get("tables")
            tables: list[TableSectionPayload] = []
            if isinstance(raw_tables, list):
                for raw_table in raw_tables:
                    if not isinstance(raw_table, dict):
                        continue
                    tables.append(
                        TableSectionPayload(
                            content=str(raw_table.get("content", "")),
                            caption=str(raw_table.get("caption", "")),
                            index=(
                                int(raw_table["index"])
                                if isinstance(raw_table.get("index"), int)
                                else None
                            ),
                        )
                    )
            metadata = item.get("metadata")
            parsed.append(
                TableExtractPayload(
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    tables=tuple(tables),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
    return parsed
