"""Data models for the source pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionProfile:
    name: str
    target_urls: tuple[str, ...]
    expected_domains: tuple[str, ...]
    preferred_tools: tuple[str, ...]
    expected_date: str | None
    expected_author: str | None
    subject_name: str | None
    text_filter: str | None
    profile_family: str | None = None
    prompt_items: tuple[str, ...] = ()
    classification_labels: dict[str, str] | None = None
    ordering_key: str | None = None
    entity_name: str | None = None
    scope: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceCandidate:
    title: str
    url: str
    snippet: str
    origin_tool: str
    score: int = 0
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    kind: str
    source_url: str
    source_type: str
    adapter_name: str
    content: str
    title_or_caption: str
    confidence: float
    extraction_method: str
    derived_from: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def source_label(self) -> str:
        """Compatibility-friendly name for ``adapter_name``."""
        return self.adapter_name
