"""Source adapter registry."""

from __future__ import annotations

from typing import Any

from .base import SourceAdapter
from .temporal_roster import (
    FightersAdapter,
    OfficialTeamDirectoryAdapter,
    WikipediaRosterAdapter,
)

__all__ = [
    "SourceAdapter",
    "FightersAdapter",
    "OfficialTeamDirectoryAdapter",
    "WikipediaRosterAdapter",
    "build_source_adapters",
]


def build_source_adapters(tools_by_name: dict[str, Any]) -> list[SourceAdapter]:
    return [
        FightersAdapter(tools_by_name),
        WikipediaRosterAdapter(tools_by_name),
        OfficialTeamDirectoryAdapter(tools_by_name),
    ]
