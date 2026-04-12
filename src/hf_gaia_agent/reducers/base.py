"""Base protocol and result model for evidence reducers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..source_pipeline import EvidenceRecord


@dataclass(frozen=True)
class ReducerResult:
    answer: str
    reducer_name: str
    confidence: float = 1.0


@runtime_checkable
class EvidenceReducer(Protocol):
    name: str
    priority: int

    def solve(
        self, question: str, evidence_records: Sequence[EvidenceRecord]
    ) -> ReducerResult | None: ...
