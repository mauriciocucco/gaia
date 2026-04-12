"""Evidence reducers — registry and public API."""

from __future__ import annotations

from typing import Sequence

from ..source_pipeline import EvidenceRecord
from .base import EvidenceReducer, ReducerResult
from ._parsing import ToolEvidence
from .metric_row import MetricRowReducer
from .table_compare import TableCompareReducer, solve_answer_from_tool_evidence
from .roster import RosterReducer
from .temporal import TemporalReducer
from .text_span import TextSpanReducer
from .award import AwardReducer

__all__ = [
    "EvidenceReducer",
    "ReducerResult",
    "ToolEvidence",
    "solve_answer_from_evidence_records",
    "solve_answer_from_tool_evidence",
    "REDUCER_REGISTRY",
]

# Ordered by priority (lower number = tried first).
REDUCER_REGISTRY: list[EvidenceReducer] = [
    MetricRowReducer(),
    TableCompareReducer(),
    RosterReducer(),
    TemporalReducer(),
    TextSpanReducer(),
    AwardReducer(),
]


def solve_answer_from_evidence_records(
    question: str,
    evidence_records: Sequence[EvidenceRecord],
) -> tuple[str | None, str | None]:
    for reducer in REDUCER_REGISTRY:
        result = reducer.solve(question, evidence_records)
        if result is not None:
            return result.answer, result.reducer_name
    return None, None
