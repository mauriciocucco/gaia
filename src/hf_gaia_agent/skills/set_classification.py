"""Shared helpers for deterministic item-set classification skills."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..normalize import normalize_submitted_answer
from ..source_pipeline import EvidenceRecord, serialize_evidence


@dataclass
class ClassifiedItemState:
    item: str
    status: str = "unknown"
    evidence: list[EvidenceRecord] = field(default_factory=list)


def build_set_classification_result(
    *,
    skill_name: str,
    included_items: list[str],
    records: list[EvidenceRecord],
    reducer_used: str = "set_classification",
    tool_trace: list[str] | None = None,
    decision_trace: list[str] | None = None,
) -> dict[str, object]:
    ordered = sorted(dict.fromkeys(included_items), key=lambda value: value.lower())
    return {
        "final_answer": ", ".join(ordered),
        "error": None,
        "reducer_used": reducer_used,
        "skill_used": skill_name,
        "skill_trace": [skill_name],
        "evidence_used": serialize_evidence(records[-6:]),
        "recovery_reason": None,
        "tool_trace": list(tool_trace or []),
        "decision_trace": list(decision_trace or []),
    }


def answer_references_item(answer_text: str, item: str) -> bool:
    normalized_answer = normalize_submitted_answer(answer_text).lower()
    normalized_item = normalize_submitted_answer(item).lower()
    return bool(normalized_answer and normalized_item and normalized_item in normalized_answer)
