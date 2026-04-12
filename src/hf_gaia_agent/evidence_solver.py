"""Structured answer extraction from previously collected tool evidence.

This module is a backward-compatible re-export layer.
All logic now lives in the ``reducers`` package.
"""

from __future__ import annotations

from .reducers import (
    ToolEvidence,
    solve_answer_from_evidence_records,
    solve_answer_from_tool_evidence,
)

__all__ = [
    "ToolEvidence",
    "solve_answer_from_evidence_records",
    "solve_answer_from_tool_evidence",
]
