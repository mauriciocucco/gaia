"""Compatibility aliases for shared recovery execution helpers.

New code inside the package may import directly from
``hf_gaia_agent.core.recoveries._shared``. This module keeps the shorter import
path used by skills and adapters.
"""

from __future__ import annotations

from ._shared import (
    RecoveryAttemptBudget,
    RecoveryExecutionContext,
    candidate_urls_from_state,
    fetch_candidate_urls,
    invoke_recovery_tool,
    quality_filtered_candidate_urls,
    ranked_candidates_from_result_text,
    recovery_result_from_records,
    recovery_trace_state,
    try_fetch_recovery,
    try_find_text_recovery,
    try_search_recovery,
    unfetched_first_candidate_urls,
    with_recovery_traces,
)

__all__ = [
    "RecoveryAttemptBudget",
    "RecoveryExecutionContext",
    "candidate_urls_from_state",
    "fetch_candidate_urls",
    "invoke_recovery_tool",
    "quality_filtered_candidate_urls",
    "ranked_candidates_from_result_text",
    "recovery_result_from_records",
    "recovery_trace_state",
    "try_fetch_recovery",
    "try_find_text_recovery",
    "try_search_recovery",
    "unfetched_first_candidate_urls",
    "with_recovery_traces",
]
