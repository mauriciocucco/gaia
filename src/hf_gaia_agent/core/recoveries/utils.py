"""Compatibility aliases for shared recovery execution helpers."""

from __future__ import annotations

from ...fallbacks.utils import (
    FallbackAttemptBudget as RecoveryAttemptBudget,
    FallbackExecutionContext as RecoveryExecutionContext,
    candidate_urls_from_state,
    fallback_result_from_records,
    fallback_trace_state as recovery_trace_state,
    fetch_candidate_urls,
    invoke_fallback_tool as invoke_recovery_tool,
    quality_filtered_candidate_urls,
    ranked_candidates_from_result_text,
    try_fetch_fallback as try_fetch_recovery,
    try_find_text_fallback as try_find_text_recovery,
    try_search_fallback as try_search_recovery,
    unfetched_first_candidate_urls,
    with_fallback_traces as with_recovery_traces,
)

__all__ = [
    "RecoveryAttemptBudget",
    "RecoveryExecutionContext",
    "candidate_urls_from_state",
    "fallback_result_from_records",
    "fetch_candidate_urls",
    "invoke_recovery_tool",
    "quality_filtered_candidate_urls",
    "ranked_candidates_from_result_text",
    "recovery_trace_state",
    "try_fetch_recovery",
    "try_find_text_recovery",
    "try_search_recovery",
    "unfetched_first_candidate_urls",
    "with_recovery_traces",
]
