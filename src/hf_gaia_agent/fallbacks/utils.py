"""Legacy compatibility layer for fallback helper imports.

The canonical implementations live in :mod:`hf_gaia_agent.core.recoveries._shared`.
This module re-exports them so older tests or imports keep working while the
main architecture depends on the newer recovery/skill layers.
"""

from __future__ import annotations

from ..core.recoveries._shared import (
    RecoveryAttemptBudget as FallbackAttemptBudget,
    RecoveryExecutionContext as FallbackExecutionContext,
    candidate_urls_from_state,
    fetch_candidate_urls,
    invoke_recovery_tool as invoke_fallback_tool,
    quality_filtered_candidate_urls,
    ranked_candidates_from_result_text,
    recovery_result_from_records as fallback_result_from_records,
    recovery_trace_state as fallback_trace_state,
    try_fetch_recovery as try_fetch_fallback,
    try_find_text_recovery as try_find_text_fallback,
    try_search_recovery as try_search_fallback,
    unfetched_first_candidate_urls,
    with_recovery_traces as with_fallback_traces,
)

__all__ = [
    "FallbackAttemptBudget",
    "FallbackExecutionContext",
    "candidate_urls_from_state",
    "fallback_result_from_records",
    "fallback_trace_state",
    "fetch_candidate_urls",
    "invoke_fallback_tool",
    "quality_filtered_candidate_urls",
    "ranked_candidates_from_result_text",
    "try_fetch_fallback",
    "try_find_text_fallback",
    "try_search_fallback",
    "unfetched_first_candidate_urls",
    "with_fallback_traces",
]
