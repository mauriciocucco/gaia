"""Generic skill for temporally grounded ordered-list lookups."""

from __future__ import annotations

from ..graph.evidence_support import grounded_temporal_ordered_list_answer
from ..graph.routing import question_profile_from_state
from ..source_pipeline import serialize_evidence


class TemporalOrderedListSkill:
    name = "temporal_ordered_list"

    def applies(self, state, profile) -> bool:
        return profile.name == "temporal_ordered_list"

    def run(self, state):
        profile = question_profile_from_state(state)
        if profile.name != "temporal_ordered_list":
            return None
        answer, records = grounded_temporal_ordered_list_answer(state, with_records=True)
        if not answer:
            return None
        return {
            "final_answer": answer,
            "error": None,
            "reducer_used": "roster_neighbor",
            "skill_used": self.name,
            "skill_trace": [self.name],
            "evidence_used": serialize_evidence(records),
            "fallback_reason": None,
        }
