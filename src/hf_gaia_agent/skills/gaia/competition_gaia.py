"""GAIA competition skill."""

from __future__ import annotations

from ...fallbacks.competition import CompetitionFallback


class CompetitionGaiaSkill(CompetitionFallback):
    name = "competition_gaia"

    def run(self, state):
        result = super().run(state)
        if result is None:
            return None
        result["skill_used"] = self.name
        result["skill_trace"] = [self.name]
        return result
