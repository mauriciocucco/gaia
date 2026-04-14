"""GAIA entity role-chain skill."""

from __future__ import annotations

from ...fallbacks.role_chain import RoleChainFallback


class RoleChainGaiaSkill(RoleChainFallback):
    name = "role_chain_gaia"

    def run(self, state):
        result = super().run(state)
        if result is None:
            return None
        result["skill_used"] = self.name
        result["skill_trace"] = [self.name]
        return result
