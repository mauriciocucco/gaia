"""Legacy compatibility wrapper for :class:`RoleChainGaiaSkill`."""

from __future__ import annotations

from ..skills.gaia.role_chain_gaia import RoleChainGaiaSkill


class RoleChainFallback(RoleChainGaiaSkill):
    """Backward-compatible alias for the canonical GAIA role-chain skill."""

    name = "role_chain"
