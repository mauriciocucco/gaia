"""Legacy compatibility wrapper for :class:`CompetitionGaiaSkill`."""

from __future__ import annotations

from ..skills.gaia.competition_gaia import CompetitionGaiaSkill


class CompetitionFallback(CompetitionGaiaSkill):
    """Backward-compatible alias for the canonical GAIA competition skill."""

    name = "competition"
