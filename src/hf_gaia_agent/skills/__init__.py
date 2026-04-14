"""Skill registry."""

from __future__ import annotations

from typing import Any

from .base import Skill
from .gaia.botanical_gaia import BotanicalGaiaSkill
from .gaia.competition_gaia import CompetitionGaiaSkill
from .gaia.role_chain_gaia import RoleChainGaiaSkill
from .temporal_ordered_list import TemporalOrderedListSkill

__all__ = [
    "Skill",
    "BotanicalGaiaSkill",
    "CompetitionGaiaSkill",
    "RoleChainGaiaSkill",
    "TemporalOrderedListSkill",
    "build_skills",
]


def build_skills(
    tools_by_name: dict[str, Any],
    answer_model: Any,
    *,
    include_benchmark_specific: bool = True,
) -> list[Skill]:
    skills: list[Skill] = [TemporalOrderedListSkill()]
    if include_benchmark_specific:
        skills.extend(
            [
                BotanicalGaiaSkill(tools_by_name),
                CompetitionGaiaSkill(tools_by_name),
                RoleChainGaiaSkill(tools_by_name, answer_model),
            ]
        )
    return skills
