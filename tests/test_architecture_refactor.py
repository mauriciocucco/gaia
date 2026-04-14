from __future__ import annotations

from inspect import signature

from hf_gaia_agent.adapters import build_source_adapters
from hf_gaia_agent.core.recoveries.article_to_paper import ArticleToPaperRecovery
from hf_gaia_agent.core.recoveries.text_span import TextSpanRecovery
from hf_gaia_agent.core.recoveries import build_core_recoveries
from hf_gaia_agent.graph.services import GraphWorkflowServices
from hf_gaia_agent.graph.workflow import GaiaGraphAgent
from hf_gaia_agent.hooks import BaseAgentHook
from hf_gaia_agent.skills import build_skills
from hf_gaia_agent.skills.gaia.competition_gaia import CompetitionGaiaSkill
from hf_gaia_agent.skills.gaia.role_chain_gaia import RoleChainGaiaSkill


class _DummyModel:
    def invoke(self, messages):
        del messages
        class _Resp:
            content = "[INSUFFICIENT]"
        return _Resp()


def test_resolution_components_use_canonical_types() -> None:
    tools_by_name: dict[str, object] = {}
    model = _DummyModel()

    core_recoveries = build_core_recoveries(tools_by_name, model)
    skills = build_skills(tools_by_name, model)
    source_adapters = build_source_adapters(tools_by_name)

    assert any(isinstance(recovery, ArticleToPaperRecovery) for recovery in core_recoveries)
    assert any(isinstance(recovery, TextSpanRecovery) for recovery in core_recoveries)
    assert any(isinstance(skill, CompetitionGaiaSkill) for skill in skills)
    assert any(isinstance(skill, RoleChainGaiaSkill) for skill in skills)
    assert source_adapters


def test_graph_services_exposes_only_canonical_resolution_methods() -> None:
    model = _DummyModel()
    tools_by_name: dict[str, object] = {}
    services = GraphWorkflowServices(
        answer_model=model,
        tools_by_name=tools_by_name,
        core_recoveries=build_core_recoveries(tools_by_name, model),
        skills=build_skills(tools_by_name, model),
        source_adapters=build_source_adapters(tools_by_name),
        hook=BaseAgentHook(),
    )

    assert services.run_resolution_pipeline({"question": "q", "messages": []}) is None
    assert services.run_targeted_resolution("roster", {"question": "q", "messages": [], "ranked_candidates": []}) is None
    assert not hasattr(services, "run_fallback_resolvers")
    assert not hasattr(services, "run_named_fallback")


def test_graph_agent_constructor_no_longer_accepts_legacy_benchmark_alias() -> None:
    params = signature(GaiaGraphAgent.__init__).parameters

    assert "include_benchmark_extensions" in params
    assert "include_benchmark_fallbacks" not in params
