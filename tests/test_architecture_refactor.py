from __future__ import annotations

from hf_gaia_agent.core.recoveries.article_to_paper import ArticleToPaperRecovery
from hf_gaia_agent.core.recoveries.text_span import TextSpanRecovery
from hf_gaia_agent.fallbacks.article_to_paper import ArticleToPaperFallback
from hf_gaia_agent.fallbacks.text_span import TextSpanFallback
from hf_gaia_agent.graph.services import GraphWorkflowServices
from hf_gaia_agent.skills.gaia.competition_gaia import CompetitionGaiaSkill
from hf_gaia_agent.skills.gaia.role_chain_gaia import RoleChainGaiaSkill
from hf_gaia_agent.fallbacks.competition import CompetitionFallback
from hf_gaia_agent.fallbacks.role_chain import RoleChainFallback


class _NoopHook:
    def on_tool_start(self, tool_name, tool_args):
        pass

    def on_tool_end(self, tool_name, result):
        pass

    def on_solve_start(self, task_id, question):
        pass

    def on_solve_end(self, task_id, result):
        pass


class _DummyModel:
    def invoke(self, messages):
        del messages
        class _Resp:
            content = "[INSUFFICIENT]"
        return _Resp()


def test_legacy_fallbacks_are_compatibility_wrappers() -> None:
    assert issubclass(ArticleToPaperFallback, ArticleToPaperRecovery)
    assert issubclass(TextSpanFallback, TextSpanRecovery)
    assert issubclass(CompetitionFallback, CompetitionGaiaSkill)
    assert issubclass(RoleChainFallback, RoleChainGaiaSkill)


def test_graph_services_exposes_resolution_and_legacy_compatibility_methods() -> None:
    services = GraphWorkflowServices(
        answer_model=_DummyModel(),
        tools_by_name={},
        core_recoveries=[],
        skills=[],
        source_adapters=[],
        hook=_NoopHook(),
    )

    assert services.run_resolution_pipeline({"question": "q", "messages": []}) is None
    assert services.run_fallback_resolvers({"question": "q", "messages": []}) is None
    assert services.run_targeted_resolution("roster", {"question": "q", "messages": [], "ranked_candidates": []}) is None
    assert services.run_named_fallback("roster", {"question": "q", "messages": [], "ranked_candidates": []}) is None
