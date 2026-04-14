from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from hf_gaia_agent.graph.candidate_support import ranked_candidates_from_state
from hf_gaia_agent.graph.contracts import ToolInvocationResult
from hf_gaia_agent.graph.finalizer import WorkflowFinalizer
from hf_gaia_agent.graph.finalization_rules import TemporalRosterFinalizationRule
from hf_gaia_agent.graph.nudges import build_search_nudge, build_stuck_search_nudge
from hf_gaia_agent.graph.tool_policy import ToolPolicyEngine
from hf_gaia_agent.source_pipeline import (
    SourceCandidate,
    evidence_records_from_tool_output,
    parse_result_blocks,
)
from hf_gaia_agent.tools._payloads import (
    SearchResultPayload,
    TableExtractPayload,
    TableSectionPayload,
    serialize_tool_payloads,
)


class _FakeFinalizationServices:
    def __init__(self) -> None:
        self.finalization_rules = []

    def last_ai_message(self, messages):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def tool_derived_answer(self, messages, question):
        del messages, question
        return None

    def structured_answer_result(self, state, *, preferred_only=False):
        del state, preferred_only
        return None

    def structured_answer_from_state(self, state):
        del state
        return None, None, []

    def salvage_answer_from_evidence(self, state):
        del state
        return None

    def verify_answer_from_evidence(self, state):
        del state
        return None

    def top_grounded_evidence_records(self, state, *, limit=6):
        del state, limit
        return []


class _FakeToolPolicyServices:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict[str, object]]] = []
        self.tool_names = {"fetch_url"}

    def ranked_candidates_from_state(self, state):
        return ranked_candidates_from_state(state)

    def merge_ranked_candidates(self, existing, new_items, *, max_items=12):
        del max_items
        return [*existing, *new_items]

    def normalize_search_query(self, query):
        return query.strip().lower()

    def is_semantically_duplicate_search(self, signature, previous_signatures):
        return signature in previous_signatures

    def pick_best_unfetched_candidate(self, state, *, fetched_urls):
        del state, fetched_urls
        return None

    def pick_better_fetch_candidate(
        self,
        *,
        requested_url,
        profile,
        ranked_candidates,
        fetched_urls,
    ):
        del requested_url, profile, ranked_candidates, fetched_urls
        return None

    def execute_python_allowed(self, state):
        del state
        return True, "prompt"

    def article_to_paper_auto_links_result(
        self, *, tool_name, tool_args, result_text, profile
    ):
        del tool_name, tool_args, result_text, profile
        return None

    def text_span_auto_follow_candidate(
        self,
        *,
        tool_name,
        tool_args,
        result_text,
        profile,
        ranked_candidates,
        fetched_urls,
    ):
        del (
            tool_name,
            tool_args,
            result_text,
            profile,
            ranked_candidates,
            fetched_urls,
        )
        return None

    def invoke_tool(self, tool_name, tool_args):
        self.invocations.append((tool_name, tool_args))
        return ToolInvocationResult(
            text="Kind: page_text\nURL: https://example.com\nTitle: Example\n\nRecovered text",
            payloads=[],
        )


def test_workflow_finalizer_depends_on_service_interface() -> None:
    services = _FakeFinalizationServices()
    finalizer = WorkflowFinalizer(services)

    result = finalizer.finalize(
        {
            "question": "What is 6 * 7?",
            "messages": [AIMessage(content="[ANSWER]42[/ANSWER]")],
            "file_name": None,
            "local_file_path": None,
            "error": None,
            "reducer_used": None,
            "evidence_used": [],
            "recovery_reason": None,
        }
    )

    assert result["final_answer"] == "42"
    assert result["error"] is None


def test_temporal_roster_rule_no_longer_reinvokes_targeted_resolution() -> None:
    class _RuleServices:
        def tool_derived_answer(self, messages, question):
            del messages, question
            return None

    rule = TemporalRosterFinalizationRule()
    state = {
        "question": "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023?",
        "messages": [],
        "question_profile": {
            "name": "temporal_ordered_list",
            "expected_date": "as of July 2023",
        },
    }

    result = rule.finalize(
        state,
        "",
        services=_RuleServices(),
        error=None,
        recovery_reason=None,
    )

    assert result == {
        "final_answer": "",
        "error": "Date-sensitive roster answer lacked temporally grounded evidence.",
        "recovery_reason": "temporal_roster_evidence_missing",
    }


def test_tool_policy_engine_runs_with_explicit_services() -> None:
    services = _FakeToolPolicyServices()
    engine = ToolPolicyEngine(services)

    state = {
        "question": "Read the page directly.",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "fetch_url",
                        "args": {"url": "https://example.com"},
                    }
                ],
            )
        ],
        "tool_trace": [],
        "decision_trace": [],
        "ranked_candidates": [],
        "search_history_normalized": [],
        "structured_tool_outputs": [],
        "question_profile": {
            "name": "direct_url",
            "target_urls": (),
            "expected_domains": (),
            "preferred_tools": ("fetch_url",),
            "expected_date": None,
            "expected_author": None,
            "subject_name": None,
            "text_filter": None,
        },
    }

    result = engine.run(state)

    assert services.invocations == [("fetch_url", {"url": "https://example.com"})]
    assert result["tool_trace"] == ["fetch_url({'url': 'https://example.com'})"]
    assert result["structured_tool_outputs"][0]["tool_name"] == "fetch_url"
    assert "Recovered text" in result["messages"][0].content


def test_tool_policy_duplicate_query_prefers_unfetched_useful_candidates() -> None:
    services = _FakeToolPolicyServices()
    engine = ToolPolicyEngine(services)

    state = {
        "question": "Find the official winners list.",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "web_search",
                        "args": {"query": "official winners list"},
                    }
                ],
            )
        ],
        "tool_trace": [],
        "decision_trace": [],
        "ranked_candidates": [
            SourceCandidate(
                title="Winners | Official Site",
                url="https://example.com/winners",
                snippet="Official winners page",
                origin_tool="web_search",
                score=72,
                reasons=("expected_domain", "token_overlap:2"),
            ),
            SourceCandidate(
                title="Downtown menu",
                url="https://noise.example.com/menu",
                snippet="Restaurant menu order online",
                origin_tool="web_search",
                score=-20,
                reasons=("commercial_noise_penalty",),
            ),
        ],
        "search_history_normalized": ["official winners list"],
        "structured_tool_outputs": [],
        "question_profile": {
            "name": "table_lookup",
            "target_urls": (),
            "expected_domains": ("example.com",),
            "preferred_tools": ("web_search", "fetch_url"),
            "expected_date": None,
            "expected_author": None,
            "subject_name": None,
            "text_filter": None,
        },
    }

    result = engine.run(state)

    assert services.invocations == []
    assert "https://example.com/winners" in result["messages"][0].content
    assert "noise.example.com/menu" not in result["messages"][0].content


def test_tool_policy_duplicate_query_detects_exhausted_candidates() -> None:
    services = _FakeToolPolicyServices()
    engine = ToolPolicyEngine(services)

    state = {
        "question": "Find the official winners list.",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "web_search",
                        "args": {"query": "official winners list"},
                    }
                ],
            )
        ],
        "tool_trace": ["fetch_url({'url': 'https://example.com/winners'})"],
        "decision_trace": ["tool:fetch_url"],
        "ranked_candidates": [
            SourceCandidate(
                title="Winners | Official Site",
                url="https://example.com/winners",
                snippet="Official winners page",
                origin_tool="web_search",
                score=72,
                reasons=("expected_domain", "token_overlap:2"),
            )
        ],
        "search_history_normalized": ["official winners list"],
        "structured_tool_outputs": [],
        "question_profile": {
            "name": "table_lookup",
            "target_urls": (),
            "expected_domains": ("example.com",),
            "preferred_tools": ("web_search", "fetch_url"),
            "expected_date": None,
            "expected_author": None,
            "subject_name": None,
            "text_filter": None,
        },
    }

    result = engine.run(state)

    assert services.invocations == []
    assert "ALL CANDIDATES EXHAUSTED" in result["messages"][0].content


def test_build_search_nudge_skips_low_quality_ranked_candidates() -> None:
    state = {
        "question": "Find the official winners list.",
        "messages": [],
        "decision_trace": ["tool:web_search", "tool:web_search"],
        "tool_trace": [],
        "ranked_candidates": [
            {
                "title": "Downtown menu",
                "url": "https://noise.example.com/menu",
                "snippet": "Restaurant menu order online",
                "origin_tool": "web_search",
                "score": -20,
                "reasons": ("commercial_noise_penalty",),
            }
        ],
    }

    nudge = build_search_nudge(state)

    assert nudge is not None
    assert "low-quality" in nudge
    assert "noise.example.com/menu" not in nudge


def test_build_stuck_search_nudge_triggers_when_repeated_search_is_exhausted() -> None:
    state = {
        "question": "Find the official winners list.",
        "messages": [ToolMessage(content="Search results", tool_call_id="search-1", name="web_search")],
        "tool_trace": ["fetch_url({'url': 'https://example.com/winners'})"],
        "search_history_normalized": [
            "official winners list",
            "official winners list",
        ],
        "ranked_candidates": [
            {
                "title": "Winners | Official Site",
                "url": "https://example.com/winners",
                "snippet": "Official winners page",
                "origin_tool": "web_search",
                "score": 72,
                "reasons": ("expected_domain", "token_overlap:2"),
            }
        ],
    }
    nudge = build_stuck_search_nudge(state)

    assert nudge is not None
    assert "SEARCH LOOP DETECTED" in nudge
    assert "Change strategy now" in nudge


def test_parse_result_blocks_accepts_structured_payloads() -> None:
    payloads = serialize_tool_payloads(
        [
            SearchResultPayload(
                title="Example Result",
                url="https://example.com/result",
                snippet="Useful snippet",
                rank=1,
            )
        ]
    )

    candidates = parse_result_blocks(payloads, origin_tool="web_search")

    assert len(candidates) == 1
    assert candidates[0].url == "https://example.com/result"
    assert candidates[0].title == "Example Result"


def test_evidence_records_accept_structured_table_payloads() -> None:
    payloads = serialize_tool_payloads(
        [
            TableExtractPayload(
                url="https://example.com/roster",
                title="Roster page",
                tables=(
                    TableSectionPayload(
                        content="Table 1\nCaption: Pitchers\n18 | Yoshida\n19 | Tamai",
                        caption="Pitchers",
                        index=1,
                    ),
                ),
            )
        ]
    )

    records = evidence_records_from_tool_output("extract_tables_from_url", payloads)

    assert len(records) == 1
    assert records[0].source_url == "https://example.com/roster"
    assert records[0].title_or_caption == "Pitchers"
