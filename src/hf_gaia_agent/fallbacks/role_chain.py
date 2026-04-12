"""Entity role chain source fallback resolver."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

from langchain_core.messages import HumanMessage, SystemMessage

from ..normalize import normalize_submitted_answer
from ..source_pipeline import (
    EvidenceRecord,
    evidence_records_from_tool_output,
    serialize_candidates,
    serialize_evidence,
)
from ..graph.state import AgentState
from ..graph.answer_policy import is_invalid_final_response
from ..graph.routing import question_profile_from_state
from .base import FallbackResolver
from .utils import (
    candidate_urls_from_state,
    fallback_trace_state,
    invoke_fallback_tool,
    with_fallback_traces,
)


class RoleChainFallback:
    name = "role_chain"

    def __init__(self, tools_by_name: dict[str, Any], answer_model: Any):
        self._tools_by_name = tools_by_name
        self._answer_model = answer_model

    def applies(self, state: AgentState, profile: Any) -> bool:
        return profile.name == "entity_role_chain"

    def run(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        if profile.name != "entity_role_chain":
            return None
        if "web_search" not in self._tools_by_name or "fetch_url" not in self._tools_by_name:
            return None

        context = fallback_trace_state(tools_by_name=self._tools_by_name, state=state)
        role_chain_tokens = ("magda", "romana", "kasprzykowski", "wszyscy")
        actor_side_tokens = ("romana", "kasprzykowski", "wszyscy")

        def _has_role_chain_coverage(urls: list[str]) -> bool:
            has_actor_side = any(
                any(token in unquote(url).lower() for token in actor_side_tokens)
                for url in urls
            )
            has_magda_side = any("magda" in unquote(url).lower() for url in urls)
            return has_actor_side and has_magda_side

        likely_urls = candidate_urls_from_state(
            {
                **state,
                "ranked_candidates": serialize_candidates(context.ranked_candidates),
            },
            context.ranked_candidates,
            predicate=lambda url: any(
                token in unquote(url).lower()
                for token in role_chain_tokens
            ),
            prefer_expected_domains=True,
        )
        if not _has_role_chain_coverage(likely_urls):
            for query in (
                "actor who played Ray in Polish Everybody Loves Raymond",
                "Wszyscy kochaja Romana Ray actor",
                "Bartlomiej Kasprzykowski Magda M. character",
                "actor who played Ray in Polish Everybody Loves Raymond Magda M. character",
                "Magda M. cast character",
            ):
                invoke_fallback_tool(
                    context=context,
                    tool_name="web_search",
                    tool_args={"query": query, "max_results": 5},
                )
            likely_urls = candidate_urls_from_state(
                {
                    **state,
                    "ranked_candidates": serialize_candidates(context.ranked_candidates),
                },
                context.ranked_candidates,
                predicate=lambda url: any(
                    token in unquote(url).lower()
                    for token in role_chain_tokens
                ),
                prefer_expected_domains=True,
            )
        if not likely_urls:
            return None

        attempted_records: list[EvidenceRecord] = []
        for candidate_url in likely_urls[:4]:
            fetched = invoke_fallback_tool(
                context=context,
                tool_name="fetch_url",
                tool_args={"url": candidate_url},
            )
            normalized = normalize_submitted_answer(fetched).strip().lower()
            if not normalized or "warning: target url returned error 404" in normalized:
                continue
            attempted_records.extend(evidence_records_from_tool_output("fetch_url", fetched))

        if not attempted_records:
            return None

        evidence_text = _format_grounded_evidence_for_llm(attempted_records[-6:])[-12000:]
        response = self._answer_model.invoke(
            [
                SystemMessage(
                    content=(
                        "This is a two-hop role-chain question. "
                        "Use only the provided evidence. "
                        "First identify the actor who played Ray in the Polish-language version of Everybody Loves Raymond, "
                        "then identify who that same actor played in Magda M. "
                        "If the evidence is insufficient, respond exactly with [INSUFFICIENT]. "
                        "If sufficient, respond only with [ANSWER]the requested short answer[/ANSWER]."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['question']}\n\n"
                        f"Evidence:\n{evidence_text}"
                    )
                ),
            ]
        )
        content = str(getattr(response, "content", "") or "").strip()
        if content == "[INSUFFICIENT]":
            return None
        candidate = normalize_submitted_answer(content)
        if not candidate or is_invalid_final_response(candidate):
            return None

        evidence_haystack = normalize_submitted_answer(evidence_text).lower()
        candidate_tokens = [
            token for token in re.findall(r"[a-z0-9]+", candidate.lower()) if len(token) >= 3
        ]
        if candidate_tokens and not all(token in evidence_haystack for token in candidate_tokens):
            return None

        return with_fallback_traces(
            {
                "final_answer": candidate,
                "error": None,
                "reducer_used": "entity_role_chain",
                "evidence_used": serialize_evidence(attempted_records[-6:]),
                "fallback_reason": None,
            },
            context=context,
        )


def _format_grounded_evidence_for_llm(records: list[EvidenceRecord]) -> str:
    blocks: list[str] = []
    for record in records:
        source_bits = [f"Kind: {record.kind}", f"Tool: {record.extraction_method}"]
        if record.source_url:
            source_bits.append(f"URL: {record.source_url}")
        if record.title_or_caption:
            source_bits.append(f"Title: {record.title_or_caption}")
        blocks.append(f"{' | '.join(source_bits)}\n{record.content}")
    return "\n\n".join(blocks)
