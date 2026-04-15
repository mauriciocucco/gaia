"""GAIA skill for grounded botanical set classification."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from ...botanical_classification import build_botanical_canonical_state
from ...core.recoveries.utils import (
    RecoveryAttemptBudget,
    fetch_candidate_urls,
    invoke_recovery_tool,
    ranked_candidates_from_result_text,
    recovery_trace_state,
    try_search_recovery,
)
from ...graph.routing import extract_prompt_list_items, question_profile_from_state
from ...source_pipeline import EvidenceRecord, evidence_records_from_tool_output
from ..set_classification import build_set_classification_result


class BotanicalGaiaSkill:
    name = "botanical_gaia"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name

    def applies(self, state, profile) -> bool:
        labels = profile.classification_labels or {}
        return (
            profile.name == "list_item_classification"
            and labels.get("include") == "vegetable"
            and labels.get("exclude") == "fruit"
        )

    def run(self, state):
        profile = question_profile_from_state(state)
        if not self.applies(state, profile):
            return None

        decision_trace = list(state.get("decision_trace") or [])
        decision_trace.append("skill:botanical_gaia:profile_match")

        if "web_search" not in self._tools_by_name or "fetch_url" not in self._tools_by_name:
            decision_trace.append("skill:botanical_gaia:missing_tools")
            return {"decision_trace": decision_trace}

        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        context.decision_trace.append("skill:botanical_gaia:profile_match")

        items = list(profile.prompt_items or extract_prompt_list_items(state["question"]))
        context.decision_trace.append(f"skill:botanical_gaia:items_extracted={len(items)}")
        if not items:
            context.decision_trace.append("skill:botanical_gaia:no_items")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        existing_records = _botanical_existing_records(state)
        has_existing_leads = (
            any(isinstance(message, ToolMessage) for message in state.get("messages", []))
            or bool(state.get("ranked_candidates"))
            or bool(existing_records)
        )
        if not has_existing_leads and len(items) > 8:
            resolution = build_botanical_canonical_state(items, existing_records)
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        all_records = existing_records
        resolution = build_botanical_canonical_state(items, all_records)
        for item in resolution.unresolved_items:
            item_budget = RecoveryAttemptBudget(remaining_searches=2, remaining_fetches=2)
            for query in (
                f"{item} botanical fruit or vegetable",
                f"{item} botany fruit vegetable",
            ):
                search_text = try_search_recovery(
                    context=context,
                    query=query,
                    max_results=5,
                    budget=item_budget,
                )
                if not search_text:
                    continue
                candidate_urls = [
                    candidate.url
                    for candidate in ranked_candidates_from_result_text(
                        context=context,
                        result_text=search_text,
                        origin_tool="web_search",
                    )
                ]
                for candidate_url in fetch_candidate_urls(
                    context=context,
                    candidate_urls=candidate_urls,
                    max_urls=2,
                ):
                    if not item_budget.consume_fetch():
                        break
                    fetched = invoke_recovery_tool(
                        context=context,
                        tool_name="fetch_url",
                        tool_args={"url": candidate_url},
                    )
                    all_records.extend(evidence_records_from_tool_output("fetch_url", fetched))
                    resolution = build_botanical_canonical_state(items, all_records)
                    if item not in resolution.unresolved_items:
                        break
                if item not in resolution.unresolved_items:
                    break

        if not resolution.is_closed:
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return {
                "decision_trace": context.decision_trace,
                "tool_trace": context.tool_trace,
            }

        context.decision_trace.append("skill:botanical_gaia:resolved")
        return build_set_classification_result(
            skill_name=self.name,
            included_items=resolution.included_items,
            records=resolution.used_records,
            reducer_used="botanical_classification",
            tool_trace=context.tool_trace,
            decision_trace=context.decision_trace,
        )


def _botanical_existing_records(state) -> list[EvidenceRecord]:
    supporting_records: list[EvidenceRecord] = []
    seen_urls: set[str] = set()
    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in {"fetch_url", "find_text_in_url"}:
            continue
        for record in evidence_records_from_tool_output(tool_name, str(message.content or "")):
            if record.source_url and record.source_url in seen_urls:
                continue
            supporting_records.append(record)
            if record.source_url:
                seen_urls.add(record.source_url)
    return supporting_records
