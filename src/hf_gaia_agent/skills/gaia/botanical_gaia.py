"""GAIA skill for grounded botanical set classification."""

from __future__ import annotations

import re
from urllib.parse import unquote

from langchain_core.messages import ToolMessage

from ...botanical_aliases import botanical_aliases_for_item
from ...botanical_classification import build_botanical_canonical_state
from ...core.recoveries.utils import (
    RecoveryAttemptBudget,
    fetch_candidate_urls,
    invoke_recovery_tool,
    ranked_candidates_from_result_text,
    recovery_trace_state,
    try_search_recovery,
)
from ...graph.candidate_support import is_botanical_recipe_noise_url
from ...graph.routing import (
    extract_prompt_list_items,
    normalize_botanical_text,
    question_profile_from_state,
)
from ...source_pipeline import (
    EvidenceRecord,
    SourceCandidate,
    evidence_records_from_tool_output,
    serialize_candidates,
    serialize_evidence,
)
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

        has_wikipedia_tools = all(
            tool_name in self._tools_by_name
            for tool_name in ("search_wikipedia", "fetch_wikipedia_page")
        )
        has_web_tools = all(
            tool_name in self._tools_by_name for tool_name in ("web_search", "fetch_url")
        )
        if not has_wikipedia_tools and not has_web_tools:
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
        botanical_item_status = _botanical_item_status_from_state(state)
        botanical_search_history = list(state.get("botanical_search_history") or [])
        has_existing_leads = (
            any(isinstance(message, ToolMessage) for message in state.get("messages", []))
            or bool(state.get("ranked_candidates"))
            or bool(existing_records)
            or bool(state.get("botanical_partial_records"))
            or bool(botanical_item_status)
            or bool(botanical_search_history)
        )
        if not has_existing_leads and len(items) > 8:
            resolution = build_botanical_canonical_state(items, existing_records)
            botanical_item_status = _sync_botanical_item_status(
                botanical_item_status,
                resolution=resolution,
            )
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return _partial_result(
                context=context,
                records=existing_records,
                botanical_item_status=botanical_item_status,
                botanical_search_history=botanical_search_history,
            )

        all_records = list(existing_records)
        resolution = build_botanical_canonical_state(items, all_records)
        botanical_item_status = _sync_botanical_item_status(
            botanical_item_status,
            resolution=resolution,
        )

        for item in resolution.unresolved_items:
            if botanical_item_status.get(item, {}).get("resolved"):
                continue
            item_budget = RecoveryAttemptBudget(remaining_searches=3, remaining_fetches=3)

            if has_wikipedia_tools and _should_try_wikipedia_first(item):
                for alias in (None, *botanical_aliases_for_item(item)):
                    resolved = _attempt_wikipedia_stage(
                        item=item,
                        alias=alias,
                        context=context,
                        item_budget=item_budget,
                        items=items,
                        all_records=all_records,
                        botanical_item_status=botanical_item_status,
                        botanical_search_history=botanical_search_history,
                    )
                    resolution = build_botanical_canonical_state(items, all_records)
                    botanical_item_status = _sync_botanical_item_status(
                        botanical_item_status,
                        resolution=resolution,
                    )
                    if resolved:
                        break
                if botanical_item_status.get(item, {}).get("resolved"):
                    continue

            if not has_web_tools:
                continue

            for query in (
                f"{item} botanical fruit or vegetable",
                f"{item} botany fruit vegetable",
            ):
                stage = f"broad_web_query:{query.removeprefix(f'{item} ')}"
                if _botanical_stage_attempted(botanical_item_status, item, stage):
                    continue
                _record_botanical_stage(
                    botanical_item_status,
                    botanical_search_history,
                    item=item,
                    stage=stage,
                    value=query,
                )
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
                        predicate=lambda candidate: _search_candidate_matches_item(
                            item, candidate
                        ),
                    )
                ]
                candidate_urls = [
                    candidate_url
                    for candidate_url in candidate_urls
                    if not is_botanical_recipe_noise_url(candidate_url)
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
                    all_records[:] = _merge_evidence_records(
                        all_records,
                        evidence_records_from_tool_output("fetch_url", fetched),
                    )
                    resolution = build_botanical_canonical_state(items, all_records)
                    botanical_item_status = _sync_botanical_item_status(
                        botanical_item_status,
                        resolution=resolution,
                    )
                    if item not in resolution.unresolved_items:
                        break
                if item not in resolution.unresolved_items:
                    break

            if item in resolution.unresolved_items:
                current = botanical_item_status.setdefault(
                    item,
                    {
                        "resolved": False,
                        "outcome": None,
                        "attempted_stages": [],
                        "last_reason": None,
                    },
                )
                current["last_reason"] = "needs_stronger_exclusion"

        resolution = build_botanical_canonical_state(items, all_records)
        botanical_item_status = _sync_botanical_item_status(
            botanical_item_status,
            resolution=resolution,
        )

        if not resolution.is_closed:
            if resolution.unresolved_items:
                unresolved = "|".join(resolution.unresolved_items)
                context.decision_trace.append(f"skill:botanical_gaia:unresolved={unresolved}")
            context.decision_trace.append("skill:botanical_gaia:aborted_partial_resolution")
            return _partial_result(
                context=context,
                records=all_records,
                botanical_item_status=botanical_item_status,
                botanical_search_history=botanical_search_history,
            )

        context.decision_trace.append("skill:botanical_gaia:resolved")
        return build_set_classification_result(
            skill_name=self.name,
            included_items=resolution.included_items,
            records=resolution.used_records,
            reducer_used="botanical_classification",
            tool_trace=context.tool_trace,
            decision_trace=context.decision_trace,
        )


def _partial_result(
    *,
    context,
    records: list[EvidenceRecord],
    botanical_item_status: dict[str, dict[str, object]],
    botanical_search_history: list[str],
) -> dict[str, object]:
    return {
        "decision_trace": context.decision_trace,
        "tool_trace": context.tool_trace,
        "ranked_candidates": serialize_candidates(context.ranked_candidates),
        "search_history_fingerprints": context.search_history,
        "botanical_partial_records": serialize_evidence(records),
        "botanical_item_status": botanical_item_status,
        "botanical_search_history": botanical_search_history,
    }


def _attempt_wikipedia_stage(
    *,
    item: str,
    alias: str | None,
    context,
    item_budget: RecoveryAttemptBudget,
    items: list[str],
    all_records: list[EvidenceRecord],
    botanical_item_status: dict[str, dict[str, object]],
    botanical_search_history: list[str],
) -> bool:
    stage_prefix = "wiki_exact" if alias is None else f"wiki_alias:{alias}"
    search_stage = f"{stage_prefix}_search"
    if _botanical_stage_attempted(botanical_item_status, item, search_stage):
        return False
    wiki_query = alias or _wikipedia_query_for_item(item)
    _record_botanical_stage(
        botanical_item_status,
        botanical_search_history,
        item=item,
        stage=search_stage,
        value=wiki_query,
    )
    wiki_search = try_search_recovery(
        context=context,
        query=wiki_query,
        max_results=5,
        budget=item_budget,
        tool_name="search_wikipedia",
    )
    if not wiki_search:
        return False
    wiki_candidates = ranked_candidates_from_result_text(
        context=context,
        result_text=wiki_search,
        origin_tool="search_wikipedia",
        predicate=lambda candidate: _wikipedia_candidate_matches_item(
            item, candidate, alias=alias
        ),
    )
    best_wiki_candidate = (
        max(
            wiki_candidates,
            key=lambda candidate: _wikipedia_candidate_selection_key(
                item, candidate, alias=alias
            ),
        )
        if wiki_candidates
        else None
    )
    wiki_title = _candidate_title(best_wiki_candidate)
    fetch_stage = f"{stage_prefix}_fetch"
    if (
        not best_wiki_candidate
        or not wiki_title
        or _botanical_stage_attempted(botanical_item_status, item, fetch_stage)
        or not item_budget.consume_fetch()
    ):
        return False
    _record_botanical_stage(
        botanical_item_status,
        botanical_search_history,
        item=item,
        stage=fetch_stage,
        value=wiki_title,
    )
    fetched = invoke_recovery_tool(
        context=context,
        tool_name="fetch_wikipedia_page",
        tool_args={"title": wiki_title},
    )
    all_records[:] = _merge_evidence_records(
        all_records,
        evidence_records_from_tool_output("fetch_wikipedia_page", fetched),
    )
    resolution = build_botanical_canonical_state(items, all_records)
    botanical_item_status.update(
        _sync_botanical_item_status(botanical_item_status, resolution=resolution)
    )
    return item not in resolution.unresolved_items


def _botanical_existing_records(state) -> list[EvidenceRecord]:
    supporting_records: list[EvidenceRecord] = []
    seen_urls: set[str] = set()
    for message in state["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        if tool_name not in {"fetch_url", "find_text_in_url", "fetch_wikipedia_page"}:
            continue
        for record in evidence_records_from_tool_output(tool_name, str(message.content or "")):
            if record.source_url and record.source_url in seen_urls:
                continue
            supporting_records.append(record)
            if record.source_url:
                seen_urls.add(record.source_url)
    for raw_record in state.get("botanical_partial_records") or []:
        if not isinstance(raw_record, dict):
            continue
        try:
            record = EvidenceRecord(
                kind=str(raw_record.get("kind", "")),
                source_url=str(raw_record.get("source_url", "")),
                source_type=str(raw_record.get("source_type", "")),
                adapter_name=str(raw_record.get("adapter_name", "")),
                content=str(raw_record.get("content", "")),
                title_or_caption=str(raw_record.get("title_or_caption", "")),
                confidence=float(raw_record.get("confidence", 0.0)),
                extraction_method=str(raw_record.get("extraction_method", "")),
                derived_from=tuple(raw_record.get("derived_from") or ()),
            )
        except Exception:
            continue
        if record.source_url and record.source_url in seen_urls:
            continue
        supporting_records.append(record)
        if record.source_url:
            seen_urls.add(record.source_url)
    return supporting_records


def _should_try_wikipedia_first(item: str) -> bool:
    normalized = _normalized_botanical_query_tokens(item)
    if not normalized:
        return False
    if len(normalized.split()) > 3:
        return False
    return bool(re.fullmatch(r"[a-z ]+", normalized))


def _wikipedia_query_for_item(item: str) -> str:
    return _normalized_botanical_query_tokens(item)


def _normalized_botanical_query_tokens(item: str) -> str:
    ignored = {"fresh", "whole", "raw", "ripe", "dried"}
    tokens = [
        token for token in normalize_botanical_text(item).split() if token and token not in ignored
    ]
    return " ".join(tokens)


def _wikipedia_candidate_matches_item(
    item: str,
    candidate: SourceCandidate,
    *,
    alias: str | None = None,
) -> bool:
    title = _candidate_title(candidate)
    return bool(title and _item_text_matches(item, title, alias=alias))


def _search_candidate_matches_item(item: str, candidate: SourceCandidate) -> bool:
    haystack = "\n".join((candidate.title, candidate.snippet, unquote(candidate.url))).strip()
    return _item_text_matches(item, haystack)


def _candidate_title(candidate: SourceCandidate | None) -> str | None:
    if candidate is None:
        return None
    title = str(candidate.title or "").strip()
    return title or None


def _wikipedia_candidate_selection_key(
    item: str,
    candidate: SourceCandidate,
    *,
    alias: str | None = None,
) -> tuple[int, int, int]:
    title = _candidate_title(candidate)
    if not title:
        return (-1, -999, -999)
    normalized_title = normalize_botanical_text(title)
    title_tokens = [token for token in normalized_title.split() if token != "disambiguation"]
    if not title_tokens:
        return (-1, -999, -999)
    token_groups = _item_token_groups(alias or item)
    if not token_groups or not all(
        any(token in variants for token in title_tokens) for variants in token_groups
    ):
        return (-1, -999, -999)
    canonical_query = _normalized_botanical_query_tokens(alias or item)
    singular_query = " ".join(sorted(variants, key=len)[0] for variants in token_groups)
    exactness = 2 if normalized_title in {canonical_query, singular_query} else 1
    matched_tokens = {
        token
        for token in title_tokens
        if any(token in variants for variants in token_groups)
    }
    extra_penalty = -len([token for token in title_tokens if token not in matched_tokens])
    return (exactness, extra_penalty, -len(title_tokens))


def _item_text_matches(item: str, text: str, *, alias: str | None = None) -> bool:
    normalized = normalize_botanical_text(text)
    candidate_groups = [_item_token_groups(item)]
    if alias:
        candidate_groups.append(_item_token_groups(alias))
    else:
        candidate_groups.extend(
            _item_token_groups(candidate_alias)
            for candidate_alias in botanical_aliases_for_item(item)
        )
    return any(
        token_groups
        and all(
            any(variant in normalized for variant in variants)
            for variants in token_groups
        )
        for token_groups in candidate_groups
    )


def _item_token_groups(item: str) -> list[set[str]]:
    ignored = {"fresh", "whole", "raw", "ripe", "dried"}
    groups: list[set[str]] = []
    for token in normalize_botanical_text(item).split():
        if token in ignored:
            continue
        variants = {token}
        if token.endswith("ies") and len(token) > 4:
            variants.add(token[:-3] + "y")
        if token.endswith("oes") and len(token) > 4:
            variants.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            variants.add(token[:-1])
        groups.append(variants)
    return groups


def _botanical_item_status_from_state(state) -> dict[str, dict[str, object]]:
    raw_status = state.get("botanical_item_status") or {}
    if not isinstance(raw_status, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for item, status in raw_status.items():
        if not isinstance(status, dict):
            continue
        normalized[str(item)] = {
            "resolved": bool(status.get("resolved")),
            "outcome": status.get("outcome"),
            "attempted_stages": list(status.get("attempted_stages") or []),
            "last_reason": status.get("last_reason"),
        }
    return normalized


def _sync_botanical_item_status(
    botanical_item_status: dict[str, dict[str, object]],
    *,
    resolution,
) -> dict[str, dict[str, object]]:
    synced = {key: dict(value) for key, value in botanical_item_status.items()}
    for item in resolution.included_items:
        current = synced.setdefault(item, {})
        current["resolved"] = True
        current["outcome"] = "include"
        current["attempted_stages"] = list(current.get("attempted_stages") or [])
        current["last_reason"] = None
    for item in resolution.excluded_items:
        current = synced.setdefault(item, {})
        current["resolved"] = True
        current["outcome"] = "exclude"
        current["attempted_stages"] = list(current.get("attempted_stages") or [])
        current["last_reason"] = None
    for item in resolution.unresolved_items:
        current = synced.setdefault(item, {})
        current["resolved"] = False
        current["outcome"] = None
        current["attempted_stages"] = list(current.get("attempted_stages") or [])
        current["last_reason"] = current.get("last_reason") or "still_unresolved"
    return synced


def _botanical_stage_attempted(
    botanical_item_status: dict[str, dict[str, object]],
    item: str,
    stage: str,
) -> bool:
    return stage in list(botanical_item_status.get(item, {}).get("attempted_stages") or [])


def _record_botanical_stage(
    botanical_item_status: dict[str, dict[str, object]],
    botanical_search_history: list[str],
    *,
    item: str,
    stage: str,
    value: str,
) -> None:
    current = botanical_item_status.setdefault(
        item,
        {
            "resolved": False,
            "outcome": None,
            "attempted_stages": [],
            "last_reason": None,
        },
    )
    attempted_stages = list(current.get("attempted_stages") or [])
    if stage not in attempted_stages:
        attempted_stages.append(stage)
    current["attempted_stages"] = attempted_stages
    botanical_search_history.append(f"{item}|{stage}|{value}")


def _merge_evidence_records(
    existing: list[EvidenceRecord],
    new_records: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    merged = list(existing)
    seen_keys = {
        (
            record.source_url.strip().lower(),
            normalize_botanical_text(record.title_or_caption),
            normalize_botanical_text(record.content[:240]),
        )
        for record in merged
    }
    for record in new_records:
        record_key = (
            record.source_url.strip().lower(),
            normalize_botanical_text(record.title_or_caption),
            normalize_botanical_text(record.content[:240]),
        )
        if record_key in seen_keys:
            continue
        merged.append(record)
        seen_keys.add(record_key)
    return merged
