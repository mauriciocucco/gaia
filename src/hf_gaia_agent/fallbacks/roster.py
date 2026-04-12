"""Roster neighbor (Fighters official roster) fallback resolver."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..normalize import normalize_submitted_answer
from ..source_pipeline import (
    EvidenceRecord,
    SourceCandidate,
    evidence_records_from_tool_output,
    parse_result_blocks,
    serialize_evidence,
)
from ..graph.state import AgentState
from ..graph.routing import question_profile_from_state
from .base import FallbackResolver
from .utils import (
    candidate_urls_from_state,
    fallback_result_from_records,
    fallback_trace_state,
    invoke_fallback_tool,
    with_fallback_traces,
)


class RosterFallback:
    name = "roster"

    def __init__(self, tools_by_name: dict[str, Any]):
        self._tools_by_name = tools_by_name

    def applies(self, state: AgentState, profile: Any) -> bool:
        return profile.name == "roster_neighbor_lookup" and bool(profile.expected_date)

    def run(self, state: AgentState) -> dict[str, Any] | None:
        profile = question_profile_from_state(state)
        if not profile.expected_date:
            return None
        target_year = _extract_year_token(profile.expected_date)
        if target_year is None:
            return None

        current_number: int | None = None
        _CONTENT_TOOL_NAMES = {"fetch_url", "find_text_in_url", "extract_tables_from_url"}
        context = fallback_trace_state(tools_by_name=self._tools_by_name, state=state)

        from langchain_core.messages import ToolMessage

        for message in state["messages"]:
            if not isinstance(message, ToolMessage):
                continue
            if (getattr(message, "name", "") or "").strip() not in _CONTENT_TOOL_NAMES:
                continue
            current_number = _extract_number_near_subject(
                text=str(message.content or ""),
                subject_name=profile.subject_name,
            )
            if current_number is not None:
                break

        if "fetch_url" not in self._tools_by_name or "extract_links_from_url" not in self._tools_by_name:
            return None

        npb_player_urls = _candidate_urls_from_messages(
            state["messages"],
            predicate=lambda url: "npb.jp/bis/eng/players/" in url,
        )
        for candidate in context.ranked_candidates:
            if "npb.jp/bis/eng/players/" in candidate.url and candidate.url not in npb_player_urls:
                npb_player_urls.append(candidate.url)

        if not npb_player_urls:
            wiki_urls = _candidate_urls_from_messages(
                state["messages"],
                predicate=lambda url: "wikipedia.org/wiki/" in url,
            )
            for candidate in context.ranked_candidates:
                if "wikipedia.org/wiki/" in candidate.url and candidate.url not in wiki_urls:
                    wiki_urls.append(candidate.url)
            for wiki_url in wiki_urls:
                wiki_links = invoke_fallback_tool(
                    context=context,
                    tool_name="extract_links_from_url",
                    tool_args={
                        "url": wiki_url,
                        "text_filter": "npb.jp",
                        "same_domain_only": False,
                        "max_results": 10,
                    },
                )
                for candidate in parse_result_blocks(wiki_links, origin_tool="extract_links_from_url"):
                    if "npb.jp/bis/eng/players/" in candidate.url and candidate.url not in npb_player_urls:
                        npb_player_urls.append(candidate.url)

        if not npb_player_urls and "web_search" in self._tools_by_name:
            subject_query = profile.subject_name or ""
            ascii_subject = (
                unicodedata.normalize("NFKD", subject_query)
                .encode("ascii", "ignore")
                .decode("ascii")
                .strip()
            )
            query_subject = ascii_subject or subject_query
            if query_subject:
                search_text = invoke_fallback_tool(
                    context=context,
                    tool_name="web_search",
                    tool_args={"query": f"{query_subject} npb.jp players", "max_results": 5},
                )
                for candidate in parse_result_blocks(search_text, origin_tool="web_search"):
                    if "npb.jp/bis/eng/players/" in candidate.url and candidate.url not in npb_player_urls:
                        npb_player_urls.append(candidate.url)

        official_list_url: str | None = None
        if current_number is not None:
            wiki_urls = _candidate_urls_from_messages(
                state["messages"],
                predicate=lambda url: "wikipedia.org/wiki/" in url,
            )
            for candidate in context.ranked_candidates:
                if "wikipedia.org/wiki/" in candidate.url and candidate.url not in wiki_urls:
                    wiki_urls.append(candidate.url)
            for wiki_url in wiki_urls:
                wiki_links = invoke_fallback_tool(
                    context=context,
                    tool_name="extract_links_from_url",
                    tool_args={
                        "url": wiki_url,
                        "text_filter": "fighters.co.jp",
                        "same_domain_only": False,
                        "max_results": 10,
                    },
                )
                wiki_candidates = parse_result_blocks(wiki_links, origin_tool="extract_links_from_url")
                root_url = next(
                    (c.url for c in wiki_candidates if "fighters.co.jp" in c.url),
                    None,
                )
                if root_url:
                    official_list_url = root_url.rstrip("/") + "/team/player/list/"
                    break

        if not npb_player_urls and not official_list_url:
            return None

        npb_player_url: str | None = None
        if current_number is None:
            for candidate_url in npb_player_urls:
                fetched = invoke_fallback_tool(
                    context=context,
                    tool_name="fetch_url",
                    tool_args={"url": candidate_url},
                )
                number = _extract_number_near_subject(
                    text=fetched,
                    subject_name=profile.subject_name,
                )
                if number is not None:
                    current_number = number
                    npb_player_url = candidate_url
                    break
        elif npb_player_urls:
            npb_player_url = npb_player_urls[0]

        if official_list_url is None and npb_player_url is not None:
            official_links = invoke_fallback_tool(
                context=context,
                tool_name="extract_links_from_url",
                tool_args={
                    "url": npb_player_url,
                    "text_filter": "Official HP",
                    "same_domain_only": False,
                    "max_results": 10,
                },
            )
            official_candidates = parse_result_blocks(official_links, origin_tool="extract_links_from_url")
            official_list_url = next(
                (c.url for c in official_candidates if "fighters.co.jp/team/player/list/" in c.url),
                None,
            )

        if current_number is None or not official_list_url:
            return None

        detail_links = invoke_fallback_tool(
            context=context,
            tool_name="extract_links_from_url",
            tool_args={
                "url": official_list_url,
                "text_filter": str(current_number),
                "same_domain_only": True,
                "max_results": 20,
            },
        )
        detail_candidates = parse_result_blocks(detail_links, origin_tool="extract_links_from_url")
        current_detail_url = next(
            (
                c.url
                for c in detail_candidates
                if re.search(r"/team/player/detail/\d{4}_\d+\.html", c.url)
                and re.match(rf"^{current_number}\b", c.title)
            ),
            None,
        )
        if not current_detail_url:
            current_detail_url = next(
                (
                    c.url
                    for c in detail_candidates
                    if re.search(r"/team/player/detail/\d{4}_\d+\.html", c.url)
                ),
                None,
            )
        if not current_detail_url:
            return None

        current_match = re.search(
            r"/team/player/detail/(?P<year>\d{4})_(?P<id>\d+)\.html",
            current_detail_url,
        )
        if not current_match:
            return None
        current_year = int(current_match.group("year"))
        current_id = int(current_match.group("id"))
        if current_year <= target_year:
            return None

        predicted_id = current_id - (42 * (current_year - target_year))
        attempted_records: list[EvidenceRecord] = []
        for candidate_id in _fighters_detail_candidate_order(predicted_id):
            candidate_url = (
                f"https://www.fighters.co.jp/team/player/detail/{target_year}_{candidate_id:08d}.html?lang=en"
            )
            candidate_fetch = invoke_fallback_tool(
                context=context,
                tool_name="fetch_url",
                tool_args={"url": candidate_url},
            )
            normalized_fetch = normalize_submitted_answer(candidate_fetch).lower()
            if "page not found" in normalized_fetch or "404" in normalized_fetch:
                continue
            candidate_records = evidence_records_from_tool_output("fetch_url", candidate_fetch)
            if not candidate_records:
                continue
            attempted_records.extend(candidate_records)
            result = fallback_result_from_records(
                state["question"],
                attempted_records,
                expected_reducer="roster_neighbor",
            )
            if result:
                return with_fallback_traces(
                    result,
                    context=context,
                )
        return None


def _extract_year_token(text: str) -> int | None:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text or "")
    return int(match.group(1)) if match else None


def _extract_number_near_subject(*, text: str, subject_name: str | None) -> int | None:
    if not text or not subject_name:
        return None
    subject_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_submitted_answer(subject_name).lower())
        if token
    }
    if not subject_tokens:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for index, line in enumerate(lines):
        line_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", normalize_submitted_answer(line).lower())
            if token
        }
        if not subject_tokens <= line_tokens:
            continue
        for back_index in range(max(0, index - 4), index):
            candidate = lines[back_index]
            if re.fullmatch(r"\d{1,3}", candidate):
                return int(candidate)
    number_hint = re.search(
        r"number(?:\s+will\s+be\s+changed\s+to)?\s*[\[\(]?(?P<number>\d{1,3})[\]\)]?",
        text,
        flags=re.IGNORECASE,
    )
    if number_hint:
        return int(number_hint.group("number"))
    return None


def _candidate_urls_from_messages(
    messages: list[Any],
    *,
    predicate: Any,
) -> list[str]:
    from langchain_core.messages import ToolMessage

    urls: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (getattr(message, "name", "") or "").strip()
        content = str(message.content or "")
        if tool_name in {"web_search", "search_wikipedia", "extract_links_from_url"}:
            for candidate in parse_result_blocks(content, origin_tool=tool_name):
                if predicate(candidate.url):
                    urls.append(candidate.url)
        else:
            metadata = evidence_records_from_tool_output(tool_name, content)
            for record in metadata:
                if predicate(record.source_url):
                    urls.append(record.source_url)
    return list(dict.fromkeys(urls))


def _fighters_detail_candidate_order(predicted_id: int, *, radius: int = 20) -> list[int]:
    ordered = [predicted_id]
    for offset in range(1, radius + 1):
        ordered.append(predicted_id - offset)
        ordered.append(predicted_id + offset)
    return [candidate for candidate in ordered if candidate > 0]
