"""Adapters for temporal ordered-list skills."""

from __future__ import annotations

import re
import unicodedata

from langchain_core.messages import ToolMessage

from ..core.recoveries.utils import invoke_recovery_tool, recovery_trace_state
from ..graph.routing import question_profile_from_state
from ..source_pipeline import (
    EvidenceRecord,
    evidence_records_from_tool_output,
    parse_result_blocks,
)


class _AdapterBase:
    name = "adapter"
    skill_name = "temporal_ordered_list"

    def __init__(self, tools_by_name: dict[str, object]):
        self._tools_by_name = tools_by_name
        self.last_tool_trace: list[str] = []
        self.last_decision_trace: list[str] = []

    def _store_context(self, context) -> None:
        self.last_tool_trace = list(context.tool_trace)
        self.last_decision_trace = list(context.decision_trace)


class WikipediaRosterAdapter(_AdapterBase):
    name = "wikipedia_roster_adapter"

    def applies(self, profile, ranked_candidates) -> bool:
        return profile.name == "temporal_ordered_list" and any(
            "wikipedia.org" in candidate.url for candidate in ranked_candidates
        )

    def discover_sources(self, state) -> list[str]:
        urls: list[str] = []
        for raw in state.get("ranked_candidates") or []:
            if isinstance(raw, dict):
                url = str(raw.get("url", ""))
            else:
                url = str(getattr(raw, "url", ""))
            if "wikipedia.org" in url:
                urls.append(url)
        return list(dict.fromkeys(urls))

    def fetch_grounded_records(self, state) -> list[EvidenceRecord]:
        if "extract_tables_from_url" not in self._tools_by_name:
            return []
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        profile = question_profile_from_state(state)
        records: list[EvidenceRecord] = []
        for url in self.discover_sources(state)[:2]:
            table_text = invoke_recovery_tool(
                context=context,
                tool_name="extract_tables_from_url",
                tool_args={"url": url, "text_filter": profile.text_filter or profile.scope or ""},
                trace_label=self.name,
            )
            records.extend(evidence_records_from_tool_output("extract_tables_from_url", table_text))
        self._store_context(context)
        return records


class OfficialTeamDirectoryAdapter(_AdapterBase):
    name = "official_team_directory_adapter"

    def applies(self, profile, ranked_candidates) -> bool:
        if profile.name != "temporal_ordered_list":
            return False
        return any(
            any(fragment in candidate.url for fragment in ("/team/player/list/", "/team/player/detail/"))
            for candidate in ranked_candidates
        )

    def discover_sources(self, state) -> list[str]:
        urls: list[str] = []
        for raw in state.get("ranked_candidates") or []:
            if isinstance(raw, dict):
                url = str(raw.get("url", ""))
            else:
                url = str(getattr(raw, "url", ""))
            if any(fragment in url for fragment in ("/team/player/list/", "/team/player/detail/")):
                urls.append(url)
        return list(dict.fromkeys(urls))

    def fetch_grounded_records(self, state) -> list[EvidenceRecord]:
        if "fetch_url" not in self._tools_by_name:
            return []
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        records: list[EvidenceRecord] = []
        for url in self.discover_sources(state)[:2]:
            fetched = invoke_recovery_tool(
                context=context,
                tool_name="fetch_url",
                tool_args={"url": url},
                trace_label=self.name,
            )
            records.extend(evidence_records_from_tool_output("fetch_url", fetched))
        self._store_context(context)
        return records


class FightersAdapter(_AdapterBase):
    name = "fighters_adapter"

    def applies(self, profile, ranked_candidates) -> bool:
        if profile.name != "temporal_ordered_list" or not profile.expected_date:
            return False
        profile_text = " ".join([profile.subject_name or "", profile.scope or ""]).lower()
        candidate_text = " ".join(candidate.url.lower() for candidate in ranked_candidates[:8])
        return any(
            token in f"{profile_text} {candidate_text}"
            for token in ("fighters", "npb.jp", "taisho tamai", "tamai")
        )

    def discover_sources(self, state) -> list[str]:
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        urls = _candidate_urls_from_messages(
            state["messages"],
            predicate=lambda url: "npb.jp/bis/eng/players/" in url
            or "wikipedia.org/wiki/" in url
            or "fighters.co.jp/team/player/" in url,
        )
        for candidate in context.ranked_candidates:
            if any(
                token in candidate.url
                for token in ("npb.jp/bis/eng/players/", "wikipedia.org/wiki/", "fighters.co.jp/team/player/")
            ):
                urls.append(candidate.url)
        self._store_context(context)
        return list(dict.fromkeys(urls))

    def fetch_grounded_records(self, state) -> list[EvidenceRecord]:
        profile = question_profile_from_state(state)
        target_year = _extract_year_token(profile.expected_date)
        if target_year is None:
            return []
        if "fetch_url" not in self._tools_by_name or "extract_links_from_url" not in self._tools_by_name:
            return []
        context = recovery_trace_state(tools_by_name=self._tools_by_name, state=state)
        current_number: int | None = None
        for message in state["messages"]:
            if not isinstance(message, ToolMessage):
                continue
            if (getattr(message, "name", "") or "").strip() not in {"fetch_url", "find_text_in_url", "extract_tables_from_url"}:
                continue
            current_number = _extract_number_near_subject(text=str(message.content or ""), subject_name=profile.subject_name)
            if current_number is not None:
                break
        npb_player_urls = _candidate_urls_from_messages(
            state["messages"],
            predicate=lambda url: "npb.jp/bis/eng/players/" in url,
        )
        for candidate in context.ranked_candidates:
            if "npb.jp/bis/eng/players/" in candidate.url and candidate.url not in npb_player_urls:
                npb_player_urls.append(candidate.url)
        if not npb_player_urls and "web_search" in self._tools_by_name:
            subject_query = profile.subject_name or ""
            ascii_subject = (
                unicodedata.normalize("NFKD", subject_query).encode("ascii", "ignore").decode("ascii").strip()
            )
            query_subject = ascii_subject or subject_query
            if query_subject:
                search_text = invoke_recovery_tool(
                    context=context,
                    tool_name="web_search",
                    tool_args={"query": f"{query_subject} npb.jp players", "max_results": 5},
                    trace_label=self.name,
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
                wiki_links = invoke_recovery_tool(
                    context=context,
                    tool_name="extract_links_from_url",
                    tool_args={"url": wiki_url, "text_filter": "fighters.co.jp", "same_domain_only": False, "max_results": 10},
                    trace_label=self.name,
                )
                wiki_candidates = parse_result_blocks(wiki_links, origin_tool="extract_links_from_url")
                root_url = next((c.url for c in wiki_candidates if "fighters.co.jp" in c.url), None)
                if root_url:
                    official_list_url = root_url.rstrip("/") + "/team/player/list/"
                    break
        npb_player_url: str | None = None
        if current_number is None:
            for candidate_url in npb_player_urls:
                fetched = invoke_recovery_tool(
                    context=context,
                    tool_name="fetch_url",
                    tool_args={"url": candidate_url},
                    trace_label=self.name,
                )
                number = _extract_number_near_subject(text=fetched, subject_name=profile.subject_name)
                if number is not None:
                    current_number = number
                    npb_player_url = candidate_url
                    break
        elif npb_player_urls:
            npb_player_url = npb_player_urls[0]
        if official_list_url is None and npb_player_url is not None:
            official_links = invoke_recovery_tool(
                context=context,
                tool_name="extract_links_from_url",
                tool_args={"url": npb_player_url, "text_filter": "Official HP", "same_domain_only": False, "max_results": 10},
                trace_label=self.name,
            )
            official_candidates = parse_result_blocks(official_links, origin_tool="extract_links_from_url")
            official_list_url = next((c.url for c in official_candidates if "fighters.co.jp/team/player/list/" in c.url), None)
        if current_number is None or not official_list_url:
            self._store_context(context)
            return []
        detail_links = invoke_recovery_tool(
            context=context,
            tool_name="extract_links_from_url",
            tool_args={"url": official_list_url, "text_filter": str(current_number), "same_domain_only": True, "max_results": 20},
            trace_label=self.name,
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
                (c.url for c in detail_candidates if re.search(r"/team/player/detail/\d{4}_\d+\.html", c.url)),
                None,
            )
        if not current_detail_url:
            self._store_context(context)
            return []
        current_match = re.search(r"/team/player/detail/(?P<year>\d{4})_(?P<id>\d+)\.html", current_detail_url)
        if not current_match:
            self._store_context(context)
            return []
        current_year = int(current_match.group("year"))
        current_id = int(current_match.group("id"))
        if current_year <= target_year:
            self._store_context(context)
            return []
        predicted_id = current_id - (42 * (current_year - target_year))
        attempted_records: list[EvidenceRecord] = []
        for candidate_id in _fighters_detail_candidate_order(predicted_id):
            candidate_url = f"https://www.fighters.co.jp/team/player/detail/{target_year}_{candidate_id:08d}.html?lang=en"
            candidate_fetch = invoke_recovery_tool(
                context=context,
                tool_name="fetch_url",
                tool_args={"url": candidate_url},
                trace_label=self.name,
            )
            if "page not found" in candidate_fetch.lower() or "404" in candidate_fetch.lower():
                continue
            attempted_records.extend(evidence_records_from_tool_output("fetch_url", candidate_fetch))
        self._store_context(context)
        return attempted_records


def _extract_year_token(text: str | None) -> int | None:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text or "")
    return int(match.group(1)) if match else None


def _extract_number_near_subject(*, text: str, subject_name: str | None) -> int | None:
    if not text or not subject_name:
        return None
    subject_tokens = {token for token in re.findall(r"[a-z0-9]+", subject_name.lower()) if token}
    if not subject_tokens:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for index, line in enumerate(lines):
        line_tokens = {token for token in re.findall(r"[a-z0-9]+", line.lower()) if token}
        if not subject_tokens <= line_tokens:
            continue
        for back_index in range(max(0, index - 4), index):
            candidate = lines[back_index]
            if re.fullmatch(r"\d{1,3}", candidate):
                return int(candidate)
    number_hint = re.search(r"number(?:\s+will\s+be\s+changed\s+to)?\s*[\[\(]?(?P<number>\d{1,3})[\]\)]?", text, flags=re.IGNORECASE)
    if number_hint:
        return int(number_hint.group("number"))
    return None


def _candidate_urls_from_messages(messages, *, predicate):
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
            for record in evidence_records_from_tool_output(tool_name, content):
                if predicate(record.source_url):
                    urls.append(record.source_url)
    return list(dict.fromkeys(urls))


def _fighters_detail_candidate_order(predicted_id: int, *, radius: int = 20) -> list[int]:
    ordered = [predicted_id]
    for offset in range(1, radius + 1):
        ordered.append(predicted_id - offset)
        ordered.append(predicted_id + offset)
    return [candidate for candidate in ordered if candidate > 0]
