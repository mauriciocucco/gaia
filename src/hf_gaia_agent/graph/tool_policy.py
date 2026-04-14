"""Tool execution policy for the graph workflow.

This module keeps the high-level workflow readable by moving tool-call
policies, auto-followups, and ranking updates out of ``workflow.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from langchain_core.messages import ToolMessage

from ..source_pipeline import (
    QuestionProfile,
    SourceCandidate,
    parse_result_blocks,
    score_candidates,
    serialize_candidates,
)
from .candidate_support import RankedCandidateBuckets, bucket_ranked_candidates
from .contracts import CandidateRankingServices, ToolExecutionServices
from .routing import question_is_metric_row_lookup, question_profile_from_state
from .state import AgentState


class ToolPolicyServices(CandidateRankingServices, ToolExecutionServices, Protocol):
    """Narrow service surface consumed by ToolPolicyEngine."""


@dataclass
class ToolPolicyRunContext:
    tool_messages: list[ToolMessage] = field(default_factory=list)
    tool_trace: list[str] = field(default_factory=list)
    decision_trace: list[str] = field(default_factory=list)
    structured_tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    ranked_candidates: list[SourceCandidate] = field(default_factory=list)
    question_profile: QuestionProfile | None = None
    search_history: list[str] = field(default_factory=list)
    previous_search_signatures: set[str] = field(default_factory=set)
    fetched_urls: set[str] = field(default_factory=set)
    consecutive_searches: int = 0


class ToolPolicyEngine:
    """Runs the tools node while keeping workflow orchestration thin."""

    SEARCH_TOOL_NAMES = {"web_search", "search_wikipedia"}
    FETCH_TOOL_NAMES = {
        "fetch_url",
        "find_text_in_url",
        "extract_tables_from_url",
        "extract_links_from_url",
        "fetch_wikipedia_page",
    }

    def __init__(self, services: ToolPolicyServices):
        self._services = services

    def run(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        context = self._build_context(state)

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            raw_tool_args = dict(tool_call.get("args", {}))
            tool_args = dict(raw_tool_args)

            if tool_name in self.SEARCH_TOOL_NAMES and self._handle_search_tool_call(
                context=context,
                state=state,
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                raw_tool_args=raw_tool_args,
                tool_args=tool_args,
            ):
                continue
            if tool_name not in self.SEARCH_TOOL_NAMES:
                context.consecutive_searches = 0

            if tool_name in self.FETCH_TOOL_NAMES:
                self._handle_fetch_tool_call(
                    context=context,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )

            if tool_name == "execute_python_code":
                if not self._handle_python_tool_call(
                    context=context,
                    state=state,
                    tool_call_id=tool_call["id"],
                    tool_name=tool_name,
                    raw_tool_args=raw_tool_args,
                    tool_args=tool_args,
                ):
                    continue
            else:
                self._append_trace(context, tool_name, tool_args)

            execution = self._execute_tool_call(
                context=context,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call["id"],
            )
            self._apply_article_auto_followup(
                context=context,
                state=state,
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                tool_args=tool_args,
                execution=execution,
            )
            self._apply_text_span_auto_followup(
                context=context,
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                tool_args=tool_args,
                execution=execution,
            )
            self._apply_metric_table_auto_fallback(
                context=context,
                state=state,
                tool_call_id=tool_call["id"],
                tool_name=tool_name,
                tool_args=tool_args,
                execution=execution,
            )
            if tool_name in self.SEARCH_TOOL_NAMES | {"extract_links_from_url"}:
                self._update_ranked_candidates_from_output(
                    context=context,
                    question=state["question"],
                    tool_name=tool_name,
                    result_text=execution.text,
                    payloads=execution.payloads,
                )

        return {
            "messages": context.tool_messages,
            "tool_trace": context.tool_trace,
            "decision_trace": context.decision_trace,
            "ranked_candidates": serialize_candidates(context.ranked_candidates),
            "search_history_normalized": context.search_history,
            "search_history_fingerprints": context.search_history,
            "structured_tool_outputs": context.structured_tool_outputs,
        }

    def _build_context(self, state: AgentState) -> ToolPolicyRunContext:
        tool_trace = list(state.get("tool_trace") or [])
        decision_trace = list(state.get("decision_trace") or [])
        search_history = list(
            state.get("search_history_fingerprints")
            or state.get("search_history_normalized")
            or []
        )
        return ToolPolicyRunContext(
            tool_trace=tool_trace,
            decision_trace=decision_trace,
            structured_tool_outputs=list(state.get("structured_tool_outputs") or []),
            ranked_candidates=self._services.ranked_candidates_from_state(state),
            question_profile=question_profile_from_state(state),
            search_history=search_history,
            previous_search_signatures=set(search_history),
            fetched_urls=self._fetched_urls_from_tool_trace(tool_trace),
            consecutive_searches=self._count_consecutive_searches(decision_trace),
        )

    def _handle_search_tool_call(
        self,
        *,
        context: ToolPolicyRunContext,
        state: AgentState,
        tool_call_id: str,
        tool_name: str,
        raw_tool_args: dict[str, Any],
        tool_args: dict[str, Any],
    ) -> bool:
        query = str(tool_args.get("query", "")).strip()
        search_signature = self._services.normalize_search_query(query)

        if context.consecutive_searches >= 3:
            best_candidate = self._services.pick_best_unfetched_candidate(
                state,
                fetched_urls=context.fetched_urls,
            )
            if best_candidate is not None:
                fetch_args = {"url": best_candidate.url}
                context.fetched_urls.add(best_candidate.url)
                self._append_trace(context, "fetch_url", fetch_args)
                context.consecutive_searches = 0
                if "fetch_url" in self._services.tool_names:
                    execution = self._execute_tool_call(
                        context=context,
                        tool_name="fetch_url",
                        tool_args=fetch_args,
                        tool_call_id=tool_call_id,
                        add_message=False,
                    )
                    result_text = execution.text
                else:
                    result_text = "fetch_url not available"
                context.tool_messages.append(
                    ToolMessage(
                        content=(
                            "AUTO-FETCH: Your search was replaced with a fetch of "
                            f"{best_candidate.url} because it was the highest-ranked "
                            f"unfetched candidate.\n\n{result_text}"
                        ),
                        tool_call_id=tool_call_id,
                        name="fetch_url",
                    )
                )
                return True
            self._append_trace(context, tool_name, raw_tool_args)
            context.consecutive_searches += 1
            context.tool_messages.append(
                ToolMessage(
                    content=(
                        "SEARCH STRATEGY SHIFT REQUIRED: repeated searches did not produce a strong unread candidate. "
                        "Do not auto-fetch low-signal results. Change strategy now: add a specific year, domain, site:, "
                        "or source type, or read the best grounded evidence already collected."
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            return True

        if (
            search_signature
            and context.ranked_candidates
            and self._services.is_semantically_duplicate_search(
                search_signature,
                context.previous_search_signatures,
            )
        ):
            candidate_buckets = bucket_ranked_candidates(
                context.ranked_candidates,
                fetched_urls=context.fetched_urls,
            )
            self._append_trace(context, tool_name, raw_tool_args)
            context.consecutive_searches += 1
            context.tool_messages.append(
                ToolMessage(
                    content=self._duplicate_query_message(
                        query=query,
                        candidate_buckets=candidate_buckets,
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            return True

        if search_signature:
            context.previous_search_signatures.add(search_signature)
            context.search_history.append(search_signature)
        context.consecutive_searches += 1
        return False

    def _handle_fetch_tool_call(
        self,
        *,
        context: ToolPolicyRunContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> None:
        redirected_candidate = self._services.pick_better_fetch_candidate(
            requested_url=str(tool_args.get("url", "")).strip(),
            profile=context.question_profile,
            ranked_candidates=context.ranked_candidates,
            fetched_urls=context.fetched_urls,
        )
        if redirected_candidate is not None:
            tool_args["url"] = redirected_candidate.url
        if (
            tool_name in {"extract_tables_from_url", "extract_links_from_url"}
            and not str(tool_args.get("text_filter", "")).strip()
            and context.question_profile.text_filter
        ):
            tool_args["text_filter"] = context.question_profile.text_filter

    def _handle_python_tool_call(
        self,
        *,
        context: ToolPolicyRunContext,
        state: AgentState,
        tool_call_id: str,
        tool_name: str,
        raw_tool_args: dict[str, Any],
        tool_args: dict[str, Any],
    ) -> bool:
        allowed, grounded_reason = self._services.execute_python_allowed(state)
        self._append_trace(context, tool_name, raw_tool_args)
        if not allowed:
            context.tool_messages.append(
                ToolMessage(
                    content=(
                        "UNGROUNDED PYTHON BLOCKED: execute_python_code may only be used "
                        "for arithmetic, prompt-contained data, attachments, or "
                        "transforming previously fetched evidence. Read a source first or "
                        "answer from the prompt."
                    ),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            return False
        if grounded_reason == "fetched_evidence":
            code = str(tool_args.get("code", ""))
            tool_args["code"] = (
                "# Grounded evidence transformation only.\n"
                "# Use only facts already present in the prompt, attachment, or previous tool outputs.\n"
                "# Do not reconstruct missing web data from memory.\n"
                f"{code}"
            )
        return True

    def _execute_tool_call(
        self,
        *,
        context: ToolPolicyRunContext,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
        add_message: bool = True,
        message_name: str | None = None,
        message_content: str | None = None,
    ):
        execution = self._services.invoke_tool(tool_name, tool_args)
        context.structured_tool_outputs.append(
            {
                "tool_name": tool_name,
                "content": execution.text,
                "payloads": execution.payloads,
            }
        )
        executed_url = str(tool_args.get("url", "")).strip()
        if tool_name in self.FETCH_TOOL_NAMES and executed_url:
            context.fetched_urls.add(executed_url)
        if add_message:
            context.tool_messages.append(
                ToolMessage(
                    content=message_content or execution.text,
                    tool_call_id=tool_call_id,
                    name=message_name or tool_name,
                )
            )
        return execution

    def _apply_article_auto_followup(
        self,
        *,
        context: ToolPolicyRunContext,
        state: AgentState,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        execution: Any,
    ) -> None:
        auto_tool = self._services.article_to_paper_auto_links_result(
            tool_name=tool_name,
            tool_args=tool_args,
            result_text=execution.text,
            profile=context.question_profile,
        )
        if auto_tool is None:
            return

        auto_args, auto_name = auto_tool
        self._append_trace(
            context,
            auto_name,
            auto_args,
            decision_label=f"tool:{auto_name}:auto_retry_without_text_filter",
            trace_label="auto_retry_without_text_filter",
        )
        auto_execution = self._execute_tool_call(
            context=context,
            tool_name=auto_name,
            tool_args=auto_args,
            tool_call_id=f"auto-links-{tool_call_id}",
        )
        self._update_ranked_candidates_from_output(
            context=context,
            question=state["question"],
            tool_name=auto_name,
            result_text=auto_execution.text,
            payloads=auto_execution.payloads,
        )
        best_candidate = self._services.pick_best_unfetched_candidate(
            {
                **state,
                "ranked_candidates": serialize_candidates(context.ranked_candidates),
            },
            fetched_urls=context.fetched_urls,
        )
        if (
            context.question_profile.name == "article_to_paper"
            and best_candidate is not None
            and best_candidate.url != auto_args["url"]
            and "fetch_url" in self._services.tool_names
        ):
            fetch_args = {"url": best_candidate.url}
            context.fetched_urls.add(best_candidate.url)
            self._append_trace(
                context,
                "fetch_url",
                fetch_args,
                decision_label="tool:fetch_url:auto_followup_from_links",
                trace_label="auto_followup_from_links",
            )
            self._execute_tool_call(
                context=context,
                tool_name="fetch_url",
                tool_args=fetch_args,
                tool_call_id=f"auto-fetch-links-{tool_call_id}",
            )

    def _apply_text_span_auto_followup(
        self,
        *,
        context: ToolPolicyRunContext,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        execution: Any,
    ) -> None:
        follow_candidate = self._services.text_span_auto_follow_candidate(
            tool_name=tool_name,
            tool_args=tool_args,
            result_text=execution.text,
            profile=context.question_profile,
            ranked_candidates=context.ranked_candidates,
            fetched_urls=context.fetched_urls,
        )
        if follow_candidate is None:
            return

        follow_args = {
            "url": follow_candidate.url,
            "query": str(tool_args.get("query", "")).strip(),
            "max_matches": tool_args.get("max_matches", 8),
        }
        context.fetched_urls.add(follow_candidate.url)
        self._append_trace(
            context,
            "find_text_in_url",
            follow_args,
            decision_label="tool:find_text_in_url:auto_followup",
            trace_label="auto_followup_from_failed_find_text",
        )
        self._execute_tool_call(
            context=context,
            tool_name="find_text_in_url",
            tool_args=follow_args,
            tool_call_id=f"auto-find-{tool_call_id}",
        )

    def _apply_metric_table_auto_fallback(
        self,
        *,
        context: ToolPolicyRunContext,
        state: AgentState,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        execution: Any,
    ) -> None:
        if (
            tool_name != "extract_tables_from_url"
            or not question_is_metric_row_lookup(state["question"])
            or "No readable HTML tables found." not in execution.text
            or "fetch_url" not in self._services.tool_names
        ):
            return

        executed_url = str(tool_args.get("url", "")).strip()
        if not executed_url:
            return

        auto_args = {"url": executed_url}
        self._append_trace(
            context,
            "fetch_url",
            auto_args,
            decision_label="tool:fetch_url:auto_fallback",
            trace_label="auto_fallback_from_extract_tables",
        )
        self._execute_tool_call(
            context=context,
            tool_name="fetch_url",
            tool_args=auto_args,
            tool_call_id=f"auto-fetch-{tool_call_id}",
        )

    def _update_ranked_candidates_from_output(
        self,
        *,
        context: ToolPolicyRunContext,
        question: str,
        tool_name: str,
        result_text: str,
        payloads: list[dict[str, Any]],
    ) -> None:
        parsed_candidates = parse_result_blocks(
            payloads or result_text,
            origin_tool=tool_name,
        )
        if not parsed_candidates:
            return
        scored_candidates = score_candidates(
            parsed_candidates,
            question=question,
            profile=context.question_profile,
        )
        context.ranked_candidates = self._services.merge_ranked_candidates(
            context.ranked_candidates,
            scored_candidates,
        )

    @staticmethod
    def _duplicate_query_message(
        *,
        query: str,
        candidate_buckets: RankedCandidateBuckets,
    ) -> str:
        if candidate_buckets.useful_unfetched:
            candidate_lines = "\n".join(
                f"- {candidate.title or candidate.url}\n  URL: {candidate.url}"
                for candidate in candidate_buckets.useful_unfetched[:3]
            )
            return (
                f"DUPLICATE QUERY: '{query}' is too similar to a previous search. "
                "Do not repeat this search. You already have unread ranked candidates. "
                "Read one of these now with fetch_url, find_text_in_url, or extract_tables_from_url:\n"
                f"{candidate_lines}"
            )
        if candidate_buckets.exhausted_useful and candidate_buckets.low_quality_unfetched:
            return (
                f"DUPLICATE QUERY: '{query}' is too similar to a previous search.\n"
                "ALL GOOD CANDIDATES EXHAUSTED: the useful ranked candidates were already read, "
                "and the remaining unfetched candidates are low-quality or off-topic. "
                "Do not reread the same results and do not fetch the low-quality ones. "
                "Change strategy completely: use search_wikipedia with 1-3 named entities, "
                "use fetch_wikipedia_page if the title is clear, fetch a likely official URL directly, "
                "or answer from the evidence already collected."
            )
        if candidate_buckets.exhausted_useful:
            return (
                f"DUPLICATE QUERY: '{query}' is too similar to a previous search.\n"
                "ALL CANDIDATES EXHAUSTED: this search strategy is spent. "
                "Do not reread the same URLs and do not repeat this search. "
                "Change strategy completely: use search_wikipedia with 1-3 named entities, "
                "use fetch_wikipedia_page if the title is clear, fetch a likely official URL directly, "
                "or answer from the evidence already collected."
            )
        if candidate_buckets.low_quality_unfetched:
            return (
                f"DUPLICATE QUERY: '{query}' is too similar to a previous search.\n"
                "LOW-QUALITY CANDIDATES: the remaining ranked candidates are low-quality or off-topic. "
                "Do not fetch them and do not repeat this search. "
                "Change strategy completely: use search_wikipedia with 1-3 named entities, "
                "use fetch_wikipedia_page if the title is clear, fetch a likely official URL directly, "
                "or answer from the evidence already collected."
            )
        return (
            f"DUPLICATE QUERY: '{query}' is too similar to a previous search. "
            "Do not repeat it. Change strategy instead: use search_wikipedia with 1-3 named entities, "
            "use fetch_wikipedia_page if the title is clear, fetch a likely official URL directly, "
            "or answer from the evidence already collected."
        )

    def _append_trace(
        self,
        context: ToolPolicyRunContext,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        decision_label: str | None = None,
        trace_label: str | None = None,
    ) -> None:
        trace_entry = f"{tool_name}({tool_args})"
        if trace_label:
            trace_entry = f"{trace_entry} [{trace_label}]"
        context.tool_trace.append(trace_entry)
        context.decision_trace.append(decision_label or f"tool:{tool_name}")

    @classmethod
    def _count_consecutive_searches(cls, decision_trace: list[str]) -> int:
        consecutive_searches = 0
        for entry in reversed(decision_trace):
            if entry.removeprefix("tool:") in cls.SEARCH_TOOL_NAMES:
                consecutive_searches += 1
            else:
                break
        return consecutive_searches

    @classmethod
    def _fetched_urls_from_tool_trace(cls, tool_trace: list[str]) -> set[str]:
        fetched_urls: set[str] = set()
        for entry in tool_trace:
            if any(
                entry.startswith(f"{name}(")
                for name in (
                    "fetch_url",
                    "find_text_in_url",
                    "extract_tables_from_url",
                    "extract_links_from_url",
                )
            ):
                url_match = re.search(r"'url':\s*'([^']+)'", entry)
                if url_match:
                    fetched_urls.add(url_match.group(1))
        return fetched_urls
