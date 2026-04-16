"""Refactored GaiaGraphAgent: LangGraph workflow using decomposed modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..adapters import build_source_adapters
from ..api_client import Question
from ..core.recoveries import RecoveryStrategy, build_core_recoveries
from ..hooks import AgentHook, BaseAgentHook, CompositeHook
from ..skills import Skill, build_skills
from ..source_pipeline import profile_question
from ..tools import read_file_content
from .answer_policy import (
    canonicalize_final_answer,
    is_invalid_final_response,
    looks_like_placeholder_answer,
)
from .evidence_support import last_ai_message
from .finalizer import WorkflowFinalizer
from .nudges import (
    build_ranked_candidate_nudge,
    build_search_nudge,
    build_stuck_search_nudge,
)
from .prompts import MODEL_TOOL_MESSAGE_MAX_CHARS, SYSTEM_PROMPT
from .retry_rules import build_retry_guidance, should_retry_answer
from .routing import (
    build_profile_guidance_block,
    build_research_hint_block,
    extract_urls,
    is_youtube_url,
    maybe_decode_reversed_question,
    try_prompt_reducer,
)
from .services import GraphWorkflowServices
from .state import AgentState
from .tool_policy import ToolPolicyEngine


class _GraphRenderOnlyModel:
    """Stub model used to compile the graph for documentation."""

    def bind_tools(self, _tools: list[Any]) -> "_GraphRenderOnlyModel":
        return self

    def invoke(self, _messages: list[Any], *args: Any, **kwargs: Any) -> AIMessage:
        raise RuntimeError("GraphRenderOnlyModel cannot execute the agent.")


def _build_model() -> Any:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "timeout": 60,
        "max_retries": 15,
    }
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if provider == "huggingface":
        token = os.getenv("HF_TOKEN")
        base_url = os.getenv("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
        if token:
            kwargs["api_key"] = token
        kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    raise ValueError(f"Unsupported MODEL_PROVIDER '{provider}'.")


def _prepare_context(state: AgentState) -> dict[str, Any]:
    file_path = state.get("local_file_path")
    file_name = state.get("file_name")
    question_profile = profile_question(
        state["question"],
        file_name=file_name,
        local_file_path=file_path,
    )
    attachment_block = ""
    decoded_block = ""
    youtube_block = ""
    research_hint_block = build_research_hint_block(state["question"])
    profile_guidance_block = build_profile_guidance_block(
        question=state["question"],
        profile=question_profile,
    )
    decoded_question = maybe_decode_reversed_question(state["question"])
    if decoded_question:
        decoded_block = (
            "\n\nDecoded question hint:\n"
            f"The original text appears reversed. Decoded version:\n{decoded_question}"
        )
    youtube_urls = [url for url in extract_urls(state["question"]) if is_youtube_url(url)]
    if youtube_urls:
        youtube_block = (
            "\n\nYouTube hint:\n"
            f"This question includes YouTube URL(s): {', '.join(youtube_urls)}\n"
            "Use the get_youtube_transcript tool first if a transcript is available.\n"
            "If the question requires visual analysis (counting objects, identifying items on screen, "
            "describing what is shown), use the analyze_youtube_video tool with the URL and the question."
        )
    if file_path:
        try:
            attachment_body = read_file_content(file_path)
            attachment_block = (
                "\n\nAttached file context:\n"
                f"File path: {file_path}\n"
                f"File name: {file_name or Path(file_path).name}\n"
                f"Contents:\n{attachment_body}"
            )
        except Exception as exc:
            attachment_block = (
                "\n\nAttached file context could not be preloaded.\n"
                f"File path: {file_path}\n"
                f"Read error: {exc}"
            )
    elif file_name:
        attachment_block = (
            "\n\nAttachment status:\n"
            f"The task references an attachment named {file_name}, but no local attachment file is available.\n"
            "Do not invent local file paths and do not claim to have inspected the attachment."
        )

    user_prompt = (
        f"Task ID: {state['task_id']}\n"
        f"Question:\n{state['question']}"
        f"{decoded_block}"
        f"{youtube_block}"
        f"{research_hint_block}"
        f"{profile_guidance_block}"
        f"{attachment_block}\n\n"
        "Work carefully. Use tools if needed. "
        "Return the final answer only as [ANSWER]...[/ANSWER]."
    )
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        "tool_trace": [],
        "decision_trace": [],
        "evidence_used": [],
        "reducer_used": None,
        "recovery_reason": None,
        "question_profile": question_profile.as_dict(),
        "ranked_candidates": [],
        "search_history_normalized": [],
        "search_history_fingerprints": [],
        "structured_tool_outputs": [],
        "skill_trace": [],
        "botanical_partial_records": [],
        "botanical_item_status": {},
        "botanical_search_history": [],
        "error": None,
        "iterations": 0,
        "final_answer": None,
    }


class GaiaGraphAgent:
    """Question solver backed by a LangGraph workflow."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        max_iterations: int | None = None,
        hooks: list[AgentHook] | None = None,
        include_benchmark_extensions: bool | None = None,
    ):
        import hf_gaia_agent.graph as _graph_ns

        self.tools = _graph_ns.build_tools()
        self.tools_by_name = {tool_.name: tool_ for tool_ in self.tools}
        self.answer_model = model or _graph_ns._build_model()
        self.model = self.answer_model.bind_tools(self.tools)
        self.max_iterations = max_iterations or int(
            os.getenv("GAIA_MAX_ITERATIONS", "15")
        )
        benchmark_extensions_enabled = (
            True if include_benchmark_extensions is None else include_benchmark_extensions
        )
        self.core_recoveries: list[RecoveryStrategy] = build_core_recoveries(
            self.tools_by_name,
            self.answer_model,
        )
        self.skills: list[Skill] = build_skills(
            self.tools_by_name,
            self.answer_model,
            include_benchmark_specific=benchmark_extensions_enabled,
        )
        self.source_adapters = build_source_adapters(self.tools_by_name)
        self._hook: AgentHook = (
            (hooks[0] if len(hooks) == 1 else CompositeHook(hooks))
            if hooks
            else BaseAgentHook()
        )
        self._services = GraphWorkflowServices(
            answer_model=self.answer_model,
            tools_by_name=self.tools_by_name,
            core_recoveries=self.core_recoveries,
            skills=self.skills,
            source_adapters=self.source_adapters,
            hook=self._hook,
        )
        self._tool_policy = ToolPolicyEngine(self._services)
        self._finalizer = WorkflowFinalizer(self._services)
        self.app = self._build_graph().compile()

    @classmethod
    def for_graph_introspection(cls) -> "GaiaGraphAgent":
        return cls(model=_GraphRenderOnlyModel(), max_iterations=1)

    def render_graph(self, *, format: str = "mermaid") -> str:
        graph = self.app.get_graph()
        if format == "mermaid":
            return graph.draw_mermaid()
        if format == "ascii":
            try:
                return graph.draw_ascii()
            except ImportError:
                return self._render_graph_ascii_fallback(graph)
        raise ValueError(f"Unsupported graph render format '{format}'.")

    def render_graph_mermaid(self) -> str:
        return self.render_graph(format="mermaid")

    def render_graph_ascii(self) -> str:
        return self.render_graph(format="ascii")

    @staticmethod
    def _render_graph_ascii_fallback(graph: Any) -> str:
        lines: list[str] = []
        for node_id in graph.nodes:
            lines.append(f"[{node_id}]")
            outgoing = [edge for edge in graph.edges if edge.source == node_id]
            if not outgoing:
                lines.append("  (no outgoing edges)")
                continue
            for edge in outgoing:
                connector = "-?->" if getattr(edge, "conditional", False) else "-->"
                lines.append(f"  {connector} {edge.target}")
        return "\n".join(lines)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("prepare_context", _prepare_context)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("resolve_after_tools", self._resolve_after_tools_node)
        workflow.add_node("retry_invalid_answer", self._retry_invalid_answer_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.add_edge(START, "prepare_context")
        workflow.add_edge("prepare_context", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "tools": "tools",
                "retry_invalid_answer": "retry_invalid_answer",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("tools", "resolve_after_tools")
        workflow.add_conditional_edges(
            "resolve_after_tools",
            self._route_after_tools,
            {"agent": "agent", "finalize": "finalize"},
        )
        workflow.add_edge("retry_invalid_answer", "agent")
        workflow.add_edge("finalize", END)
        return workflow

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        msgs = self._messages_for_model(state["messages"])
        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            msgs.append(
                SystemMessage(
                    content=(
                        "CRITICAL: You have reached the maximum number of tool calls. "
                        "You CANNOT use tools anymore. You MUST provide your final guess or answer "
                        "using the [ANSWER]...[/ANSWER] wrapper immediately in this message based on the "
                        "evidence collected so far."
                    )
                )
            )
            response = (
                self.model.invoke(msgs, tool_choice="none")
                if hasattr(self.model, "invoke")
                else self.model.invoke(msgs)
            )
        else:
            stuck_search_nudge = build_stuck_search_nudge(state)
            nudges = [
                nudge
                for nudge in (
                    build_ranked_candidate_nudge(state),
                    stuck_search_nudge,
                    None if stuck_search_nudge else build_search_nudge(state),
                )
                if nudge
            ]
            if nudges:
                msgs.append(HumanMessage(content="\n\n".join(nudges)))
            response = self.model.invoke(msgs)
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        return self._tool_policy.run(state)

    def _retry_invalid_answer_node(self, state: AgentState) -> dict[str, Any]:
        last_content = str(getattr(last_ai_message(state["messages"]), "content", "") or "")
        extra = "".join(build_retry_guidance(state, last_content))
        reminder = HumanMessage(
            content=(
                "Your previous reply was invalid because it was not a concrete answer. "
                "Do not apologize, do not mention access issues, and do not ask to try again later. "
                "Answer from the existing evidence first. "
                "If the current tool outputs already contain a relevant table, list, passage, or transcript excerpt, compute the answer from that evidence and stop. "
                "Only make another tool call if you can point to a concrete gap in the existing evidence. "
                f"Respond only with [ANSWER]the factual answer[/ANSWER].{extra}"
            )
        )
        return {"messages": [reminder]}

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        return self._finalizer.finalize(state)

    def _resolve_after_tools_node(self, state: AgentState) -> dict[str, Any]:
        return self._services.run_resolution_pipeline(state) or {}

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        last_content = str(getattr(last_message, "content", ""))
        if getattr(last_message, "tool_calls", None):
            return "tools"
        if (
            (
                is_invalid_final_response(last_content)
                or looks_like_placeholder_answer(state["question"], last_content)
            )
            and state.get("iterations", 0) < state.get("max_iterations", 0)
        ):
            return "retry_invalid_answer"
        if (
            should_retry_answer(state, last_content)
            and state.get("iterations", 0) < state.get("max_iterations", 0)
        ):
            return "retry_invalid_answer"
        return "finalize"

    def _route_after_tools(self, state: AgentState) -> str:
        return "finalize" if state.get("final_answer") is not None else "agent"

    @staticmethod
    def _messages_for_model(messages: list[Any]) -> list[Any]:
        prepared: list[Any] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                prepared.append(message)
                continue
            if str(getattr(message, "tool_call_id", "") or "").startswith("auto-"):
                continue
            content = str(message.content)
            if len(content) <= MODEL_TOOL_MESSAGE_MAX_CHARS:
                prepared.append(message)
                continue
            prepared.append(
                ToolMessage(
                    content=GaiaGraphAgent._truncate_for_model_context(
                        content,
                        max_chars=MODEL_TOOL_MESSAGE_MAX_CHARS,
                    ),
                    tool_call_id=str(getattr(message, "tool_call_id", "") or "tool"),
                    name=(getattr(message, "name", "") or None),
                )
            )
        return prepared

    @staticmethod
    def _truncate_for_model_context(value: str, *, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        head_chars = max_chars * 3 // 4
        tail_chars = max_chars - head_chars - 36
        if tail_chars <= 0:
            return value[:max_chars]
        return (
            value[:head_chars]
            + "\n...[truncated for model context]...\n"
            + value[-tail_chars:]
        )

    def solve(
        self, question: Question, *, local_file_path: str | Path | None = None
    ) -> dict[str, Any]:
        self._hook.on_solve_start(question.task_id, question.question)

        prompt_reducer_answer, prompt_reducer_used = try_prompt_reducer(question.question)
        if prompt_reducer_answer:
            result = {
                "task_id": question.task_id,
                "question": question.question,
                "submitted_answer": canonicalize_final_answer(
                    question.question,
                    prompt_reducer_answer,
                ),
                "file_name": question.file_name,
                "tool_trace": [],
                "decision_trace": [f"prompt_reducer:{prompt_reducer_used}"],
                "evidence_used": [],
                "reducer_used": prompt_reducer_used,
                "skill_trace": [],
                "recovery_reason": None,
                "error": None,
            }
            self._hook.on_solve_end(question.task_id, result)
            return result

        final_state = self.app.invoke(
            {
                "task_id": question.task_id,
                "question": question.question,
                "file_name": question.file_name,
                "local_file_path": str(local_file_path) if local_file_path else None,
                "messages": [],
                "tool_trace": [],
                "decision_trace": [],
                "evidence_used": [],
                "reducer_used": None,
                "recovery_reason": None,
                "error": None,
                "final_answer": None,
                "iterations": 0,
                "max_iterations": self.max_iterations,
                "question_profile": {},
                "ranked_candidates": [],
                "search_history_normalized": [],
                "search_history_fingerprints": [],
                "structured_tool_outputs": [],
                "skill_trace": [],
                "botanical_partial_records": [],
                "botanical_item_status": {},
                "botanical_search_history": [],
            },
            config={"recursion_limit": 50},
        )
        result = {
            "task_id": question.task_id,
            "question": question.question,
            "submitted_answer": canonicalize_final_answer(
                question.question,
                final_state.get("final_answer", "") or "",
            ),
            "file_name": question.file_name,
            "tool_trace": final_state.get("tool_trace", []),
            "decision_trace": final_state.get("decision_trace", []),
            "evidence_used": final_state.get("evidence_used", []),
            "reducer_used": final_state.get("reducer_used"),
            "skill_trace": final_state.get("skill_trace", []),
            "recovery_reason": final_state.get("recovery_reason"),
            "error": final_state.get("error"),
        }
        self._hook.on_solve_end(question.task_id, result)
        return result
