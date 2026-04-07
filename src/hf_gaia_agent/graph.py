"""LangGraph-based GAIA agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from .api_client import Question
from .normalize import normalize_submitted_answer
from .tools import build_tools, read_file_content


SYSTEM_PROMPT = """You are a GAIA benchmark assistant.

Rules:
- Solve the user's task as accurately as possible.
- Use tools when needed, especially for web lookup, file reading, and arithmetic.
- If a task includes an attachment, treat that attachment as part of the question context.
- When you are ready to answer, return only the final answer wrapped as [ANSWER]...[/ANSWER].
- Do not include explanations outside the answer wrapper in the final response.
- Preserve commas, ordering, pluralization, and formatting constraints requested by the task.
"""


class AgentState(MessagesState):
    task_id: str
    question: str
    file_name: str | None
    local_file_path: str | None
    final_answer: str | None
    tool_trace: list[str]
    error: str | None
    iterations: int
    max_iterations: int


def _build_model() -> Any:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "timeout": 60,
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

    raise ValueError(
        f"Unsupported MODEL_PROVIDER '{provider}'. Use 'openai' or 'huggingface'."
    )


def _prepare_context(state: AgentState) -> dict[str, Any]:
    file_path = state.get("local_file_path")
    attachment_block = ""
    if file_path:
        try:
            attachment_body = read_file_content(file_path)
            attachment_block = (
                "\n\nAttached file context:\n"
                f"File path: {file_path}\n"
                f"File name: {state.get('file_name') or Path(file_path).name}\n"
                f"Contents:\n{attachment_body}"
            )
        except Exception as exc:
            attachment_block = (
                "\n\nAttached file context could not be preloaded.\n"
                f"File path: {file_path}\n"
                f"Read error: {exc}"
            )

    user_prompt = (
        f"Task ID: {state['task_id']}\n"
        f"Question:\n{state['question']}"
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
        "error": None,
        "iterations": 0,
        "final_answer": None,
    }


class GaiaGraphAgent:
    """Question solver backed by a small LangGraph workflow."""

    def __init__(self, *, model: Any | None = None, max_iterations: int | None = None):
        self.tools = build_tools()
        self.tools_by_name = {tool_.name: tool_ for tool_ in self.tools}
        self.model = (model or _build_model()).bind_tools(self.tools)
        self.max_iterations = max_iterations or int(os.getenv("GAIA_MAX_ITERATIONS", "6"))
        self.app = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("prepare_context", _prepare_context)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.add_edge(START, "prepare_context")
        workflow.add_edge("prepare_context", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"agent": "agent", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)
        return workflow

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        response = self.model.invoke(state["messages"])
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        tool_trace = list(state.get("tool_trace") or [])

        for tool_call in getattr(last_message, "tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_trace.append(f"{tool_name}({tool_args})")
            tool_ = self.tools_by_name[tool_name]
            try:
                result = tool_.invoke(tool_args)
            except Exception as exc:
                result = f"Tool error: {exc}"
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
        return {"messages": tool_messages, "tool_trace": tool_trace}

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "finalize"

    def _route_after_tools(self, state: AgentState) -> str:
        if state.get("iterations", 0) >= state.get("max_iterations", self.max_iterations):
            return "finalize"
        return "agent"

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        last_ai = self._last_ai_message(state["messages"])
        raw_answer = last_ai.content if last_ai else ""
        final_answer = normalize_submitted_answer(str(raw_answer))
        error = state.get("error")
        if not final_answer:
            error = error or "Model did not produce a final answer."
        return {"final_answer": final_answer, "error": error}

    @staticmethod
    def _last_ai_message(messages: list[Any]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    def solve(
        self,
        question: Question,
        *,
        local_file_path: str | Path | None = None,
    ) -> dict[str, Any]:
        final_state = self.app.invoke(
            {
                "task_id": question.task_id,
                "question": question.question,
                "file_name": question.file_name,
                "local_file_path": str(local_file_path) if local_file_path else None,
                "messages": [],
                "tool_trace": [],
                "error": None,
                "final_answer": None,
                "iterations": 0,
                "max_iterations": self.max_iterations,
            }
        )
        return {
            "task_id": question.task_id,
            "question": question.question,
            "submitted_answer": final_state.get("final_answer", "") or "",
            "file_name": question.file_name,
            "tool_trace": final_state.get("tool_trace", []),
            "error": final_state.get("error"),
        }
