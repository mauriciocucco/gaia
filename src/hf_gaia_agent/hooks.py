"""Agent lifecycle hooks for observability and debugging."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentHook(Protocol):
    """Observer protocol for agent workflow events.

    Implement any subset of methods; unimplemented ones default to no-ops
    when using :class:`BaseAgentHook`.
    """

    def on_tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None: ...
    def on_tool_end(self, tool_name: str, result: str) -> None: ...
    def on_solve_start(self, task_id: str, question: str) -> None: ...
    def on_solve_end(self, task_id: str, result: dict[str, Any]) -> None: ...


class BaseAgentHook:
    """No-op base — override only the callbacks you need."""

    def on_tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        pass

    def on_tool_end(self, tool_name: str, result: str) -> None:
        pass

    def on_solve_start(self, task_id: str, question: str) -> None:
        pass

    def on_solve_end(self, task_id: str, result: dict[str, Any]) -> None:
        pass


class VerboseHook(BaseAgentHook):
    """Prints tool calls and results to stdout — replaces the old monkeypatch debug."""

    def __init__(self, *, args_max_len: int = 150, result_max_len: int = 200) -> None:
        self.args_max_len = args_max_len
        self.result_max_len = result_max_len

    def on_tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        args_str = str(tool_args)[:self.args_max_len]
        print(f"  >> {tool_name}({args_str})", flush=True)

    def on_tool_end(self, tool_name: str, result: str) -> None:
        preview = result[:self.result_max_len].replace("\n", " ")
        print(f"  << {tool_name} => {preview}", flush=True)

    def on_solve_start(self, task_id: str, question: str) -> None:
        print(f"\n{'=' * 80}")
        print(f"Task: {task_id}")
        print(f"Question: {question[:120]}")
        print(f"{'=' * 80}\n", flush=True)

    def on_solve_end(self, task_id: str, result: dict[str, Any]) -> None:
        separator = "-" * 80
        print(f"\n{separator}")
        print(f"ANSWER: {result.get('submitted_answer', '')}")
        tool_trace = result.get("tool_trace", [])
        print(f"Tool trace ({len(tool_trace)} calls):")
        for entry in tool_trace:
            print(f"  * {entry[:120]}")
        print(f"Decision trace: {result.get('decision_trace', [])}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("recovery_reason"):
            print(f"Recovery reason: {result['recovery_reason']}")
        print(separator, flush=True)


class CompositeHook(BaseAgentHook):
    """Dispatches events to multiple hooks."""

    def __init__(self, hooks: list[AgentHook]) -> None:
        self._hooks = list(hooks)

    def on_tool_start(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        for hook in self._hooks:
            hook.on_tool_start(tool_name, tool_args)

    def on_tool_end(self, tool_name: str, result: str) -> None:
        for hook in self._hooks:
            hook.on_tool_end(tool_name, result)

    def on_solve_start(self, task_id: str, question: str) -> None:
        for hook in self._hooks:
            hook.on_solve_start(task_id, question)

    def on_solve_end(self, task_id: str, result: dict[str, Any]) -> None:
        for hook in self._hooks:
            hook.on_solve_end(task_id, result)
