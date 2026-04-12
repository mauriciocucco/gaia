from pathlib import Path
import sys
from uuid import uuid4

import hf_gaia_agent.cli as cli_module
import hf_gaia_agent.graph as graph_module
from hf_gaia_agent.api_client import Question
from hf_gaia_agent.cli import build_parser, graph_command, debug_command
from hf_gaia_agent.hooks import BaseAgentHook
from hf_gaia_agent.runner import solve_question_by_id, solve_questions
from hf_gaia_agent.graph import GaiaGraphAgent


class FakeClient:
    def __init__(self, question: Question) -> None:
        self.question = question

    def list_questions(self) -> list[Question]:
        return [self.question]

    def download_file(self, task_id: str, file_name: str | None = None) -> Path:
        raise RuntimeError(f"No file path associated with task_id {task_id}.")


class FakeAgent:
    def solve(self, question: Question, *, local_file_path: str | Path | None = None) -> dict[str, object]:
        assert local_file_path is None
        return {
            "task_id": question.task_id,
            "question": question.question,
            "submitted_answer": "",
            "file_name": question.file_name,
            "tool_trace": [],
            "error": "Required attachment was not available locally.",
        }


class FakeAgentWithHook:
    def __init__(self, hook: BaseAgentHook) -> None:
        self._hook = hook

    def solve(
        self,
        question: Question,
        *,
        local_file_path: str | Path | None = None,
    ) -> dict[str, object]:
        del local_file_path
        self._hook.on_solve_start(question.task_id, question.question)
        result = {
            "task_id": question.task_id,
            "question": question.question,
            "submitted_answer": "ok",
            "file_name": question.file_name,
            "tool_trace": [],
            "decision_trace": [],
            "evidence_used": [],
            "reducer_used": None,
            "fallback_reason": None,
            "error": None,
        }
        self._hook.on_solve_end(question.task_id, result)
        return result


class RecordingHook(BaseAgentHook):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def on_solve_start(self, task_id: str, question: str) -> None:
        del question
        self.events.append(("start", task_id))

    def on_solve_end(self, task_id: str, result: dict[str, object]) -> None:
        del result
        self.events.append(("end", task_id))


def test_solve_questions_preserves_attachment_error_when_agent_also_errors() -> None:
    question = Question(
        task_id="attach-1",
        question="Review the chess position provided in the image.",
        file_name="board.png",
    )

    results = solve_questions(FakeClient(question), FakeAgent())

    assert len(results) == 1
    assert results[0]["attachment_path"] is None
    assert results[0]["attachment_error"] == "No file path associated with task_id attach-1."
    assert results[0]["error"] == "Required attachment was not available locally."


def test_solve_question_by_id_preserves_attachment_error_without_printing() -> None:
    question = Question(
        task_id="attach-2",
        question="Review the attached board image.",
        file_name="board.png",
    )

    results = solve_question_by_id(FakeClient(question), FakeAgent(), "attach")

    assert len(results) == 1
    assert results[0]["attachment_path"] is None
    assert results[0]["attachment_error"] == "No file path associated with task_id attach-2."
    assert results[0]["error"] == "Required attachment was not available locally."


def test_runner_does_not_duplicate_agent_lifecycle_hooks() -> None:
    question = Question(task_id="hook-1", question="What is 2+2?")
    hook = RecordingHook()

    results = solve_question_by_id(
        FakeClient(question),
        FakeAgentWithHook(hook),
        "hook",
        hook=hook,
    )

    assert len(results) == 1
    assert hook.events == [("start", "hook-1"), ("end", "hook-1")]


def test_build_parser_accepts_graph_subcommand_output_after_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["graph", "--format", "mermaid", "--output", "graph.mmd"])

    assert args.command == "graph"
    assert args.format == "mermaid"
    assert args.output == Path("graph.mmd")
    assert args.func is graph_command


def test_graph_command_main_renders_mermaid_without_api_key(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_module, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        graph_module,
        "_build_model",
        lambda: (_ for _ in ()).throw(
            AssertionError("Runtime model should not be initialized for graph rendering.")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["hf-gaia-agent", "graph", "--format", "mermaid"])

    exit_code = cli_module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "prepare_context" in captured.out
    assert "retry_invalid_answer" in captured.out
    assert "finalize" in captured.out


def test_graph_command_writes_output_file() -> None:
    destination = Path(".tmp_pytest") / f"graph-{uuid4().hex}.mmd"
    args = build_parser().parse_args(
        ["graph", "--format", "mermaid", "--output", str(destination)]
    )

    exit_code = graph_command(args)

    assert exit_code == 0
    assert destination.exists()
    content = destination.read_text(encoding="utf-8")
    assert "prepare_context" in content
    assert "tools" in content


def test_graph_introspection_mermaid_contains_expected_nodes() -> None:
    agent = GaiaGraphAgent.for_graph_introspection()

    rendered = agent.render_graph_mermaid()

    assert "prepare_context" in rendered
    assert "agent" in rendered
    assert "tools" in rendered
    assert "retry_invalid_answer" in rendered
    assert "finalize" in rendered


def test_graph_introspection_ascii_contains_expected_edges() -> None:
    agent = GaiaGraphAgent.for_graph_introspection()

    rendered = agent.render_graph_ascii()

    assert "[prepare_context]" in rendered
    assert "--> agent" in rendered
    assert "-?-> tools" in rendered
    assert "-?-> finalize" in rendered


def test_build_parser_accepts_debug_question_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["debug-question", "abc123", "7"])

    assert args.command == "debug-question"
    assert args.targets == ["abc123", "7"]
    assert args.func is debug_command
