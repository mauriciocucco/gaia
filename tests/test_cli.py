from pathlib import Path

from hf_gaia_agent.api_client import Question
from hf_gaia_agent.cli import solve_questions


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
