import json
from pathlib import Path

import httpx
import pytest

from hf_gaia_agent.api_client import AnswerPayload, Question, ScoringAPIClient

from conftest import make_case_dir as _case_dir


def test_list_questions_parses_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/questions"
        return httpx.Response(
            200,
            json=[
                {
                    "task_id": "1",
                    "question": "What is 2+2?",
                    "Level": "1",
                    "file_name": None,
                }
            ],
        )

    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        download_dir=_case_dir("list-questions"),
    )
    questions = client.list_questions()
    assert len(questions) == 1
    assert questions[0].task_id == "1"
    assert questions[0].question == "What is 2+2?"
    assert questions[0].level == "1"


def test_download_file_writes_response_to_disk() -> None:
    body = b"hello world"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/files/abc"
        return httpx.Response(
            200,
            content=body,
            headers={"content-disposition": 'attachment; filename="note.txt"'},
        )

    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        download_dir=_case_dir("download-file"),
    )
    result = client.download_file("abc")
    assert result.read_bytes() == body
    assert result.name == "abc__note.txt"


def test_download_file_falls_back_to_task_and_filename_path() -> None:
    body = b"png-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/abc":
            return httpx.Response(404)
        if request.url.path == "/files/abc/board.png":
            return httpx.Response(
                200,
                content=body,
                headers={"content-disposition": 'attachment; filename="board.png"'},
            )
        raise AssertionError(f"Unexpected path {request.url.path}")

    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        download_dir=_case_dir("download-file-fallback"),
    )
    result = client.download_file("abc", "board.png")
    assert result.read_bytes() == body
    assert result.name == "abc__board.png"


def test_submit_answers_serializes_expected_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/submit"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {
            "username": "mauri",
            "agent_code": "https://huggingface.co/spaces/example/tree/main",
            "answers": [{"task_id": "1", "submitted_answer": "4"}],
        }
        return httpx.Response(
            200,
            json={
                "username": "mauri",
                "score": 50.0,
                "correct_count": 1,
                "total_attempted": 1,
                "message": "ok",
                "timestamp": "2026-04-06T00:00:00Z",
            },
        )

    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        download_dir=_case_dir("submit-answers"),
    )
    response = client.submit_answers(
        "mauri",
        "https://huggingface.co/spaces/example/tree/main",
        [AnswerPayload(task_id="1", submitted_answer="4")],
    )
    assert response.score == 50.0


def test_list_questions_propagates_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        download_dir=_case_dir("timeout"),
    )
    with pytest.raises(httpx.ReadTimeout):
        client.list_questions()


def test_retry_recovers_from_transient_503(monkeypatch) -> None:
    """_RetryTransport retries on 503 then returns the good response."""
    monkeypatch.setattr("time.sleep", lambda _: None)  # skip real delays

    from hf_gaia_agent.api_client import _RetryTransport

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=[])

    inner = httpx.MockTransport(handler)
    transport = _RetryTransport(inner, retries=3)
    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=transport,
        download_dir=_case_dir("retry-503"),
    )
    questions = client.list_questions()
    assert questions == []
    assert call_count == 3


def test_retry_gives_up_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _: None)

    from hf_gaia_agent.api_client import _RetryTransport

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    inner = httpx.MockTransport(handler)
    transport = _RetryTransport(inner, retries=2)
    client = ScoringAPIClient(
        base_url="https://example.test",
        transport=transport,
        download_dir=_case_dir("retry-fail"),
    )
    response = client._client.get("/questions")
    assert response.status_code == 503


def test_question_from_api_handles_missing_optional_fields() -> None:
    raw = {"task_id": "42", "question": "Hello?"}
    q = Question.from_api(raw)
    assert q.task_id == "42"
    assert q.level is None
    assert q.file_name is None
