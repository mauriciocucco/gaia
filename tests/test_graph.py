from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import hf_gaia_agent.graph as graph_module

from hf_gaia_agent.api_client import Question
from hf_gaia_agent.graph import GaiaGraphAgent


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "calculate",
                        "args": {"expression": "6 * 7"},
                    }
                ],
            )
        return AIMessage(content="[ANSWER]42[/ANSWER]")


class FakeModelWithInvalidAnswer:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="The answer is not explicitly stated in the available information."
            )
        return AIMessage(content="[ANSWER]7[/ANSWER]")


class FakeModelWithInvalidFinalAfterTool:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "analyze_youtube_video",
                        "args": {
                            "url": "https://www.youtube.com/watch?v=L1vXCYZAYYM",
                            "question": "What is the highest number of bird species to be on camera simultaneously?",
                        },
                    }
                ],
            )
        return AIMessage(
            content="The available information does not make the answer explicit."
        )


class ExplodingModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise AssertionError("Model should not be invoked for heuristic solve.")


def test_graph_runs_tools_and_normalizes_answer() -> None:
    base = Path(".test-artifacts") / f"graph-{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    attachment = base / "task.txt"
    attachment.write_text("Ignore this file.", encoding="utf-8")

    agent = GaiaGraphAgent(model=FakeModel(), max_iterations=3)
    result = agent.solve(
        Question(task_id="1", question="What is six times seven?", file_name="task.txt"),
        local_file_path=attachment,
    )

    assert result["submitted_answer"] == "42"
    assert any("calculate" in item for item in result["tool_trace"])


def test_graph_retries_when_model_returns_non_answer() -> None:
    agent = GaiaGraphAgent(model=FakeModelWithInvalidAnswer(), max_iterations=3)
    result = agent.solve(
        Question(task_id="2", question="How many days are in a week?", file_name=None)
    )

    assert result["submitted_answer"] == "7"
    assert result["error"] is None


def test_graph_falls_back_to_tool_answer_when_final_response_is_invalid(monkeypatch) -> None:
    @tool
    def analyze_youtube_video(url: str, question: str) -> str:
        """Return a concrete count from mocked video analysis."""
        assert url == "https://www.youtube.com/watch?v=L1vXCYZAYYM"
        assert "highest number of bird species" in question.lower()
        return "The highest number of bird species visible simultaneously is 3."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [analyze_youtube_video])

    agent = GaiaGraphAgent(model=FakeModelWithInvalidFinalAfterTool(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="2b",
            question="In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "3"
    assert result["error"] is None


def test_graph_solves_reversed_opposite_prompt_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="3",
            question='.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "right"
    assert result["tool_trace"] == ["heuristic(reversed_text_opposite_word)"]


def test_graph_solves_wikipedia_album_count_heuristically(monkeypatch) -> None:
    def fake_counter(artist_name: str, start_year: int, end_year: int) -> int:
        assert artist_name == "Mercedes Sosa"
        assert start_year == 2000
        assert end_year == 2009
        return 3

    monkeypatch.setattr(
        graph_module,
        "count_wikipedia_studio_album_count_for_artist",
        fake_counter,
    )

    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="4",
            question="How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "3"
    assert result["tool_trace"] == ["heuristic(wikipedia_studio_album_count:Mercedes Sosa:2000-2009)"]
