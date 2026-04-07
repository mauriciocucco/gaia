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


class FakeModelWithToolFailureAndInvalidFinal:
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
            content="The answer cannot be determined from the available information."
        )


class FakeModelWithEvidenceSalvage:
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
                        "name": "extract_tables_from_url",
                        "args": {
                            "url": "https://example.com/olympics",
                            "text_filter": "athletes",
                        },
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="The available information does not make the answer explicit."
            )
        raise AssertionError(
            "Structured evidence solver should answer before a salvage model call."
        )


class FakeModelWithMissingAttachmentMeta:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(
            content="No chess position image or data provided to analyze and determine the winning move for black."
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


def test_graph_does_not_submit_tool_failure_as_answer(monkeypatch) -> None:
    @tool
    def analyze_youtube_video(url: str, question: str) -> str:
        """Return an operational failure message from mocked video analysis."""
        assert url == "https://www.youtube.com/watch?v=L1vXCYZAYYM"
        assert "highest number of bird species" in question.lower()
        return "Failed to download video L1vXCYZAYYM: 'yt-dlp' not found on PATH. Install it first."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [analyze_youtube_video])

    agent = GaiaGraphAgent(model=FakeModelWithToolFailureAndInvalidFinal(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="2c",
            question="In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == ""
    assert result["error"] == "Model produced an invalid non-answer."


def test_graph_salvages_answer_from_existing_tool_evidence(monkeypatch) -> None:
    @tool
    def extract_tables_from_url(url: str, text_filter: str = "") -> str:
        """Return table evidence from mocked page extraction."""
        assert url == "https://example.com/olympics"
        assert text_filter == "athletes"
        return (
            "Table 1\n"
            "Participating National Olympic Committees\n"
            "Cuba (1)\n"
            "Panama (1)\n"
            "Argentina (81)"
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [extract_tables_from_url])

    agent = GaiaGraphAgent(model=FakeModelWithEvidenceSalvage(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="2c-salvage",
            question=(
                "What country had the least number of athletes at the 1928 Summer Olympics? "
                "If there's a tie for a number of athletes, return the first in alphabetical order. "
                "Give the IOC country code as your answer."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "CUB"
    assert any("extract_tables_from_url" in item for item in result["tool_trace"])
    assert result["error"] is None


def test_graph_marks_missing_attachment_meta_answer_invalid() -> None:
    agent = GaiaGraphAgent(model=FakeModelWithMissingAttachmentMeta(), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="2d",
            question="Review the chess position provided in the image. It is black's turn.",
            file_name="board.png",
        )
    )

    assert result["submitted_answer"] == ""
    assert result["error"] == "Required attachment was not available locally."


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


def test_graph_solves_non_commutative_subset_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="5",
            question=(
                "Given this table defining * on the set S = {a, b, c, d, e}\n\n"
                "|*|a|b|c|d|e|\n"
                "|---|---|---|---|---|---|\n"
                "|a|a|b|c|b|d|\n"
                "|b|b|c|a|e|c|\n"
                "|c|c|a|b|b|a|\n"
                "|d|b|e|b|e|d|\n"
                "|e|d|b|a|d|c|\n\n"
                "provide the subset of S involved in any possible counter-examples that prove * is not commutative. "
                "Provide your answer as a comma separated list of the elements in the set in alphabetical order."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "b, e"
    assert result["tool_trace"] == ["heuristic(non_commutative_subset)"]


def test_graph_no_longer_short_circuits_benchmark_specific_questions() -> None:
    class FakeModelWithoutTools:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            self.calls += 1
            return AIMessage(content="[ANSWER]model-driven answer[/ANSWER]")

    model = FakeModelWithoutTools()
    agent = GaiaGraphAgent(model=model, max_iterations=2)
    result = agent.solve(
        Question(
            task_id="7b",
            question="Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.",
            file_name=None,
        )
    )

    assert model.calls == 1
    assert result["submitted_answer"] == "model-driven answer"
    assert result["tool_trace"] == []


def test_prepare_context_includes_research_hints_for_linked_article_questions() -> None:
    state = {
        "task_id": "hint-1",
        "question": (
            "On June 6, 2023, an article was published. The article links to a paper at the bottom. "
            "Find the paper and answer a question about it."
        ),
        "file_name": None,
        "local_file_path": None,
        "messages": [],
    }

    prepared = graph_module._prepare_context(state)  # type: ignore[arg-type]
    prompt = prepared["messages"][1].content

    assert "Research hints:" in prompt
    assert "extract_links_from_url" in prompt


def test_graph_marks_audio_access_meta_answer_invalid() -> None:
    class FakeModelWithMissingAudioMeta:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(
                content=(
                    "I could not access the audio file to listen to the recipe. Could you please provide the audio file "
                    "or a link to it so I can help you extract the ingredients for the filling?"
                )
            )

    agent = GaiaGraphAgent(model=FakeModelWithMissingAudioMeta(), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="10",
            question="Audio attachment question",
            file_name="recipe.mp3",
        )
    )

    assert result["submitted_answer"] == ""
    assert result["error"] == "Required attachment was not available locally."


def test_graph_rejects_hallucinated_answer_when_required_attachment_is_missing() -> None:
    class FakeModelWithHallucinatedAttachmentAnswer:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="cornstarch, ripe strawberries, sugar, water")

    agent = GaiaGraphAgent(model=FakeModelWithHallucinatedAttachmentAnswer(), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="11",
            question=(
                "I've attached the recipe as Strawberry pie.mp3. Could you please listen to the recipe and list all "
                "of the ingredients that my friend described?"
            ),
            file_name="recipe.mp3",
        )
    )

    assert result["submitted_answer"] == ""
    assert result["error"] == "Required attachment was not available locally."
