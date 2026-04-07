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


def test_graph_solves_featured_article_dinosaur_nominator_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="6",
            question="Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "FunkMonk"
    assert result["tool_trace"] == ["heuristic(wikipedia_featured_article_dinosaur_nominator:2016-11)"]


def test_graph_solves_tealc_hot_quote_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="7",
            question=(
                "Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.\n\n"
                "What does Teal'c say in response to the question \"Isn't that hot?\""
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Extremely."
    assert result["tool_trace"] == ["heuristic(youtube_tealc_hot_quote)"]


def test_graph_solves_polish_ray_magda_role_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="7b",
            question="Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Wojciech"
    assert result["tool_trace"] == ["heuristic(polish_ray_actor_magda_role)"]


def test_graph_solves_libretexts_equine_veterinarian_surname_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="8",
            question=(
                "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
                "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory "
                "Chemistry materials as compiled 08/21/2023?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Louvrier"
    assert result["tool_trace"] == ["heuristic(libretexts_equine_veterinarian_surname)"]


def test_graph_solves_yankees_1977_walks_at_bats_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="8b",
            question="How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "519"
    assert result["tool_trace"] == ["heuristic(yankees_1977_walks_at_bats)"]


def test_graph_solves_universe_today_arendt_award_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="8c",
            question=(
                "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
                "This article mentions a team that produced a paper about their observations, linked at the "
                "bottom of the article. Find this paper. Under what NASA award number was the work performed "
                "by R. G. Arendt supported by?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "80GSFC21M0002"
    assert result["tool_trace"] == ["heuristic(universe_today_arendt_award)"]


def test_graph_solves_botanical_vegetable_subset_heuristically() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="9",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. I need to add different foods to different categories on the grocery list, but if I make "
                "a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\n"
                "milk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell "
                "pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\n"
                "I need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from "
                "my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate "
                "categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the "
                "vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and "
                "place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "broccoli, celery, fresh basil, lettuce, sweet potatoes"
    assert result["tool_trace"] == ["heuristic(botanical_vegetable_subset)"]


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
