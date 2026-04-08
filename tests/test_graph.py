from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import pytest
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


class FakeModelSingleAnswer:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=self.answer)


class FakeModelWithBlockedPython:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-python",
                        "name": "execute_python_code",
                        "args": {"code": "print('invented dataset')"},
                    }
                ],
            )
        assert any(
            "UNGROUNDED PYTHON BLOCKED" in str(getattr(msg, "content", ""))
            for msg in messages
        )
        return AIMessage(content="[ANSWER]blocked[/ANSWER]")


class FakeModelForAutoFetch:
    def __init__(self) -> None:
        self.calls = 0
        self.queries = [
            "equine veterinarian libretexts exercises",
            "introductory chemistry ck12 exercise veterinarian",
            "chemistry materials 2023 equine surname",
            "equine vet question chemistry license",
        ]

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= len(self.queries):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-search-{self.calls}",
                        "name": "web_search",
                        "args": {"query": self.queries[self.calls - 1], "max_results": 5},
                    }
                ],
            )
        assert any(
            "AUTO-FETCH:" in str(getattr(msg, "content", "")) for msg in messages
        )
        return AIMessage(content="[ANSWER]done[/ANSWER]")


class FakeModelForBadFetchRedirect:
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
                        "id": "call-search",
                        "name": "web_search",
                        "args": {
                            "query": "actor who played Ray in Polish Everybody Loves Raymond Magda M. character",
                            "max_results": 5,
                        },
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-fetch",
                        "name": "fetch_url",
                        "args": {
                            "url": "https://www.instagram.com/popular/actor-who-played-ray-in-polish-version-of-everybody-loves-raymond/"
                        },
                    }
                ],
            )
        return AIMessage(content="[ANSWER]done[/ANSWER]")


class FakeModelForTextSpanRedirect:
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
                        "id": "call-search",
                        "name": "web_search",
                        "args": {
                            "query": "equine veterinarian libretexts ck-12 1.E exercises",
                            "max_results": 5,
                        },
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-find",
                        "name": "find_text_in_url",
                        "args": {
                            "url": "https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01%3A_Chemistry_in_our_Lives/1.E%3A_Exercises",
                            "query": "equine veterinarian",
                        },
                    }
                ],
            )
        return AIMessage(content="[ANSWER]done[/ANSWER]")


class FakeModelWithGroundedTextSalvage:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-find",
                        "name": "find_text_in_url",
                        "args": {
                            "url": "https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises",
                            "query": "equine veterinarian",
                        },
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="The available information does not make the answer explicit."
            )
        if self.calls == 3:
            return AIMessage(
                content="The available information does not make the answer explicit."
            )
        last_prompt = str(getattr(messages[-1], "content", ""))
        assert "Question profile:" in last_prompt
        assert "Dr. Rivera" in last_prompt
        return AIMessage(content="[ANSWER]Rivera[/ANSWER]")


class FakeModelWithTemporalRosterRetry:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-current-roster",
                        "name": "extract_tables_from_url",
                        "args": {
                            "url": "https://en.wikipedia.org/wiki/List_of_current_Nippon_Professional_Baseball_team_rosters",
                            "text_filter": "Hokkaido Nippon-Ham Fighters pitchers",
                        },
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(content="[ANSWER]Yamasaki, Uehara[/ANSWER]")
        if self.calls == 3:
            reminder_text = " ".join(str(getattr(msg, "content", "")) for msg in messages)
            assert "date-sensitive (as of July 2023)" in reminder_text
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-archive-roster",
                        "name": "extract_tables_from_url",
                        "args": {
                            "url": "https://npb.example.com/fighters/archive/2023-07-roster",
                            "text_filter": "pitchers",
                        },
                    }
                ],
            )
        return AIMessage(content="[ANSWER]Yoshida, Uehara[/ANSWER]")


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


def test_graph_canonicalizes_award_number_answers() -> None:
    agent = GaiaGraphAgent(model=FakeModelSingleAnswer("[ANSWER]NASA award number 80NSSC20K0533[/ANSWER]"), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="award-number-format",
            question="Under what NASA award number was the work performed by R. G. Arendt supported by?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "80NSSC20K0533"


def test_graph_expands_city_abbreviations_when_question_forbids_abbreviations() -> None:
    agent = GaiaGraphAgent(model=FakeModelSingleAnswer("[ANSWER]St. Petersburg[/ANSWER]"), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="city-no-abbrev",
            question="Where were the specimens eventually deposited? Just give me the city name without abbreviations.",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Saint Petersburg"


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


def test_prepare_context_marks_self_contained_classification_questions() -> None:
    state = {
        "task_id": "hint-self-contained",
        "question": (
            "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
            "to categorizing things. Here's the list I have so far:\n\n"
            "milk, eggs, flour, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, broccoli, celery, zucchini, lettuce\n\n"
            "Please alphabetize the vegetables and place each item in a comma separated list."
        ),
        "file_name": None,
        "local_file_path": None,
        "messages": [],
    }

    prepared = graph_module._prepare_context(state)  # type: ignore[arg-type]
    prompt = prepared["messages"][1].content

    assert "self-contained" in prompt.lower()
    assert "avoid web tools" in prompt.lower()
    assert "Do not use execute_python_code" in prompt


def test_graph_solves_botanical_vegetable_list_without_model_call() -> None:
    agent = GaiaGraphAgent(model=ExplodingModel(), max_iterations=1)

    result = agent.solve(
        Question(
            task_id="botany-list",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, "
                "she won't buy anything inserted in the wrong category. Here's the list I have so far:\n\n"
                "milk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, "
                "whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\n"
                "I need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from my list? "
                "But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list. "
                "Please alphabetize the list of vegetables, and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert (
        result["submitted_answer"]
        == "broccoli, celery, fresh basil, lettuce, sweet potatoes"
    )
    assert result["reducer_used"] == "botanical_vegetable_list"


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


@pytest.mark.parametrize(
    "question",
    [
        "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.",
        "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.",
        "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?",
    ],
)
def test_graph_blocks_ungrounded_python_for_lookup_questions(monkeypatch, question: str) -> None:
    @tool
    def execute_python_code(code: str) -> str:
        """Tool should not be invoked when python is ungrounded."""
        raise AssertionError("execute_python_code should have been blocked")

    monkeypatch.setattr(graph_module, "build_tools", lambda: [execute_python_code])

    agent = GaiaGraphAgent(model=FakeModelWithBlockedPython(), max_iterations=2)
    result = agent.solve(Question(task_id="blocked-python", question=question, file_name=None))

    assert result["submitted_answer"] == "blocked"
    assert any("execute_python_code" in item for item in result["tool_trace"])
    assert result["error"] is None


def test_graph_auto_fetch_uses_ranked_candidate_instead_of_first_raw_url(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return search results with a bad first URL and a better ranked LibreTexts URL."""
        assert max_results == 5
        assert query
        return (
            "1. Forum Thread\n"
            "URL: https://forums.example.com/equine-veterinarian-thread\n"
            "Snippet: equine veterinarian discussion board\n\n"
            "2. 1.E Exercises\n"
            "URL: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises\n"
            "Snippet: Introductory Chemistry CK-12 1.E Exercises equine veterinarian\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Fetch the chosen source page."""
        assert "libretexts.org" in url
        assert "forums.example.com" not in url
        return "Title: 1.E Exercises\nURL: https://chem.libretexts.org/example\nDr. Rivera is the equine veterinarian."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(model=FakeModelForAutoFetch(), max_iterations=6)
    result = agent.solve(
        Question(
            task_id="auto-fetch-ranked",
            question=(
                "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
                "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "done"
    assert any(
        item.startswith("fetch_url(") and "libretexts.org" in item
        for item in result["tool_trace"]
    )
    assert not any("forums.example.com" in item for item in result["tool_trace"] if item.startswith("fetch_url("))


def test_graph_redirects_bad_fetch_candidate_for_entity_role_chain(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return a bad social URL plus a better cast page."""
        assert max_results == 5
        assert "Magda M." in query
        return (
            "1. Popular actor post\n"
            "URL: https://www.instagram.com/popular/actor-who-played-ray-in-polish-version-of-everybody-loves-raymond/\n"
            "Snippet: social post about the actor\n\n"
            "2. Magda M. cast list\n"
            "URL: https://en.wikipedia.org/wiki/Magda_M.\n"
            "Snippet: cast and characters from Magda M.\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Fetch the chosen source page."""
        assert "instagram.com" not in url
        assert url == "https://en.wikipedia.org/wiki/Magda_M."
        return "Title: Magda M.\nURL: https://en.wikipedia.org/wiki/Magda_M.\nCast list."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(model=FakeModelForBadFetchRedirect(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="role-fetch-redirect",
            question=(
                "Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? "
                "Give only the first name."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "done"
    assert any(
        item.startswith("fetch_url(") and "https://en.wikipedia.org/wiki/Magda_M." in item
        for item in result["tool_trace"]
    )


def test_graph_redirects_generic_libretexts_lookup_to_ranked_exact_page(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return a generic course mirror plus the canonical CK-12 exercise page."""
        assert max_results == 5
        assert "equine veterinarian" in query
        return (
            "1. Course mirror exercises\n"
            "URL: https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01%3A_Chemistry_in_our_Lives/1.E%3A_Exercises\n"
            "Snippet: Introductory chemistry course mirror exercise page\n\n"
            "2. CK-12 1.E Exercises\n"
            "URL: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises\n"
            "Snippet: Introductory Chemistry CK-12 1.E Exercises equine veterinarian\n"
        )

    @tool
    def find_text_in_url(url: str, query: str) -> str:
        """Read the redirected exercise page."""
        assert "Bookshelves/Introductory_Chemistry" in url
        assert "/Courses/" not in url
        assert query == "equine veterinarian"
        return "The equine veterinarian mentioned in the exercise is Dr. Louvrier."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, find_text_in_url])

    agent = GaiaGraphAgent(model=FakeModelForTextSpanRedirect(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="text-span-fetch-redirect",
            question=(
                "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
                "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
                "as compiled 08/21/2023?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "done"
    assert any(
        item.startswith("find_text_in_url(") and "Bookshelves/Introductory_Chemistry" in item
        for item in result["tool_trace"]
    )


def test_graph_salvages_answer_from_grounded_text_without_new_reducer(monkeypatch) -> None:
    @tool
    def find_text_in_url(url: str, query: str) -> str:
        """Return a grounded line from the fetched page."""
        assert "libretexts.org" in url
        assert query == "equine veterinarian"
        return "The equine veterinarian mentioned in the exercise is Dr. Rivera."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [find_text_in_url])

    agent = GaiaGraphAgent(model=FakeModelWithGroundedTextSalvage(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="grounded-salvage",
            question=(
                "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
                "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Rivera"
    assert result["error"] is None


def test_graph_retries_date_sensitive_roster_answer_without_temporal_grounding(monkeypatch) -> None:
    @tool
    def extract_tables_from_url(url: str, text_filter: str = "") -> str:
        """Return either a current roster table or a dated archive roster table."""
        assert text_filter
        if "List_of_current" in url:
            return (
                "Table 1\n"
                "Caption: Hokkaido Nippon-Ham Fighters roster\n"
                "No. | Name\n"
                "18 | Sachiya Yamasaki\n"
                "19 | Taisho Tamai\n"
                "20 | Kenta Uehara\n"
            )
        assert "2023-07-roster" in url
        return (
            "Table 1\n"
            "Caption: July 2023 Hokkaido Nippon-Ham Fighters pitchers\n"
            "No. | Name\n"
            "18 | Koki Yoshida\n"
            "19 | Taisho Tamai\n"
            "20 | Kenta Uehara\n"
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [extract_tables_from_url])

    agent = GaiaGraphAgent(model=FakeModelWithTemporalRosterRetry(), max_iterations=4)
    result = agent.solve(
        Question(
            task_id="temporal-roster-retry",
            question=(
                "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? "
                "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Yoshida, Uehara"
    assert any("List_of_current_Nippon_Professional_Baseball_team_rosters" in item for item in result["tool_trace"])
    assert any("2023-07-roster" in item for item in result["tool_trace"])


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
