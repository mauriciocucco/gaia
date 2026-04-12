from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import pytest
import hf_gaia_agent.graph as graph_module

from hf_gaia_agent.api_client import Question
from hf_gaia_agent.fallbacks.article_to_paper import ArticleToPaperFallback
from hf_gaia_agent.fallbacks.role_chain import RoleChainFallback
from hf_gaia_agent.fallbacks.roster import RosterFallback
from hf_gaia_agent.fallbacks.text_span import TextSpanFallback
from hf_gaia_agent.graph import GaiaGraphAgent
from hf_gaia_agent.graph.evidence_support import (
    grounded_temporal_roster_answer,
    has_temporal_roster_grounding_gap,
    has_temporally_grounded_roster_evidence,
    requires_temporal_roster_retry,
)


def _tools_by_name() -> dict[str, object]:
    return {tool_.name: tool_ for tool_ in graph_module.build_tools()}


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
        raise AssertionError("Model should not be invoked for prompt reducer solve.")


class FakeModelSingleAnswer:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=self.answer)


class FakeModelForBotanicalGroundingRetry:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="[ANSWER]Bell pepper, Broccoli, Celery, Corn, Green beans, Lettuce, Sweet potatoes, Zucchini[/ANSWER]"
            )
        if self.calls == 2:
            assert any(
                "botanical classification task" in str(getattr(msg, "content", "")).lower()
                for msg in messages
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-botany-fetch",
                        "name": "fetch_url",
                        "args": {"url": "https://example.com/botany"},
                    }
                ],
            )
        return AIMessage(
            content="[ANSWER]broccoli, celery, fresh basil, lettuce, sweet potatoes[/ANSWER]"
        )


class RecordingModel:
    def __init__(self) -> None:
        self.messages = None

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="[ANSWER]ok[/ANSWER]")


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


class FakeModelForEntityRoleChainFallback:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if any(
            isinstance(message, SystemMessage)
            and "two-hop role-chain question" in str(getattr(message, "content", "")).lower()
            for message in messages
        ):
            return AIMessage(content="[ANSWER]Wojciech[/ANSWER]")
        return AIMessage(content="[ANSWER]Ryszard[/ANSWER]")


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
            assert "season-specific official player directory" in reminder_text
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


class FakeModelWithWrongConcreteMetricAnswer:
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
                        "id": "call-table",
                        "name": "extract_tables_from_url",
                        "args": {
                            "url": "https://www.baseball-reference.com/teams/NYY/1977.shtml",
                            "text_filter": "walks at bats",
                        },
                    }
                ],
            )
        return AIMessage(content="[ANSWER]595[/ANSWER]")


class FakeModelWithMetricTextFallback:
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
                        "id": "call-metric-fallback",
                        "name": "extract_tables_from_url",
                        "args": {
                            "url": "https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA",
                            "text_filter": "walks at bats",
                        },
                    }
                ],
            )
        raise AssertionError("Graph should finalize from auto fetch fallback before another model turn.")


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


def test_graph_prefers_structured_metric_answer_over_wrong_concrete_model_answer(monkeypatch) -> None:
    @tool
    def extract_tables_from_url(url: str, text_filter: str = "") -> str:
        """Return a batting table with enough evidence for the structured reducer."""
        assert url == "https://www.baseball-reference.com/teams/NYY/1977.shtml"
        assert text_filter == "walks at bats"
        return (
            "Table 1\n"
            "Caption: Batting\n"
            "Player | AB | BB\n"
            "Thurman Munson | 519 | 82\n"
            "Reggie Jackson | 589 | 66\n"
            "Graig Nettles | 566 | 80\n"
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [extract_tables_from_url])
    monkeypatch.setattr(GaiaGraphAgent, "_route_after_tools", lambda self, state: "agent")

    agent = GaiaGraphAgent(model=FakeModelWithWrongConcreteMetricAnswer(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="metric-preferred",
            question="How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "519"
    assert result["reducer_used"] == "metric_row_lookup"
    assert result["error"] is None


def test_graph_auto_fetches_text_when_metric_table_extraction_finds_no_html_tables(monkeypatch) -> None:
    @tool
    def extract_tables_from_url(url: str, text_filter: str = "") -> str:
        """Simulate a stats page whose table is only available in fetched markdown/text."""
        assert url == "https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA"
        assert text_filter == "walks at bats"
        return "URL: https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA\nNo readable HTML tables found."

    @tool
    def fetch_url(url: str) -> str:
        """Return a markdown stats table from the same page."""
        assert url == "https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA"
        return (
            "Title: 1977 New York Yankees Hitting Stats by Baseball Almanac\n"
            "URL: https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA\n\n"
            "| Name | G | AB | RBI | BB |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Thurman Munson | 149 | 519 | 100 | 82 |\n"
            "| Reggie Jackson | 146 | 525 | 110 | 74 |\n"
            "| Graig Nettles | 158 | 589 | 107 | 68 |\n"
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [extract_tables_from_url, fetch_url])

    agent = GaiaGraphAgent(model=FakeModelWithMetricTextFallback(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="metric-text-fallback",
            question="How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "519"
    assert result["reducer_used"] == "metric_row_lookup"
    assert any("[auto_fallback_from_extract_tables]" in item for item in result["tool_trace"])
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


def test_graph_prefers_structured_award_number_over_placeholder_model_answer(monkeypatch) -> None:
    @tool
    def find_text_in_url(url: str, query: str) -> str:
        """Return a grounded award number snippet."""
        assert "northwestern.edu" in url
        assert query == "NASA award number"
        return "The study was supported by NASA (award number 80GSFC21M0002)."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [find_text_in_url])

    class FakeModelWithPlaceholderAwardAnswer:
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
                            "id": "call-award",
                            "name": "find_text_in_url",
                            "args": {
                                "url": "https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1",
                                "query": "NASA award number",
                            },
                        }
                    ],
                )
            return AIMessage(content="[ANSWER]N/A[/ANSWER]")

    agent = GaiaGraphAgent(model=FakeModelWithPlaceholderAwardAnswer(), max_iterations=2)
    result = agent.solve(
        Question(
            task_id="award-number-evidence-wins",
            question="Under what NASA award number was the work performed by R. G. Arendt supported by?",
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "80GSFC21M0002"
    assert result["reducer_used"] == "award_number"
    assert result["error"] is None


def test_graph_retries_non_identifier_award_answer() -> None:
    state = {
        "question": "Under what NASA award number was the work performed by R. G. Arendt supported by?",
        "messages": [AIMessage(content="National")],
        "iterations": 1,
        "max_iterations": 3,
    }

    assert GaiaGraphAgent._route_after_agent(state) == "retry_invalid_answer"


def test_graph_article_to_paper_auto_retries_links_and_redirects_fetch(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return the target Universe Today article."""
        assert "Carolyn Collins Petersen" in query
        return (
            "1. There Are Hundreds of Mysterious Filaments at the Center of the Milky Way\n"
            "URL: https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way\n"
            "Snippet: Carolyn Collins Petersen June 6, 2023 Universe Today article\n"
        )

    @tool
    def extract_links_from_url(url: str, text_filter: str = "", max_results: int = 20, same_domain_only: bool = False) -> str:
        """Fail with the filter, succeed without it."""
        assert "universetoday.com/articles/" in url
        if text_filter:
            return "No matching links found."
        return (
            "1. Mysterious dashes revealed in Milky Way's Center\n"
            "URL: https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1\n"
            "Snippet: For Journalists news release with the published paper link\n\n"
            "2. The Population of the Galactic Center Filaments: Position Angle Distribution Reveals a Degree-scale Collimated Outflow from Sgr A* along the Galactic Plane\n"
            "URL: https://iopscience.iop.org/article/10.3847/2041-8213/acd54b\n"
            "Snippet: Published paper in The Astrophysical Journal Letters\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Fetch the redirected linked source instead of the original article."""
        assert "universetoday.com/articles/" not in url
        assert "northwestern.edu" in url or "iopscience.iop.org" in url
        if "northwestern.edu" in url:
            return (
                "Title: Mysterious dashes revealed in Milky Way's center\n"
                "URL: https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1\n\n"
                "The study was supported by NASA (award number 80GSFC21M0002)."
            )
        return (
            "Title: Published paper\n"
            "URL: https://iopscience.iop.org/article/10.3847/2041-8213/acd54b\n\n"
            "R. G. Arendt was supported by NASA award number 80GSFC21M0002."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, extract_links_from_url, fetch_url])

    class FakeModelArticleToPaper:
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
                                "query": "site:universetoday.com Carolyn Collins Petersen June 6 2023",
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
                            "id": "call-links",
                            "name": "extract_links_from_url",
                            "args": {
                                "url": "https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way",
                                "text_filter": "paper",
                            },
                        }
                    ],
                )
            return AIMessage(content="[ANSWER]N/A[/ANSWER]")

    agent = GaiaGraphAgent(model=FakeModelArticleToPaper(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="article-to-paper-redirect",
            question=(
                "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
                "This article mentions a team that produced a paper about their observations, linked at the bottom of the article. "
                "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "80GSFC21M0002"
    assert any("[auto_retry_without_text_filter]" in item for item in result["tool_trace"])
    assert any(
        "[auto_followup_from_links]" in item and "universetoday.com/articles/" not in item
        for item in result["tool_trace"]
    )


def test_graph_article_identifier_fallback_fetches_external_candidate(monkeypatch) -> None:
    @tool
    def fetch_url(url: str) -> str:
        """Return the Northwestern for-journalists page with the award number."""
        assert "northwestern.edu" in url
        return (
            "Title: Mysterious dashes revealed in Milky Way's center\n"
            "URL: https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1\n\n"
            "The study was supported by NASA (award number 80GSFC21M0002)."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [fetch_url])

    state = {
        "question": (
            "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
            "This article mentions a team that produced a paper about their observations, linked at the bottom of the article. "
            "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
        ),
        "messages": [],
        "ranked_candidates": [
            {
                "title": "Mysterious dashes revealed in Milky Way's center",
                "url": "https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1",
                "snippet": "For Journalists release with the linked paper",
                "origin_tool": "extract_links_from_url",
                "score": 95,
                "reasons": ("linked_source", "primary_source_hint"),
            }
        ],
    }

    result = ArticleToPaperFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "80GSFC21M0002"
    assert result["reducer_used"] == "award_number"


def test_graph_article_identifier_fallback_searches_by_supported_subject(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return a paper page when searching by supported subject."""
        assert max_results == 5
        assert "R. G. Arendt" in query
        return (
            "1. Published paper\n"
            "URL: https://iopscience.iop.org/article/10.3847/2041-8213/ac982a/pdf\n"
            "Snippet: Work by R.G.A. was supported by NASA under award No. 80GSFC21M0002.\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Return the fetched paper text with the award number."""
        assert "iopscience.iop.org" in url
        return (
            "Title: Published paper\n"
            "URL: https://iopscience.iop.org/article/10.3847/2041-8213/ac982a/pdf\n\n"
            "Work by R.G.A. was supported by NASA under award No. 80GSFC21M0002."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    state = {
        "question": (
            "This article links to a paper at the bottom. "
            "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
        ),
        "messages": [],
    }

    result = ArticleToPaperFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "80GSFC21M0002"
    assert result["reducer_used"] == "award_number"


def test_graph_article_identifier_fallback_searches_by_exact_paper_title_when_primary_source_is_blocked(
    monkeypatch,
) -> None:
    search_queries: list[str] = []

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return an external metadata page when searching by the exact linked paper title."""
        assert max_results == 5
        search_queries.append(query)
        if "Population of the Galactic Center Filaments" in query and "R. G. Arendt" in query:
            return (
                "1. Northwestern Scholars publication\n"
                "URL: https://arch.library.northwestern.edu/concern/publications/gx41mm28v\n"
                "Snippet: Work by R.G.A. was supported by NASA under award No. 80GSFC21M0002.\n"
            )
        return "No results found."

    @tool
    def find_text_in_url(url: str, query: str, max_matches: int = 1) -> str:
        """Fail direct text extraction for the blocked paper page."""
        assert max_matches == 1
        assert query in {"NASA award number", "supported by NASA"}
        return "No matches found."

    @tool
    def fetch_url(url: str) -> str:
        """Return captcha text for the paper, and grounded funding text for the external metadata page."""
        if "iopscience.iop.org" in url:
            return (
                "Title: Radware Bot Manager Captcha\n"
                "URL: https://validate.perfdrive.com/example\n\n"
                "Access blocked by captcha."
            )
        assert "arch.library.northwestern.edu" in url
        return (
            "Title: Northwestern Scholars publication\n"
            "URL: https://arch.library.northwestern.edu/concern/publications/gx41mm28v\n\n"
            "Work by R.G.A. was supported by NASA under award No. 80GSFC21M0002."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, find_text_in_url, fetch_url])

    state = {
        "question": (
            "This article links to a paper at the bottom. "
            "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
        ),
        "messages": [],
        "ranked_candidates": [
            {
                "title": (
                    "The Population of the Galactic Center Filaments: Position Angle Distribution "
                    "Reveals a Degree-scale Collimated Outflow from Sgr A* along the Galactic Plane"
                ),
                "url": "https://iopscience.iop.org/article/10.3847/2041-8213/acd54b",
                "snippet": "Published paper in The Astrophysical Journal Letters",
                "origin_tool": "extract_links_from_url",
                "score": 95,
                "reasons": ("linked_source", "primary_source_hint"),
            }
        ],
    }

    result = ArticleToPaperFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "80GSFC21M0002"
    assert result["reducer_used"] == "award_number"
    assert any(
        "Population of the Galactic Center Filaments" in query and "R. G. Arendt" in query
        for query in search_queries
    )


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


def test_graph_reversed_prompt_no_longer_short_circuits_before_graph() -> None:
    model = FakeModelSingleAnswer("[ANSWER]right[/ANSWER]")
    agent = GaiaGraphAgent(model=model, max_iterations=3)
    result = agent.solve(
        Question(
            task_id="3",
            question='.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "right"
    assert model.calls == 1
    assert result["tool_trace"] == []
    assert result["reducer_used"] is None


def test_graph_solves_non_commutative_subset_with_prompt_reducer() -> None:
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
    assert result["tool_trace"] == []
    assert result["decision_trace"] == ["prompt_reducer:non_commutative_subset"]
    assert result["reducer_used"] == "non_commutative_subset"


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
    assert "Do not answer from culinary/common usage or from memory." in prompt


def test_prepare_context_includes_decoded_question_hint() -> None:
    state = {
        "task_id": "hint-decoded",
        "question": '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI',
        "file_name": None,
        "local_file_path": None,
        "messages": [],
    }

    prepared = graph_module._prepare_context(state)  # type: ignore[arg-type]
    prompt = prepared["messages"][1].content

    assert "Decoded question hint:" in prompt
    assert 'write the opposite of the word "left" as the answer.' in prompt


def test_graph_botanical_prompt_no_longer_short_circuits_before_graph() -> None:
    model = FakeModelSingleAnswer("[ANSWER]broccoli, celery, fresh basil, lettuce, sweet potatoes[/ANSWER]")
    agent = GaiaGraphAgent(model=model, max_iterations=1)

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

    assert result["submitted_answer"] == ""
    assert model.calls == 1
    assert result["tool_trace"] == []
    assert result["reducer_used"] is None
    assert result["error"] == "Botanical classification answer lacked grounded evidence."


def test_graph_retries_botanical_classification_until_it_has_grounding(monkeypatch) -> None:
    @tool
    def fetch_url(url: str) -> str:
        """Return grounded botanical classification text."""
        assert url == "https://example.com/botany"
        return (
            "Title: Botanical food classification\n"
            "URL: https://example.com/botany\n\n"
            "Fresh basil, broccoli, celery, lettuce, and sweet potatoes are vegetables in the botanical sense. "
            "Bell peppers, corn, green beans, peanuts, plums, and zucchini are botanical fruits."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [fetch_url])

    agent = GaiaGraphAgent(model=FakeModelForBotanicalGroundingRetry(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="botany-grounded",
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

    assert result["submitted_answer"] == "broccoli, celery, fresh basil, lettuce, sweet potatoes"
    assert result["tool_trace"] == ["fetch_url({'url': 'https://example.com/botany'})"]


def test_graph_botanical_fallback_classifies_items_from_fetched_sources(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return one source candidate per botanical query."""
        assert max_results == 5
        url_map = {
            "fresh basil botanical fruit or vegetable": "https://example.com/basil",
            "broccoli botanical fruit or vegetable": "https://example.com/broccoli",
            "bell pepper botanical fruit or vegetable": "https://example.com/bell-pepper",
            "sweet potatoes botanical fruit or vegetable": "https://example.com/sweet-potato",
        }
        url = url_map.get(query, "https://example.com/unknown")
        return f"1. Source\nURL: {url}\nSnippet: botanical classification for {query}"

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched botanical classification text."""
        payloads = {
            "https://example.com/basil": (
                "Title: Basil\nURL: https://example.com/basil\n\n"
                "Basil is a culinary herb whose edible leaves are used fresh."
            ),
            "https://example.com/broccoli": (
                "Title: Broccoli\nURL: https://example.com/broccoli\n\n"
                "Broccoli is eaten for its flowering head and stalk."
            ),
            "https://example.com/bell-pepper": (
                "Title: Bell pepper\nURL: https://example.com/bell-pepper\n\n"
                "Botanically, bell pepper is a fruit because it contains seeds."
            ),
            "https://example.com/sweet-potato": (
                "Title: Sweet potato\nURL: https://example.com/sweet-potato\n\n"
                "Sweet potato is an edible root vegetable."
            ),
        }
        return payloads[url]

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(
        model=FakeModelSingleAnswer("[ANSWER]Bell pepper, Broccoli, Fresh basil, Sweet potatoes[/ANSWER]"),
        max_iterations=1,
    )
    result = agent.solve(
        Question(
            task_id="botany-fallback",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. Here's the list I have so far:\n\n"
                "fresh basil, broccoli, bell pepper, sweet potatoes\n\n"
                "Please alphabetize the vegetables and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "broccoli, fresh basil, sweet potatoes"
    assert result["reducer_used"] == "botanical_classification"


def test_graph_botanical_fallback_recovers_prompt_item_omitted_by_model(monkeypatch) -> None:
    search_queries: list[str] = []

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return one source candidate per botanical query."""
        assert max_results == 5
        search_queries.append(query)
        url_map = {
            "fresh basil botanical fruit or vegetable": "https://example.com/basil",
            "broccoli botanical fruit or vegetable": "https://example.com/broccoli",
            "bell pepper botanical fruit or vegetable": "https://example.com/bell-pepper",
            "sweet potatoes botanical fruit or vegetable": "https://example.com/sweet-potato",
        }
        url = url_map.get(query, "https://example.com/unknown")
        return f"1. Source\nURL: {url}\nSnippet: botanical classification for {query}"

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched botanical classification text."""
        payloads = {
            "https://example.com/basil": (
                "Title: Basil\nURL: https://example.com/basil\n\n"
                "Basil is a culinary herb whose edible leaves are used fresh."
            ),
            "https://example.com/broccoli": (
                "Title: Broccoli\nURL: https://example.com/broccoli\n\n"
                "Broccoli is eaten for its flowering head and stalk."
            ),
            "https://example.com/bell-pepper": (
                "Title: Bell pepper\nURL: https://example.com/bell-pepper\n\n"
                "Botanically, bell pepper is a fruit because it contains seeds."
            ),
            "https://example.com/sweet-potato": (
                "Title: Sweet potato\nURL: https://example.com/sweet-potato\n\n"
                "Sweet potato is an edible root vegetable."
            ),
        }
        return payloads.get(
            url,
            f"Title: Unknown\nURL: {url}\n\nThis page does not contain a relevant botanical classification.",
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(
        model=FakeModelSingleAnswer("[ANSWER]Broccoli, Sweet potatoes[/ANSWER]"),
        max_iterations=1,
    )
    result = agent.solve(
        Question(
            task_id="botany-fallback-omitted-item",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. Here's the list I have so far:\n\n"
                "fresh basil, broccoli, bell pepper, sweet potatoes\n\n"
                "Please alphabetize the vegetables and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "broccoli, fresh basil, sweet potatoes"
    assert result["reducer_used"] == "botanical_classification"
    assert any(query.startswith("fresh basil botanical fruit or vegetable") for query in search_queries)


def test_graph_botanical_fallback_ignores_low_signal_fruit_metadata_pages(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return one source candidate per botanical query."""
        assert max_results == 5
        url_map = {
            "broccoli botanical fruit or vegetable": "https://example.com/broccoli",
            "broccoli botany fruit vegetable": "https://example.com/broccoli",
            "plums botanical fruit or vegetable": "https://example.com/plum-quora",
            "plums botany fruit vegetable": "https://example.com/plum-britannica",
            "sweet potatoes botanical fruit or vegetable": "https://example.com/sweet-potato",
            "sweet potatoes botany fruit vegetable": "https://example.com/sweet-potato",
        }
        url = url_map.get(query, "https://example.com/unknown")
        return f"1. Source\nURL: {url}\nSnippet: botanical classification for {query}"

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched botanical classification text."""
        payloads = {
            "https://example.com/broccoli": (
                "Title: Broccoli\nURL: https://example.com/broccoli\n\n"
                "Broccoli is eaten for its flowering head and stalk."
            ),
            "https://example.com/plum-quora": (
                "URL Source: https://example.com/Is-plum-a-vegetable-Why-or-why-not"
            ),
            "https://example.com/plum-britannica": (
                "Title: Plum\nURL: https://example.com/plum\n\n"
                "Plum | Description, Uses, Cultivation, History, & Facts | Britannica\n"
                "purple-leaf plum"
            ),
            "https://example.com/sweet-potato": (
                "Title: Sweet potato\nURL: https://example.com/sweet-potato\n\n"
                "Sweet potato is an edible root vegetable."
            ),
        }
        return payloads.get(
            url,
            f"Title: Unknown\nURL: {url}\n\nThis page does not contain a relevant botanical classification.",
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(
        model=FakeModelSingleAnswer("[ANSWER]Broccoli, Plums, Sweet potatoes[/ANSWER]"),
        max_iterations=1,
    )
    result = agent.solve(
        Question(
            task_id="botany-fallback-low-signal-fruit",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. Here's the list I have so far:\n\n"
                "broccoli, plums, sweet potatoes\n\n"
                "Please alphabetize the vegetables and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "broccoli, sweet potatoes"
    assert result["reducer_used"] == "botanical_classification"


def test_graph_botanical_fallback_ignores_ambiguous_culinary_zucchini_page(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return one source candidate per botanical query."""
        assert max_results == 5
        url_map = {
            "broccoli botanical fruit or vegetable": "https://example.com/broccoli",
            "broccoli botany fruit vegetable": "https://example.com/broccoli",
            "sweet potatoes botanical fruit or vegetable": "https://example.com/sweet-potato",
            "sweet potatoes botany fruit vegetable": "https://example.com/sweet-potato",
            "zucchini botanical fruit or vegetable": "https://example.com/zucchini-ambiguous",
            "zucchini botany fruit vegetable": "https://example.com/zucchini-ambiguous",
        }
        url = url_map.get(query, "https://example.com/unknown")
        return f"1. Source\nURL: {url}\nSnippet: botanical classification for {query}"

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched botanical classification text."""
        payloads = {
            "https://example.com/broccoli": (
                "Title: Broccoli\nURL: https://example.com/broccoli\n\n"
                "Broccoli is eaten for its flowering head and stalk."
            ),
            "https://example.com/sweet-potato": (
                "Title: Sweet potato\nURL: https://example.com/sweet-potato\n\n"
                "Sweet potato is an edible root vegetable."
            ),
            "https://example.com/zucchini-ambiguous": (
                "Title: Zucchini: fruit or vegetable?\n"
                "URL: https://example.com/zucchini-ambiguous\n\n"
                "Did you know that zucchini is not a vegetable but a fruit? "
                "Vegetables can be roots, tubers, leaves, stems, flowers, or even fruits such as tomatoes or zucchini. "
                "So here is the answer: zucchini is both a fruit and a vegetable."
            ),
        }
        return payloads.get(
            url,
            f"Title: Unknown\nURL: {url}\n\nThis page does not contain a relevant botanical classification.",
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(
        model=FakeModelSingleAnswer("[ANSWER]Broccoli, Sweet potatoes, Zucchini[/ANSWER]"),
        max_iterations=1,
    )
    result = agent.solve(
        Question(
            task_id="botany-fallback-ambiguous-zucchini",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. Here's the list I have so far:\n\n"
                "broccoli, sweet potatoes, zucchini\n\n"
                "Please alphabetize the vegetables and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "broccoli, sweet potatoes"
    assert result["reducer_used"] == "botanical_classification"


def test_graph_rejects_ungrounded_botanical_classification_when_retry_budget_is_exhausted() -> None:
    agent = GaiaGraphAgent(
        model=FakeModelSingleAnswer(
            "[ANSWER]Bell pepper, Broccoli, Celery, Corn, Green beans, Lettuce, Sweet potatoes, Zucchini[/ANSWER]"
        ),
        max_iterations=1,
    )

    result = agent.solve(
        Question(
            task_id="botany-ungrounded",
            question=(
                "I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes "
                "to categorizing things. Here's the list I have so far:\n\n"
                "sweet potatoes, fresh basil, plums, green beans, corn, bell pepper, broccoli, celery, zucchini, lettuce\n\n"
                "Please alphabetize the vegetables and place each item in a comma separated list."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == ""
    assert result["error"] == "Botanical classification answer lacked grounded evidence."
    assert result["fallback_reason"] == "botanical_classification_evidence_missing"


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
    "case",
    [
        (
            "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.",
            "blocked",
            None,
        ),
        "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.",
        (
            "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?",
            "blocked",
            None,
        ),
    ],
)
def test_graph_blocks_ungrounded_python_for_lookup_questions(monkeypatch, case) -> None:
    if isinstance(case, tuple):
        question, expected_answer, expected_error = case
    else:
        question = case
        expected_answer = ""
        expected_error = "Date-sensitive roster answer lacked temporally grounded evidence."

    @tool
    def execute_python_code(code: str) -> str:
        """Tool should not be invoked when python is ungrounded."""
        raise AssertionError("execute_python_code should have been blocked")

    monkeypatch.setattr(graph_module, "build_tools", lambda: [execute_python_code])

    agent = GaiaGraphAgent(model=FakeModelWithBlockedPython(), max_iterations=2)
    result = agent.solve(Question(task_id="blocked-python", question=question, file_name=None))

    assert result["submitted_answer"] == expected_answer
    assert any("execute_python_code" in item for item in result["tool_trace"])
    assert result["error"] == expected_error


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

    assert result["submitted_answer"] == "Louvrier"
    assert any(
        item.startswith("find_text_in_url(") and "Bookshelves/Introductory_Chemistry" in item
        for item in result["tool_trace"]
    )


def test_graph_auto_follows_ranked_text_span_candidate_after_no_match(monkeypatch) -> None:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return a generic mirror plus the canonical CK-12 exercise page."""
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
    def find_text_in_url(url: str, query: str, max_matches: int = 8) -> str:
        """Fail on the mirror page and succeed on the canonical exercise page."""
        assert query == "equine veterinarian"
        assert max_matches == 8
        if "/Courses/" in url:
            return "No matches found."
        assert "Bookshelves/Introductory_Chemistry" in url
        return "The equine veterinarian mentioned in the exercise is Dr. Louvrier."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, find_text_in_url])

    class FakeModelMirrorThenInvalid:
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
            return AIMessage(content="The available information does not make the answer explicit.")

    agent = GaiaGraphAgent(model=FakeModelMirrorThenInvalid(), max_iterations=3)
    result = agent.solve(
        Question(
            task_id="text-span-auto-follow",
            question=(
                "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
                "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
                "as compiled 08/21/2023?"
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Louvrier"
    assert any(
        item.startswith("find_text_in_url(") and "Bookshelves/Introductory_Chemistry" in item
        for item in result["tool_trace"]
    )
    assert result["error"] is None


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
                "Caption: Hokkaido Nippon-Ham Fighters current 2023 season roster\n"
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


def test_temporal_roster_retry_ignores_player_profile_with_year_and_season() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after TaishÅ Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: [Official] Taisho Tamai (Hokkaido Nippon-Ham) | Pacific League | "
                    "Player directory | 2023 season pitcher stats\n"
                    "URL: https://pacificleague.com/en/player/517064"
                ),
                tool_call_id="player-page",
                name="fetch_url",
            ),
            ToolMessage(
                content=(
                    "URL: https://en.wikipedia.org/wiki/List_of_current_Nippon_Professional_Baseball_team_rosters\n"
                    "Table 1\n"
                    "Caption: Hokkaido Nippon-Ham Fighters current 2023 season roster\n"
                    "No. | Name\n"
                    "18 | Sachiya Yamasaki\n"
                    "19 | Taisho Tamai\n"
                    "20 | Kenta Uehara\n"
                ),
                tool_call_id="current-roster",
                name="extract_tables_from_url",
            ),
            ToolMessage(
                content=(
                    "URL: https://en.wikipedia.org/wiki/Template:Hokkaido_Nippon-Ham_Fighters_roster\n"
                    "Table 1\n"
                    "Caption: Hokkaido Nippon-Ham Fighters 2023 season roster\n"
                    "No. | Name\n"
                    "18 | Sachiya Yamasaki\n"
                    "19 | Taisho Tamai\n"
                    "20 | Kenta Uehara\n"
                ),
                tool_call_id="template-roster",
                name="extract_tables_from_url",
            ),
            ToolMessage(
                content=(
                    "Title: 2023 Hokkaido Nippon-Ham Fighters Individual Pitching (Pacific League) | NPB.jp\n"
                    "URL: https://npb.jp/bis/eng/2023/stats/idp1_f.html\n\n"
                    "2023 Hokkaido Nippon-Ham Fighters Individual Pitching\n"
                    "Pitcher\n"
                    "Tamai, Taisho\n"
                    "Uehara, Kenta\n"
                    "Yoshida, Kosei\n"
                    "Website\n"
                    "Roster\n"
                    "Team Roster Listing\n"
                ),
                tool_call_id="stats-page",
                name="fetch_url",
            ),
        ],
    }

    assert has_temporal_roster_grounding_gap(state) is True
    assert requires_temporal_roster_retry(state, "Yamasaki, Uehara") is True


def test_temporal_roster_retry_requires_positive_grounded_roster_evidence() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: [Official] Taisho Tamai (Hokkaido Nippon-Ham) | Pacific League | "
                    "Player directory | 2023 season pitcher stats\n"
                    "URL: https://pacificleague.com/en/player/517064"
                ),
                tool_call_id="player-page",
                name="fetch_url",
            ),
            ToolMessage(
                content=(
                    "Title: 2023 Hokkaido Nippon-Ham Fighters Individual Pitching (Pacific League) | NPB.jp\n"
                    "URL: https://npb.jp/bis/eng/2023/stats/idp1_f.html\n\n"
                    "2023 Hokkaido Nippon-Ham Fighters Individual Pitching\n"
                    "Pitcher\n"
                    "Tamai, Taisho\n"
                    "Uehara, Kenta\n"
                    "Yoshida, Kosei\n"
                ),
                tool_call_id="stats-page",
                name="fetch_url",
            ),
        ],
    }

    assert has_temporally_grounded_roster_evidence(state) is False
    assert requires_temporal_roster_retry(state, "Yamasaki, Uehara") is True


def test_graph_targeted_fighters_roster_fallback_solves_from_projected_season_page(monkeypatch) -> None:
    @tool
    def fetch_url(url: str) -> str:
        """Return mocked Fighters roster pages."""
        if url == "https://npb.jp/bis/eng/players/91295134.html":
            return (
                "Title: Tamai,Taisho（Hokkaido Nippon-Ham Fighters） | Players | Nippon Professional Baseball Organization\n"
                "URL: https://npb.jp/bis/eng/players/91295134.html\n\n"
                "Players\n"
                "Roster (Hokkaido Nippon-Ham Fighters)\n"
                "Thursday, April 9, 2026\n"
                "19\n"
                "Hokkaido Nippon-Ham Fighters\n"
                "Tamai, Taisho\n"
            )
        if url == "https://www.fighters.co.jp/team/player/detail/2023_00001560.html?lang=en":
            return (
                "Title: 25 Naoki Miyanishi player directory 2023\n"
                "URL: https://www.fighters.co.jp/team/player/detail/2023_00001560.html?lang=en\n\n"
                "25 Naoki Miyanishi\n"
                "2023\n"
                "Pitchers\n"
                "Show Other Players\n"
                "17 Hiromi Ito\n"
                "18 Kosei Yoshida\n"
                "19 Taisho Tamai\n"
                "20 Kenta Uehara\n"
                "22 Toshihiro Sugiura\n"
            )
        return "Title: Page not found\nURL Source: {url}\nWarning: Target URL returned error 404: Not Found"

    @tool
    def extract_links_from_url(
        url: str,
        text_filter: str = "",
        max_results: int = 20,
        same_domain_only: bool = False,
    ) -> str:
        """Return mocked link extraction results."""
        del max_results, same_domain_only
        if url == "https://npb.jp/bis/eng/players/91295134.html" and text_filter == "Official HP":
            return (
                "1. Roster (Hokkaido Nippon-Ham Fighters Official HP)\n"
                "URL: https://www.fighters.co.jp/team/player/list/\n"
                "Snippet: \n"
            )
        if url == "https://www.fighters.co.jp/team/player/list/" and text_filter == "19":
            return (
                "1. 19 玉井 大翔 たまい たいしょう 33歳 北海道\n"
                "URL: https://www.fighters.co.jp/team/player/detail/2026_00001686.html\n"
                "Snippet: \n"
            )
        return "No matching links found."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [fetch_url, extract_links_from_url])

    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "1. Taisho Tamai - Wikipedia\n"
                    "URL: https://en.wikipedia.org/wiki/Taisho_Tamai\n"
                    "Snippet: current player bio\n\n"
                    "2. Tamai,Taisho（Hokkaido Nippon-Ham Fighters） | Players | Nippon Professional Baseball Organization\n"
                    "URL: https://npb.jp/bis/eng/players/91295134.html\n"
                    "Snippet: NPB player page\n"
                ),
                tool_call_id="search-1",
                name="web_search",
            )
        ],
    }

    result = RosterFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "Yoshida, Uehara"
    assert result["reducer_used"] == "roster_neighbor"


def test_graph_targeted_fighters_roster_fallback_can_start_from_pacificleague_and_team_wiki(monkeypatch) -> None:
    @tool
    def fetch_url(url: str) -> str:
        """Return mocked Pacific League and Fighters pages."""
        if url == "https://pacificleague.com/en/player/517064":
            return (
                "Title: [Official] Taisho Tamai (Hokkaido Nippon-Ham) | Pacific League\n"
                "URL: https://pacificleague.com/en/player/517064\n\n"
                "Player Directory\n"
                "Hokkaido Nippon-Ham Fighters\n"
                "Taisho Tamai\n"
                "19\n"
                "From the 2023 season, number will be changed to [19].\n"
            )
        if url == "https://www.fighters.co.jp/team/player/detail/2023_00001560.html?lang=en":
            return (
                "Title: 25 Naoki Miyanishi player directory 2023\n"
                "URL: https://www.fighters.co.jp/team/player/detail/2023_00001560.html?lang=en\n\n"
                "25 Naoki Miyanishi\n"
                "2023\n"
                "Pitchers\n"
                "Show Other Players\n"
                "18 Kosei Yoshida\n"
                "19 Taisho Tamai\n"
                "20 Kenta Uehara\n"
            )
        return "Title: Page not found\nWarning: Target URL returned error 404: Not Found"

    @tool
    def extract_links_from_url(
        url: str,
        text_filter: str = "",
        max_results: int = 20,
        same_domain_only: bool = False,
    ) -> str:
        """Return mocked link extraction results."""
        del max_results, same_domain_only
        if url == "https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters" and text_filter == "fighters.co.jp":
            return (
                "1. https://www.fighters.co.jp/\n"
                "URL: https://www.fighters.co.jp/\n"
                "Snippet: \n"
            )
        if url == "https://www.fighters.co.jp/team/player/list/" and text_filter == "19":
            return (
                "1. 19 玉井 大翔 たまい たいしょう 33歳 北海道\n"
                "URL: https://www.fighters.co.jp/team/player/detail/2026_00001686.html\n"
                "Snippet: \n"
            )
        return "No matching links found."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [fetch_url, extract_links_from_url])

    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: [Official] Taisho Tamai (Hokkaido Nippon-Ham) | Pacific League\n"
                    "URL: https://pacificleague.com/en/player/517064\n\n"
                    "19\n"
                    "Taisho Tamai\n"
                ),
                tool_call_id="pacificleague-page",
                name="fetch_url",
            ),
            ToolMessage(
                content=(
                    "1. Hokkaido Nippon-Ham Fighters\n"
                    "URL: https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters\n"
                    "Snippet: team page\n"
                ),
                tool_call_id="wiki-search",
                name="search_wikipedia",
            ),
        ],
    }

    result = RosterFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "Yoshida, Uehara"


def test_graph_text_span_source_fallback_solves_from_ranked_candidate(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    @tool
    def find_text_in_url(url: str, query: str, max_matches: int = 8) -> str:
        """Return the live CK-12-compatible passage."""
        del max_matches
        calls.append((url, query))
        assert query == "equine veterinarian"
        assert "chem.libretexts.org" in url
        return "The equine veterinarian mentioned in the exercise is Dr. Louvrier."

    monkeypatch.setattr(graph_module, "build_tools", lambda: [find_text_in_url])

    state = {
        "question": (
            "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
            "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
            "as compiled 08/21/2023?"
        ),
        "messages": [],
        "ranked_candidates": [
            {
                "title": "1.E Exercises",
                "url": "https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01:_Chemistry_in_our_Lives/1.E:_Exercises",
                "snippet": "license:ck12 author@Marisa Alviar-Agnew author@Henry Agnew",
                "origin_tool": "web_search",
                "score": 92,
                "reasons": ("exercise_page", "expected_domain"),
            }
        ],
    }

    result = TextSpanFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "Louvrier"
    assert result["reducer_used"] == "text_span_attribute"
    assert any("Chabot_College" in url for url, _ in calls)


def test_graph_text_span_source_fallback_fetches_candidate_page_after_find_miss(
    monkeypatch,
) -> None:
    calls: list[str] = []

    @tool
    def find_text_in_url(url: str, query: str, max_matches: int = 8) -> str:
        """Return no match for the direct text lookup."""
        del max_matches
        calls.append(f"find:{url}:{query}")
        if "Chabot_College" in url and query == "equine veterinarian":
            return "No matches found."
        return "No matches found."

    @tool
    def fetch_url(url: str) -> str:
        """Return the full candidate page so the reducer can recover the answer from page text."""
        calls.append(f"fetch:{url}")
        assert "Chabot_College" in url
        return (
            "Title: 1.E: Exercises - Chemistry LibreTexts\n"
            "URL: https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01:_Chemistry_in_our_Lives/1.E:_Exercises\n\n"
            "During Pasteur's time, anthrax was a widespread and disastrous disease for livestock. "
            "Around 1876, a horse doctor in eastern France named Louvrier, claimed to have invented a cure for anthrax."
        )

    monkeypatch.setattr(graph_module, "build_tools", lambda: [find_text_in_url, fetch_url])

    state = {
        "question": (
            "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
            "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
            "as compiled 08/21/2023?"
        ),
        "messages": [],
        "ranked_candidates": [
            {
                "title": "1.E Exercises",
                "url": "https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01:_Chemistry_in_our_Lives/1.E:_Exercises",
                "snippet": "license:ck12 author@Marisa Alviar-Agnew author@Henry Agnew",
                "origin_tool": "web_search",
                "score": 92,
                "reasons": ("exercise_page", "expected_domain"),
            }
        ],
    }

    result = TextSpanFallback(_tools_by_name()).run(state)

    assert result is not None
    assert result["final_answer"] == "Louvrier"
    assert any(item.startswith("find_text_in_url(") for item in result["tool_trace"])
    assert any(item.startswith("fetch_url(") for item in result["tool_trace"])
    assert any(entry.startswith("tool:find_text_in_url") for entry in result["decision_trace"])
    assert any(entry.startswith("tool:fetch_url") for entry in result["decision_trace"])
    assert any(call.endswith(":equine veterinarian") for call in calls if call.startswith("find:"))
    assert any(call.startswith("fetch:") and "Chabot_College" in call for call in calls)


def test_graph_entity_role_chain_fallback_overrides_ungrounded_final_answer(
    monkeypatch,
) -> None:
    search_queries: list[str] = []

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return ranked entity-role-chain candidates."""
        assert max_results == 5
        search_queries.append(query)
        return (
            "1. Bartłomiej Kasprzykowski\n"
            "URL: https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski\n"
            "Snippet: Polish actor who played the titular hero in Wszyscy kochają Romana.\n\n"
            "2. Magda M. cast list\n"
            "URL: https://en.wikipedia.org/wiki/Magda_M.\n"
            "Snippet: cast and characters from Magda M.\n\n"
            "3. Popular actor post\n"
            "URL: https://www.instagram.com/popular/actor-who-played-ray-in-polish-version-of-everybody-loves-raymond/\n"
            "Snippet: social post about the actor\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched pages with enough evidence to resolve the role chain."""
        assert "instagram.com" not in url
        if url == "https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski":
            return (
                "Title: Bartłomiej Kasprzykowski\n"
                "URL: https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski\n\n"
                "Bartłomiej Kasprzykowski starred as Roman in Wszyscy kochają Romana, "
                "the Polish-language version of Everybody Loves Raymond."
            )
        if url == "https://en.wikipedia.org/wiki/Magda_M.":
            return (
                "Title: Magda M.\n"
                "URL: https://en.wikipedia.org/wiki/Magda_M.\n\n"
                "Cast\n"
                "Wojciech Płaska - Bartłomiej Kasprzykowski\n"
            )
        raise AssertionError(f"Unexpected URL fetched: {url}")

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    agent = GaiaGraphAgent(model=FakeModelForEntityRoleChainFallback(), max_iterations=1)
    result = agent.solve(
        Question(
            task_id="role-chain-fallback",
            question=(
                "Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? "
                "Give only the first name."
            ),
            file_name=None,
        )
    )

    assert result["submitted_answer"] == "Wojciech"
    assert result["reducer_used"] == "entity_role_chain"
    assert any("Magda M." in query for query in search_queries)
    assert any(item.startswith("web_search(") for item in result["tool_trace"])
    assert any(
        item.startswith("fetch_url(") and "Bart%C5%82omiej_Kasprzykowski" in item
        for item in result["tool_trace"]
    )
    assert any(
        item.startswith("fetch_url(") and "Magda_M." in item
        for item in result["tool_trace"]
    )


def test_graph_entity_role_chain_fallback_searches_past_weak_raymond_candidates(
    monkeypatch,
) -> None:
    search_queries: list[str] = []

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Return strong two-hop candidates only after targeted fallback searches."""
        assert max_results == 5
        search_queries.append(query)
        return (
            "1. BartÅ‚omiej Kasprzykowski\n"
            "URL: https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski\n"
            "Snippet: Polish actor who played Roman in Wszyscy kochajÄ… Romana.\n\n"
            "2. Magda M. cast list\n"
            "URL: https://en.wikipedia.org/wiki/Magda_M.\n"
            "Snippet: cast and characters from Magda M.\n"
        )

    @tool
    def fetch_url(url: str) -> str:
        """Return fetched pages with enough evidence to resolve the role chain."""
        if url == "https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski":
            return (
                "Title: BartÅ‚omiej Kasprzykowski\n"
                "URL: https://en.wikipedia.org/wiki/Bart%C5%82omiej_Kasprzykowski\n\n"
                "BartÅ‚omiej Kasprzykowski starred as Roman in Wszyscy kochajÄ… Romana, "
                "the Polish-language version of Everybody Loves Raymond."
            )
        if url == "https://en.wikipedia.org/wiki/Magda_M.":
            return (
                "Title: Magda M.\n"
                "URL: https://en.wikipedia.org/wiki/Magda_M.\n\n"
                "Cast\n"
                "Wojciech PÅ‚aska - BartÅ‚omiej Kasprzykowski\n"
            )
        raise AssertionError(f"Unexpected URL fetched: {url}")

    monkeypatch.setattr(graph_module, "build_tools", lambda: [web_search, fetch_url])

    state = {
        "question": (
            "Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? "
            "Give only the first name."
        ),
        "messages": [],
        "tool_trace": [],
        "decision_trace": [],
        "ranked_candidates": [
            {
                "title": "Everybody Loves Raymond",
                "url": "https://en.wikipedia.org/wiki/Everybody_Loves_Raymond",
                "snippet": "American sitcom",
                "origin_tool": "web_search",
                "score": 90,
                "reasons": ("expected_domain",),
            },
            {
                "title": "Magda M.",
                "url": "https://en.wikipedia.org/wiki/Magda_M.",
                "snippet": "Polish drama series",
                "origin_tool": "web_search",
                "score": 89,
                "reasons": ("expected_domain",),
            }
        ],
        "question_profile": {
            "name": "entity_role_chain",
            "target_urls": (),
            "expected_domains": ("wikipedia.org",),
            "preferred_tools": ("web_search", "fetch_url", "find_text_in_url"),
            "expected_date": None,
            "expected_author": None,
            "subject_name": None,
            "text_filter": "cast character",
        },
    }

    result = RoleChainFallback(
        _tools_by_name(),
        FakeModelForEntityRoleChainFallback(),
    ).run(state)

    assert result is not None
    assert result["final_answer"] == "Wojciech"
    assert result["reducer_used"] == "entity_role_chain"
    assert any("Magda M." in query for query in search_queries)
    assert any("Bartlomiej Kasprzykowski" in query or "Wszyscy kochaja Romana" in query for query in search_queries)
    assert any(
        item.startswith("fetch_url(") and "Bart%C5%82omiej_Kasprzykowski" in item
        for item in result["tool_trace"]
    )
    assert any(
        item.startswith("fetch_url(") and "Magda_M." in item
        for item in result["tool_trace"]
    )


def test_temporal_roster_grounding_ignores_off_scope_dated_roster_pages() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: Team Japan World Baseball Classic 2023 roster\n"
                    "URL: https://www.mlb.com/news/team-japan-world-baseball-classic-2023-roster\n\n"
                    "Team Japan World Baseball Classic 2023 roster\n"
                    "Pitchers\n"
                    "Yu Darvish\n"
                    "Shota Imanaga\n"
                    "Roki Sasaki\n"
                ),
                tool_call_id="wbc-roster",
                name="fetch_url",
            )
        ],
    }

    assert has_temporally_grounded_roster_evidence(state) is False


def test_temporal_roster_grounding_accepts_year_specific_player_directory_with_neighbors() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: 25 Naoki Miyanishi player directory 2023\n"
                    "URL: https://www.fighters.co.jp/team/player/detail/2023_00001560.html?lang=en\n\n"
                    "25 Naoki Miyanishi\n"
                    "2023\n"
                    "Pitchers\n"
                    "Show Other Players\n"
                    "18 Kosei Yoshida\n"
                    "19 Taisho Tamai\n"
                    "20 Kenta Uehara\n"
                ),
                tool_call_id="player-dir-2023",
                name="fetch_url",
            )
        ],
    }

    assert has_temporally_grounded_roster_evidence(state) is True


def test_temporal_roster_grounding_treats_generic_wikipedia_team_roster_as_current_only() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "URL: https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters\n"
                    "Table 1\n"
                    "Caption: Hokkaido Nippon-Ham Fighters roster view talk edit First team | Second team\n"
                    "Pitchers | 18 Sachiya Yamasaki | 19 Taisho Tamai | 20 Kenta Uehara\n"
                ),
                tool_call_id="wiki-team-roster",
                name="extract_tables_from_url",
            )
        ],
    }

    assert has_temporally_grounded_roster_evidence(state) is False


def test_temporal_roster_retry_requires_answer_grounded_in_temporal_records() -> None:
    state = {
        "question": (
            "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
            "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
        ),
        "messages": [
            ToolMessage(
                content=(
                    "Title: 2023 Hokkaido Nippon Ham Fighters minor league baseball Roster on StatsCrew.com\n"
                    "URL: https://www.statscrew.com/minorbaseball/roster/t-nf13423/y-2023\n\n"
                    "2023 Hokkaido Nippon Ham Fighters roster\n"
                    "Taisho Tamai\n"
                    "Kenta Uehara\n"
                    "Kosei Yoshida\n"
                ),
                tool_call_id="statscrew-roster",
                name="fetch_url",
            )
        ],
    }

    assert grounded_temporal_roster_answer(state) is None
    assert requires_temporal_roster_retry(state, "Yamasaki, Uehara") is True


def test_agent_node_truncates_tool_messages_for_model_context() -> None:
    model = RecordingModel()
    agent = GaiaGraphAgent(model=model, max_iterations=2)
    long_tool_output = "URL: https://example.com/data\n" + ("A" * 10000)

    state = {
        "question": "Who won the event?",
        "messages": [
            SystemMessage(content="system"),
            HumanMessage(content="question"),
            ToolMessage(
                content=long_tool_output,
                tool_call_id="tool-1",
                name="fetch_url",
            ),
        ],
        "iterations": 0,
        "max_iterations": 2,
        "question_profile": {},
        "ranked_candidates": [],
        "decision_trace": [],
    }

    agent._agent_node(state)

    assert model.messages is not None
    tool_messages = [message for message in model.messages if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert len(str(tool_messages[0].content)) < len(long_tool_output)
    assert "[truncated for model context]" in str(tool_messages[0].content)


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
