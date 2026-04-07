from hf_gaia_agent.evidence_solver import ToolEvidence, solve_answer_from_tool_evidence


def test_evidence_solver_returns_ioc_code_for_minimum_with_tie_break() -> None:
    question = (
        "What country had the least number of athletes at the 1928 Summer Olympics? "
        "If there's a tie for a number of athletes, return the first in alphabetical order. "
        "Give the IOC country code as your answer."
    )
    tool_outputs = [
        ToolEvidence(
            tool_name="extract_tables_from_url",
            content=(
                "Table 1\n"
                "Country | Athletes\n"
                "Cuba | 1\n"
                "Panama | 1\n"
                "Argentina | 81"
            ),
        )
    ]

    result = solve_answer_from_tool_evidence(question, tool_outputs)

    assert result == "CUB"


def test_evidence_solver_reads_parenthetical_rows_from_table_extraction() -> None:
    question = (
        "What country had the least number of athletes at the 1928 Summer Olympics? "
        "If there's a tie for a number of athletes, return the first in alphabetical order. "
        "Give the IOC country code as your answer."
    )
    tool_outputs = [
        ToolEvidence(
            tool_name="extract_tables_from_url",
            content=(
                "Table 1\n"
                "Participating National Olympic Committees\n"
                "Cuba (1)\n"
                "Panama (1)\n"
                "Argentina (81)"
            ),
        )
    ]

    result = solve_answer_from_tool_evidence(question, tool_outputs)

    assert result == "CUB"


def test_evidence_solver_returns_numeric_answer_for_maximum_question() -> None:
    question = "What is the highest number of athletes listed in the table?"
    tool_outputs = [
        ToolEvidence(
            tool_name="extract_tables_from_url",
            content=(
                "Table 1\n"
                "Country | Athletes\n"
                "Cuba | 1\n"
                "Panama | 1\n"
                "Argentina | 81"
            ),
        )
    ]

    result = solve_answer_from_tool_evidence(question, tool_outputs)

    assert result == "81"
