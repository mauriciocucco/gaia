from hf_gaia_agent.normalize import normalize_submitted_answer


def test_removes_final_answer_prefix() -> None:
    assert normalize_submitted_answer("FINAL ANSWER: Madrid") == "Madrid"


def test_removes_answer_wrappers() -> None:
    assert normalize_submitted_answer("[ANSWER]42[/ANSWER]") == "42"


def test_preserves_numeric_answers() -> None:
    assert normalize_submitted_answer("  3.1415  ") == "3.1415"


def test_preserves_comma_separated_lists() -> None:
    assert (
        normalize_submitted_answer(' "apples,bananas, pears" ')
        == "apples, bananas, pears"
    )
