import pytest

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


# --- Acronyms and codes ---


def test_preserves_uppercase_acronyms() -> None:
    assert normalize_submitted_answer("NASA") == "NASA"
    assert normalize_submitted_answer("Answer: UNESCO") == "UNESCO"


def test_preserves_alphanumeric_codes() -> None:
    assert normalize_submitted_answer("ABC-123") == "ABC-123"
    assert normalize_submitted_answer("  R2-D2  ") == "R2-D2"


def test_preserves_ioc_country_codes() -> None:
    assert normalize_submitted_answer("USA") == "USA"
    assert normalize_submitted_answer("GBR") == "GBR"


def test_preserves_doi_identifier() -> None:
    doi = "10.1038/s41586-020-2649-2"
    assert normalize_submitted_answer(f"Answer: {doi}") == doi


# --- Sensitive punctuation ---


def test_preserves_decimal_numbers() -> None:
    assert normalize_submitted_answer("0.75") == "0.75"
    assert normalize_submitted_answer("-12.5") == "-12.5"


def test_preserves_dollar_amounts() -> None:
    assert normalize_submitted_answer("$1,234.56") == "$1, 234.56"


def test_preserves_percentage() -> None:
    assert normalize_submitted_answer("42.3%") == "42.3%"


def test_preserves_parenthetical_info() -> None:
    assert normalize_submitted_answer("Paris (France)") == "Paris (France)"


def test_preserves_slash_separated_values() -> None:
    assert normalize_submitted_answer("yes/no") == "yes/no"


# --- Lists ---


def test_normalizes_comma_separated_names() -> None:
    assert (
        normalize_submitted_answer("Alice,  Bob,Charlie")
        == "Alice, Bob, Charlie"
    )


def test_preserves_semicolon_separated_list() -> None:
    assert normalize_submitted_answer("red; green; blue") == "red; green; blue"


def test_preserves_numbered_list_collapsed_to_single_line() -> None:
    raw = "1. Mercury\n2. Venus\n3. Earth"
    result = normalize_submitted_answer(raw)
    assert "Mercury" in result
    assert "Venus" in result
    assert "Earth" in result


# --- Wrapper stripping ---


@pytest.mark.parametrize("wrapper", [
    "[ANSWER]hello[/ANSWER]",
    "[answer]hello[/answer]",
    "[ANSWER] hello ",
])
def test_answer_block_variants(wrapper: str) -> None:
    assert normalize_submitted_answer(wrapper) == "hello"


def test_strips_code_fences() -> None:
    assert normalize_submitted_answer("```\n42\n```") == "42"
    assert normalize_submitted_answer("```python\n42\n```") == "42"


def test_strips_nested_label_and_quotes() -> None:
    assert normalize_submitted_answer('Answer: "Berlin"') == "Berlin"


def test_strips_backtick_quotes() -> None:
    assert normalize_submitted_answer("`Tokyo`") == "Tokyo"


# --- Edge cases ---


def test_empty_and_whitespace() -> None:
    assert normalize_submitted_answer("") == ""
    assert normalize_submitted_answer("   ") == ""
    assert normalize_submitted_answer(None) == ""  # type: ignore[arg-type]


def test_multiline_collapses_whitespace() -> None:
    assert normalize_submitted_answer("  hello \n  world  ") == "hello world"


def test_preserves_single_character_answer() -> None:
    assert normalize_submitted_answer("A") == "A"
    assert normalize_submitted_answer("7") == "7"
