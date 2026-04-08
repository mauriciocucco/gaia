from hf_gaia_agent.source_pipeline import (
    evidence_records_from_tool_output,
    parse_result_blocks,
    profile_question,
    score_candidates,
)


def test_profile_question_classifies_article_to_paper() -> None:
    profile = profile_question(
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
        "This article mentions a team that produced a paper about their observations, linked at the bottom of the article. "
        "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
    )

    assert profile.name == "article_to_paper"
    assert "universetoday.com" in profile.expected_domains
    assert profile.expected_author == "Carolyn Collins Petersen"
    assert profile.expected_date == "June 6, 2023"


def test_profile_question_classifies_olympics_ioc_as_wikipedia_lookup() -> None:
    profile = profile_question(
        "What country had the least number of athletes at the 1928 Summer Olympics? "
        "If there's a tie for a number of athletes, return the first in alphabetical order. "
        "Give the IOC country code as your answer."
    )

    assert profile.name == "wikipedia_lookup"
    assert "wikipedia.org" in profile.expected_domains
    assert profile.text_filter == "athletes"


def test_score_candidates_prefers_expected_domain_author_and_date() -> None:
    raw = (
        "1. There Are Hundreds of Mysterious Filaments at the Center of the Milky Way\n"
        "URL: https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way\n"
        "Snippet: Carolyn Collins Petersen June 6, 2023 Universe Today article\n\n"
        "2. Wrong Result\n"
        "URL: https://www.universetoday.com/169059/astronomers-see-a-rare-giant-star-explosion-in-the-nearby-galaxy-messier-83/\n"
        "Snippet: Carolyn Collins Petersen June 4, 2023 Universe Today article\n"
    )
    profile = profile_question(
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
        "This article mentions a team that produced a paper about their observations, linked at the bottom of the article."
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question="Carolyn Collins Petersen Universe Today June 6, 2023", profile=profile)

    assert scored[0].url.endswith("mysterious-filaments-at-the-center-of-the-milky-way")
    assert scored[0].score > scored[1].score


def test_score_candidates_penalizes_expected_domain_miss() -> None:
    raw = (
        "1. Wrong Result\n"
        "URL: https://www.zhihu.com/question/578992181\n"
        "Snippet: many and much usage examples\n\n"
        "2. Better Result\n"
        "URL: https://www.baseball-reference.com/teams/NYY/1977.shtml\n"
        "Snippet: 1977 New York Yankees batting statistics and team roster\n"
    )
    profile = profile_question(
        "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question="1977 New York Yankees walks at bats", profile=profile)

    assert scored[0].url == "https://www.baseball-reference.com/teams/NYY/1977.shtml"
    assert "expected_domain" in scored[0].reasons
    assert "expected_domain_miss" in scored[1].reasons


def test_evidence_records_from_table_output_splits_tables() -> None:
    content = (
        "Table 1\n"
        "Caption: Roster\n"
        "No. | Name\n"
        "18 | Yoshida\n\n"
        "Table 2\n"
        "Caption: Batting\n"
        "Player | AB\n"
        "Munson | 519"
    )

    records = evidence_records_from_tool_output("extract_tables_from_url", content)

    assert len(records) == 2
    assert records[0].kind == "table"
    assert records[0].title_or_caption == "Roster"
    assert "18 | Yoshida" in records[0].content


def test_score_candidates_prefers_libretexts_for_text_span_lookup() -> None:
    question = (
        "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
        "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
        "as compiled 08/21/2023?"
    )
    profile = profile_question(question)
    raw = (
        "1. Forum Thread\n"
        "URL: https://forums.example.com/equine-veterinarian-thread\n"
        "Snippet: equine veterinarian discussion board\n\n"
        "2. 1.E Exercises\n"
        "URL: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises\n"
        "Snippet: Introductory Chemistry CK-12 1.E Exercises equine veterinarian\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert "libretexts.org" in scored[0].url
    assert "expected_domain" in scored[0].reasons


def test_score_candidates_prefers_roster_page_for_roster_neighbor_lookup() -> None:
    question = (
        "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Taisho Tamai player profile\n"
        "URL: https://example.com/player/tamai\n"
        "Snippet: player profile and season overview\n\n"
        "2. Hokkaido Nippon-Ham Fighters roster\n"
        "URL: https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters\n"
        "Snippet: roster, pitchers, numbers and staff\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters"
    assert "tableish_title" in scored[0].reasons


def test_score_candidates_prefers_official_winners_page_for_competition_lookup() -> None:
    question = (
        "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) "
        "whose nationality on record is a country that no longer exists?"
    )
    profile = profile_question(question)
    raw = (
        "1. Malko Competition\n"
        "URL: https://grokipedia.com/page/malko_competition\n"
        "Snippet: general overview of the competition\n\n"
        "2. Winners | Malko Competition\n"
        "URL: https://malkocompetition.dk/winners/all\n"
        "Snippet: winners list with years and nationalities\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://malkocompetition.dk/winners/all"
