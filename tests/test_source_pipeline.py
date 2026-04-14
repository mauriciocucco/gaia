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


def test_profile_question_extracts_month_year_expected_date() -> None:
    profile = profile_question(
        "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023?"
    )

    assert profile.name == "temporal_ordered_list"
    assert profile.expected_date == "as of July 2023"


def test_profile_question_prioritizes_attachment_required_before_specialized_families() -> None:
    profile = profile_question(
        "What is the surname of the equine veterinarian mentioned in the attached chapter notes?",
        file_name="chapter.pdf",
        local_file_path=None,
    )

    assert profile.name == "attachment_required"
    assert profile.preferred_tools == ("read_local_file",)


def test_profile_question_prioritizes_youtube_before_direct_url_detection() -> None:
    profile = profile_question(
        "What is said in this video https://www.youtube.com/watch?v=L1vXCYZAYYM and this page https://example.com/context?"
    )

    assert profile.name == "transcript_or_video"
    assert profile.target_urls == ("https://www.youtube.com/watch?v=L1vXCYZAYYM",)
    assert profile.expected_domains == ("youtube.com", "youtu.be")


def test_profile_question_falls_back_to_direct_url_via_registry() -> None:
    profile = profile_question("Summarize the claims on https://example.com/report")

    assert profile.name == "direct_url"
    assert profile.target_urls == ("https://example.com/report",)


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


def test_score_candidates_prefers_linked_primary_source_for_article_to_paper() -> None:
    question = (
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
        "This article mentions a team that produced a paper about their observations, linked at the bottom of the article. "
        "Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?"
    )
    profile = profile_question(question)
    raw = (
        "1. There Are Hundreds of Mysterious Filaments at the Center of the Milky Way\n"
        "URL: https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way\n"
        "Snippet: Universe Today article by Carolyn Collins Petersen\n\n"
        "2. Mysterious dashes revealed in Milky Way's Center\n"
        "URL: https://news.northwestern.edu/stories/2023/06/mysterious-dashes-revealed-in-milky-ways-center/?fj=1\n"
        "Snippet: For Journalists news release with the published paper link\n\n"
        "3. The Population of the Galactic Center Filaments: Position Angle Distribution Reveals a Degree-scale Collimated Outflow from Sgr A* along the Galactic Plane\n"
        "URL: https://iopscience.iop.org/article/10.3847/2041-8213/acd54b\n"
        "Snippet: Published paper in The Astrophysical Journal Letters\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="extract_links_from_url")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert "universetoday.com/articles/" not in scored[0].url
    assert any(reason in scored[0].reasons for reason in ("linked_source", "primary_source_hint", "paper_mention"))


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


def test_score_candidates_filters_offtopic_commercial_noise() -> None:
    question = (
        "On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. "
        "Find that article."
    )
    profile = profile_question(question)
    raw = (
        "1. Downtown restaurant menu\n"
        "URL: https://noise.example.com/menu\n"
        "Snippet: Restaurant menu, delivery, order online, book now\n\n"
        "2. There Are Hundreds of Mysterious Filaments at the Center of the Milky Way\n"
        "URL: https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way\n"
        "Snippet: Carolyn Collins Petersen June 6, 2023 Universe Today article\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert [candidate.url for candidate in scored] == [
        "https://www.universetoday.com/articles/there-are-hundreds-of-mysterious-filaments-at-the-center-of-the-milky-way"
    ]


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


def test_evidence_records_from_table_output_preserves_source_url_metadata() -> None:
    content = (
        "URL: https://en.wikipedia.org/wiki/List_of_current_Nippon_Professional_Baseball_team_rosters\n"
        "Title: List of current Nippon Professional Baseball team rosters\n"
        "Table 1\n"
        "Caption: Hokkaido Nippon-Ham Fighters roster\n"
        "No. | Name\n"
        "19 | Taisho Tamai\n"
    )

    records = evidence_records_from_tool_output("extract_tables_from_url", content)

    assert records[0].source_url == "https://en.wikipedia.org/wiki/List_of_current_Nippon_Professional_Baseball_team_rosters"
    assert records[0].title_or_caption == "Hokkaido Nippon-Ham Fighters roster"


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


def test_score_candidates_prefers_canonical_textbook_page_over_course_mirror_for_text_span_lookup() -> None:
    question = (
        "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
        "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
        "as compiled 08/21/2023?"
    )
    profile = profile_question(question)
    raw = (
        "1. Course mirror exercises\n"
        "URL: https://chem.libretexts.org/Courses/Chabot_College/Introduction_to_General_Organic_and_Biochemistry/01%3A_Chemistry_in_our_Lives/1.E%3A_Exercises\n"
        "Snippet: Introductory chemistry exercise page at a course mirror\n\n"
        "2. CK-12 1.E Exercises\n"
        "URL: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises\n"
        "Snippet: Introductory Chemistry CK-12 1.E Exercises equine veterinarian\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert "Bookshelves/Introductory_Chemistry" in scored[0].url
    assert "canonical_textbook_path" in scored[0].reasons
    assert "mirror_course_penalty" in scored[1].reasons


def test_score_candidates_prefers_exact_exercise_page_over_bulk_pdf_for_text_span_lookup() -> None:
    question = (
        "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
        "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
        "as compiled 08/21/2023?"
    )
    profile = profile_question(question)
    raw = (
        "1. Full LibreTexts PDF\n"
        "URL: https://batch.libretexts.org/print/Letter/Finished/chem-521975/Full.pdf\n"
        "Snippet: Introductory Chemistry CK-12 compiled 08/21/2023 by Marisa Alviar-Agnew and Henry Agnew\n\n"
        "2. 1.E Exercises\n"
        "URL: https://chem.libretexts.org/Bookshelves/Introductory_Chemistry/Book%3A_Introductory_Chemistry_(CK-12)/1%3A_Atoms_Molecules_and_Ions/1.E%3A_Exercises\n"
        "Snippet: Introductory Chemistry CK-12 1.E Exercises equine veterinarian\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert "1.E%3A_Exercises" in scored[0].url
    assert "exercise_page" in scored[0].reasons
    assert "bulk_pdf_penalty" in scored[1].reasons


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


def test_score_candidates_penalizes_subject_profile_for_roster_neighbor_lookup() -> None:
    question = (
        "Who are the pitchers with the number before and after TaishÅ Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. TaishÅ Tamai - Wikipedia\n"
        "URL: https://en.wikipedia.org/wiki/Taish%C5%8D_Tamai\n"
        "Snippet: professional Japanese baseball pitcher for the Fighters\n\n"
        "2. Fighters roster archive\n"
        "URL: https://npb.example.com/fighters/archive/2023-07-roster\n"
        "Snippet: July 2023 pitchers roster and numbers archive\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://npb.example.com/fighters/archive/2023-07-roster"
    assert "subject_profile_penalty" in scored[1].reasons


def test_score_candidates_prefers_temporally_matching_roster_sources() -> None:
    question = (
        "Who are the pitchers with the number before and after Taishō Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Current Hokkaido Nippon-Ham Fighters roster\n"
        "URL: https://en.wikipedia.org/wiki/Hokkaido_Nippon-Ham_Fighters\n"
        "Snippet: current roster and staff\n\n"
        "2. 2023 roster update\n"
        "URL: https://npb.example.com/fighters/roster-2023\n"
        "Snippet: July 2023 pitchers roster and numbers\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://npb.example.com/fighters/roster-2023"
    assert "expected_year" in scored[0].reasons or "expected_date_partial" in scored[0].reasons
    assert "expected_date_miss" in scored[1].reasons


def test_score_candidates_prefers_official_yearbook_page_for_dated_roster_lookup() -> None:
    question = (
        "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Template:Hokkaido Nippon-Ham Fighters roster\n"
        "URL: https://en.wikipedia.org/wiki/Template:Hokkaido_Nippon-Ham_Fighters_roster\n"
        "Snippet: current roster template for the Fighters\n\n"
        "2. 20 Kenta Uehara player directory 2023\n"
        "URL: https://www.fighters.co.jp/team/player/detail/2023_00001561.html?lang=en\n"
        "Snippet: 2023 player directory, Pitchers, Show Other Players, 19 Taisho Tamai\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://www.fighters.co.jp/team/player/detail/2023_00001561.html?lang=en"
    assert "official_yearbook_hint" in scored[0].reasons
    assert "current_roster_penalty" in scored[1].reasons


def test_score_candidates_penalizes_current_roster_list_for_dated_roster_lookup() -> None:
    question = (
        "Who are the pitchers with the number before and after TaishÅ Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Current NPB roster list\n"
        "URL: https://en.wikipedia.org/wiki/List_of_current_Nippon_Professional_Baseball_team_rosters\n"
        "Snippet: current roster and staff for Nippon Professional Baseball teams\n\n"
        "2. Fighters roster archive\n"
        "URL: https://npb.example.com/fighters/archive/2023-07-roster\n"
        "Snippet: July 2023 pitchers roster and numbers archive\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://npb.example.com/fighters/archive/2023-07-roster"
    assert "dated_roster_hint" in scored[0].reasons
    assert "current_roster_penalty" in scored[1].reasons


def test_score_candidates_prefers_scoped_roster_archive_over_minor_league_or_discussion() -> None:
    question = (
        "Who are the pitchers with the number before and after TaishÅ Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Fighters roster discussion\n"
        "URL: https://www.reddit.com/r/NPB/comments/example/fighters_roster_july_2023/\n"
        "Snippet: discussion about July 2023 Fighters roster and numbers\n\n"
        "2. Fighters minor league roster\n"
        "URL: https://www.statscrew.com/minorbaseball/roster/t-nf13423/y-2023\n"
        "Snippet: minor league roster and staff for 2023\n\n"
        "3. Fighters roster archive\n"
        "URL: https://npb.example.com/fighters/archive/2023-07-roster\n"
        "Snippet: July 2023 pitchers roster and numbers archive\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://npb.example.com/fighters/archive/2023-07-roster"
    assert any("discussion_source_penalty" in candidate.reasons for candidate in scored)
    assert any("off_scope_roster_penalty" in candidate.reasons for candidate in scored)


def test_score_candidates_prefers_stats_page_over_roster_for_metric_row_lookup() -> None:
    question = "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?"
    profile = profile_question(question)
    raw = (
        "1. 1977 New York Yankees Roster | Baseball Almanac\n"
        "URL: https://www.baseball-almanac.com/teamstats/roster.php?y=1977&t=NYA\n"
        "Snippet: team roster and player list for the 1977 Yankees\n\n"
        "2. 1977 New York Yankees Hitting Stats by Baseball Almanac\n"
        "URL: https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA\n"
        "Snippet: AB and BB for every 1977 Yankees hitter\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://www.baseball-almanac.com/teamstats/hitting.php?y=1977&t=NYA"
    assert "stats_page_hint" in scored[0].reasons
    assert "roster_page_penalty" in scored[1].reasons


def test_score_candidates_penalizes_stats_pages_for_dated_roster_lookup() -> None:
    question = (
        "Who are the pitchers with the number before and after TaishÃ…Â Tamai's number as of July 2023? "
        "Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters."
    )
    profile = profile_question(question)
    raw = (
        "1. Fighters individual pitching stats\n"
        "URL: https://npb.jp/bis/eng/2023/stats/idp1_f.html\n"
        "Snippet: 2023 Hokkaido Nippon-Ham Fighters individual pitching stats and roster links\n\n"
        "2. Fighters roster archive\n"
        "URL: https://npb.example.com/fighters/archive/2023-07-roster\n"
        "Snippet: July 2023 pitchers roster and numbers archive\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert scored[0].url == "https://npb.example.com/fighters/archive/2023-07-roster"
    assert "stats_page_penalty" in scored[1].reasons


def test_score_candidates_penalizes_low_signal_domains_for_entity_role_chain() -> None:
    question = (
        "Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? "
        "Give only the first name."
    )
    profile = profile_question(question)
    raw = (
        "1. Popular actor post\n"
        "URL: https://www.instagram.com/popular/actor-who-played-ray-in-polish-version-of-everybody-loves-raymond/\n"
        "Snippet: social post about the actor\n\n"
        "2. Magda M. cast list\n"
        "URL: https://en.wikipedia.org/wiki/Magda_M.\n"
        "Snippet: cast and characters from Magda M.\n"
    )

    candidates = parse_result_blocks(raw, origin_tool="web_search")
    scored = score_candidates(candidates, question=question, profile=profile)

    assert profile.name == "entity_role_chain"
    assert scored[0].url == "https://en.wikipedia.org/wiki/Magda_M."
    assert "low_signal_domain" in scored[1].reasons


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
