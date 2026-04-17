from hf_gaia_agent.botanical_classification import classify_botanical_item_from_records
from hf_gaia_agent.source_pipeline import EvidenceRecord


def test_classify_botanical_item_uses_adjacent_context_for_sweet_potato_root_sentence() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Sweet_potato",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Sweet potato\n"
            "URL: https://en.wikipedia.org/wiki/Sweet_potato\n\n"
            "The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant in the morning glory family, "
            "Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, "
            "which is a staple food in parts of the world."
        ),
        title_or_caption="Sweet potato",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records(
        "sweet potatoes",
        [record],
    )

    assert outcome == "include"


def test_sweet_potatoes_do_not_match_unrelated_zucchini_page() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Zucchini",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Zucchini\n"
            "URL: https://en.wikipedia.org/wiki/Zucchini\n\n"
            "Zucchini is treated as a vegetable in cookery and is occasionally used in sweeter "
            "cooking. Other vegetables played include carrots, bell peppers, potatoes and parsnips."
        ),
        title_or_caption="Zucchini",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records(
        "sweet potatoes",
        [record],
    )

    assert outcome is None


def test_zucchini_pepo_excludes_from_vegetables() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Cucurbita_pepo",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Cucurbita pepo\n"
            "URL: https://en.wikipedia.org/wiki/Cucurbita_pepo\n\n"
            "Cucurbita pepo is a cultivated plant. Zucchini is a pepo, a botanical fruit "
            "developed from the flower ovary and containing seeds."
        ),
        title_or_caption="Cucurbita pepo",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records("zucchini", [record])

    assert outcome == "exclude"


def test_bell_pepper_capsicum_fruit_signal_excludes() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Capsicum_annuum",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Capsicum annuum\n"
            "URL: https://en.wikipedia.org/wiki/Capsicum_annuum\n\n"
            "The bell pepper is the fruit of the plant Capsicum annuum. "
            "It is a botanical fruit because it develops from the ovary and contains seeds."
        ),
        title_or_caption="Capsicum annuum",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records("bell pepper", [record])

    assert outcome == "exclude"


def test_peanut_legume_pod_signal_excludes() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Peanut",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Peanut\n"
            "URL: https://en.wikipedia.org/wiki/Peanut\n\n"
            "Peanut is the common name for Arachis hypogaea. The peanut fruit is a legume pod "
            "that contains seeds."
        ),
        title_or_caption="Peanut",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records("peanuts", [record])

    assert outcome == "exclude"


def test_peanut_live_wikipedia_style_text_excludes_despite_processed_food_phrase() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Peanut",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Peanut\n"
            "URL: https://en.wikipedia.org/wiki/Peanut\n\n"
            "The peanut (Arachis hypogaea), also known as the groundnut, is a legume crop grown "
            "mainly for its edible seeds, contained in underground pods. It is widely grown both "
            "as a grain legume and as an oil crop. Peanut fruits develop underground, an unusual "
            "feature known as geocarpy. These pods, technically called legumes, normally contain "
            "one to four seeds. Peanut butter is made from ground dry roasted peanuts."
        ),
        title_or_caption="Peanut",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records("peanuts", [record])

    assert outcome == "exclude"


def test_scientific_alias_page_can_support_common_name_item() -> None:
    record = EvidenceRecord(
        kind="text",
        source_url="https://en.wikipedia.org/wiki/Arachis_hypogaea",
        source_type="page_text",
        adapter_name="wikipedia.org",
        content=(
            "Title: Arachis hypogaea\n"
            "URL: https://en.wikipedia.org/wiki/Arachis_hypogaea\n\n"
            "Arachis hypogaea produces a legume pod with edible seeds."
        ),
        title_or_caption="Arachis hypogaea",
        confidence=0.9,
        extraction_method="fetch_wikipedia_page",
        derived_from=("fetch_wikipedia_page",),
    )

    outcome, _supporting = classify_botanical_item_from_records("peanuts", [record])

    assert outcome == "exclude"
