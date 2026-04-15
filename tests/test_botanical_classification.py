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
