from hf_gaia_agent.evidence_solver import solve_answer_from_evidence_records
from hf_gaia_agent.source_pipeline import EvidenceRecord


def test_solve_answer_from_evidence_records_extracts_text_span_attribute() -> None:
    question = (
        "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials "
        "licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials "
        "as compiled 08/21/2023?"
    )
    records = [
        EvidenceRecord(
            kind="text",
            source_url="https://chem.libretexts.org/example",
            source_type="page_text",
            adapter_name="ReferenceTextAdapter",
            content=(
                "Around 1876, a horse doctor in eastern France named Louvrier, claimed to have invented a cure for anthrax."
            ),
            title_or_caption="1.E: Exercises",
            confidence=0.8,
            extraction_method="find_text_in_url",
            derived_from=("find_text_in_url",),
        )
    ]

    answer, reducer = solve_answer_from_evidence_records(question, records)

    assert answer == "Louvrier"
    assert reducer == "text_span_attribute"


def test_solve_answer_from_evidence_records_filters_temporal_rows() -> None:
    question = (
        "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) "
        "whose nationality on record is a country that no longer exists?"
    )
    records = [
        EvidenceRecord(
            kind="table",
            source_url="https://en.wikipedia.org/wiki/Malko_Competition",
            source_type="table",
            adapter_name="TableExtraction",
            content=(
                "Table 1\n"
                "Year | Recipient | Nationality\n"
                "1977 | Philip Greenberg | United States\n"
                "1980 | Maximiano Valdes | Chile\n"
                "1983 | Claus Peter Flor | East Germany\n"
                "1986 | Kazufumi Yamashita | Japan\n"
            ),
            title_or_caption="Year | Recipient | Nationality",
            confidence=0.8,
            extraction_method="extract_tables_from_url",
            derived_from=("extract_tables_from_url",),
        )
    ]

    answer, reducer = solve_answer_from_evidence_records(question, records)

    assert answer == "Claus"
    assert reducer == "temporal_row_filter"
