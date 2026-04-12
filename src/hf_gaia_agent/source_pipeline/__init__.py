"""Source-aware profiling, candidate selection, and evidence normalization.

This package is a refactored split of the former monolithic ``source_pipeline``
module.  All public symbols are re-exported here so that existing import
statements (``from …source_pipeline import X``) continue to work.
"""

from ._models import EvidenceRecord, QuestionProfile, SourceCandidate
from ._utils import (
    URL_RE,
    SEARCH_STOP_WORDS,
    extract_urls,
    is_metric_row_lookup_question,
    is_youtube_url,
    query_tokens,
    registered_host,
)
from .candidate_ranker import CANDIDATE_SCORES, score_candidates
from .evidence_normalizer import (
    evidence_records_from_tool_output,
    parse_fetch_metadata,
    parse_result_blocks,
    serialize_candidates,
    serialize_evidence,
)
from .question_classifier import profile_question
from .source_labels import (
    select_adapter_name,
    source_family_for_url,
    source_label_for_url,
)

__all__ = [
    # Models
    "EvidenceRecord",
    "QuestionProfile",
    "SourceCandidate",
    # Classifier
    "profile_question",
    # Ranker
    "CANDIDATE_SCORES",
    "score_candidates",
    # Evidence normalizer
    "evidence_records_from_tool_output",
    "parse_fetch_metadata",
    "parse_result_blocks",
    "serialize_candidates",
    "serialize_evidence",
    # Source labels
    "source_family_for_url",
    "select_adapter_name",
    "source_label_for_url",
    # Shared utils
    "URL_RE",
    "SEARCH_STOP_WORDS",
    "extract_urls",
    "is_metric_row_lookup_question",
    "is_youtube_url",
    "query_tokens",
    "registered_host",
]
