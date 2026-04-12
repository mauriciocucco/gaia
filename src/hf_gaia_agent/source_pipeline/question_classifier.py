"""Question profiling and classification."""

from __future__ import annotations

from ._models import QuestionProfile
from ._question_classifiers import (
    QUESTION_CLASSIFIERS,
    QuestionClassificationContext,
)
from ._question_extractors import (
    extract_expected_author,
    extract_expected_date,
    extract_subject_name,
    extract_subject_name_unicode,
    infer_text_filter,
)
from ._utils import extract_urls, is_youtube_url


def profile_question(
    question: str,
    *,
    file_name: str | None = None,
    local_file_path: str | None = None,
) -> QuestionProfile:
    urls = tuple(extract_urls(question))
    context = QuestionClassificationContext(
        question=question,
        lowered_question=question.lower(),
        urls=urls,
        generic_urls=tuple(url for url in urls if not is_youtube_url(url)),
        file_name=file_name,
        local_file_path=local_file_path,
        expected_date=extract_expected_date(question),
        expected_author=extract_expected_author(question),
        subject_name=extract_subject_name_unicode(question)
        or extract_subject_name(question),
        text_filter=infer_text_filter(question),
    )
    for classifier in QUESTION_CLASSIFIERS:
        profile = classifier.classify(context)
        if profile is not None:
            return profile
    raise AssertionError("Question classifier registry must include a fallback classifier.")
