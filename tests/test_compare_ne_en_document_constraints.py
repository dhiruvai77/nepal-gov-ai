"""Tests for the targeted NE->EN document-constraint diagnostic."""

from __future__ import annotations

import pytest

from src.evaluation.compare_ne_en_document_constraints import (
    DEFAULT_DATASET_PATH,
    calculate_metrics,
    expected_document_id,
    first_primary_rank,
    format_rank,
    select_ne_en_records,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
    load_evaluation_records,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


EXPECTED_NE_EN_IDS = (
    "ne_en_001",
    "ne_en_002",
    "ne_en_003",
    "ne_en_004",
    "ne_en_005",
    "ne_en_006",
)


def make_record(
    *,
    question_id: str = "ne_en_001",
    query_language: str = "ne",
    target_language: str = "en",
    expected_document_ids: tuple[
        str,
        ...,
    ] = (
        "constitution_nepal_current_en",
    ),
    primary_ids: tuple[
        str,
        ...,
    ] = (
        "primary",
    ),
    relevant_ids: tuple[
        str,
        ...,
    ] = (
        "primary",
        "supporting",
    ),
) -> EvaluationRecord:
    """Build one deterministic benchmark fixture."""

    return EvaluationRecord(
        question_id=(
            question_id
        ),
        query=(
            "नेपालको संविधानले "
            "स्वास्थ्यसम्बन्धी अधिकारबारे "
            "के व्यवस्था गरेको छ?"
        ),
        query_language=(
            query_language
        ),
        target_language=(
            target_language
        ),
        category=(
            "constitution_law"
        ),
        expected_document_ids=(
            expected_document_ids
        ),
        primary_relevant_chunk_ids=(
            primary_ids
        ),
        relevant_chunk_ids=(
            relevant_ids
        ),
        notes=(
            "Test record."
        ),
    )


def make_result(
    point_id: str,
) -> RetrievalResult:
    """Build one deterministic normalized retrieval result."""

    return RetrievalResult(
        point_id=(
            point_id
        ),
        score=1.0,
        chunk_id=(
            f"chunk-{point_id}"
        ),
        document_id=(
            "constitution_nepal_current_en"
        ),
        title=(
            "Constitution of Nepal"
        ),
        organization=(
            "Nepal Law Commission"
        ),
        language="en",
        page_start=1,
        page_end=1,
        source_url=(
            "https://example.gov.np/"
        ),
        chunk_text=(
            f"Evidence {point_id}."
        ),
        chunk_index=1,
        token_count=100,
        category=(
            "constitution_law"
        ),
        document_type=(
            "constitution"
        ),
        publication_date=None,
        section=None,
        subsection=None,
        article_number=None,
        article_title=None,
        extraction_method=(
            "native"
        ),
    )


def test_dataset_contains_expected_ne_en_slice() -> None:
    records = (
        load_evaluation_records(
            DEFAULT_DATASET_PATH
        )
    )

    selected = (
        select_ne_en_records(
            records
        )
    )

    assert tuple(
        record.question_id
        for record in selected
    ) == EXPECTED_NE_EN_IDS


def test_select_ne_en_records_filters_and_sorts() -> None:
    records = [
        make_record(
            question_id="ne_en_002"
        ),
        make_record(
            question_id="en_en_001",
            query_language="en",
            target_language="en",
        ),
        make_record(
            question_id="ne_en_001"
        ),
    ]

    selected = (
        select_ne_en_records(
            records
        )
    )

    assert [
        record.question_id
        for record in selected
    ] == [
        "ne_en_001",
        "ne_en_002",
    ]


def test_expected_document_id_returns_single_document() -> None:
    record = (
        make_record(
            expected_document_ids=(
                "economic_survey_2023_24_en",
            )
        )
    )

    assert (
        expected_document_id(
            record
        )
        == "economic_survey_2023_24_en"
    )


def test_expected_document_id_rejects_multiple_documents() -> None:
    record = (
        make_record(
            expected_document_ids=(
                "doc-a",
                "doc-b",
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "requires exactly one"
        ),
    ):
        expected_document_id(
            record
        )


def test_expected_document_id_rejects_no_documents() -> None:
    record = (
        make_record(
            expected_document_ids=()
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "requires exactly one"
        ),
    ):
        expected_document_id(
            record
        )


def test_expected_document_id_rejects_blank_document() -> None:
    record = (
        make_record(
            expected_document_ids=(
                "   ",
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "must contain non-whitespace text"
        ),
    ):
        expected_document_id(
            record
        )


def test_first_primary_rank_returns_match() -> None:
    record = (
        make_record(
            primary_ids=(
                "primary-a",
                "primary-b",
            ),
            relevant_ids=(
                "primary-a",
                "primary-b",
                "supporting",
            ),
        )
    )

    results = [
        make_result(
            "irrelevant"
        ),
        make_result(
            "primary-b"
        ),
    ]

    assert (
        first_primary_rank(
            record,
            results,
        )
        == 2
    )


def test_first_primary_rank_returns_none_when_missing() -> None:
    record = (
        make_record()
    )

    assert (
        first_primary_rank(
            record,
            [
                make_result(
                    "irrelevant"
                )
            ],
        )
        is None
    )


def test_format_rank() -> None:
    assert (
        format_rank(
            4,
            depth=100,
        )
        == "4"
    )

    assert (
        format_rank(
            None,
            depth=100,
        )
        == ">100"
    )


def test_calculate_metrics_uses_standard_evaluator() -> None:
    record = (
        make_record()
    )

    rankings = {
        record.question_id: [
            make_result(
                "irrelevant"
            ),
            make_result(
                "primary"
            ),
            make_result(
                "supporting"
            ),
        ],
    }

    metrics = (
        calculate_metrics(
            [
                record,
            ],
            rankings,
            cutoff=3,
        )
    )

    assert len(
        metrics
    ) == 1

    assert (
        metrics[
            0
        ].hit_rate
        == 1.0
    )

    assert (
        metrics[
            0
        ].reciprocal_rank
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            0
        ].recall
        == 1.0
    )