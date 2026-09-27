"""Tests for response-level answer completeness and factual fidelity."""

from __future__ import annotations

import pytest

from src.evaluation.answer_quality_review import (
    REVIEW_CONFIG_ID,
    REVIEW_SCHEMA_VERSION,
    REVIEW_STATUS_COMPLETED,
    aggregate_answer_quality,
    apply_review_labels,
    build_review_row,
    collect_required_point_ids,
    validate_review_row,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)


def make_record(
    *,
    question_id: str = "en_en_001",
    query_language: str = "en",
    target_language: str = "en",
) -> EvaluationRecord:
    """Build one deterministic retrieval benchmark fixture."""

    return EvaluationRecord(
        question_id=question_id,
        query="What does the document say?",
        query_language=query_language,
        target_language=target_language,
        category="test",
        expected_document_ids=(
            "document-1",
        ),
        primary_relevant_chunk_ids=(
            "point-primary",
        ),
        relevant_chunk_ids=(
            "point-primary",
            "point-supporting",
        ),
        notes="Test record.",
    )


def make_payload(
    point_id: str,
) -> dict:
    """Build one exact Qdrant-style payload fixture."""

    return {
        "chunk_id": (
            f"chunk-{point_id}"
        ),
        "document_id": "document-1",
        "title": "Government Document",
        "organization": "Government of Nepal",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": (
            "https://example.gov.np/"
        ),
        "chunk_text": (
            f"Exact source passage for {point_id}."
        ),
    }


def make_selected(
    point_id: str,
    evidence_id: str,
) -> dict:
    """Build persisted production selected-evidence metadata."""

    payload = (
        make_payload(
            point_id
        )
    )

    return {
        "evidence_id": evidence_id,
        "point_id": point_id,
        "chunk_id": (
            payload[
                "chunk_id"
            ]
        ),
        "document_id": (
            payload[
                "document_id"
            ]
        ),
        "title": (
            payload[
                "title"
            ]
        ),
        "organization": (
            payload[
                "organization"
            ]
        ),
        "language": (
            payload[
                "language"
            ]
        ),
        "page_start": (
            payload[
                "page_start"
            ]
        ),
        "page_end": (
            payload[
                "page_end"
            ]
        ),
        "source_url": (
            payload[
                "source_url"
            ]
        ),
        "original_rank": 1,
        "rerank_score": 0.9,
    }


def make_rag_row(
    *,
    question_id: str = "en_en_001",
    query_language: str = "en",
    target_language: str = "en",
) -> dict:
    """Build one persisted production RAG fixture."""

    return {
        "schema_version": 1,
        "run_config_id": (
            "production-rag-v2-interactions"
        ),
        "question_id": question_id,
        "query": (
            "What does the document say?"
        ),
        "query_language": (
            query_language
        ),
        "target_language": (
            target_language
        ),
        "answer_language": (
            query_language
        ),
        "category": "test",
        "expected_document_ids": [
            "document-1",
        ],
        "primary_relevant_chunk_ids": [
            "point-primary",
        ],
        "relevant_chunk_ids": [
            "point-primary",
            "point-supporting",
        ],
        "answer_text": (
            "The document gives the "
            "supported answer [E1]."
        ),
        "accepted": True,
        "withheld": False,
        "selected_evidence": [
            make_selected(
                "point-primary",
                "E1",
            ),
            make_selected(
                "point-other",
                "E2",
            ),
        ],
        "metrics": {
            "selected_primary_hit": 1.0,
            "selected_relevant_recall": 0.5,
        },
    }


def payloads() -> dict:
    """Return payloads for selected and gold evidence."""

    return {
        "point-primary": (
            make_payload(
                "point-primary"
            )
        ),
        "point-supporting": (
            make_payload(
                "point-supporting"
            )
        ),
        "point-other": (
            make_payload(
                "point-other"
            )
        ),
    }


def pending_row() -> dict:
    """Build one valid pending review row."""

    return (
        build_review_row(
            make_rag_row(),
            make_record(),
            payloads_by_point_id=(
                payloads()
            ),
        )
    )


def completed_row(
    *,
    question_id: str,
    completeness: str,
    fidelity: str,
) -> dict:
    """Build one completed review fixture."""

    record = (
        make_record(
            question_id=(
                question_id
            )
        )
    )

    rag = (
        make_rag_row(
            question_id=(
                question_id
            )
        )
    )

    row = (
        build_review_row(
            rag,
            record,
            payloads_by_point_id=(
                payloads()
            ),
        )
    )

    return (
        apply_review_labels(
            row,
            completeness_label=(
                completeness
            ),
            factual_fidelity_label=(
                fidelity
            ),
        )
    )


def test_build_review_row_creates_pending_review() -> None:
    """The builder should preserve exact benchmark evidence and answer."""

    row = (
        pending_row()
    )

    assert (
        row[
            "review_schema_version"
        ]
        == REVIEW_SCHEMA_VERSION
    )

    assert (
        row[
            "review_config_id"
        ]
        == REVIEW_CONFIG_ID
    )

    assert (
        row[
            "review_status"
        ]
        == "pending"
    )

    assert (
        row[
            "completeness_label"
        ]
        is None
    )

    assert (
        row[
            "factual_fidelity_label"
        ]
        is None
    )

    assert [
        item[
            "point_id"
        ]
        for item in (
            row[
                "gold_reference_evidence"
            ]
        )
    ] == [
        "point-primary",
        "point-supporting",
    ]


def test_build_review_row_marks_primary_gold_evidence() -> None:
    """Primary evidence must be explicit for human completeness review."""

    row = (
        pending_row()
    )

    assert (
        row[
            "gold_reference_evidence"
        ][0][
            "is_primary"
        ]
        is True
    )

    assert (
        row[
            "gold_reference_evidence"
        ][1][
            "is_primary"
        ]
        is False
    )


def test_selected_evidence_tracks_gold_membership() -> None:
    """Selected passages should indicate whether they belong to gold evidence."""

    row = (
        pending_row()
    )

    assert (
        row[
            "selected_evidence"
        ][0][
            "is_gold_relevant"
        ]
        is True
    )

    assert (
        row[
            "selected_evidence"
        ][1][
            "is_gold_relevant"
        ]
        is False
    )


def test_collect_required_point_ids_deduplicates_stably() -> None:
    """Selected and gold evidence should be fetched only once."""

    record = (
        make_record()
    )

    result = (
        collect_required_point_ids(
            [
                make_rag_row(),
            ],
            {
                record.question_id: (
                    record
                ),
            },
        )
    )

    assert result == (
        "point-primary",
        "point-other",
        "point-supporting",
    )


def test_validate_pending_row_accepts_valid_row() -> None:
    """A newly built unlabeled review row should validate."""

    validate_review_row(
        pending_row()
    )


def test_apply_review_labels_completes_row() -> None:
    """Applying both dimensions should complete the review row."""

    completed = (
        apply_review_labels(
            pending_row(),
            completeness_label=(
                "complete"
            ),
            factual_fidelity_label=(
                "fully_faithful"
            ),
            review_notes=(
                "Covers the verified answer."
            ),
        )
    )

    assert (
        completed[
            "review_status"
        ]
        == REVIEW_STATUS_COMPLETED
    )

    assert (
        completed[
            "completeness_label"
        ]
        == "complete"
    )

    assert (
        completed[
            "factual_fidelity_label"
        ]
        == "fully_faithful"
    )

    assert (
        completed[
            "review_notes"
        ]
        == "Covers the verified answer."
    )


def test_apply_review_labels_accepts_needs_review() -> None:
    """Ambiguous human cases should remain explicit rather than forced."""

    completed = (
        apply_review_labels(
            pending_row(),
            completeness_label=(
                "needs_review"
            ),
            factual_fidelity_label=(
                "needs_review"
            ),
        )
    )

    assert (
        completed[
            "review_status"
        ]
        == REVIEW_STATUS_COMPLETED
    )


def test_apply_review_labels_rejects_invalid_completeness() -> None:
    """Unknown completeness labels must fail explicitly."""

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported completeness label"
        ),
    ):
        apply_review_labels(
            pending_row(),
            completeness_label=(
                "perfect"
            ),
            factual_fidelity_label=(
                "fully_faithful"
            ),
        )


def test_apply_review_labels_rejects_invalid_fidelity() -> None:
    """Unknown fidelity labels must fail explicitly."""

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported factual-fidelity label"
        ),
    ):
        apply_review_labels(
            pending_row(),
            completeness_label=(
                "complete"
            ),
            factual_fidelity_label=(
                "perfect"
            ),
        )


def test_pending_row_cannot_contain_labels() -> None:
    """Pending state and completed labels cannot coexist."""

    row = (
        pending_row()
    )

    row[
        "completeness_label"
    ] = "complete"

    with pytest.raises(
        ValueError,
        match=(
            "pending row cannot contain"
        ),
    ):
        validate_review_row(
            row
        )


def test_gold_evidence_must_match_relevant_ids() -> None:
    """Review data cannot silently reorder or lose benchmark reference evidence."""

    row = (
        pending_row()
    )

    row[
        "gold_reference_evidence"
    ] = list(
        reversed(
            row[
                "gold_reference_evidence"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "gold evidence order"
        ),
    ):
        validate_review_row(
            row
        )


def test_aggregate_answer_quality_metrics() -> None:
    """Aggregate metrics should keep completeness and fidelity separate."""

    rows = [
        completed_row(
            question_id="q1",
            completeness="complete",
            fidelity="fully_faithful",
        ),
        completed_row(
            question_id="q2",
            completeness="mostly_complete",
            fidelity="minor_issue",
        ),
        completed_row(
            question_id="q3",
            completeness="incomplete",
            fidelity="major_issue",
        ),
    ]

    metrics = (
        aggregate_answer_quality(
            rows
        )
    )

    assert (
        metrics[
            "completion_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "scorable_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "fully_complete_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "at_least_mostly_complete_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics[
            "fully_faithful_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "no_major_factual_error_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics[
            "major_factual_issue_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "strong_answer_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "acceptable_answer_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )


def test_needs_review_is_excluded_from_scored_denominator() -> None:
    """Ambiguous rows should not be silently scored as quality failures."""

    clear = (
        completed_row(
            question_id="clear",
            completeness="complete",
            fidelity="fully_faithful",
        )
    )

    ambiguous = (
        completed_row(
            question_id="ambiguous",
            completeness="needs_review",
            fidelity="needs_review",
        )
    )

    metrics = (
        aggregate_answer_quality(
            [
                clear,
                ambiguous,
            ]
        )
    )

    assert (
        metrics[
            "completion_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "scored_count"
        ]
        == 1
    )

    assert (
        metrics[
            "scorable_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "fully_complete_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "fully_faithful_rate"
        ]
        == 1.0
    )


def test_pending_rows_count_in_completion_denominator() -> None:
    """Incomplete review progress should remain visible in summary metrics."""

    completed = (
        completed_row(
            question_id="done",
            completeness="complete",
            fidelity="fully_faithful",
        )
    )

    pending = (
        pending_row()
    )

    metrics = (
        aggregate_answer_quality(
            [
                completed,
                pending,
            ]
        )
    )

    assert (
        metrics[
            "question_count"
        ]
        == 2
    )

    assert (
        metrics[
            "reviewed_count"
        ]
        == 1
    )

    assert (
        metrics[
            "completion_rate"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_aggregate_requires_rows() -> None:
    """Empty evaluation input should fail explicitly."""

    with pytest.raises(
        ValueError,
        match=(
            "At least one answer-quality"
        ),
    ):
        aggregate_answer_quality(
            []
        )