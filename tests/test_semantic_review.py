"""Tests for human semantic citation review."""

from __future__ import annotations

import json

import pytest

from src.evaluation.build_semantic_review_subset import (
    SAMPLE_CONFIG_ID,
)
from src.evaluation.semantic_review import (
    REVIEW_STATUS_COMPLETED,
    aggregate_semantic_review,
    apply_review_labels,
    initialize_review_output,
    load_review_rows,
    validate_review_row,
)


def make_evidence(
    *,
    evidence_id: str = "E1",
) -> dict:
    """Create one semantic evidence fixture."""

    return {
        "evidence_id": evidence_id,
        "point_id": (
            f"point-{evidence_id}"
        ),
        "chunk_id": (
            f"chunk-{evidence_id}"
        ),
        "document_id": "document-1",
        "title": "Government Document",
        "organization": "Government of Nepal",
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": "https://example.gov.np/",
        "chunk_text": (
            "Exact government evidence."
        ),
        "individual_support_label": None,
        "individual_support_notes": None,
    }


def make_row(
    *,
    claim_id: str = "q1_c001",
    query_language: str = "en",
    target_language: str = "en",
    cited: bool = True,
) -> dict:
    """Create one pending review fixture."""

    evidence_ids = (
        [
            "E1",
        ]
        if cited
        else []
    )

    cited_evidence = (
        [
            make_evidence()
        ]
        if cited
        else []
    )

    return {
        "schema_version": 1,
        "source_run_config_id": (
            "production-rag-v2-interactions"
        ),
        "source_schema_version": 1,
        "question_id": "q1",
        "query": "Question?",
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
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "claim_id": claim_id,
        "claim_index": 1,
        "claim_text": "Government claim.",
        "raw_text": (
            "Government claim [E1]."
            if cited
            else "Government claim."
        ),
        "has_citation": cited,
        "evidence_ids": evidence_ids,
        "cited_evidence": cited_evidence,
        "semantic_support_label": None,
        "citation_requirement_label": None,
        "semantic_notes": None,
        "review_sample_config_id": (
            SAMPLE_CONFIG_ID
        ),
        "review_language_pair": (
            f"{query_language}->"
            f"{target_language}"
        ),
        "review_stratum": (
            "single_other"
            if cited
            else "uncited_other"
        ),
        "review_status": "pending",
    }


def write_rows(
    path,
    rows,
) -> None:
    """Write JSONL fixtures."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            handle.write(
                "\n"
            )


def test_pending_row_is_valid() -> None:
    validate_review_row(
        make_row()
    )


def test_apply_review_labels_completes_cited_claim() -> None:
    completed = (
        apply_review_labels(
            make_row(),
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "supported",
            ],
            semantic_notes=(
                "Directly stated."
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
            "semantic_support_label"
        ]
        == "supported"
    )

    assert (
        completed[
            "cited_evidence"
        ][0][
            "individual_support_label"
        ]
        == "supported"
    )


def test_uncited_claim_can_be_marked_unsupported() -> None:
    completed = (
        apply_review_labels(
            make_row(
                cited=False
            ),
            semantic_support_label=(
                "unsupported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[],
        )
    )

    assert (
        completed[
            "semantic_support_label"
        ]
        == "unsupported"
    )


def test_uncited_claim_cannot_be_marked_supported() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "uncited claim cannot"
        ),
    ):
        apply_review_labels(
            make_row(
                cited=False
            ),
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[],
        )


def test_rejects_wrong_individual_label_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "requires 1 individual "
            "evidence labels"
        ),
    ):
        apply_review_labels(
            make_row(),
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[],
        )


def test_rejects_invalid_semantic_label() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unsupported semantic "
            "support label"
        ),
    ):
        apply_review_labels(
            make_row(),
            semantic_support_label=(
                "perfect"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "supported",
            ],
        )


def test_initialize_review_output_copies_source(
    tmp_path,
) -> None:
    source = (
        tmp_path
        / "source.jsonl"
    )

    output = (
        tmp_path
        / "output.jsonl"
    )

    rows = [
        make_row(),
    ]

    write_rows(
        source,
        rows,
    )

    initialized = (
        initialize_review_output(
            source,
            output_path=output,
        )
    )

    assert len(
        initialized
    ) == 1

    assert output.exists()

    persisted = (
        load_review_rows(
            output
        )
    )

    assert (
        persisted[0][
            "claim_id"
        ]
        == "q1_c001"
    )


def test_initialize_review_output_resumes_labels(
    tmp_path,
) -> None:
    source = (
        tmp_path
        / "source.jsonl"
    )

    output = (
        tmp_path
        / "output.jsonl"
    )

    pending = (
        make_row()
    )

    completed = (
        apply_review_labels(
            pending,
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "supported",
            ],
        )
    )

    write_rows(
        source,
        [
            pending,
        ],
    )

    write_rows(
        output,
        [
            completed,
        ],
    )

    resumed = (
        initialize_review_output(
            source,
            output_path=output,
        )
    )

    assert (
        resumed[0][
            "review_status"
        ]
        == REVIEW_STATUS_COMPLETED
    )


def test_initialize_rejects_mismatched_existing_output(
    tmp_path,
) -> None:
    source = (
        tmp_path
        / "source.jsonl"
    )

    output = (
        tmp_path
        / "output.jsonl"
    )

    write_rows(
        source,
        [
            make_row(
                claim_id="q1_c001"
            ),
        ],
    )

    write_rows(
        output,
        [
            make_row(
                claim_id="q1_c999"
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match"
        ),
    ):
        initialize_review_output(
            source,
            output_path=output,
        )


def test_load_rejects_duplicate_claim_id(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "review.jsonl"
    )

    row = (
        make_row()
    )

    write_rows(
        path,
        [
            row,
            row,
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "duplicate claim_id"
        ),
    ):
        load_review_rows(
            path
        )


def test_aggregate_semantic_review_metrics() -> None:
    supported = (
        apply_review_labels(
            make_row(
                claim_id="q1"
            ),
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "supported",
            ],
        )
    )

    partial = (
        apply_review_labels(
            make_row(
                claim_id="q2"
            ),
            semantic_support_label=(
                "partially_supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "partially_supported",
            ],
        )
    )

    unsupported = (
        apply_review_labels(
            make_row(
                claim_id="q3",
                cited=False,
            ),
            semantic_support_label=(
                "unsupported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[],
        )
    )

    metrics = (
        aggregate_semantic_review(
            [
                supported,
                partial,
                unsupported,
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
            "fully_supported_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "at_least_partially_supported_rate"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics[
            "unsupported_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics[
            "required_citation_coverage"
        ]
        == pytest.approx(
            2 / 3
        )
    )

    assert (
        metrics[
            "individual_at_least_partial_support_rate"
        ]
        == 1.0
    )


def test_pending_rows_count_toward_completion_denominator() -> None:
    completed = (
        apply_review_labels(
            make_row(
                claim_id="q1"
            ),
            semantic_support_label=(
                "supported"
            ),
            citation_requirement_label=(
                "required"
            ),
            individual_support_labels=[
                "supported",
            ],
        )
    )

    pending = (
        make_row(
            claim_id="q2"
        )
    )

    metrics = (
        aggregate_semantic_review(
            [
                completed,
                pending,
            ]
        )
    )

    assert (
        metrics[
            "claim_count"
        ]
        == 2
    )

    assert (
        metrics[
            "reviewed_claim_count"
        ]
        == 1
    )

    assert (
        metrics[
            "completion_rate"
        ]
        == 0.5
    )


def test_unnecessary_citation_rate() -> None:
    row = (
        apply_review_labels(
            make_row(),
            semantic_support_label=(
                "not_a_factual_claim"
            ),
            citation_requirement_label=(
                "not_required"
            ),
            individual_support_labels=[
                "unsupported",
            ],
        )
    )

    metrics = (
        aggregate_semantic_review(
            [
                row,
            ]
        )
    )

    assert (
        metrics[
            "unnecessary_citation_count"
        ]
        == 1
    )

    assert (
        metrics[
            "unnecessary_citation_rate"
        ]
        == 1.0
    )