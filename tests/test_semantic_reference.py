"""Tests for generic completed semantic-reference validation."""

from __future__ import annotations

import json

import pytest

from src.evaluation.build_semantic_review_subset import (
    SAMPLE_CONFIG_ID,
)
from src.evaluation.semantic_reference import (
    load_semantic_reference_rows,
    validate_semantic_reference_row,
)


HARD_CASE_CONFIG_ID = (
    "semantic-judge-hard-cases-v1"
)


def make_evidence(
    *,
    evidence_id: str = "E1",
    individual_support_label: str = (
        "supported"
    ),
) -> dict:
    """Build one completed evidence fixture."""

    return {
        "evidence_id": (
            evidence_id
        ),
        "point_id": (
            f"point-{evidence_id}"
        ),
        "chunk_id": (
            f"chunk-{evidence_id}"
        ),
        "document_id": (
            "example_document"
        ),
        "title": (
            "Example Government Document"
        ),
        "organization": (
            "Government of Nepal"
        ),
        "language": "en",
        "page_start": 1,
        "page_end": 1,
        "source_url": (
            "https://example.gov.np/"
        ),
        "chunk_text": (
            "The document directly states "
            "the example provision."
        ),
        "individual_support_label": (
            individual_support_label
        ),
        "individual_support_notes": None,
    }


def make_reference_row(
    *,
    claim_id: str = "q1_c001",
    config_id: str = (
        SAMPLE_CONFIG_ID
    ),
    query_language: str = "en",
    target_language: str = "en",
    semantic_support_label: str = (
        "supported"
    ),
    citation_requirement_label: str = (
        "required"
    ),
    cited: bool = True,
    individual_support_label: str = (
        "supported"
    ),
    review_status: str = (
        "completed"
    ),
) -> dict:
    """Build one generic semantic-reference fixture."""

    evidence_ids = (
        [
            "E1",
        ]
        if cited
        else []
    )

    cited_evidence = (
        [
            make_evidence(
                individual_support_label=(
                    individual_support_label
                )
            ),
        ]
        if cited
        else []
    )

    return {
        "schema_version": 1,
        "source_run_config_id": (
            "reference-test-source"
        ),
        "source_schema_version": 1,
        "question_id": "q1",
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
        "provider": "reference",
        "model": "none",
        "claim_id": (
            claim_id
        ),
        "claim_index": 1,
        "claim_text": (
            "The document directly states "
            "the example provision."
        ),
        "raw_text": (
            "The document directly states "
            "the example provision [E1]."
            if cited
            else (
                "The document directly states "
                "the example provision."
            )
        ),
        "has_citation": (
            cited
        ),
        "evidence_ids": (
            evidence_ids
        ),
        "cited_evidence": (
            cited_evidence
        ),
        "semantic_support_label": (
            semantic_support_label
        ),
        "citation_requirement_label": (
            citation_requirement_label
        ),
        "semantic_notes": None,
        "review_sample_config_id": (
            config_id
        ),
        "review_language_pair": (
            f"{query_language}"
            f"->{target_language}"
        ),
        "review_stratum": (
            "reference_test"
        ),
        "review_status": (
            review_status
        ),
    }


def write_rows(
    path,
    rows,
) -> None:
    """Write reference fixtures as JSONL."""

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


def test_accepts_original_production_reference_config() -> None:
    validate_semantic_reference_row(
        make_reference_row()
    )


def test_accepts_separate_hard_case_reference_config() -> None:
    validate_semantic_reference_row(
        make_reference_row(
            config_id=(
                HARD_CASE_CONFIG_ID
            )
        )
    )


def test_accepts_unsupported_reference() -> None:
    validate_semantic_reference_row(
        make_reference_row(
            semantic_support_label=(
                "unsupported"
            ),
            individual_support_label=(
                "unsupported"
            ),
        )
    )


def test_accepts_needs_review_reference() -> None:
    validate_semantic_reference_row(
        make_reference_row(
            semantic_support_label=(
                "needs_review"
            ),
            individual_support_label=(
                "needs_review"
            ),
        )
    )


def test_accepts_unclear_citation_requirement() -> None:
    validate_semantic_reference_row(
        make_reference_row(
            semantic_support_label=(
                "not_a_factual_claim"
            ),
            citation_requirement_label=(
                "unclear"
            ),
            cited=False,
        )
    )


def test_rejects_pending_reference() -> None:
    with pytest.raises(
        ValueError,
        match="must be completed",
    ):
        validate_semantic_reference_row(
            make_reference_row(
                review_status="pending"
            )
        )


def test_rejects_empty_reference_config() -> None:
    with pytest.raises(
        ValueError,
        match="review_sample_config_id",
    ):
        validate_semantic_reference_row(
            make_reference_row(
                config_id="   "
            )
        )


def test_rejects_inconsistent_language_pair() -> None:
    row = (
        make_reference_row(
            query_language="en",
            target_language="ne",
        )
    )

    row[
        "review_language_pair"
    ] = "en->en"

    with pytest.raises(
        ValueError,
        match=(
            "inconsistent "
            "review_language_pair"
        ),
    ):
        validate_semantic_reference_row(
            row
        )


def test_rejects_inconsistent_citation_state() -> None:
    row = (
        make_reference_row()
    )

    row[
        "has_citation"
    ] = False

    with pytest.raises(
        ValueError,
        match=(
            "inconsistent citation state"
        ),
    ):
        validate_semantic_reference_row(
            row
        )


def test_rejects_duplicate_evidence_ids() -> None:
    row = (
        make_reference_row()
    )

    row[
        "evidence_ids"
    ] = [
        "E1",
        "E1",
    ]

    row[
        "cited_evidence"
    ] = [
        make_evidence(
            evidence_id="E1"
        ),
        make_evidence(
            evidence_id="E1"
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "duplicate evidence IDs"
        ),
    ):
        validate_semantic_reference_row(
            row
        )


def test_rejects_mismatched_evidence_identity() -> None:
    row = (
        make_reference_row()
    )

    row[
        "cited_evidence"
    ][
        0
    ][
        "evidence_id"
    ] = "E2"

    with pytest.raises(
        ValueError,
        match=(
            "do not match evidence_ids"
        ),
    ):
        validate_semantic_reference_row(
            row
        )


def test_rejects_invalid_individual_support_label() -> None:
    row = (
        make_reference_row()
    )

    row[
        "cited_evidence"
    ][
        0
    ][
        "individual_support_label"
    ] = "maybe"

    with pytest.raises(
        ValueError,
        match=(
            "invalid "
            "individual_support_label"
        ),
    ):
        validate_semantic_reference_row(
            row
        )


def test_uncited_supported_reference_is_invalid() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "cannot be supported"
        ),
    ):
        validate_semantic_reference_row(
            make_reference_row(
                cited=False,
                semantic_support_label=(
                    "supported"
                ),
            )
        )


def test_load_reference_rows_supports_hard_case_config(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "reference.jsonl"
    )

    write_rows(
        path,
        [
            make_reference_row(
                config_id=(
                    HARD_CASE_CONFIG_ID
                )
            ),
        ],
    )

    rows = (
        load_semantic_reference_rows(
            path
        )
    )

    assert len(
        rows
    ) == 1

    assert (
        rows[
            0
        ][
            "review_sample_config_id"
        ]
        == HARD_CASE_CONFIG_ID
    )


def test_load_reference_rows_rejects_duplicate_claim_id(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "reference.jsonl"
    )

    row = (
        make_reference_row()
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
        match="duplicate claim_id",
    ):
        load_semantic_reference_rows(
            path
        )


def test_load_reference_rows_requires_content(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "empty.jsonl"
    )

    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="contains no rows",
    ):
        load_semantic_reference_rows(
            path
        )