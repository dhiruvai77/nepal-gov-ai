"""Tests for human response-level abstention review."""

from __future__ import annotations

import json

import pytest

from src.evaluation.response_behavior_review import (
    REVIEW_CONFIG_ID,
    REVIEW_SCHEMA_VERSION,
    REVIEW_STATUS_COMPLETED,
    aggregate_response_behavior_review,
    apply_response_behavior_label,
    build_review_row,
    initialize_review_output,
    load_review_rows,
    source_record_sha256,
)


def make_source_row(
    *,
    question_id: str = "ie_en_en_001",
    expected_behavior: str = "answer",
    accepted: bool = True,
    answer_text: str = "Supported answer [E1].",
) -> dict:
    """Build one persisted insufficient-evidence benchmark row."""

    return {
        "schema_version": 1,
        "run_config_id": (
            "insufficient-evidence-v1"
        ),
        "question_id": (
            question_id
        ),
        "query": (
            "What does the law say?"
        ),
        "query_language": "en",
        "target_language": "en",
        "category": "law",
        "expected_behavior": (
            expected_behavior
        ),
        "case_type": (
            "answerable_control"
            if expected_behavior
            == "answer"
            else "out_of_corpus_document"
        ),
        "expected_document_ids": (
            ["example"]
            if expected_behavior
            == "answer"
            else []
        ),
        "reference_question_id": (
            "en_en_001"
            if expected_behavior
            == "answer"
            else None
        ),
        "notes": "Test.",
        "accepted": (
            accepted
        ),
        "withheld": (
            not accepted
        ),
        "reason": (
            None
            if accepted
            else "missing_citations"
        ),
        "answer_text": (
            answer_text
        ),
        "generated_answer_text": (
            answer_text
        ),
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "selected_evidence": [
            {
                "evidence_id": "E1",
                "document_id": "example",
            }
        ],
        "cited_evidence": (
            [
                {
                    "evidence_id": "E1",
                    "document_id": "example",
                }
            ]
            if accepted
            else []
        ),
        "invalid_evidence_ids": [],
        "sources": [],
        "metrics": {},
    }


def write_source(
    path,
    rows,
) -> None:
    """Write source rows as JSONL."""

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


def test_source_record_sha256_is_deterministic() -> None:
    row = (
        make_source_row()
    )

    assert (
        source_record_sha256(
            row
        )
        == source_record_sha256(
            row
        )
    )


def test_build_review_row_creates_pending_row() -> None:
    row = (
        build_review_row(
            make_source_row()
        )
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
            "response_behavior_label"
        ]
        is None
    )

    assert (
        row[
            "selected_document_ids"
        ]
        == [
            "example",
        ]
    )


def test_apply_response_behavior_label_completes_row() -> None:
    row = (
        build_review_row(
            make_source_row()
        )
    )

    completed = (
        apply_response_behavior_label(
            row,
            response_behavior_label=(
                "substantive_answer"
            ),
            review_notes=(
                "Direct answer."
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
            "response_behavior_label"
        ]
        == "substantive_answer"
    )


def test_apply_rejects_invalid_label() -> None:
    row = (
        build_review_row(
            make_source_row()
        )
    )

    with pytest.raises(
        ValueError,
        match="Unsupported response",
    ):
        apply_response_behavior_label(
            row,
            response_behavior_label="invalid",
        )


def test_initialize_review_output_creates_rows(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "source.jsonl"
    )

    output_path = (
        tmp_path
        / "review.jsonl"
    )

    write_source(
        source_path,
        [
            make_source_row(),
        ],
    )

    rows = (
        initialize_review_output(
            source_path,
            output_path=(
                output_path
            ),
        )
    )

    assert len(
        rows
    ) == 1

    assert (
        output_path.exists()
    )


def test_initialize_review_output_resumes_labels(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "source.jsonl"
    )

    output_path = (
        tmp_path
        / "review.jsonl"
    )

    source_row = (
        make_source_row()
    )

    write_source(
        source_path,
        [
            source_row,
        ],
    )

    rows = (
        initialize_review_output(
            source_path,
            output_path=(
                output_path
            ),
        )
    )

    completed = [
        apply_response_behavior_label(
            rows[
                0
            ],
            response_behavior_label=(
                "substantive_answer"
            ),
        )
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                completed[
                    0
                ],
                ensure_ascii=False,
            )
        )

        handle.write(
            "\n"
        )

    resumed = (
        initialize_review_output(
            source_path,
            output_path=(
                output_path
            ),
        )
    )

    assert (
        resumed[
            0
        ][
            "review_status"
        ]
        == "completed"
    )


def test_initialize_rejects_changed_source(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "source.jsonl"
    )

    output_path = (
        tmp_path
        / "review.jsonl"
    )

    source_row = (
        make_source_row()
    )

    write_source(
        source_path,
        [
            source_row,
        ],
    )

    initialize_review_output(
        source_path,
        output_path=(
            output_path
        ),
    )

    changed = dict(
        source_row
    )

    changed[
        "answer_text"
    ] = "Changed answer."

    write_source(
        source_path,
        [
            changed,
        ],
    )

    with pytest.raises(
        ValueError,
        match="different benchmark output",
    ):
        initialize_review_output(
            source_path,
            output_path=(
                output_path
            ),
        )


def test_load_review_rows_round_trip(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "source.jsonl"
    )

    output_path = (
        tmp_path
        / "review.jsonl"
    )

    write_source(
        source_path,
        [
            make_source_row(),
        ],
    )

    initialize_review_output(
        source_path,
        output_path=(
            output_path
        ),
    )

    rows = (
        load_review_rows(
            output_path
        )
    )

    assert len(
        rows
    ) == 1


def test_aggregate_perfect_behavior() -> None:
    answer = (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    question_id=(
                        "answer"
                    ),
                    expected_behavior=(
                        "answer"
                    ),
                    accepted=True,
                )
            ),
            response_behavior_label=(
                "substantive_answer"
            ),
        )
    )

    abstain = (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    question_id=(
                        "withhold"
                    ),
                    expected_behavior=(
                        "withhold"
                    ),
                    accepted=False,
                    answer_text=(
                        "Insufficient evidence."
                    ),
                )
            ),
            response_behavior_label=(
                "abstained"
            ),
        )
    )

    metrics = (
        aggregate_response_behavior_review(
            [
                answer,
                abstain,
            ]
        )
    )

    assert (
        metrics[
            "behavior_accuracy"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "answer_delivery_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "semantic_abstention_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "unsafe_substantive_answer_rate"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_aggregate_detects_safe_abstention_despite_structural_accept() -> None:
    row = (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    question_id="false_accept",
                    expected_behavior="withhold",
                    accepted=True,
                    answer_text=(
                        "The supplied evidence "
                        "is insufficient."
                    ),
                )
            ),
            response_behavior_label=(
                "abstained"
            ),
        )
    )

    metrics = (
        aggregate_response_behavior_review(
            [
                row,
            ]
        )
    )

    assert (
        metrics[
            "structural_false_accept_count"
        ]
        == 1
    )

    assert (
        metrics[
            "safe_abstention_despite_structural_accept_count"
        ]
        == 1
    )

    assert (
        metrics[
            "safe_abstention_despite_structural_accept_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "semantic_abstention_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_aggregate_detects_unsafe_substantive_answer() -> None:
    row = (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    question_id="unsafe",
                    expected_behavior="withhold",
                    accepted=True,
                )
            ),
            response_behavior_label=(
                "substantive_answer"
            ),
        )
    )

    metrics = (
        aggregate_response_behavior_review(
            [
                row,
            ]
        )
    )

    assert (
        metrics[
            "unsafe_substantive_answer_count"
        ]
        == 1
    )

    assert (
        metrics[
            "unsafe_substantive_answer_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_aggregate_detects_unnecessary_abstention() -> None:
    row = (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    expected_behavior="answer",
                    accepted=True,
                    answer_text=(
                        "Insufficient evidence."
                    ),
                )
            ),
            response_behavior_label=(
                "abstained"
            ),
        )
    )

    metrics = (
        aggregate_response_behavior_review(
            [
                row,
            ]
        )
    )

    assert (
        metrics[
            "unnecessary_abstention_count"
        ]
        == 1
    )

    assert (
        metrics[
            "unnecessary_abstention_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_aggregate_requires_rows() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        aggregate_response_behavior_review(
            []
        )