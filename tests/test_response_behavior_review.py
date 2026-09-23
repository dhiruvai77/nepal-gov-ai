"""Tests for human response-level evidence-sufficiency review."""

from __future__ import annotations

import json

import pytest

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_ANSWER,
    EXPECTED_BEHAVIOR_PARTIAL,
    EXPECTED_BEHAVIOR_WITHHOLD,
)
from src.evaluation.response_behavior_review import (
    REVIEW_CONFIG_ID,
    REVIEW_SCHEMA_VERSION,
    REVIEW_STATUS_COMPLETED,
    aggregate_response_behavior_review,
    apply_response_behavior_label,
    build_review_row,
    initialize_review_output,
    load_review_rows,
    load_source_rows,
    source_record_sha256,
)


def make_source_row(
    *,
    question_id: str = "ie_en_en_001",
    expected_behavior: str = (
        EXPECTED_BEHAVIOR_ANSWER
    ),
    accepted: bool = True,
    answer_text: str = (
        "Supported answer [E1]."
    ),
) -> dict:
    """Build one persisted evidence-sufficiency benchmark row."""

    if (
        expected_behavior
        == EXPECTED_BEHAVIOR_ANSWER
    ):
        case_type = (
            "answerable_control"
        )

        expected_document_ids = [
            "example",
        ]

        reference_question_id = (
            "en_en_001"
        )

    elif (
        expected_behavior
        == EXPECTED_BEHAVIOR_PARTIAL
    ):
        case_type = (
            "partial_evidence"
        )

        expected_document_ids = [
            "example",
        ]

        reference_question_id = None

    else:
        case_type = (
            "out_of_corpus_document"
        )

        expected_document_ids = []

        reference_question_id = None

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
            case_type
        ),
        "expected_document_ids": (
            expected_document_ids
        ),
        "reference_question_id": (
            reference_question_id
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


def completed_row(
    *,
    question_id: str,
    expected_behavior: str,
    accepted: bool,
    response_behavior_label: str,
    answer_text: str = (
        "Supported answer [E1]."
    ),
) -> dict:
    """Build and label one review row."""

    return (
        apply_response_behavior_label(
            build_review_row(
                make_source_row(
                    question_id=(
                        question_id
                    ),
                    expected_behavior=(
                        expected_behavior
                    ),
                    accepted=accepted,
                    answer_text=(
                        answer_text
                    ),
                )
            ),
            response_behavior_label=(
                response_behavior_label
            ),
        )
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


def test_build_review_row_supports_partial_expected_behavior() -> None:
    row = (
        build_review_row(
            make_source_row(
                question_id=(
                    "partial"
                ),
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                accepted=True,
            )
        )
    )

    assert (
        row[
            "expected_behavior"
        ]
        == EXPECTED_BEHAVIOR_PARTIAL
    )

    assert (
        row[
            "case_type"
        ]
        == "partial_evidence"
    )


def test_load_source_rows_accepts_partial_behavior(
    tmp_path,
) -> None:
    source_path = (
        tmp_path
        / "source.jsonl"
    )

    write_source(
        source_path,
        [
            make_source_row(
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                )
            ),
        ],
    )

    rows = (
        load_source_rows(
            source_path
        )
    )

    assert len(
        rows
    ) == 1

    assert (
        rows[
            0
        ][
            "expected_behavior"
        ]
        == EXPECTED_BEHAVIOR_PARTIAL
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


def test_apply_partial_response_behavior_label() -> None:
    row = (
        build_review_row(
            make_source_row(
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                )
            )
        )
    )

    completed = (
        apply_response_behavior_label(
            row,
            response_behavior_label=(
                "partial_answer_with_limitation"
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
        == "partial_answer_with_limitation"
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


def test_initialize_partial_review_output_creates_rows(
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
            make_source_row(
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                )
            ),
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
        rows[
            0
        ][
            "expected_behavior"
        ]
        == EXPECTED_BEHAVIOR_PARTIAL
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


def test_aggregate_perfect_legacy_behavior() -> None:
    answer = (
        completed_row(
            question_id="answer",
            expected_behavior=(
                EXPECTED_BEHAVIOR_ANSWER
            ),
            accepted=True,
            response_behavior_label=(
                "substantive_answer"
            ),
        )
    )

    abstain = (
        completed_row(
            question_id="withhold",
            expected_behavior=(
                EXPECTED_BEHAVIOR_WITHHOLD
            ),
            accepted=False,
            response_behavior_label=(
                "abstained"
            ),
            answer_text=(
                "Insufficient evidence."
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
            "partial_response_success_rate"
        ]
        == pytest.approx(
            0.0
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


def test_aggregate_perfect_three_way_behavior() -> None:
    answer = (
        completed_row(
            question_id="answer",
            expected_behavior=(
                EXPECTED_BEHAVIOR_ANSWER
            ),
            accepted=True,
            response_behavior_label=(
                "substantive_answer"
            ),
        )
    )

    partial = (
        completed_row(
            question_id="partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            accepted=True,
            response_behavior_label=(
                "partial_answer_with_limitation"
            ),
            answer_text=(
                "Supported part [E1]. "
                "The evidence is insufficient "
                "for the remaining part."
            ),
        )
    )

    abstain = (
        completed_row(
            question_id="withhold",
            expected_behavior=(
                EXPECTED_BEHAVIOR_WITHHOLD
            ),
            accepted=False,
            response_behavior_label=(
                "abstained"
            ),
            answer_text=(
                "Insufficient evidence."
            ),
        )
    )

    metrics = (
        aggregate_response_behavior_review(
            [
                answer,
                partial,
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
            "partial_response_success_rate"
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
            "structural_behavior_agreement_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_aggregate_detects_safe_abstention_despite_structural_accept() -> None:
    row = (
        completed_row(
            question_id="false_accept",
            expected_behavior=(
                EXPECTED_BEHAVIOR_WITHHOLD
            ),
            accepted=True,
            response_behavior_label=(
                "abstained"
            ),
            answer_text=(
                "The supplied evidence "
                "is insufficient."
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
        completed_row(
            question_id="unsafe",
            expected_behavior=(
                EXPECTED_BEHAVIOR_WITHHOLD
            ),
            accepted=True,
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
        completed_row(
            question_id="answer",
            expected_behavior=(
                EXPECTED_BEHAVIOR_ANSWER
            ),
            accepted=True,
            response_behavior_label=(
                "abstained"
            ),
            answer_text=(
                "Insufficient evidence."
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


def test_aggregate_detects_correct_partial_response() -> None:
    row = (
        completed_row(
            question_id="partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            accepted=True,
            response_behavior_label=(
                "partial_answer_with_limitation"
            ),
            answer_text=(
                "Supported part [E1]. "
                "The remaining part is not "
                "supported by the evidence."
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
            "partial_response_success_count"
        ]
        == 1
    )

    assert (
        metrics[
            "partial_response_success_rate"
        ]
        == pytest.approx(
            1.0
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


def test_aggregate_detects_unqualified_substantive_on_partial() -> None:
    row = (
        completed_row(
            question_id="partial_unsafe",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            accepted=True,
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
            "partial_response_success_rate"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        metrics[
            "unqualified_substantive_on_partial_count"
        ]
        == 1
    )

    assert (
        metrics[
            "unqualified_substantive_on_partial_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "behavior_accuracy"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_aggregate_detects_full_abstention_on_partial() -> None:
    row = (
        completed_row(
            question_id="partial_abstain",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            accepted=False,
            response_behavior_label=(
                "abstained"
            ),
            answer_text=(
                "Insufficient evidence."
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
            "partial_response_success_rate"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        metrics[
            "full_abstention_on_partial_count"
        ]
        == 1
    )

    assert (
        metrics[
            "full_abstention_on_partial_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "behavior_accuracy"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_partial_answer_counts_as_structural_accept_behavior() -> None:
    row = (
        completed_row(
            question_id="partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            accepted=True,
            response_behavior_label=(
                "partial_answer_with_limitation"
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
            "structural_behavior_agreement_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_partial_answer_on_withhold_remains_incorrect() -> None:
    row = (
        completed_row(
            question_id="withhold_partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_WITHHOLD
            ),
            accepted=True,
            response_behavior_label=(
                "partial_answer_with_limitation"
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
            "partial_answer_on_withhold_count"
        ]
        == 1
    )

    assert (
        metrics[
            "partial_answer_on_withhold_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "behavior_accuracy"
        ]
        == pytest.approx(
            0.0
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