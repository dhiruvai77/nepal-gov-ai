"""Tests for insufficient-evidence benchmark contracts and metrics."""

from __future__ import annotations

import json

import pytest

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_ANSWER,
    EXPECTED_BEHAVIOR_PARTIAL,
    EXPECTED_BEHAVIOR_WITHHOLD,
    InsufficientEvidenceRecord,
    aggregate_insufficient_evidence_metrics,
    evaluate_insufficient_evidence_result,
    load_insufficient_evidence_records,
    parse_insufficient_evidence_record,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
)
from src.rag.pipeline import (
    RAGResult,
)


def make_record(
    *,
    question_id: str = "ie_en_en_001",
    expected_behavior: str = (
        EXPECTED_BEHAVIOR_ANSWER
    ),
    case_type: str = (
        "answerable_control"
    ),
) -> InsufficientEvidenceRecord:
    """Build one evaluator record."""

    return (
        InsufficientEvidenceRecord(
            question_id=question_id,
            query=(
                "What does the law say?"
            ),
            query_language="en",
            target_language="en",
            category="law",
            expected_behavior=(
                expected_behavior
            ),
            case_type=case_type,
            expected_document_ids=(
                (
                    "example_document",
                )
                if expected_behavior
                in {
                    EXPECTED_BEHAVIOR_ANSWER,
                    EXPECTED_BEHAVIOR_PARTIAL,
                }
                else ()
            ),
            reference_question_id=(
                "en_en_001"
                if case_type
                == "answerable_control"
                else None
            ),
            notes="Test case.",
        )
    )


def make_result(
    *,
    accepted: bool,
    reason: (
        EvidenceGuardReason
        | None
    ) = None,
) -> RAGResult:
    """Build one minimal RAG result."""

    return (
        RAGResult(
            answer_text=(
                "Supported answer."
                if accepted
                else "Insufficient evidence."
            ),
            accepted=accepted,
            reason=reason,
            sources=(),
            selected_context=(),
            citation_result=None,
            provider=(
                "test"
                if accepted
                else None
            ),
            model=(
                "test-model"
                if accepted
                else None
            ),
        )
    )


def valid_data() -> dict:
    """Return one valid benchmark JSON object."""

    return {
        "question_id": (
            "ie_en_en_001"
        ),
        "query": (
            "What does the Constitution "
            "say about health?"
        ),
        "query_language": "en",
        "target_language": "en",
        "category": (
            "constitution_law"
        ),
        "expected_behavior": (
            "answer"
        ),
        "case_type": (
            "answerable_control"
        ),
        "expected_document_ids": [
            "constitution_nepal_current_en",
        ],
        "reference_question_id": (
            "en_en_007"
        ),
        "notes": (
            "Existing answerable control."
        ),
    }


def test_parse_valid_answerable_control() -> None:
    record = (
        parse_insufficient_evidence_record(
            valid_data()
        )
    )

    assert (
        record.question_id
        == "ie_en_en_001"
    )

    assert (
        record.expected_behavior
        == "answer"
    )

    assert (
        record.reference_question_id
        == "en_en_007"
    )


def test_parse_valid_withhold_case() -> None:
    data = (
        valid_data()
    )

    data[
        "question_id"
    ] = "ie_en_en_003"

    data[
        "expected_behavior"
    ] = "withhold"

    data[
        "case_type"
    ] = "out_of_corpus_document"

    data[
        "expected_document_ids"
    ] = []

    data[
        "reference_question_id"
    ] = None

    record = (
        parse_insufficient_evidence_record(
            data
        )
    )

    assert (
        record.expected_behavior
        == "withhold"
    )

    assert (
        record.expected_document_ids
        == ()
    )


def test_parse_valid_partial_evidence_case() -> None:
    data = {
        "question_id": "partial",
        "query": (
            "Compare the 2025/26 and 2026/27 "
            "Budget Speech scholarship allocations."
        ),
        "query_language": "en",
        "target_language": "en",
        "category": "education",
        "expected_behavior": "partial",
        "case_type": "partial_evidence",
        "expected_document_ids": [
            "budget_speech_2025_26_en",
        ],
        "reference_question_id": None,
        "notes": (
            "Only the 2025/26 Budget Speech "
            "is indexed."
        ),
    }

    record = (
        parse_insufficient_evidence_record(
            data
        )
    )

    assert (
        record.expected_behavior
        == EXPECTED_BEHAVIOR_PARTIAL
    )

    assert (
        record.case_type
        == "partial_evidence"
    )


def test_parse_valid_mixed_supported_unsupported_case() -> None:
    data = {
        "question_id": "mixed",
        "query": (
            "What constitutional right is "
            "supported, and what does the "
            "unindexed law additionally require?"
        ),
        "query_language": "en",
        "target_language": "en",
        "category": "mixed",
        "expected_behavior": "partial",
        "case_type": (
            "mixed_supported_unsupported"
        ),
        "expected_document_ids": [
            "constitution_nepal_current_en",
        ],
        "reference_question_id": None,
        "notes": (
            "One requested component is "
            "supported and one is not."
        ),
    }

    record = (
        parse_insufficient_evidence_record(
            data
        )
    )

    assert (
        record.expected_behavior
        == EXPECTED_BEHAVIOR_PARTIAL
    )

    assert (
        record.case_type
        == "mixed_supported_unsupported"
    )


def test_parse_rejects_unknown_behavior() -> None:
    data = (
        valid_data()
    )

    data[
        "expected_behavior"
    ] = "maybe"

    with pytest.raises(
        ValueError,
        match="unsupported expected_behavior",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_answerable_control_requires_documents() -> None:
    data = (
        valid_data()
    )

    data[
        "expected_document_ids"
    ] = []

    with pytest.raises(
        ValueError,
        match="at least one expected document",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_partial_case_requires_documents() -> None:
    data = {
        "question_id": "partial",
        "query": "Partially answerable question.",
        "query_language": "en",
        "target_language": "en",
        "category": "education",
        "expected_behavior": "partial",
        "case_type": "partial_evidence",
        "expected_document_ids": [],
        "reference_question_id": None,
        "notes": "Test.",
    }

    with pytest.raises(
        ValueError,
        match="at least one expected document",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_answerable_control_requires_reference_question() -> None:
    data = (
        valid_data()
    )

    data[
        "reference_question_id"
    ] = None

    with pytest.raises(
        ValueError,
        match="reference_question_id",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_partial_evidence_rejects_withhold_behavior() -> None:
    data = {
        "question_id": "partial",
        "query": "Partial question.",
        "query_language": "en",
        "target_language": "en",
        "category": "education",
        "expected_behavior": "withhold",
        "case_type": "partial_evidence",
        "expected_document_ids": [
            "budget_speech_2025_26_en",
        ],
        "reference_question_id": None,
        "notes": "Test.",
    }

    with pytest.raises(
        ValueError,
        match="partial_evidence must use",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_mixed_case_rejects_answer_behavior() -> None:
    data = {
        "question_id": "mixed",
        "query": "Mixed question.",
        "query_language": "en",
        "target_language": "en",
        "category": "mixed",
        "expected_behavior": "answer",
        "case_type": (
            "mixed_supported_unsupported"
        ),
        "expected_document_ids": [
            "constitution_nepal_current_en",
        ],
        "reference_question_id": None,
        "notes": "Test.",
    }

    with pytest.raises(
        ValueError,
        match=(
            "mixed_supported_unsupported "
            "must use"
        ),
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_out_of_corpus_case_requires_withhold_behavior() -> None:
    data = {
        "question_id": "out",
        "query": "Out-of-corpus question.",
        "query_language": "en",
        "target_language": "en",
        "category": "out_of_corpus",
        "expected_behavior": "partial",
        "case_type": "out_of_corpus_document",
        "expected_document_ids": [
            "example",
        ],
        "reference_question_id": None,
        "notes": "Test.",
    }

    with pytest.raises(
        ValueError,
        match="out_of_corpus_document must use",
    ):
        parse_insufficient_evidence_record(
            data
        )


def test_load_records_rejects_duplicate_question_ids(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "dataset.jsonl"
    )

    row = (
        valid_data()
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row
            )
            + "\n"
        )

        handle.write(
            json.dumps(
                row
            )
            + "\n"
        )

    with pytest.raises(
        ValueError,
        match="Duplicate question_id",
    ):
        load_insufficient_evidence_records(
            path
        )


def test_load_records_reads_valid_dataset(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "dataset.jsonl"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                valid_data()
            )
            + "\n"
        )

    records = (
        load_insufficient_evidence_records(
            path
        )
    )

    assert len(
        records
    ) == 1


def test_answerable_accept_is_correct() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(),
            make_result(
                accepted=True
            ),
        )
    )

    assert (
        metric.correct_decision
        is True
    )

    assert (
        metric.false_withhold
        is False
    )


def test_answerable_withhold_is_false_withhold() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .MISSING_CITATIONS
                ),
            ),
        )
    )

    assert (
        metric.correct_decision
        is False
    )

    assert (
        metric.false_withhold
        is True
    )


def test_partial_accept_is_structurally_correct() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="partial",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                case_type="partial_evidence",
            ),
            make_result(
                accepted=True
            ),
        )
    )

    assert (
        metric.correct_decision
        is True
    )

    assert (
        metric.false_accept
        is False
    )

    assert (
        metric.false_withhold
        is False
    )


def test_partial_withhold_is_false_withhold() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="partial",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                case_type="partial_evidence",
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .MISSING_CITATIONS
                ),
            ),
        )
    )

    assert (
        metric.correct_decision
        is False
    )

    assert (
        metric.false_accept
        is False
    )

    assert (
        metric.false_withhold
        is True
    )


def test_mixed_accept_is_structurally_correct() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="mixed",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                case_type=(
                    "mixed_supported_unsupported"
                ),
            ),
            make_result(
                accepted=True
            ),
        )
    )

    assert (
        metric.correct_decision
        is True
    )


def test_unanswerable_withhold_is_correct() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(
                expected_behavior=(
                    EXPECTED_BEHAVIOR_WITHHOLD
                ),
                case_type=(
                    "out_of_corpus_document"
                ),
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .NO_SELECTED_EVIDENCE
                ),
            ),
        )
    )

    assert (
        metric.correct_decision
        is True
    )

    assert (
        metric.false_accept
        is False
    )


def test_unanswerable_accept_is_false_accept() -> None:
    metric = (
        evaluate_insufficient_evidence_result(
            make_record(
                expected_behavior=(
                    EXPECTED_BEHAVIOR_WITHHOLD
                ),
                case_type=(
                    "out_of_corpus_document"
                ),
            ),
            make_result(
                accepted=True
            ),
        )
    )

    assert (
        metric.correct_decision
        is False
    )

    assert (
        metric.false_accept
        is True
    )


def test_aggregate_metrics_without_partial_cases() -> None:
    metrics = [
        evaluate_insufficient_evidence_result(
            make_record(
                question_id=(
                    "answer_1"
                )
            ),
            make_result(
                accepted=True
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id=(
                    "answer_2"
                )
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .MISSING_CITATIONS
                ),
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id=(
                    "withhold_1"
                ),
                expected_behavior=(
                    EXPECTED_BEHAVIOR_WITHHOLD
                ),
                case_type=(
                    "out_of_corpus_document"
                ),
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .NO_SELECTED_EVIDENCE
                ),
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id=(
                    "withhold_2"
                ),
                expected_behavior=(
                    EXPECTED_BEHAVIOR_WITHHOLD
                ),
                case_type=(
                    "out_of_corpus_period"
                ),
            ),
            make_result(
                accepted=True
            ),
        ),
    ]

    summary = (
        aggregate_insufficient_evidence_metrics(
            metrics
        )
    )

    assert (
        summary[
            "decision_accuracy"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "answer_acceptance_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "expected_partial_count"
        ]
        == 0
    )

    assert (
        summary[
            "partial_case_acceptance_rate"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        summary[
            "withholding_success_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "guard_false_accept_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "guard_false_withhold_rate"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_aggregate_metrics_with_partial_cases() -> None:
    metrics = [
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="answer",
            ),
            make_result(
                accepted=True
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="partial_ok",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                case_type="partial_evidence",
            ),
            make_result(
                accepted=True
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="partial_withheld",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_PARTIAL
                ),
                case_type=(
                    "mixed_supported_unsupported"
                ),
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .MISSING_CITATIONS
                ),
            ),
        ),
        evaluate_insufficient_evidence_result(
            make_record(
                question_id="withhold",
                expected_behavior=(
                    EXPECTED_BEHAVIOR_WITHHOLD
                ),
                case_type=(
                    "out_of_corpus_document"
                ),
            ),
            make_result(
                accepted=False,
                reason=(
                    EvidenceGuardReason
                    .NO_SELECTED_EVIDENCE
                ),
            ),
        ),
    ]

    summary = (
        aggregate_insufficient_evidence_metrics(
            metrics
        )
    )

    assert (
        summary[
            "question_count"
        ]
        == 4
    )

    assert (
        summary[
            "expected_answer_count"
        ]
        == 1
    )

    assert (
        summary[
            "expected_partial_count"
        ]
        == 2
    )

    assert (
        summary[
            "expected_withhold_count"
        ]
        == 1
    )

    assert (
        summary[
            "decision_accuracy"
        ]
        == pytest.approx(
            0.75
        )
    )

    assert (
        summary[
            "answer_acceptance_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        summary[
            "partial_case_acceptance_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary[
            "withholding_success_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        summary[
            "guard_false_accept_rate"
        ]
        == pytest.approx(
            0.0
        )
    )

    # There are three cases that structurally should be accepted:
    # one full-answer case and two partial cases. One partial case was
    # incorrectly withheld.
    assert (
        summary[
            "guard_false_withhold_rate"
        ]
        == pytest.approx(
            1 / 3
        )
    )


def test_aggregate_requires_metrics() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        aggregate_insufficient_evidence_metrics(
            []
        )