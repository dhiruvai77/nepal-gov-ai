"""Tests for the resumable insufficient-evidence production runner."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.evaluation.insufficient_evidence_evaluator import (
    EXPECTED_BEHAVIOR_ANSWER,
    EXPECTED_BEHAVIOR_PARTIAL,
    EXPECTED_BEHAVIOR_WITHHOLD,
    InsufficientEvidenceRecord,
    evaluate_insufficient_evidence_result,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.evaluation.run_insufficient_evidence_evaluation import (
    RUN_CONFIG_ID,
    RESULT_SCHEMA_VERSION,
    append_output_record,
    build_output_record,
    load_existing_output,
    metric_from_output_record,
    run_evaluation,
    validate_answerable_controls,
    validate_limit,
    validate_run_config_id,
)
from src.generation.evidence_guard import (
    EvidenceGuardReason,
)
from src.rag.pipeline import (
    RAGResult,
)


def make_record(
    question_id: str = "ie_en_en_001",
    *,
    expected_behavior: str = (
        EXPECTED_BEHAVIOR_ANSWER
    ),
    case_type: str = (
        "answerable_control"
    ),
    reference_question_id: str | None = (
        "en_en_007"
    ),
) -> InsufficientEvidenceRecord:
    """Build one benchmark record."""

    return (
        InsufficientEvidenceRecord(
            question_id=question_id,
            query=(
                "What does the Constitution "
                "of Nepal guarantee regarding "
                "the right to health?"
            ),
            query_language="en",
            target_language="en",
            category="constitution_law",
            expected_behavior=(
                expected_behavior
            ),
            case_type=(
                case_type
            ),
            expected_document_ids=(
                (
                    "constitution_nepal_current_en",
                )
                if expected_behavior
                in {
                    EXPECTED_BEHAVIOR_ANSWER,
                    EXPECTED_BEHAVIOR_PARTIAL,
                }
                else ()
            ),
            reference_question_id=(
                reference_question_id
                if expected_behavior
                == EXPECTED_BEHAVIOR_ANSWER
                else None
            ),
            notes="Test benchmark case.",
        )
    )


def make_reference_record(
) -> EvaluationRecord:
    """Build the verified source benchmark record for one control."""

    return (
        EvaluationRecord(
            question_id="en_en_007",
            query=(
                "What does the Constitution "
                "of Nepal guarantee regarding "
                "the right to health?"
            ),
            query_language="en",
            target_language="en",
            category="constitution_law",
            expected_document_ids=(
                "constitution_nepal_current_en",
            ),
            primary_relevant_chunk_ids=(
                "point-1",
            ),
            relevant_chunk_ids=(
                "point-1",
            ),
            notes="Verified reference.",
        )
    )


def make_result(
    *,
    accepted: bool = True,
    reason: (
        EvidenceGuardReason
        | None
    ) = None,
) -> RAGResult:
    """Build one minimal pipeline result."""

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
                "fake"
                if accepted
                else None
            ),
            model=(
                "fake-model"
                if accepted
                else None
            ),
        )
    )


def record_to_json(
    record: InsufficientEvidenceRecord,
) -> dict:
    """Convert one benchmark record to the JSONL schema."""

    return {
        "question_id": (
            record.question_id
        ),
        "query": (
            record.query
        ),
        "query_language": (
            record.query_language
        ),
        "target_language": (
            record.target_language
        ),
        "category": (
            record.category
        ),
        "expected_behavior": (
            record.expected_behavior
        ),
        "case_type": (
            record.case_type
        ),
        "expected_document_ids": list(
            record.expected_document_ids
        ),
        "reference_question_id": (
            record.reference_question_id
        ),
        "notes": (
            record.notes
        ),
    }


def write_dataset(
    path,
    records,
) -> None:
    """Write benchmark records to JSONL."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record_to_json(
                        record
                    )
                )
            )

            handle.write(
                "\n"
            )


def write_reference_dataset(
    path,
) -> None:
    """Write one compatible verified retrieval reference row."""

    row = {
        "question_id": "en_en_007",
        "query": (
            "What does the Constitution "
            "of Nepal guarantee regarding "
            "the right to health?"
        ),
        "query_language": "en",
        "target_language": "en",
        "category": "constitution_law",
        "expected_document_ids": [
            "constitution_nepal_current_en",
        ],
        "primary_relevant_chunk_ids": [
            "point-1",
        ],
        "relevant_chunk_ids": [
            "point-1",
        ],
        "notes": "Verified reference.",
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                row
            )
        )

        handle.write(
            "\n"
        )


class FakePipeline:
    """Deterministic fake RAG pipeline."""

    def __init__(
        self,
        *,
        results: list[
            RAGResult
        ]
        | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.results = (
            list(
                results
            )
            if results is not None
            else []
        )

        self.fail_on_call = (
            fail_on_call
        )

        self.calls: list[
            tuple[
                str,
                str,
                dict[
                    str,
                    str,
                ]
                | None,
            ]
        ] = []

        self.closed = False

    def answer(
        self,
        query: str,
        *,
        answer_language: str,
        filters=None,
    ) -> RAGResult:
        """Return one deterministic result."""

        self.calls.append(
            (
                query,
                answer_language,
                filters,
            )
        )

        if (
            self.fail_on_call
            is not None
            and len(
                self.calls
            )
            == self.fail_on_call
        ):
            raise RuntimeError(
                "simulated hosted failure"
            )

        index = (
            len(
                self.calls
            )
            - 1
        )

        if (
            index
            < len(
                self.results
            )
        ):
            return (
                self.results[
                    index
                ]
            )

        return (
            make_result(
                accepted=True
            )
        )

    def close(
        self,
    ) -> None:
        self.closed = True


def make_output_row(
    record: InsufficientEvidenceRecord,
    *,
    accepted: bool = True,
    run_config_id: str = RUN_CONFIG_ID,
) -> dict:
    """Build one valid persisted output row."""

    result = (
        make_result(
            accepted=accepted,
            reason=(
                None
                if accepted
                else (
                    EvidenceGuardReason
                    .MISSING_CITATIONS
                )
            ),
        )
    )

    metrics = (
        evaluate_insufficient_evidence_result(
            record,
            result,
        )
    )

    return (
        build_output_record(
            record,
            result,
            metrics,
            run_config_id=(
                run_config_id
            ),
        )
    )


def test_validate_answerable_controls_accepts_matching_reference() -> None:
    validate_answerable_controls(
        [
            make_record(),
        ],
        [
            make_reference_record(),
        ],
    )


def test_validate_answerable_controls_rejects_unknown_reference() -> None:
    record = (
        make_record(
            reference_question_id=(
                "missing"
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="unknown verified question",
    ):
        validate_answerable_controls(
            [
                record,
            ],
            [
                make_reference_record(),
            ],
        )


def test_validate_answerable_controls_rejects_drifted_query() -> None:
    record = (
        make_record()
    )

    reference = (
        make_reference_record()
    )

    different = (
        EvaluationRecord(
            question_id=(
                reference.question_id
            ),
            query="Different question.",
            query_language=(
                reference.query_language
            ),
            target_language=(
                reference.target_language
            ),
            category=(
                reference.category
            ),
            expected_document_ids=(
                reference.expected_document_ids
            ),
            primary_relevant_chunk_ids=(
                reference.primary_relevant_chunk_ids
            ),
            relevant_chunk_ids=(
                reference.relevant_chunk_ids
            ),
            notes=(
                reference.notes
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="'query'",
    ):
        validate_answerable_controls(
            [
                record,
            ],
            [
                different,
            ],
        )


def test_validate_answerable_controls_ignores_partial_cases() -> None:
    record = (
        make_record(
            question_id="partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            case_type="partial_evidence",
        )
    )

    validate_answerable_controls(
        [
            record,
        ],
        [],
    )


def test_build_output_record_preserves_decision_fields() -> None:
    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    assert (
        row[
            "schema_version"
        ]
        == RESULT_SCHEMA_VERSION
    )

    assert (
        row[
            "run_config_id"
        ]
        == RUN_CONFIG_ID
    )

    assert (
        row[
            "expected_behavior"
        ]
        == "answer"
    )

    assert (
        row[
            "accepted"
        ]
        is True
    )


def test_build_output_record_supports_custom_run_config() -> None:
    record = (
        make_record()
    )

    result = (
        make_result(
            accepted=True
        )
    )

    metrics = (
        evaluate_insufficient_evidence_result(
            record,
            result,
        )
    )

    row = (
        build_output_record(
            record,
            result,
            metrics,
            run_config_id=(
                "partial-mixed-test"
            ),
        )
    )

    assert (
        row[
            "run_config_id"
        ]
        == "partial-mixed-test"
    )


def test_metric_from_output_record_round_trips() -> None:
    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    metric = (
        metric_from_output_record(
            row
        )
    )

    assert (
        metric.question_id
        == record.question_id
    )

    assert (
        metric.correct_decision
        is True
    )


def test_load_existing_output_returns_empty_when_missing(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "missing.jsonl"
    )

    loaded = (
        load_existing_output(
            path,
            [
                make_record(),
            ],
        )
    )

    assert (
        loaded
        == {}
    )


def test_load_existing_output_reuses_valid_row(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    path = (
        tmp_path
        / "results.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            record
        ),
    )

    loaded = (
        load_existing_output(
            path,
            [
                record,
            ],
        )
    )

    assert list(
        loaded
    ) == [
        "ie_en_en_001",
    ]


def test_load_existing_output_supports_custom_run_config(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    path = (
        tmp_path
        / "results.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            record,
            run_config_id=(
                "partial-mixed-test"
            ),
        ),
    )

    loaded = (
        load_existing_output(
            path,
            [
                record,
            ],
            run_config_id=(
                "partial-mixed-test"
            ),
        )
    )

    assert list(
        loaded
    ) == [
        record.question_id,
    ]


def test_load_existing_output_rejects_wrong_run_config(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    path = (
        tmp_path
        / "results.jsonl"
    )

    append_output_record(
        path,
        make_output_row(
            record,
            run_config_id=(
                "partial-mixed-test"
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="different run_config_id",
    ):
        load_existing_output(
            path,
            [
                record,
            ],
            run_config_id=(
                "different-config"
            ),
        )


def test_load_existing_output_rejects_changed_behavior(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    row[
        "expected_behavior"
    ] = "withhold"

    path = (
        tmp_path
        / "results.jsonl"
    )

    append_output_record(
        path,
        row,
    )

    with pytest.raises(
        ValueError,
        match="expected_behavior",
    ):
        load_existing_output(
            path,
            [
                record,
            ],
        )


def test_run_evaluation_uses_query_language_and_target_filter(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            record,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    pipeline = (
        FakePipeline()
    )

    run_evaluation(
        dataset_path,
        reference_dataset_path=(
            reference_path
        ),
        output_path=(
            output_path
        ),
        pipeline_factory=(
            lambda: pipeline
        ),
    )

    assert (
        pipeline.calls
        == [
            (
                record.query,
                "en",
                {
                    "language": "en",
                },
            ),
        ]
    )

    assert (
        pipeline.closed
        is True
    )


def test_run_evaluation_persists_custom_run_config(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            record,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    pipeline = (
        FakePipeline()
    )

    run_evaluation(
        dataset_path,
        reference_dataset_path=(
            reference_path
        ),
        output_path=(
            output_path
        ),
        run_config_id=(
            "partial-mixed-test"
        ),
        pipeline_factory=(
            lambda: pipeline
        ),
    )

    row = (
        json.loads(
            output_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )
    )

    assert (
        row[
            "run_config_id"
        ]
        == "partial-mixed-test"
    )


def test_run_evaluation_supports_partial_case_without_reference_match(
    tmp_path,
) -> None:
    record = (
        make_record(
            question_id="partial",
            expected_behavior=(
                EXPECTED_BEHAVIOR_PARTIAL
            ),
            case_type="partial_evidence",
        )
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            record,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    pipeline = (
        FakePipeline()
    )

    metrics, overall = (
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            run_config_id=(
                "partial-mixed-test"
            ),
            pipeline_factory=(
                lambda: pipeline
            ),
        )
    )

    assert len(
        metrics
    ) == 1

    assert (
        metrics[
            0
        ].correct_decision
        is True
    )

    assert (
        overall[
            "expected_partial_count"
        ]
        == 1
    )

    assert (
        overall[
            "partial_case_acceptance_rate"
        ]
        == pytest.approx(
            1.0
        )
    )


def test_run_evaluation_resumes_without_repeating_completed(
    tmp_path,
) -> None:
    first = (
        make_record(
            "ie_en_en_001"
        )
    )

    second = (
        make_record(
            "ie_en_en_002"
        )
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            first,
            second,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    append_output_record(
        output_path,
        make_output_row(
            first
        ),
    )

    pipeline = (
        FakePipeline()
    )

    metrics, _ = (
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            pipeline_factory=(
                lambda: pipeline
            ),
        )
    )

    assert len(
        pipeline.calls
    ) == 1

    assert len(
        metrics
    ) == 2


def test_run_evaluation_resumes_custom_run_without_repeating_completed(
    tmp_path,
) -> None:
    first = (
        make_record(
            "ie_en_en_001"
        )
    )

    second = (
        make_record(
            "ie_en_en_002"
        )
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            first,
            second,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    append_output_record(
        output_path,
        make_output_row(
            first,
            run_config_id=(
                "partial-mixed-test"
            ),
        ),
    )

    pipeline = (
        FakePipeline()
    )

    metrics, _ = (
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            run_config_id=(
                "partial-mixed-test"
            ),
            pipeline_factory=(
                lambda: pipeline
            ),
        )
    )

    assert len(
        pipeline.calls
    ) == 1

    assert len(
        metrics
    ) == 2


def test_run_evaluation_limit_applies_only_to_new_questions(
    tmp_path,
) -> None:
    records = [
        make_record(
            "ie_en_en_001"
        ),
        make_record(
            "ie_en_en_002"
        ),
        make_record(
            "ie_en_en_003"
        ),
    ]

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        records,
    )

    write_reference_dataset(
        reference_path
    )

    pipeline = (
        FakePipeline()
    )

    metrics, _ = (
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            limit=2,
            pipeline_factory=(
                lambda: pipeline
            ),
        )
    )

    assert len(
        pipeline.calls
    ) == 2

    assert len(
        metrics
    ) == 2


def test_completed_row_survives_later_pipeline_failure(
    tmp_path,
) -> None:
    records = [
        make_record(
            "ie_en_en_001"
        ),
        make_record(
            "ie_en_en_002"
        ),
    ]

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        records,
    )

    write_reference_dataset(
        reference_path
    )

    pipeline = (
        FakePipeline(
            fail_on_call=2
        )
    )

    with pytest.raises(
        RuntimeError,
        match="simulated hosted failure",
    ):
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            pipeline_factory=(
                lambda: pipeline
            ),
        )

    loaded = (
        load_existing_output(
            output_path,
            records,
        )
    )

    assert list(
        loaded
    ) == [
        "ie_en_en_001",
    ]

    assert (
        pipeline.closed
        is True
    )


def test_reset_discards_existing_result(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            record,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    append_output_record(
        output_path,
        make_output_row(
            record
        ),
    )

    pipeline = (
        FakePipeline()
    )

    run_evaluation(
        dataset_path,
        reference_dataset_path=(
            reference_path
        ),
        output_path=(
            output_path
        ),
        reset=True,
        pipeline_factory=(
            lambda: pipeline
        ),
    )

    assert len(
        pipeline.calls
    ) == 1


def test_completed_benchmark_avoids_pipeline_construction(
    tmp_path,
) -> None:
    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    reference_path = (
        tmp_path
        / "reference.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        [
            record,
        ],
    )

    write_reference_dataset(
        reference_path
    )

    append_output_record(
        output_path,
        make_output_row(
            record
        ),
    )

    factory = (
        Mock()
    )

    metrics, _ = (
        run_evaluation(
            dataset_path,
            reference_dataset_path=(
                reference_path
            ),
            output_path=(
                output_path
            ),
            pipeline_factory=factory,
        )
    )

    factory.assert_not_called()

    assert len(
        metrics
    ) == 1


def test_validate_limit_rejects_non_positive_values() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_limit(
            0
        )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_limit(
            -1
        )


def test_validate_limit_accepts_positive_and_none() -> None:
    assert (
        validate_limit(
            None
        )
        is None
    )

    assert (
        validate_limit(
            3
        )
        == 3
    )


def test_validate_run_config_id_accepts_and_strips_text() -> None:
    assert (
        validate_run_config_id(
            " partial-mixed-test "
        )
        == "partial-mixed-test"
    )


def test_validate_run_config_id_rejects_empty_text() -> None:
    with pytest.raises(
        ValueError,
        match="run_config_id",
    ):
        validate_run_config_id(
            "   "
        )