"""Tests for the resumable production RAG evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import (
    Mock,
)

import pytest

from src.citations.evidence import (
    process_answer_citations,
    render_cited_sources,
)
from src.evaluation.rag_evaluator import (
    evaluate_rag_result,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.evaluation.run_rag_evaluation import (
    RESULT_SCHEMA_VERSION,
    RUN_CONFIG_ID,
    append_output_record,
    build_output_record,
    load_existing_output,
    metric_from_output_record,
    run_evaluation,
    validate_limit,
)
from src.rag.pipeline import (
    RAGResult,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_record(
    question_id: str = "q1",
    *,
    query_language: str = "en",
    target_language: str = "en",
) -> EvaluationRecord:
    """Create one deterministic evaluation record."""

    return EvaluationRecord(
        question_id=question_id,
        query=(
            f"Question for {question_id}"
        ),
        query_language=(
            query_language
        ),
        target_language=(
            target_language
        ),
        category="test",
        expected_document_ids=(
            "document-1",
        ),
        primary_relevant_chunk_ids=(
            "point-1",
        ),
        relevant_chunk_ids=(
            "point-1",
        ),
        notes="Fixture.",
    )


def make_evidence() -> RerankedResult:
    """Create one gold selected-evidence fixture."""

    result = RetrievalResult(
        point_id="point-1",
        score=0.1,
        chunk_id="chunk-1",
        document_id="document-1",
        title="Government Document 1",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url=(
            "https://example.gov.np/"
            "document-1.pdf"
        ),
        chunk_text="Evidence passage.",
        chunk_index=0,
        token_count=100,
    )

    return RerankedResult(
        result=result,
        rerank_score=0.95,
        original_rank=1,
    )


def make_result(
    answer_text: str = (
        "Supported answer [E1]."
    ),
) -> RAGResult:
    """Create one accepted RAG result with real citation processing."""

    evidence = (
        make_evidence()
    )

    context = (
        evidence,
    )

    citation_result = (
        process_answer_citations(
            answer_text,
            context,
        )
    )

    sources = (
        render_cited_sources(
            citation_result
        )
    )

    return RAGResult(
        answer_text=answer_text,
        accepted=True,
        reason=None,
        sources=sources,
        selected_context=context,
        citation_result=(
            citation_result
        ),
        provider="fake",
        model="fake-model",
    )


def write_dataset(
    path: Path,
    records: list[
        EvaluationRecord
    ],
) -> None:
    """Write evaluation records using the production JSONL schema."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
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
                        "expected_document_ids": list(
                            record.expected_document_ids
                        ),
                        "primary_relevant_chunk_ids": list(
                            record.primary_relevant_chunk_ids
                        ),
                        "relevant_chunk_ids": list(
                            record.relevant_chunk_ids
                        ),
                        "notes": (
                            record.notes
                        ),
                    },
                    ensure_ascii=False,
                )
            )

            handle.write(
                "\n"
            )


def make_output_row(
    record: EvaluationRecord,
) -> dict:
    """Build one valid persisted evaluation row."""

    result = (
        make_result()
    )

    metrics = (
        evaluate_rag_result(
            record,
            result,
        )
    )

    return build_output_record(
        record,
        result,
        metrics,
    )


class FakePipeline:
    """Deterministic pipeline used to test hosted-call orchestration."""

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.calls: list[
            tuple[
                str,
                str,
                dict[str, str] | None,
            ]
        ] = []

        self.closed = False
        self.fail_on_call = (
            fail_on_call
        )

    def answer(
        self,
        query: str,
        *,
        answer_language: str,
        filters: dict[
            str,
            str,
        ]
        | None = None,
    ) -> RAGResult:
        """Record the call and optionally simulate a hosted failure."""

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

        return make_result()

    def close(
        self,
    ) -> None:
        """Record cleanup."""

        self.closed = True


def test_build_output_record_preserves_generation_and_evidence() -> None:
    """Persisted output should retain answer and evidence diagnostics."""

    record = (
        make_record()
    )

    result = (
        make_result()
    )

    metrics = (
        evaluate_rag_result(
            record,
            result,
        )
    )

    row = (
        build_output_record(
            record,
            result,
            metrics,
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
            "answer_text"
        ]
        == "Supported answer [E1]."
    )

    assert (
        row[
            "generated_answer_text"
        ]
        == "Supported answer [E1]."
    )

    assert (
        row[
            "selected_evidence"
        ][0][
            "point_id"
        ]
        == "point-1"
    )

    assert (
        row[
            "cited_evidence"
        ][0][
            "evidence_id"
        ]
        == "E1"
    )


def test_metric_from_output_record_reconstructs_metrics() -> None:
    """Persisted deterministic metrics should remain reusable."""

    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    metrics = (
        metric_from_output_record(
            row
        )
    )

    assert (
        metrics.question_id
        == "q1"
    )

    assert (
        metrics.cited_primary_hit
        == 1.0
    )


def test_load_existing_output_returns_empty_for_missing_file(
    tmp_path,
) -> None:
    """A first benchmark run should begin with no completed rows."""

    path = (
        tmp_path
        / "missing.jsonl"
    )

    assert (
        load_existing_output(
            path,
            [
                make_record(),
            ],
        )
        == {}
    )


def test_load_existing_output_reuses_valid_row(
    tmp_path,
) -> None:
    """Compatible completed rows should be reusable."""

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
        "q1",
    ]


def test_load_existing_output_rejects_different_run_config(
    tmp_path,
) -> None:
    """Stale production configurations must not be silently mixed."""

    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    row[
        "run_config_id"
    ] = "old-config"

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
        match="different run_config_id",
    ):
        load_existing_output(
            path,
            [
                record,
            ],
        )


def test_load_existing_output_rejects_changed_question(
    tmp_path,
) -> None:
    """Changed benchmark questions require an intentional fresh run."""

    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    row[
        "query"
    ] = "Old question text"

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
        match="does not match current 'query'",
    ):
        load_existing_output(
            path,
            [
                record,
            ],
        )


def test_load_existing_output_rejects_duplicate_question(
    tmp_path,
) -> None:
    """A question must never be counted twice in one benchmark run."""

    record = (
        make_record()
    )

    row = (
        make_output_row(
            record
        )
    )

    path = (
        tmp_path
        / "results.jsonl"
    )

    append_output_record(
        path,
        row,
    )

    append_output_record(
        path,
        row,
    )

    with pytest.raises(
        ValueError,
        match="duplicate question_id",
    ):
        load_existing_output(
            path,
            [
                record,
            ],
        )


def test_run_evaluation_uses_query_language_for_answer_and_target_filter(
    tmp_path,
) -> None:
    """Cross-lingual evaluation should separate answer and evidence languages."""

    record = (
        make_record(
            query_language="en",
            target_language="ne",
        )
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
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

    pipeline = (
        FakePipeline()
    )

    run_evaluation(
        dataset_path,
        output_path=output_path,
        pipeline_factory=(
            lambda: pipeline
        ),
    )

    assert pipeline.calls == [
        (
            record.query,
            "en",
            {
                "language": "ne",
            },
        ),
    ]

    assert pipeline.closed is True


def test_run_evaluation_resumes_without_repeating_completed_question(
    tmp_path,
) -> None:
    """Resume should execute only questions missing from persisted output."""

    first = (
        make_record(
            "q1"
        )
    )

    second = (
        make_record(
            "q2"
        )
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
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
            output_path=output_path,
            pipeline_factory=(
                lambda: pipeline
            ),
        )
    )

    assert len(
        pipeline.calls
    ) == 1

    assert (
        pipeline.calls[
            0
        ][0]
        == second.query
    )

    assert len(
        metrics
    ) == 2


def test_run_evaluation_limit_applies_only_to_new_questions(
    tmp_path,
) -> None:
    """A controlled live run should stop after the requested new-call count."""

    records = [
        make_record(
            "q1"
        ),
        make_record(
            "q2"
        ),
        make_record(
            "q3"
        ),
    ]

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        records,
    )

    pipeline = (
        FakePipeline()
    )

    metrics, _ = (
        run_evaluation(
            dataset_path,
            output_path=output_path,
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


def test_completed_rows_survive_later_pipeline_failure(
    tmp_path,
) -> None:
    """Every successful question should be persisted before the next call."""

    records = [
        make_record(
            "q1"
        ),
        make_record(
            "q2"
        ),
    ]

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
    )

    output_path = (
        tmp_path
        / "results.jsonl"
    )

    write_dataset(
        dataset_path,
        records,
    )

    pipeline = (
        FakePipeline(
            fail_on_call=2
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "simulated hosted failure"
        ),
    ):
        run_evaluation(
            dataset_path,
            output_path=output_path,
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
        "q1",
    ]

    assert pipeline.closed is True


def test_reset_discards_existing_rows_and_reruns(
    tmp_path,
) -> None:
    """Reset should intentionally replace reusable benchmark state."""

    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
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
        output_path=output_path,
        reset=True,
        pipeline_factory=(
            lambda: pipeline
        ),
    )

    assert len(
        pipeline.calls
    ) == 1


def test_all_completed_questions_avoid_pipeline_construction(
    tmp_path,
) -> None:
    """A completed benchmark should require zero additional hosted resources."""

    record = (
        make_record()
    )

    dataset_path = (
        tmp_path
        / "dataset.jsonl"
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
            output_path=output_path,
            pipeline_factory=factory,
        )
    )

    factory.assert_not_called()

    assert len(
        metrics
    ) == 1


def test_validate_limit_rejects_non_positive_values() -> None:
    """The controlled-run limit must always represent actual work."""

    with pytest.raises(
        ValueError,
        match=(
            "limit must be greater than zero"
        ),
    ):
        validate_limit(
            0
        )

    with pytest.raises(
        ValueError,
        match=(
            "limit must be greater than zero"
        ),
    ):
        validate_limit(
            -1
        )


def test_validate_limit_accepts_none_and_positive_values() -> None:
    """No limit or a positive limit should be accepted."""

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