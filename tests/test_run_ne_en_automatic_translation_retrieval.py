"""Tests for automatic NE->EN translated-query retrieval evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation.run_ne_en_automatic_translation_retrieval import (
    BENCHMARK_CONFIG_ID,
    BENCHMARK_SCHEMA_VERSION,
    aggregate_point_id_metrics,
    build_benchmark_row,
    build_summary,
    evaluate_point_ids,
    primary_rank_from_ids,
    validate_baseline_artifact,
    validate_persisted_benchmark_row,
)
from src.evaluation.retrieval_evaluator import (
    EvaluationRecord,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_record(
    *,
    question_id: str = "ne_en_001",
) -> EvaluationRecord:
    """Build one deterministic benchmark record."""

    return EvaluationRecord(
        question_id=question_id,
        query="नेपाली प्रश्न",
        query_language="ne",
        target_language="en",
        category="test",
        expected_document_ids=(
            "doc",
        ),
        primary_relevant_chunk_ids=(
            "primary",
        ),
        relevant_chunk_ids=(
            "primary",
            "supporting",
        ),
        notes="Test.",
    )


def make_result(
    point_id: str,
) -> RetrievalResult:
    """Build one deterministic retrieval result."""

    return RetrievalResult(
        point_id=point_id,
        score=1.0,
        chunk_id=(
            f"chunk-{point_id}"
        ),
        document_id="doc",
        title="Document",
        organization="Government",
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
        category="test",
        document_type="report",
        publication_date=None,
        section=None,
        subsection=None,
        article_number=None,
        article_title=None,
        extraction_method="native",
    )


def make_translation(
    *,
    question_id: str = "ne_en_001",
) -> dict:
    """Build one persisted automatic translation fixture."""

    return {
        "schema_version": 1,
        "translation_config_id": (
            "ne-en-query-translation-v1"
        ),
        "question_id": question_id,
        "query_language": "ne",
        "target_language": "en",
        "original_query": "नेपाली प्रश्न",
        "translated_query": (
            "What does the document say?"
        ),
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "prompt_sha256": (
            "a" * 64
        ),
    }


def make_reranked(
    results: list[
        RetrievalResult
    ],
) -> list[
    RerankedResult
]:
    """Build deterministic reranker results preserving candidate identity."""

    return [
        RerankedResult(
            result=result,
            rerank_score=(
                1.0
                / index
            ),
            original_rank=index,
        )
        for (
            index,
            result,
        ) in enumerate(
            results,
            start=1,
        )
    ]


def make_baseline(
    records: list[
        EvaluationRecord
    ],
) -> dict:
    """Build a minimal valid prior manual-control benchmark artifact."""

    ranks = {
        record.question_id: {
            "production_first_stage": 1,
            "production_bge": 1,
            "translated_first_stage": 1,
            "translated_bge": 1,
        }
        for record in records
    }

    metric_block = {
        "production_first_stage": {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "recall": 0.5,
        },
        "production_bge": {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "recall": 0.5,
        },
        "translated_first_stage": {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "recall": 0.5,
        },
        "translated_bge": {
            "hit_rate": 1.0,
            "mrr": 1.0,
            "recall": 0.5,
        },
    }

    return {
        "run_config_id": (
            "ne-en-targeted-retrieval-v1"
        ),
        "production_changed": False,
        "evaluation_slice": {
            "question_count": len(
                records
            ),
            "question_ids": [
                record.question_id
                for record in records
            ],
        },
        "candidate_generation": {
            "translated_hybrid": {
                "dense_component_depth": 100,
                "bm25_component_depth": 100,
                "final_candidate_depth": 20,
            },
        },
        "reranked_primary_ranks": (
            ranks
        ),
        "first_stage_metrics": {
            "5": metric_block,
            "10": metric_block,
            "20": metric_block,
        },
        "reranked_metrics": {
            "5": metric_block,
            "10": metric_block,
            "20": metric_block,
        },
    }


def test_primary_rank_from_ids() -> None:
    """Primary rank should use manually verified primary IDs."""

    record = (
        make_record()
    )

    assert (
        primary_rank_from_ids(
            record,
            [
                "other",
                "primary",
                "supporting",
            ],
        )
        == 2
    )

    assert (
        primary_rank_from_ids(
            record,
            [
                "other",
                "supporting",
            ],
        )
        is None
    )


def test_evaluate_point_ids_matches_retrieval_metric_contract() -> None:
    """Persisted point-ID metrics should match Hit/MRR/Recall semantics."""

    metrics = (
        evaluate_point_ids(
            make_record(),
            [
                "other",
                "primary",
                "supporting",
            ],
            cutoff=3,
        )
    )

    assert (
        metrics[
            "hit_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "mrr"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "recall"
        ]
        == 1.0
    )


def test_evaluate_point_ids_rejects_invalid_cutoff() -> None:
    """Metric cutoff must remain positive."""

    with pytest.raises(
        ValueError,
        match="cutoff",
    ):
        evaluate_point_ids(
            make_record(),
            [],
            cutoff=0,
        )


def test_validate_baseline_artifact_accepts_matching_configuration() -> None:
    """The previous manual-control experiment should be reusable."""

    records = [
        make_record(
            question_id=(
                f"ne_en_00{index}"
            )
        )
        for index in range(
            1,
            7,
        )
    ]

    validate_baseline_artifact(
        make_baseline(
            records
        ),
        records,
    )


def test_validate_baseline_artifact_rejects_wrong_depth() -> None:
    """Comparison must not silently use a different manual baseline."""

    records = [
        make_record()
    ]

    baseline = (
        make_baseline(
            records
        )
    )

    baseline[
        "candidate_generation"
    ][
        "translated_hybrid"
    ][
        "dense_component_depth"
    ] = 50

    with pytest.raises(
        ValueError,
        match=(
            "dense component depth"
        ),
    ):
        validate_baseline_artifact(
            baseline,
            records,
        )


def test_build_benchmark_row_records_rankings() -> None:
    """One automatic case should retain dense, hybrid, and BGE rankings."""

    record = (
        make_record()
    )

    dense = [
        make_result(
            "dense-other"
        ),
        make_result(
            "primary"
        ),
    ]

    hybrid = [
        make_result(
            f"candidate-{index}"
        )
        for index in range(
            1,
            20,
        )
    ]

    hybrid.insert(
        4,
        make_result(
            "primary"
        ),
    )

    reranked_results = list(
        hybrid
    )

    primary_result = (
        reranked_results.pop(
            4
        )
    )

    reranked_results.insert(
        0,
        primary_result,
    )

    reranked = (
        make_reranked(
            reranked_results
        )
    )

    row = (
        build_benchmark_row(
            record,
            make_translation(),
            dense_results=dense,
            hybrid_results=hybrid,
            reranked_results=(
                reranked
            ),
        )
    )

    assert (
        row[
            "schema_version"
        ]
        == BENCHMARK_SCHEMA_VERSION
    )

    assert (
        row[
            "run_config_id"
        ]
        == BENCHMARK_CONFIG_ID
    )

    assert (
        row[
            "automatic_dense_primary_rank"
        ]
        == 2
    )

    assert (
        row[
            "automatic_hybrid_primary_rank"
        ]
        == 5
    )

    assert (
        row[
            "automatic_bge_primary_rank"
        ]
        == 1
    )


def test_validate_persisted_benchmark_row_accepts_valid_row() -> None:
    """Current automatic benchmark rows should validate."""

    record = (
        make_record()
    )

    hybrid = [
        make_result(
            "primary"
        )
    ] + [
        make_result(
            f"candidate-{index}"
        )
        for index in range(
            2,
            21,
        )
    ]

    row = (
        build_benchmark_row(
            record,
            make_translation(),
            dense_results=(
                hybrid
            ),
            hybrid_results=(
                hybrid
            ),
            reranked_results=(
                make_reranked(
                    hybrid
                )
            ),
        )
    )

    validate_persisted_benchmark_row(
        row,
        record,
        make_translation(),
    )


def test_validate_persisted_benchmark_row_rejects_translation_change() -> None:
    """Changing translation text must invalidate stale retrieval results."""

    record = (
        make_record()
    )

    hybrid = [
        make_result(
            "primary"
        )
    ] + [
        make_result(
            f"candidate-{index}"
        )
        for index in range(
            2,
            21,
        )
    ]

    row = (
        build_benchmark_row(
            record,
            make_translation(),
            dense_results=(
                hybrid
            ),
            hybrid_results=(
                hybrid
            ),
            reranked_results=(
                make_reranked(
                    hybrid
                )
            ),
        )
    )

    changed = (
        make_translation()
    )

    changed[
        "translated_query"
    ] = (
        "A different English translation."
    )

    with pytest.raises(
        ValueError,
        match=(
            "automatic_query"
        ),
    ):
        validate_persisted_benchmark_row(
            row,
            record,
            changed,
        )


def test_aggregate_point_id_metrics() -> None:
    """Aggregate metrics should operate over persisted automatic rankings."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
        make_record(
            question_id="ne_en_002"
        ),
    ]

    rows = {
        "ne_en_001": {
            "automatic_hybrid_point_ids": [
                "primary",
                "supporting",
            ],
        },
        "ne_en_002": {
            "automatic_hybrid_point_ids": [
                "other",
                "primary",
            ],
        },
    }

    metrics = (
        aggregate_point_id_metrics(
            records,
            rows,
            ranking_field=(
                "automatic_hybrid_point_ids"
            ),
            cutoff=2,
        )
    )

    assert (
        metrics[
            "hit_rate"
        ]
        == 1.0
    )

    assert (
        metrics[
            "mrr"
        ]
        == pytest.approx(
            0.75
        )
    )

    assert (
        metrics[
            "recall"
        ]
        == pytest.approx(
            0.75
        )
    )


def test_build_summary_requires_all_questions() -> None:
    """The summary should not silently describe an incomplete hosted run."""

    records = [
        make_record(
            question_id="ne_en_001"
        ),
        make_record(
            question_id="ne_en_002"
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "incomplete"
        ),
    ):
        build_summary(
            records,
            {},
            make_baseline(
                records
            ),
        )