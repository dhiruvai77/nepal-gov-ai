"""Tests for the targeted NE->EN candidate-pool reranker benchmark."""

from __future__ import annotations

import pytest

from src.evaluation.compare_ne_en_reranked_variants import (
    calculate_metrics,
    first_primary_rank,
    format_rank,
    reranked_results_only,
    validate_candidate_identity,
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
        title="Title",
        organization="Organization",
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


def make_record() -> EvaluationRecord:
    """Build one deterministic NE->EN evaluation record."""

    return EvaluationRecord(
        question_id="ne_en_test",
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


def test_validate_candidate_identity_accepts_reordering() -> None:
    """Reranking may reorder candidates without changing membership."""

    candidates = [
        make_result(
            "a"
        ),
        make_result(
            "b"
        ),
    ]

    reranked = [
        RerankedResult(
            result=candidates[
                1
            ],
            rerank_score=0.9,
            original_rank=2,
        ),
        RerankedResult(
            result=candidates[
                0
            ],
            rerank_score=0.8,
            original_rank=1,
        ),
    ]

    validate_candidate_identity(
        candidates,
        reranked,
    )


def test_validate_candidate_identity_rejects_count_change() -> None:
    """Dropping candidates should fail benchmark validation."""

    candidates = [
        make_result(
            "a"
        ),
        make_result(
            "b"
        ),
    ]

    reranked = [
        RerankedResult(
            result=candidates[
                0
            ],
            rerank_score=0.9,
            original_rank=1,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            "changed candidate count"
        ),
    ):
        validate_candidate_identity(
            candidates,
            reranked,
        )


def test_validate_candidate_identity_rejects_identity_change() -> None:
    """Reranking must not substitute unseen candidates."""

    candidates = [
        make_result(
            "a"
        ),
        make_result(
            "b"
        ),
    ]

    reranked = [
        RerankedResult(
            result=make_result(
                "a"
            ),
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=make_result(
                "c"
            ),
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            "changed candidate identity"
        ),
    ):
        validate_candidate_identity(
            candidates,
            reranked,
        )


def test_validate_candidate_identity_rejects_duplicates() -> None:
    """Duplicate reranker output must fail explicitly."""

    candidates = [
        make_result(
            "a"
        ),
        make_result(
            "b"
        ),
    ]

    reranked = [
        RerankedResult(
            result=make_result(
                "a"
            ),
            rerank_score=0.9,
            original_rank=1,
        ),
        RerankedResult(
            result=make_result(
                "a"
            ),
            rerank_score=0.8,
            original_rank=1,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            "changed candidate identity"
        ),
    ):
        validate_candidate_identity(
            candidates,
            reranked,
        )


def test_reranked_results_only_preserves_order() -> None:
    """The extraction helper should preserve reranker order exactly."""

    first = (
        make_result(
            "first"
        )
    )

    second = (
        make_result(
            "second"
        )
    )

    items = [
        RerankedResult(
            result=second,
            rerank_score=0.9,
            original_rank=2,
        ),
        RerankedResult(
            result=first,
            rerank_score=0.8,
            original_rank=1,
        ),
    ]

    assert [
        item.point_id
        for item in (
            reranked_results_only(
                items
            )
        )
    ] == [
        "second",
        "first",
    ]


def test_first_primary_rank_returns_expected_rank() -> None:
    """Primary evidence rank should follow benchmark annotations."""

    record = (
        make_record()
    )

    results = [
        make_result(
            "irrelevant"
        ),
        make_result(
            "primary"
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
    """Missing primary evidence should remain explicit."""

    assert (
        first_primary_rank(
            make_record(),
            [
                make_result(
                    "irrelevant"
                )
            ],
        )
        is None
    )


def test_format_rank_handles_missing_evidence() -> None:
    """Rank output should distinguish misses from valid ranks."""

    assert (
        format_rank(
            3,
            depth=20,
        )
        == "3"
    )

    assert (
        format_rank(
            None,
            depth=20,
        )
        == ">20"
    )


def test_calculate_metrics_uses_existing_evaluator() -> None:
    """Metric calculation should preserve the project evaluation contract."""

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

    assert (
        len(
            metrics
        )
        == 1
    )

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