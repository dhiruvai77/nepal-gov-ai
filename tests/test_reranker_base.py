"""Unit tests for the provider-independent reranking contract."""

import pytest

from src.reranking.base import (
    RerankedResult,
    select_top_reranked_results,
    validate_rerank_request,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
) -> RetrievalResult:
    """Create a minimal retrieval result for reranking tests."""

    return RetrievalResult(
        point_id=point_id,
        score=0.8,
        chunk_id=f"chunk-{point_id}",
        document_id="test-document",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document.pdf",
        chunk_text="Example government document passage.",
        category="test",
        document_type="report",
    )


def test_validate_rerank_request_returns_trimmed_query() -> None:
    """Valid reranking requests should normalize surrounding whitespace."""

    candidates = [
        make_result(
            "point-1"
        ),
    ]

    result = validate_rerank_request(
        "  education policy  ",
        candidates,
        top_k=5,
    )

    assert (
        result
        == "education policy"
    )


def test_validate_rerank_request_rejects_blank_query() -> None:
    """Blank queries should fail before reaching a reranker provider."""

    candidates = [
        make_result(
            "point-1"
        ),
    ]

    with pytest.raises(
        ValueError,
        match="query must contain non-whitespace text",
    ):
        validate_rerank_request(
            "   ",
            candidates,
            top_k=5,
        )


def test_validate_rerank_request_rejects_empty_candidates() -> None:
    """Reranking an empty candidate set is a caller error."""

    with pytest.raises(
        ValueError,
        match="at least one retrieval candidate",
    ):
        validate_rerank_request(
            "education policy",
            [],
            top_k=5,
        )


def test_validate_rerank_request_rejects_invalid_top_k() -> None:
    """The requested output count must be positive."""

    candidates = [
        make_result(
            "point-1"
        ),
    ]

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        validate_rerank_request(
            "education policy",
            candidates,
            top_k=0,
        )


def test_select_top_reranked_results_orders_by_score() -> None:
    """Higher reranker relevance scores should rank first."""

    first = RerankedResult(
        result=make_result(
            "point-1"
        ),
        rerank_score=0.42,
        original_rank=1,
    )

    second = RerankedResult(
        result=make_result(
            "point-2"
        ),
        rerank_score=0.91,
        original_rank=2,
    )

    ranked = select_top_reranked_results(
        [
            first,
            second,
        ],
        top_k=None,
    )

    assert [
        item.result.point_id
        for item in ranked
    ] == [
        "point-2",
        "point-1",
    ]


def test_select_top_reranked_results_applies_top_k() -> None:
    """The reranker output should support a final candidate cutoff."""

    results = [
        RerankedResult(
            result=make_result(
                f"point-{index}"
            ),
            rerank_score=float(
                index
            ),
            original_rank=index,
        )
        for index in range(
            1,
            4,
        )
    ]

    ranked = select_top_reranked_results(
        results,
        top_k=2,
    )

    assert [
        item.result.point_id
        for item in ranked
    ] == [
        "point-3",
        "point-2",
    ]


def test_select_top_reranked_results_uses_original_rank_for_ties() -> None:
    """Equal reranker scores should preserve first-stage ranking order."""

    earlier = RerankedResult(
        result=make_result(
            "point-a"
        ),
        rerank_score=0.75,
        original_rank=2,
    )

    later = RerankedResult(
        result=make_result(
            "point-b"
        ),
        rerank_score=0.75,
        original_rank=5,
    )

    ranked = select_top_reranked_results(
        [
            later,
            earlier,
        ],
        top_k=None,
    )

    assert [
        item.result.point_id
        for item in ranked
    ] == [
        "point-a",
        "point-b",
    ]