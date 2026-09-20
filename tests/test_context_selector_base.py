"""Tests for context-selection interfaces and fixed top-k selection."""

import pytest

from src.context_selection.base import (
    ContextSelector,
)
from src.context_selection.fixed_top_k import (
    DEFAULT_CONTEXT_COUNT,
    FixedTopKContextSelector,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_reranked_result(
    point_id: str,
    *,
    score: float,
    original_rank: int,
    chunk_index: int,
    token_count: int,
) -> RerankedResult:
    """Create one deterministic reranked result for selector tests."""

    result = RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"document_chunk_{chunk_index:05d}",
        document_id="test-document",
        title="Test Government Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document",
        chunk_text=f"Evidence passage {point_id}.",
        chunk_index=chunk_index,
        token_count=token_count,
    )

    return RerankedResult(
        result=result,
        rerank_score=score,
        original_rank=original_rank,
    )


def test_context_selector_is_abstract() -> None:
    """The common context-selector contract should not be instantiated."""

    with pytest.raises(
        TypeError,
    ):
        ContextSelector()


def test_default_context_count_is_five() -> None:
    """The initial evaluated fixed-context baseline should use five passages."""

    assert (
        DEFAULT_CONTEXT_COUNT
        == 5
    )


def test_fixed_top_k_preserves_reranker_order() -> None:
    """Selection must not silently rerank the candidates again."""

    candidates = [
        make_reranked_result(
            "point-1",
            score=0.90,
            original_rank=4,
            chunk_index=3,
            token_count=300,
        ),
        make_reranked_result(
            "point-2",
            score=0.80,
            original_rank=1,
            chunk_index=8,
            token_count=250,
        ),
        make_reranked_result(
            "point-3",
            score=0.70,
            original_rank=2,
            chunk_index=9,
            token_count=200,
        ),
    ]

    selector = FixedTopKContextSelector(
        top_k=2
    )

    selected = selector.select(
        candidates
    )

    assert [
        item.result.point_id
        for item in selected
    ] == [
        "point-1",
        "point-2",
    ]


def test_fixed_top_k_returns_all_when_pool_is_smaller() -> None:
    """A short candidate pool should not cause an error."""

    candidates = [
        make_reranked_result(
            "point-1",
            score=0.90,
            original_rank=1,
            chunk_index=1,
            token_count=200,
        ),
    ]

    selector = FixedTopKContextSelector(
        top_k=5
    )

    assert (
        selector.select(
            candidates
        )
        == candidates
    )


def test_fixed_top_k_handles_empty_pool() -> None:
    """No retrieved evidence should produce no selected context."""

    selector = (
        FixedTopKContextSelector()
    )

    assert (
        selector.select([])
        == []
    )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_fixed_top_k_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """A context count must always be positive."""

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        FixedTopKContextSelector(
            top_k=top_k
        )