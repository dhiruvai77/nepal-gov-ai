"""Tests for adjacent-chunk redundancy-aware context selection."""

import pytest

from src.context_selection.adjacent_chunk import (
    DEFAULT_CONTEXT_COUNT,
    AdjacentChunkContextSelector,
)
from src.reranking.base import (
    RerankedResult,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
    *,
    document_id: str = "document-a",
    chunk_index: int | None,
    rerank_score: float,
    original_rank: int,
) -> RerankedResult:
    """Create one deterministic reranked result."""

    result = RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"{document_id}-{point_id}",
        document_id=document_id,
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document",
        chunk_text=f"Evidence passage {point_id}.",
        chunk_index=chunk_index,
        token_count=300,
    )

    return RerankedResult(
        result=result,
        rerank_score=rerank_score,
        original_rank=original_rank,
    )


def test_default_context_count_is_five() -> None:
    """Redundancy-aware selection should use the top-5 baseline by default."""

    assert (
        DEFAULT_CONTEXT_COUNT
        == 5
    )


def test_selector_skips_adjacent_chunk_and_fills_from_lower_rank() -> None:
    """Adjacent overlap should be replaced by the next eligible candidate."""

    candidates = [
        make_result(
            "a",
            chunk_index=10,
            rerank_score=0.90,
            original_rank=1,
        ),
        make_result(
            "b",
            chunk_index=11,
            rerank_score=0.85,
            original_rank=2,
        ),
        make_result(
            "c",
            chunk_index=15,
            rerank_score=0.80,
            original_rank=3,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=2
        )
    )

    selected = selector.select(
        candidates
    )

    assert [
        item.result.point_id
        for item in selected
    ] == [
        "a",
        "c",
    ]


def test_selector_preserves_reranker_order() -> None:
    """The selector should filter candidates rather than rerank them."""

    candidates = [
        make_result(
            "a",
            chunk_index=2,
            rerank_score=0.9,
            original_rank=3,
        ),
        make_result(
            "b",
            chunk_index=7,
            rerank_score=0.8,
            original_rank=1,
        ),
        make_result(
            "c",
            chunk_index=12,
            rerank_score=0.7,
            original_rank=2,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=3
        )
    )

    assert [
        item.result.point_id
        for item in selector.select(
            candidates
        )
    ] == [
        "a",
        "b",
        "c",
    ]


def test_selector_allows_same_index_from_different_documents() -> None:
    """Chunk positions are meaningful only within their source document."""

    candidates = [
        make_result(
            "a",
            document_id="document-a",
            chunk_index=10,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            document_id="document-b",
            chunk_index=10,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=2
        )
    )

    assert len(
        selector.select(
            candidates
        )
    ) == 2


def test_selector_allows_nonadjacent_chunks_from_same_document() -> None:
    """Separate portions of one document may provide complementary evidence."""

    candidates = [
        make_result(
            "a",
            chunk_index=10,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            chunk_index=12,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=2
        )
    )

    assert len(
        selector.select(
            candidates
        )
    ) == 2


def test_selector_keeps_candidate_with_missing_chunk_index() -> None:
    """Unknown adjacency should not cause potentially useful evidence removal."""

    candidates = [
        make_result(
            "a",
            chunk_index=10,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            chunk_index=None,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=2
        )
    )

    assert len(
        selector.select(
            candidates
        )
    ) == 2


def test_selector_handles_empty_pool() -> None:
    """No reranked candidates should produce no context."""

    selector = (
        AdjacentChunkContextSelector()
    )

    assert (
        selector.select([])
        == []
    )


def test_selector_returns_available_nonredundant_candidates() -> None:
    """The selector may return fewer than k when all remaining chunks overlap."""

    candidates = [
        make_result(
            "a",
            chunk_index=10,
            rerank_score=0.9,
            original_rank=1,
        ),
        make_result(
            "b",
            chunk_index=11,
            rerank_score=0.8,
            original_rank=2,
        ),
    ]

    selector = (
        AdjacentChunkContextSelector(
            top_k=5
        )
    )

    selected = selector.select(
        candidates
    )

    assert len(
        selected
    ) == 1

    assert (
        selected[0].result.point_id
        == "a"
    )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_selector_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """The requested context size must be positive."""

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        AdjacentChunkContextSelector(
            top_k=top_k
        )