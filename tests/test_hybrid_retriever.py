"""Unit tests for Reciprocal Rank Fusion hybrid retrieval."""

from src.retrieval.dense_retriever import (
    RetrievalResult,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)


def make_result(
    point_id: str,
    score: float,
) -> RetrievalResult:
    """Create a minimal normalized retrieval result for fusion tests."""

    return RetrievalResult(
        point_id=point_id,
        score=score,
        chunk_id=f"chunk-{point_id}",
        document_id="doc-1",
        title="Test Document",
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/document",
        chunk_text=f"Text for {point_id}",
    )


class FakeRetriever:
    """Return a deterministic ranking without contacting Qdrant."""

    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.calls: list[
            tuple[
                str,
                int,
                dict[str, str] | None,
            ]
        ] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Record the call and return the requested result prefix."""

        self.calls.append(
            (
                query,
                top_k,
                filters,
            )
        )

        return self.results[
            :top_k
        ]


def test_hybrid_retrieval_rewards_cross_retriever_agreement() -> None:
    """A point found by both retrievers should outrank isolated candidates."""

    dense = FakeRetriever(
        [
            make_result(
                "dense-only",
                0.95,
            ),
            make_result(
                "shared",
                0.90,
            ),
        ]
    )

    sparse = FakeRetriever(
        [
            make_result(
                "sparse-only",
                20.0,
            ),
            make_result(
                "shared",
                18.0,
            ),
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    results = retriever.retrieve(
        "education rights",
        top_k=3,
    )

    assert (
        results[0].point_id
        == "shared"
    )


def test_hybrid_retrieval_uses_rank_not_raw_scores() -> None:
    """RRF must ignore incompatible dense and BM25 score magnitudes."""

    dense = FakeRetriever(
        [
            make_result(
                "dense-first",
                0.91,
            ),
        ]
    )

    sparse = FakeRetriever(
        [
            make_result(
                "sparse-first",
                150.0,
            ),
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    results = retriever.retrieve(
        "budget",
        top_k=2,
    )

    # Both rank first in their individual lists, so both receive the same RRF
    # contribution even though their original score scales are very different.
    assert (
        results[0].score
        == results[1].score
    )


def test_hybrid_retrieval_passes_filters_to_both_retrievers() -> None:
    """Dense and sparse retrieval must receive identical metadata filters."""

    dense = FakeRetriever(
        [
            make_result(
                "a",
                0.9,
            )
        ]
    )

    sparse = FakeRetriever(
        [
            make_result(
                "b",
                10.0,
            )
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    filters = {
        "language": "en",
        "document_type": "constitution",
    }

    retriever.retrieve(
        "fundamental rights",
        top_k=4,
        filters=filters,
    )

    # The default multiplier is three, so top_k=4 requests twelve candidates
    # from each component retriever before fusion.
    assert dense.calls == [
        (
            "fundamental rights",
            12,
            filters,
        )
    ]

    assert sparse.calls == [
        (
            "fundamental rights",
            12,
            filters,
        )
    ]


def test_hybrid_retrieval_limits_final_result_count() -> None:
    """Fusion should return at most the requested number of results."""

    dense = FakeRetriever(
        [
            make_result(
                "a",
                0.9,
            ),
            make_result(
                "b",
                0.8,
            ),
            make_result(
                "c",
                0.7,
            ),
        ]
    )

    sparse = FakeRetriever(
        [
            make_result(
                "d",
                12.0,
            ),
            make_result(
                "e",
                11.0,
            ),
            make_result(
                "f",
                10.0,
            ),
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    results = retriever.retrieve(
        "test",
        top_k=2,
    )

    assert len(
        results
    ) == 2


def test_hybrid_retrieval_rejects_blank_query() -> None:
    """Blank hybrid queries should fail before component retrieval."""

    retriever = HybridRetriever(
        dense_retriever=FakeRetriever(
            []
        ),
        sparse_retriever=FakeRetriever(
            []
        ),
    )

    try:
        retriever.retrieve(
            "   "
        )

    except ValueError as exc:
        assert (
            "query must contain non-whitespace text"
            in str(exc)
        )

    else:
        raise AssertionError(
            "Expected ValueError for blank query."
        )


def test_hybrid_retrieval_rejects_invalid_top_k() -> None:
    """Hybrid retrieval requires a positive final result count."""

    retriever = HybridRetriever(
        dense_retriever=FakeRetriever(
            []
        ),
        sparse_retriever=FakeRetriever(
            []
        ),
    )

    try:
        retriever.retrieve(
            "budget",
            top_k=0,
        )

    except ValueError as exc:
        assert (
            "top_k must be greater than zero"
            in str(exc)
        )

    else:
        raise AssertionError(
            "Expected ValueError for invalid top_k."
        )


def test_hybrid_retriever_rejects_negative_rrf_k() -> None:
    """RRF rank constant cannot be negative."""

    try:
        HybridRetriever(
            dense_retriever=FakeRetriever(
                []
            ),
            sparse_retriever=FakeRetriever(
                []
            ),
            rrf_k=-1,
        )

    except ValueError as exc:
        assert (
            "rrf_k must be zero or greater"
            in str(exc)
        )

    else:
        raise AssertionError(
            "Expected ValueError for negative rrf_k."
        )


def test_hybrid_retriever_rejects_invalid_candidate_multiplier() -> None:
    """Candidate expansion must request at least one ranking depth."""

    try:
        HybridRetriever(
            dense_retriever=FakeRetriever(
                []
            ),
            sparse_retriever=FakeRetriever(
                []
            ),
            candidate_multiplier=0,
        )

    except ValueError as exc:
        assert (
            "candidate_multiplier must be greater than zero"
            in str(exc)
        )

    else:
        raise AssertionError(
            "Expected ValueError for invalid candidate multiplier."
        )