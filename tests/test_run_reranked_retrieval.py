"""Tests for the production retrieval-and-reranking entry point."""

from unittest.mock import (
    Mock,
    patch,
)

import pytest

from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)
from src.retrieval.run_reranked_retrieval import (
    DEFAULT_RERANK_CANDIDATE_COUNT,
    run_reranked_retrieval,
)


def make_result(
    point_id: str,
) -> RetrievalResult:
    """Create one normalized retrieval result for orchestration tests."""

    return RetrievalResult(
        point_id=point_id,
        score=0.031,
        chunk_id=f"chunk-{point_id}",
        document_id="constitution_nepal_current_en",
        title="Constitution of Nepal",
        organization="Nepal Law Commission",
        language="en",
        page_start=16,
        page_end=16,
        source_url="https://example.gov.np/constitution",
        chunk_text=(
            "Every citizen shall have the right "
            "to education."
        ),
        category="constitution_law",
        document_type="constitution",
    )


class FakeReranker(
    Reranker
):
    """Deterministic reranker used to verify production orchestration."""

    def __init__(
        self,
    ) -> None:
        self.query: str | None = None
        self.candidates: list[
            RetrievalResult
        ] = []
        self.top_k: int | None = None

    def rerank(
        self,
        query,
        candidates,
        *,
        top_k=None,
    ):
        self.query = query
        self.candidates = list(
            candidates
        )
        self.top_k = top_k

        ranked = [
            RerankedResult(
                result=candidate,
                rerank_score=float(
                    len(self.candidates)
                    - index
                ),
                original_rank=index + 1,
            )
            for index, candidate
            in enumerate(
                self.candidates
            )
        ]

        if top_k is None:
            return ranked

        return ranked[
            :top_k
        ]


def test_run_reranked_retrieval_connects_first_stage_and_reranker() -> None:
    """Production wrapper should rerank the exact retrieved candidate pool."""

    candidates = [
        make_result(
            "point-1"
        ),
        make_result(
            "point-2"
        ),
    ]

    reranker = FakeReranker()

    embedding_service = Mock()
    client = Mock()

    with patch(
        "src.retrieval.run_reranked_retrieval.run_hybrid_retrieval",
        return_value=candidates,
    ) as retrieval_mock:
        results = run_reranked_retrieval(
            "What education rights are guaranteed?",
            candidate_count=20,
            top_k=1,
            filters={
                "language": "en",
            },
            embedding_service=embedding_service,
            client=client,
            reranker=reranker,
        )

    retrieval_mock.assert_called_once_with(
        query=(
            "What education rights are guaranteed?"
        ),
        top_k=20,
        filters={
            "language": "en",
        },
        embedding_service=embedding_service,
        client=client,
    )

    assert (
        reranker.query
        == "What education rights are guaranteed?"
    )

    assert (
        reranker.candidates
        == candidates
    )

    assert (
        reranker.top_k
        == 1
    )

    assert len(
        results
    ) == 1

    assert (
        results[0].result
        is candidates[0]
    )


def test_default_candidate_count_matches_evaluated_depth() -> None:
    """Production should default to the validated top-20 candidate pool."""

    assert (
        DEFAULT_RERANK_CANDIDATE_COUNT
        == 20
    )


def test_run_reranked_retrieval_returns_all_candidates_by_default() -> None:
    """Context selection should receive the full reranked candidate pool."""

    candidates = [
        make_result(
            "point-1"
        ),
        make_result(
            "point-2"
        ),
    ]

    reranker = FakeReranker()

    with patch(
        "src.retrieval.run_reranked_retrieval.run_hybrid_retrieval",
        return_value=candidates,
    ):
        results = run_reranked_retrieval(
            "education rights",
            reranker=reranker,
        )

    assert len(
        results
    ) == 2

    assert (
        reranker.top_k
        is None
    )


def test_run_reranked_retrieval_skips_reranker_when_no_candidates() -> None:
    """An empty first-stage result should not call hosted inference."""

    reranker = Mock(
        spec=Reranker
    )

    with patch(
        "src.retrieval.run_reranked_retrieval.run_hybrid_retrieval",
        return_value=[],
    ):
        results = run_reranked_retrieval(
            "education rights",
            reranker=reranker,
        )

    assert results == []

    reranker.rerank.assert_not_called()


@pytest.mark.parametrize(
    "candidate_count",
    [
        0,
        -1,
    ],
)
def test_run_reranked_retrieval_rejects_invalid_candidate_count(
    candidate_count: int,
) -> None:
    """First-stage candidate depth must always be positive."""

    with pytest.raises(
        ValueError,
        match="candidate_count must be greater than zero",
    ):
        run_reranked_retrieval(
            "education rights",
            candidate_count=candidate_count,
            reranker=FakeReranker(),
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_run_reranked_retrieval_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """Optional output truncation must use a positive cutoff."""

    with pytest.raises(
        ValueError,
        match=(
            "top_k must be greater than zero "
            "when provided"
        ),
    ):
        run_reranked_retrieval(
            "education rights",
            top_k=top_k,
            reranker=FakeReranker(),
        )