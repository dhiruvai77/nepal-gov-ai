"""Tests for document-title-aware reranker input experiments."""

from src.evaluation.compare_reranker_inputs import (
    TitleAwareReranker,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.retrieval.dense_retriever import (
    RetrievalResult,
)


def make_result(
    point_id: str,
    title: str,
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        point_id=point_id,
        score=0.5,
        chunk_id=f"chunk-{point_id}",
        document_id="test-document",
        title=title,
        organization="Government of Nepal",
        language="en",
        page_start=1,
        page_end=1,
        source_url="https://example.gov.np/test.pdf",
        chunk_text=text,
    )


class CapturingReranker(
    Reranker
):
    """Fake reranker that records the exact candidate text it receives."""

    def __init__(
        self,
    ) -> None:
        self.received = []

    def rerank(
        self,
        query,
        candidates,
        *,
        top_k=None,
    ):
        candidates = list(
            candidates
        )

        self.received = candidates

        ranked = [
            RerankedResult(
                result=candidate,
                rerank_score=float(
                    len(candidates) - index
                ),
                original_rank=index + 1,
            )
            for index, candidate
            in enumerate(
                candidates
            )
        ]

        if top_k is None:
            return ranked

        return ranked[
            :top_k
        ]


def test_title_aware_reranker_prefixes_title_but_returns_original() -> None:
    candidate = make_result(
        "point-1",
        "Constitution of Nepal",
        "Every citizen shall have the right to health.",
    )

    inner = CapturingReranker()

    reranker = TitleAwareReranker(
        inner
    )

    results = reranker.rerank(
        "What does the Constitution guarantee?",
        [
            candidate
        ],
    )

    assert (
        inner.received[0].chunk_text
        == (
            "Document: Constitution of Nepal\n\n"
            "Every citizen shall have the right to health."
        )
    )

    # The experiment changes only cross-encoder input. Downstream retrieval
    # payloads must retain their original source passage.
    assert (
        results[0].result.chunk_text
        == candidate.chunk_text
    )

    assert (
        results[0].result
        is candidate
    )