"""Provider-independent reranking interfaces for NepalGov AI.

The reranking layer receives candidates from first-stage retrieval and assigns
query-dependent relevance scores without changing their citation metadata.

Keeping the interface provider-independent allows the project to benchmark
different hosted or local reranker implementations without changing the RAG
pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from src.retrieval.dense_retriever import (
    RetrievalResult,
)


@dataclass(frozen=True)
class RerankedResult:
    """One retrieval candidate after cross-encoder reranking.

    `result` preserves the original retrieval/citation metadata.

    `rerank_score` is the score produced by the reranking model.

    `original_rank` records the candidate's position before reranking, which is
    useful for evaluation and retrieval diagnostics.
    """

    result: RetrievalResult
    rerank_score: float
    original_rank: int


class Reranker(ABC):
    """Abstract interface implemented by concrete reranker providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Rerank retrieved candidates for one query.

        Implementations must return candidates ordered from highest to lowest
        reranker relevance score.

        When `top_k` is None, all candidates should be returned.
        """


def validate_rerank_request(
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    top_k: int | None,
) -> str:
    """Validate common reranking inputs and return the normalized query.

    Centralizing validation keeps hosted and future local providers consistent
    and prevents malformed requests from reaching external inference services.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "query must contain non-whitespace text."
        )

    if top_k is not None and top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero when provided."
        )

    if not candidates:
        raise ValueError(
            "at least one retrieval candidate is required."
        )

    return clean_query


def select_top_reranked_results(
    results: Sequence[RerankedResult],
    *,
    top_k: int | None,
) -> list[RerankedResult]:
    """Sort reranker output deterministically and apply the requested cutoff.

    Original rank is used as the tie-breaker so identical reranker scores do
    not cause unstable result ordering between repeated evaluations.
    """

    ordered = sorted(
        results,
        key=lambda item: (
            -item.rerank_score,
            item.original_rank,
            item.result.point_id,
        ),
    )

    if top_k is None:
        return ordered

    return ordered[
        :top_k
    ]