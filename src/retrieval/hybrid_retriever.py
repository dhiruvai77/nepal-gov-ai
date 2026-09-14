"""Hybrid dense + sparse retrieval using Reciprocal Rank Fusion.

This module combines the independently ranked results from DenseRetriever and
SparseRetriever. RRF operates on rank positions rather than raw similarity
scores, which is important because dense cosine scores and BM25 scores are not
directly comparable.
"""

from dataclasses import replace
from typing import Protocol

from src.retrieval.dense_retriever import (
    RetrievalResult,
)


DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 3


class Retriever(Protocol):
    """Minimal retrieval contract required by the hybrid fusion layer."""

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked retrieval results for one query."""


class HybridRetriever:
    """Fuse dense semantic and sparse lexical retrieval with RRF."""

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        """Configure the retrievers and rank-fusion parameters."""

        if rrf_k < 0:
            raise ValueError(
                "rrf_k must be zero or greater."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than zero."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = (
            candidate_multiplier
        )

    def _rrf_score(
        self,
        rank: int,
    ) -> float:
        """Calculate one Reciprocal Rank Fusion contribution."""

        # Ranks are one-based. The conventional RRF constant of 60 reduces the
        # effect of small rank differences while still rewarding agreement
        # between independent retrieval systems.
        return 1.0 / (
            self.rrf_k + rank
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates from both paths and return their RRF ranking."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "query must contain non-whitespace text."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Retrieve more candidates than the final requested count so documents
        # ranked moderately by both systems still have a chance to rise after
        # reciprocal-rank fusion.
        candidate_count = (
            top_k
            * self.candidate_multiplier
        )

        dense_results = (
            self.dense_retriever.retrieve(
                clean_query,
                top_k=candidate_count,
                filters=filters,
            )
        )

        sparse_results = (
            self.sparse_retriever.retrieve(
                clean_query,
                top_k=candidate_count,
                filters=filters,
            )
        )

        # point_id is the stable Qdrant identifier shared by both retrieval
        # paths, so it is the safest key for deduplicating the candidate sets.
        results_by_id: dict[
            str,
            RetrievalResult,
        ] = {}

        fused_scores: dict[
            str,
            float,
        ] = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            results_by_id[
                result.point_id
            ] = result

            fused_scores[
                result.point_id
            ] = (
                fused_scores.get(
                    result.point_id,
                    0.0,
                )
                + self._rrf_score(
                    rank
                )
            )

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            # Preserve one normalized result payload even when the same point
            # appears in both lists. Metadata should be identical because both
            # retrievers query the same Qdrant point.
            results_by_id.setdefault(
                result.point_id,
                result,
            )

            fused_scores[
                result.point_id
            ] = (
                fused_scores.get(
                    result.point_id,
                    0.0,
                )
                + self._rrf_score(
                    rank
                )
            )

        ranked_point_ids = sorted(
            fused_scores,
            key=lambda point_id: (
                -fused_scores[point_id],
                point_id,
            ),
        )

        fused_results: list[
            RetrievalResult
        ] = []

        for point_id in ranked_point_ids[
            :top_k
        ]:
            # Reuse RetrievalResult for now and place the fused RRF score in its
            # score field. A richer retrieval trace can be added later when we
            # introduce evaluation and reranking diagnostics.
            fused_results.append(
                replace(
                    results_by_id[
                        point_id
                    ],
                    score=fused_scores[
                        point_id
                    ],
                )
            )

        return fused_results