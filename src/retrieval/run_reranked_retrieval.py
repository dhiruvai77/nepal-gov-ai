"""Production entry point for retrieval followed by BGE reranking.

This module preserves first-stage retrieval as an independent component and
adds the evaluated multilingual reranker as a separate second-stage ranking
step.

The default pipeline retrieves 20 candidates using the selected production
retrieval architecture and reranks that fixed candidate pool with hosted
BAAI/bge-reranker-v2-m3.

Final context selection is intentionally not performed here. A later context
selection layer will decide how many reranked passages should be supplied to
the generation model.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from src.embeddings.base import (
    EmbeddingService,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.reranking.hf_bge_reranker import (
    HuggingFaceBGEReranker,
)
from src.retrieval.run_hybrid_retrieval import (
    run_hybrid_retrieval,
)


DEFAULT_RERANK_CANDIDATE_COUNT = 20


def run_reranked_retrieval(
    query: str,
    *,
    candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
    reranker: Reranker | None = None,
) -> list[RerankedResult]:
    """Run production first-stage retrieval followed by BGE reranking.

    `candidate_count` controls only the size of the first-stage candidate pool.
    The production default of 20 matches the depth used in the validated
    reranker benchmark.

    `top_k` optionally truncates the reranked output. Its default is None so
    the complete reranked candidate pool remains available to the upcoming
    context-selection layer.

    Retrieval and reranker dependencies can be injected by tests and future
    application layers without changing production defaults.
    """

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be greater than zero."
        )

    if (
        top_k is not None
        and top_k <= 0
    ):
        raise ValueError(
            "top_k must be greater than zero when provided."
        )

    candidates = run_hybrid_retrieval(
        query=query,
        top_k=candidate_count,
        filters=filters,
        embedding_service=embedding_service,
        client=client,
    )

    # An empty first-stage result is a valid retrieval outcome. Avoid sending
    # an invalid empty candidate list to the hosted reranker in that case.
    if not candidates:
        return []

    active_reranker = (
        reranker
        if reranker is not None
        else HuggingFaceBGEReranker()
    )

    return active_reranker.rerank(
        query,
        candidates,
        top_k=top_k,
    )


def print_reranked_results(
    results: list[RerankedResult],
) -> None:
    """Print reranked candidates with ranking and citation diagnostics."""

    if not results:
        print(
            "No reranked retrieval results found."
        )
        return

    for rank, item in enumerate(
        results,
        start=1,
    ):
        result = item.result

        print(
            f"[{rank}] "
            f"rerank_score={item.rerank_score:.6f} "
            f"original_rank={item.original_rank} "
            f"title={result.title} "
            f"pages={result.page_start}-{result.page_end}"
        )

        print(
            f"    first_stage_score={result.score:.6f}"
        )

        print(
            f"    chunk_id={result.chunk_id}"
        )

        print(
            f"    source={result.source_url}"
        )

        print(
            f"    text={result.chunk_text}"
        )


def main() -> None:
    """Run one interactive production retrieval-and-reranking query."""

    query = input(
        "Enter a Nepal government question: "
    ).strip()

    results = run_reranked_retrieval(
        query=query,
    )

    print_reranked_results(
        results
    )


if __name__ == "__main__":
    main()