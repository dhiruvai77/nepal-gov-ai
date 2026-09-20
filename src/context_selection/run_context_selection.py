"""Production context-selection entry point for NepalGov AI.

This module connects the validated retrieval-and-reranking pipeline to the
selected generation-context strategy.

The production pipeline retrieves 20 first-stage candidates, reranks the full
candidate pool with the hosted multilingual BGE reranker, and supplies the
highest-ranked five passages to downstream generation.

Fixed top-5 was selected after comparing fixed context sizes, token budgets,
and adjacent-chunk redundancy suppression on the multilingual retrieval
evaluation set.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from src.context_selection.base import (
    ContextSelector,
)
from src.context_selection.fixed_top_k import (
    FixedTopKContextSelector,
)
from src.embeddings.base import (
    EmbeddingService,
)
from src.reranking.base import (
    RerankedResult,
    Reranker,
)
from src.retrieval.run_reranked_retrieval import (
    DEFAULT_RERANK_CANDIDATE_COUNT,
    run_reranked_retrieval,
)


DEFAULT_CONTEXT_COUNT = 5


def run_context_selection(
    query: str,
    *,
    candidate_count: int = DEFAULT_RERANK_CANDIDATE_COUNT,
    filters: dict[str, str] | None = None,
    embedding_service: EmbeddingService | None = None,
    client: QdrantClient | None = None,
    reranker: Reranker | None = None,
    selector: ContextSelector | None = None,
) -> list[RerankedResult]:
    """Run production retrieval, reranking, and context selection.

    The reranker always receives the complete first-stage candidate pool.
    Context truncation happens only after reranking so retrieval, reranking,
    and context selection remain independently testable pipeline stages.

    The default selector is fixed top-5, the strategy selected by the measured
    multilingual context-selection benchmark.

    Dependencies remain injectable for tests and future application layers.
    """

    active_selector = (
        selector
        if selector is not None
        else FixedTopKContextSelector(
            top_k=DEFAULT_CONTEXT_COUNT
        )
    )

    reranked_candidates = (
        run_reranked_retrieval(
            query=query,
            candidate_count=candidate_count,
            top_k=None,
            filters=filters,
            embedding_service=embedding_service,
            client=client,
            reranker=reranker,
        )
    )

    if not reranked_candidates:
        return []

    return active_selector.select(
        reranked_candidates
    )


def print_selected_context(
    results: list[
        RerankedResult
    ],
) -> None:
    """Print selected evidence with ranking and citation diagnostics."""

    if not results:
        print(
            "No context passages selected."
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
            f"    chunk_id={result.chunk_id}"
        )

        print(
            f"    chunk_index={result.chunk_index}"
        )

        print(
            f"    token_count={result.token_count}"
        )

        print(
            f"    source={result.source_url}"
        )

        print(
            f"    text={result.chunk_text}"
        )


def main() -> None:
    """Run one interactive production context-selection query."""

    query = input(
        "Enter a Nepal government question: "
    ).strip()

    results = run_context_selection(
        query=query,
    )

    print_selected_context(
        results
    )


if __name__ == "__main__":
    main()